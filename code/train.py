# code/train.py
import sys
import os


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pickle
import numpy as np
import pandas as pd

from code.mlp_model import CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing, fit_quantile_edges, to_tensors, train_ensemble, make_bundle, predict_bundle, get_device
from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model, make_blend_bundle
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix, PITCHMIX_COLS

# 트랙맨 pitcher-std 통합, 2026-08-17 세션 최종본 (상세 경위는 code/trackman_pitcher_features.py
# 모듈 docstring 참고): tier A(투수x구종군)->MLP만 채택. tier B(+압박)/C(+타자손)는 한때
# cutoff=7 스플릿으로 플러스처럼 보였으나, `clean_trackman()`의 결측치 오필터링 버그를 고쳐
# 2023(pre-ABS)+cutoff7(ABS) 듀얼체크로 재검증한 결과 B는 CatBoost 단독 점수를 실제로는
# 깎아먹는 가짜 신호(메타모델 재가중치로 블렌드만 구제됨)였고, C는 2023에서 그냥 마이너스였다
# — 둘 다 제외. 대신 팀원이 제보한 coarse pitchmix(merge_coarse_pitchmix, ->CatBoost)를
# 추가해 A+pitchmix 조합으로 2023 +9.57 / cutoff7 +34.43 둘 다 통과 확인 후 채택.
#
# 2026-08-18 세션: 위 A+pitchmix 조합이 실전 리더보드 950.81(-31.41, 982.22 대비)로
# 확인됨. season==2023 레짐 로컬 결과(tier A 있으면 -4.02)와 방향이 일치 — tier A를
# 빼고 pitchmix만 남긴 구성(pitchmix-only)을 다음 실전 후보로 시험한다. cutoff7
# 레짐에서는 로컬상 tier A가 여전히 크게 이기지만(§43), 단일-레짐 로컬 검증의
# 신뢰성 한계(CLAUDE.md "Known reliability gap" 참고)로 실전 증거를 우선한다.
TRACKMAN_TIER_FEED = {}

def apply_f1_filter(df):
    """2022년 이하 시즌의 game_type=='F'(퓨처스/2군) 행을 학습에서만 제거.

    2023년부터 F의 제구 성공률이 R(1군)보다 낮아지는 방향으로 관계가 역전됐는데
    (2022 이전: F가 R보다 최대 +20.5%p 높음 -> 2023~2024: 오히려 -3.0%p 낮음),
    game_type이 CatBoost 피처 중요도 1위라 이 역전이 학습을 크게 오염시킨다
    (season==2023 단일 홀드아웃 검증 시 스코어가 0에 가깝게 붕괴). 2023년 이후 F는
    새 관계가 유효하므로 남긴다 — 전량 제거는 검증에서 더 낮은 점수를 보였다.
    가중치로 희석하는 방식(2023 이후 F에 2배 가중)도 시도됐으나 조기 종료가 첫 트리에서
    멈추는 등 실패해, 오염 구간은 제거가 유일한 해법으로 확인됐다 (EXPERIMENTS.md 참고).
    검증/추론 데이터에는 적용하지 않는다 — 학습 데이터에만 적용한다."""
    before = len(df)
    filtered = df[~((df['game_type'] == 'F') & (df['season'] <= 2022))].reset_index(drop=True)
    print(f"[F1 필터] game_type=='F' & season<=2022 제거: {before} -> {len(filtered)}행 ({before - len(filtered)}행 제거)")
    return filtered

SEASON_PROGRESSION_SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
]


def build_season_end_lookup(df):
    """투수/타자별 "시즌 마지막 행" 누적치 정적 lookup 테이블을 만든다 (팀원 제보 피처,
    2026-08-18 세션 채택). control_success가 있는 학습 데이터(train.csv, 스플릿 이전
    전체)에서만 계산 가능 — test.csv에는 정답이 없고 과거 시즌 행 자체가 없으므로
    (test는 항상 season 2025뿐), pitcher_map.csv/pitchmix_lookup.csv와 동일하게 이
    테이블을 한 번 만들어 정적으로 재사용한다(dopip.py Full Retrain에서 계산해
    submit/model/season_end_lookup.csv로 동봉, submit/script.py는 재계산 없이 병합만).
    반환: role별 next_season(해당 시즌의 "다음" 시즌 = 이 값이 적용될 시즌) 키의 DataFrame."""
    tables = []
    for role, id_col, n_col, rate_col in SEASON_PROGRESSION_SPECS:
        idx = df.groupby([id_col, "season"])[n_col].idxmax()
        season_end = df.loc[idx, [id_col, "season", n_col, rate_col, "control_success"]].copy()
        season_end.columns = ["id", "season", "end_n", "end_rate", "end_success"]
        season_end["season"] = season_end["season"] + 1  # 이 값이 적용되는(=다음) 시즌으로 키 이동
        season_end.insert(0, "role", role)
        tables.append(season_end)
    return pd.concat(tables, ignore_index=True)


def apply_season_progression_features(df, lookup):
    """투수/타자 "시즌 진행분" 8개를 계산한다. asof_{pitcher,batter}_success_rate는
    커리어 전체 누적값이라 베테랑일수록 최근 시즌 컨디션 신호가 희석되는데, lookup(직전
    시즌 마지막 행의 누적치)을 시즌 시작 시점 기준값으로 빼서 "이번 시즌만의" 성공률과
    커리어 누적 성공률 대비 격차(rate_gap)를 분리해낸다. df 자신의 다른 행이나
    control_success에 의존하지 않으므로 test.csv에도 그대로 안전하게 쓸 수 있다.
    CatBoost 단독 재검증에서 cutoff7 +34.66 / season==2023 +150.76, 프로덕션 블렌드
    재검증에서 cutoff7 +35.90 / season==2023 +164.90 — 트랙맨류와 달리 양쪽 레짐·양쪽
    모델 전부 같은 방향으로 크게 개선되어 채택 (code/experiment_season_progression.py,
    code/experiment_season_progression_blend.py, EXPERIMENTS.md §45 참고)."""
    df = df.copy()
    for role, id_col, n_col, rate_col in SEASON_PROGRESSION_SPECS:
        lut = lookup.loc[lookup["role"] == role, ["id", "season", "end_n", "end_rate", "end_success"]]
        merged = df[[id_col, "season"]].merge(
            lut, left_on=[id_col, "season"], right_on=["id", "season"], how="left",
        )
        pre_n = (merged["end_n"] + 1).fillna(0).values
        pre_success = ((merged["end_n"] * merged["end_rate"]).round().fillna(0) + merged["end_success"].fillna(0)).values

        cum_n = df[n_col].values
        cum_success = np.round(df[n_col].values * df[rate_col].values)
        season_n = np.maximum(cum_n - pre_n, 0)
        season_success = np.maximum(cum_success - pre_success, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            season_rate = np.where(season_n > 0, season_success / season_n, np.nan)

        df[f"{role}_season_n"] = season_n
        df[f"{role}_season_success_count"] = season_success
        df[f"{role}_season_success_rate"] = season_rate
        df[f"{role}_season_rate_gap"] = season_rate - df[rate_col].values
    return df


def add_engineered_features(df, league_success_mean):
    """asof_* 및 카운트 정보를 조합한 파생 피처를 추가합니다.

    df는 control_success를 포함한 학습 데이터(스플릿 이전 전체)여야 한다 — 시즌
    진행분 피처가 lookup을 df 자신으로부터 만들기 때문. test.csv(정답 없음, 과거
    시즌 행도 없음)에는 이 함수를 쓸 수 없다 — submit/script.py는 build_season_end_lookup
    없이 apply_season_progression_features만 정적 CSV lookup과 함께 인라인 복제해 쓴다."""
    df = df.copy()
    df = apply_season_progression_features(df, build_season_end_lookup(df))

    df['pitcher_recent1_gap'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_success_rate']
    df['pitcher_recent3_gap'] = df['asof_pitcher_prev3_game_success_rate'] - df['asof_pitcher_success_rate']
    df['pitcher_recent5_gap'] = df['asof_pitcher_prev5_game_success_rate'] - df['asof_pitcher_success_rate']

    df['pitcher_relative_success'] = df['asof_pitcher_success_rate'] - league_success_mean

    df['count_diff'] = df['strikes_before'] - df['balls_before']
    df['is_full_count'] = ((df['balls_before'] == 3) & (df['strikes_before'] == 2)).astype(np.int64)

    df['pitcher_count_advantage_raw'] = df['asof_pitcher_success_rate'] * df['count_diff']
    df['pitcher_count_advantage_rel'] = df['pitcher_relative_success'] * df['count_diff']

    df['pitcher_trend'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_prev5_game_success_rate']
    df['pitcher_consistency'] = df[[
        'asof_pitcher_prev1_game_success_rate',
        'asof_pitcher_prev3_game_success_rate',
        'asof_pitcher_prev5_game_success_rate',
    ]].std(axis=1)

    df['matchup'] = df['asof_pitcher_success_rate'] - df['asof_batter_success_rate']

    pressure_signal = df['li'] * ((df['strikes_before'] >= 2) | (df['balls_before'] >= 3)).astype(np.int64)
    df['count_pressure'] = df['pitcher_relative_success'] * pressure_signal

    return df

def main():
    DATA_DIR = "./open/data"
    target_col = 'control_success'
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)

    train_df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # ABS(자동 볼 판정 시스템) 레짐 시프트 가설 검증(EXPERIMENTS.md §35) 결과 채택:
    # 2024시즌부터 KBO가 ABS를 전면 도입해 2019~2023(구 판정체계) 데이터만으로는
    # 2024/2025(신 판정체계)를 예측하기 어렵다는 가설을, 학습에 2024 초반(3~6월)을
    # 일부 포함시켜 검증했다. cutoff=7(7월부터 검증)이 여러 cutoff(4~10) 중 블렌드
    # 기준 가장 크고 신뢰할 만한 이득(+58.03)을 보여 채택. 가중치(sample_weight)는
    # 표본이 큰 cutoff에서 오히려 baseline보다 나빠 불채택(weight=1 유지).
    train_mask = (train_df['season'] < 2024) | ((train_df['season'] == 2024) & (train_df['game_month'] < 7))
    val_mask = (train_df['season'] == 2024) & (train_df['game_month'] >= 7)

    # 트랙맨 pitcher-std 피처(tier A) 병합. holdout=2024로 넘겨 season==2024 행은
    # 학습/검증 구분 없이 own-season 트랙맨을 못 보게 클램프한다 — 이 스플릿(cutoff=7)으로
    # 실측 검증한 조건과 정확히 동일하게 맞추기 위함(TRACKMAN_TIER_FEED 주석 참고).
    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    train_df, trk_tier_cols = add_all_tiers(train_df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=2024)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]

    # coarse pitchmix(볼카운트x손 조합별 구종 비중, ->CatBoost) 병합. 크로스워크가 필요
    # 없으므로 원본(미클렌징) df_trm을 그대로 쓴다 — clean_trackman이 걸러내는 물리 지표
    # 이상치와 무관한 컬럼만 쓴다(merge_coarse_pitchmix 문서 참고). holdout=2024로 val
    # 시즌(2024) 자기 자신의 트랙맨은 테이블 계산에서 제외한다(리크 방지).
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=2024)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    print(f"[트랙맨] tier별 피처 수: { {t: len(c) for t, c in trk_tier_cols.items()} }, pitchmix 피처 {len(PITCHMIX_COLS)}개 | MLP행 {len(trk_mlp_cols)}개, CatBoost행 {len(trk_cat_cols)}개")

    league_success_mean = train_df.loc[train_mask, target_col].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ['row_id', target_col]
    features = [col for col in train_df.columns if col not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS and c not in trk_cat_cols]
    cat_feature_cols = [c for c in features if c not in trk_mlp_cols]

    train_split = train_df.loc[train_mask, features + [target_col]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [target_col]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)
    print(f"훈련 데이터 (2019~2023 + 2024 3~6월, F1 필터 적용): {len(train_split)} 행 | 검증 데이터 (2024 7~10월): {len(val_split)} 행")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, target_col)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, target_col)
    y_val_np = val_proc[target_col].values

    device = get_device()
    print(f"[Device] {device}")

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
        seeds=ENSEMBLE_SEEDS, device=device,
    )

    mlp_bundle = make_bundle(
        members, CAT_COLS, num_cols, cat_dims, embed_dims,
        cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges,
    )

    print("\n--- [CatBoost] 블렌딩용 CatBoost 모델 학습 ---")
    X_train_raw, y_train_raw = train_split[cat_feature_cols], train_split[target_col].values
    X_val_raw, y_val_raw = val_split[cat_feature_cols], val_split[target_col].values
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=True)
    print(f"[CatBoost] 학습 완료 (best_iteration={catboost_best_iteration})")

    mlp_val_preds = predict_bundle(mlp_bundle, val_split[features], device=device)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    w_cat, w_mlp, intercept, blend_score, blend_brier = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)
    meta_model = {"w_cat": w_cat, "w_mlp": w_mlp, "intercept": intercept}
    print(f"[Blend] 스태킹 메타모델 w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f} | 블렌드 Val Score: {blend_score:.2f}")

    bundle = make_blend_bundle(catboost_model, mlp_bundle, meta_model, cat_feature_cols=cat_feature_cols)
    bundle["catboost_best_iteration"] = catboost_best_iteration

    os.makedirs("./open/temp", exist_ok=True)
    with open("./open/temp/latest_model.pkl", 'wb') as f:
        pickle.dump(bundle, f)
    print(f"✅ Model saved to ./open/temp/latest_model.pkl (mlp best_epoch_avg={mlp_bundle['best_epoch_']}, catboost best_iteration={catboost_best_iteration}, meta_model={meta_model})")

if __name__ == "__main__":
    main()
