# code/train.py
"""2026-08-27 전면 교체: 팀원 조유담님 파이프라인(teammate/yudam/) 기반으로 피처
엔지니어링·CatBoost 하이퍼파라미터·MLP 구성·메타모델 피팅 방식을 전부 교체했다.

배경: 실전 리더보드에서 유담님 파이프라인(v2 CatBoost HP 포함)이 real 1092.55를
기록한 반면 이 repo의 최고 기록은 1046.56(candidate A)에 그쳤다. 그의 실제 코드를
그대로(하이퍼파라미터 파일 경로 1줄만 교체) 우리 데이터로 직접 실행해 재현한 결과
half_2024 fold(=우리 cutoff7과 동일 정의)에서 블렌드 815.43(30% eval subset)을
확인, 그의 재현이 실제로 되는 것과 이 repo의 예전 재구현 시도가 버그(구식
하이퍼파라미터 + same_hand 누락)로 실패했었다는 것 둘 다 확인됐다
(EXPERIMENTS.md §91). 이번 세션에서 재현 성공 여부와 무관하게 그의 순정 레시피로
파이프라인을 전면 교체하기로 결정 — code/mlp_model.py, code/catboost_model.py,
code/blend_model.py는 이미 그의 번들 스키마(catboost_models 리스트/mlp_bundle/
meta_model dict)와 100% 호환되는 범용 인프라라 변경 없이 재사용한다. 이 파일과
code/test.py, dopip.py Full Retrain 단계, submit/script.py만 그의 feature_engineering.py
+ full_retrain_blend_f1.py + submit/script.py 기준으로 다시 작성했다.

**이전 code/train.py 대비 제거된 것**: pair_matchup(=팀원과 무관, 이 repo 자체 실험,
실전 -12.65로 이미 기각됐었는데 코드에 남아 cat_feature_cols에 여전히 피드되고
있던 leftover — 이번 교체로 청소됨), team_matchup(피드 비활성 상태였던 미검증
실험, 마찬가지로 제거), tier A/B/C 트랙맨 크로스워크(TRACKMAN_TIER_FEED={}로 이미
비활성이었음, 완전 제거).

**새로 추가된 것(유담님 레시피에는 있었으나 이 repo 프로덕션엔 없었던 것)**:
- 트랙맨 상황별 물리 지표 mean/std 전체 조인(구 "tier A류"와 달리 크로스워크 없이
  상황(10-key) 조인만 사용 — `process_trackman_features_safe`, 구 트랙맨64 컬럼).
  이 repo는 이 조인을 실전 검증 후 제거했었지만(CLAUDE.md "Trackman history" 참고),
  유담님은 여전히 유지 중이고 그의 실전 결과가 이 repo보다 높으므로 "재현 성공
  여부와 무관하게 그대로 이식"이라는 이번 세션 지침에 따라 그대로 포함한다. 이후
  ablation(phase 2)에서 제거 시 영향을 별도로 측정한다.
- reverse_rate 시즌 진행분(asof_pitcher_reverse_rate의 시즌 분해, `build_rate_end_lookup`/
  `apply_rate_progression_features`) — 이 repo에서는 아직 "REOPENED"(미확정) 상태였던
  것을 유담님은 이미 채택해서 쓰고 있음.

**빠진 것(quantile PLE)**: 유담님 MLP는 QuantileEmbedding(PLE) 없이 표준화된
수치형을 그대로 concat한다(raw concat) — `code/mlp_model.py::train_ensemble`/
`make_bundle`에 `bin_edges=None`을 넘기면 그대로 이 경로로 자동 폴백되므로
mlp_model.py 자체는 수정 불필요. 이 repo는 quantile PLE가 검증된 이득이었지만
유담님 쪽에서는 과거 실측 회귀(968.15→879.54)가 있어 그대로 뺐다 — "재현
성공 여부와 무관하게 순정 레시피 그대로" 원칙 적용. quantile PLE를 이 레시피
위에 다시 얹었을 때 어떻게 되는지는 phase 3에서 별도로 측정한다
(트랙맨64 fallback-constant와의 상호작용 가설, teammate_catboost_mlp_track_983.md 참고).

**메타모델 피팅 방식**: 기존(창현님/이 repo 컨벤션)은 val 전체로 fit, val 전체로
점수 산출. 유담님은 val을 다시 70/30으로 나눠 70%로 fit, 30%로 평가한다(진짜
연속된 신규 데이터가 없어 "홀드아웃 안의 홀드아웃"으로 과적합을 다시 확인하는
취지). 이번 교체에서는 그의 방식(70/30)을 그대로 따르되, dopip.py의 test.py 비교
로직(reference 대비 NEW_BEST/KEEP_REF 판정)과의 호환을 위해 val 전체 기준 점수도
함께 계산해 로그에 남긴다 — 실제 meta_model 계수는 70% 파티션으로 학습한 것을
채택한다(그의 실전 1092.55가 이 방식으로 학습된 계수를 그대로 썼기 때문).
"""
import sys
import os


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pickle
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from code.mlp_model import CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing, to_tensors, train_ensemble, make_bundle, predict_bundle, get_device
from code.catboost_model import predict_catboost_ensemble, train_catboost_ensemble
from code.blend_model import make_blend_bundle
from code.trackman_pitcher_features import merge_coarse_pitchmix, PITCHMIX_COLS

# 유담님 파이프라인의 시드 목록을 그대로 이식(값 자체는 임의, 개수만 그의 레시피와
# 맞춘다 -- CatBoost 5-seed/MLP 7-seed). 이 repo의 code/mlp_model.py::ENSEMBLE_SEEDS
# (20개, quantile PLE 채택 이후 다른 검증 경로에서 쓰임)와 별개로 이 파일 전용 상수로 둔다.
YUDAM_ENSEMBLE_SEEDS = [42, 123, 7, 2024, 99, 555, 31337]
YUDAM_CATBOOST_SEEDS = YUDAM_ENSEMBLE_SEEDS[:5]


def process_trackman_features_safe(df_main, df_trm, is_train_split=True):
    """`teammate/yudam/feature_engineering.py::process_trackman_features_safe` 이식
    (수정 없이 그대로). 10-key 상황 지문(season/game_month/dayofweek/inning/
    top_bottom/balls_before/strikes_before/outs_before 등 df_main과 trackman의
    공통 컬럼 전부) x pitch_type_group 별 물리 지표(rel_speed/spin_rate/...)
    mean/std를 조인한다. 이 repo는 한때 이 조인을 제거했었으나(구 "트랙맨64"),
    유담님은 유지 중이고 그의 실전 결과가 더 높아 "순정 레시피 그대로 이식" 원칙에
    따라 복원한다 -- phase2 ablation으로 이 repo 데이터에서의 순효과를 별도 검증한다."""
    df_main_copy = df_main.copy()
    df_trm_copy = df_trm.copy()

    if is_train_split:
        max_season = df_main_copy['season'].max()
        max_month = df_main_copy[df_main_copy['season'] == max_season]['game_month'].max()
        future_mask = (df_trm_copy['season'] > max_season) | \
                      ((df_trm_copy['season'] == max_season) & (df_trm_copy['game_month'] > max_month))
        df_trm_copy = df_trm_copy[~future_mask].reset_index(drop=True)

    match_cols = [c for c in df_main_copy.columns if (c in df_trm_copy.columns) and c != 'row_id']

    df_main_copy['top_bottom'] = df_main_copy['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    df_trm_copy['top_bottom'] = df_trm_copy['top_bottom'].map({'Top': 0, 'Bottom': 1}).astype(np.int64)

    grouped = df_trm_copy.groupby(match_cols + ['pitch_type_group', 'auto_pitch_type'])
    grouped_phase1 = grouped[['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break', 'extension', 'rel_height', 'rel_side', 'zone_speed']].agg(['mean', 'std'])
    grouped_phase2 = grouped_phase1.reset_index()

    std_cols = [c for c in grouped_phase2.columns if 'std' in c]
    grouped_phase2[std_cols] = grouped_phase2[std_cols].fillna(0)
    grouped_phase2.columns = ['_'.join(c).strip('_') for c in grouped_phase2.columns]

    grouped_phase3 = grouped_phase2.drop(columns='auto_pitch_type')
    grouped_phase3 = grouped_phase3.groupby(match_cols + ['pitch_type_group']).agg(['mean'])

    pivoted = grouped_phase3.unstack(level='pitch_type_group')
    pivoted.columns = [f"{c[0]}_{c[1]}_{c[2]}" for c in pivoted.columns]
    tm_final = pivoted.reset_index()
    tm_final = tm_final.fillna(0)

    for col in ['batter_hand', 'pitcher_hand']:
        if col in tm_final.columns:
            tm_final[col] = tm_final[col].map({'Left': 1, 'Right': 2}).astype(np.int64)

    tr_final = pd.merge(df_main_copy, tm_final, on=match_cols, how='left')
    new_feature_cols = [c for c in tm_final.columns if c not in match_cols]
    tr_final[new_feature_cols] = tr_final[new_feature_cols].fillna(tr_final[new_feature_cols].mean())

    return tr_final, match_cols, new_feature_cols


def apply_f1_filter(df):
    """2022년 이하 시즌의 game_type=='F'(퓨처스/2군) 행을 학습에서만 제거. 이 repo와
    유담님 양쪽에서 동일하게 검증된 필터라 변경 없음 (원본 docstring 근거는
    PROJECT_HISTORY.md §16 참고)."""
    before = len(df)
    filtered = df[~((df['game_type'] == 'F') & (df['season'] <= 2022))].reset_index(drop=True)
    print(f"[F1 필터] game_type=='F' & season<=2022 제거: {before} -> {len(filtered)}행 ({before - len(filtered)}행 제거)")
    return filtered


SEASON_PROGRESSION_SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
]


def build_season_end_lookup(df):
    tables = []
    for role, id_col, n_col, rate_col in SEASON_PROGRESSION_SPECS:
        idx = df.groupby([id_col, "season"])[n_col].idxmax()
        season_end = df.loc[idx, [id_col, "season", n_col, rate_col, "control_success"]].copy()
        season_end.columns = ["id", "season", "end_n", "end_rate", "end_success"]
        season_end["season"] = season_end["season"] + 1
        season_end.insert(0, "role", role)
        tables.append(season_end)
    return pd.concat(tables, ignore_index=True)


def apply_season_progression_features(df, lookup):
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


RATE_PROGRESSION_SPECS = [
    ("pitcher_reverse", "pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate"),
]


def build_rate_end_lookup(df):
    """유담님 이식: asof_pitcher_reverse_rate 시즌 진행분 lookup. 이 repo에서는 아직
    "REOPENED"(미확정) 상태였으나 유담님은 이미 채택해서 쓰고 있어 순정 이식."""
    tables = []
    for name, id_col, n_col, rate_col in RATE_PROGRESSION_SPECS:
        idx = df.groupby([id_col, "season"])[n_col].idxmax()
        season_end = df.loc[idx, [id_col, "season", n_col, rate_col]].copy()
        season_end.columns = ["id", "season", "end_n", "end_rate"]
        season_end["season"] = season_end["season"] + 1
        season_end.insert(0, "name", name)
        tables.append(season_end)
    return pd.concat(tables, ignore_index=True)


def apply_rate_progression_features(df, lookup):
    df = df.copy()
    for name, id_col, n_col, rate_col in RATE_PROGRESSION_SPECS:
        lut = lookup.loc[lookup["name"] == name, ["id", "season", "end_n", "end_rate"]]
        merged = df[[id_col, "season"]].merge(lut, left_on=[id_col, "season"], right_on=["id", "season"], how="left")
        pre_n = merged["end_n"].fillna(0).values
        pre_count = (merged["end_n"] * merged["end_rate"]).round().fillna(0).values
        cum_n = df[n_col].values
        cum_count = np.round(df[n_col].values * df[rate_col].values)
        season_n = np.maximum(cum_n - pre_n, 0)
        season_count = np.maximum(cum_count - pre_count, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            season_rate = np.where(season_n > 0, season_count / season_n, np.nan)
        df[f"{name}_season_rate"] = season_rate
        df[f"{name}_season_rate_gap"] = season_rate - df[rate_col].values
    return df


TE_K_SMOOTH = 200
TE_AXES = [
    ("te_p_cnt", ["pitcher_id", "balls_before", "strikes_before"], "p_main"),
    ("te_p_bhand", ["pitcher_id", "batter_hand"], "p_main"),
    ("te_p_run", ["pitcher_id", "num_runners_on"], "p_main"),
    ("te_p_inn", ["pitcher_id", "inning"], "p_main"),
    ("te_b_cnt", ["batter_id", "balls_before", "strikes_before"], "b_main"),
]
TE_MAIN_AXES = [("p_main", ["pitcher_id"]), ("b_main", ["batter_id"])]
TE_RESIDUAL_COLS = [f"{name}_res" for name, *_ in TE_AXES] + ["te_covered"]


def causal_smoothed_te_encode(source_df, query_df, group_cols, prior, k=TE_K_SMOOTH):
    agg = source_df.groupby(group_cols + ["season"])["control_success"].agg(["sum", "count"]).reset_index()
    agg = agg.sort_values("season")
    agg["cum_sum"] = agg.groupby(group_cols)["sum"].cumsum()
    agg["cum_n"] = agg.groupby(group_cols)["count"].cumsum()
    agg = agg[group_cols + ["season", "cum_sum", "cum_n"]].sort_values("season")

    query = query_df[group_cols + ["season"]].reset_index()
    query_sorted = query.sort_values("season")
    merged = pd.merge_asof(
        query_sorted, agg, on="season", by=group_cols,
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.sort_values("index")
    cum_sum = merged["cum_sum"].fillna(0.0).values
    cum_n = merged["cum_n"].fillna(0.0).values
    enc = (cum_sum + prior * k) / (cum_n + k)
    covered = (cum_n > 0).astype(np.int64)
    return enc, covered


def apply_te_residual_features(source_df, query_df, prior):
    """Track A (팀원 제보 target-encoding 잔차 6개, CatBoost 전용). 이 repo와 유담님
    양쪽에서 채택된 피처라 변경 없음."""
    query_df = query_df.copy()
    mains = {}
    covered_any = np.zeros(len(query_df), dtype=np.int64)
    for name, group_cols in TE_MAIN_AXES:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        mains[name] = enc
        covered_any = np.maximum(covered_any, covered)

    for name, group_cols, main_key in TE_AXES:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        query_df[f"{name}_res"] = enc - mains[main_key]
        covered_any = np.maximum(covered_any, covered)

    query_df["te_covered"] = covered_any
    return query_df


def add_engineered_features(df, league_success_mean):
    df = df.copy()
    df = apply_season_progression_features(df, build_season_end_lookup(df))
    df = apply_rate_progression_features(df, build_rate_end_lookup(df))

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


SAME_HAND_COLS = ['same_hand', 'same_hand_advantage']

# 트랙맨 상황(10-key) 물리조인 산출물 64컬럼. process_trackman_features_safe 는
# match_cols 에 season 이 들어가 2025 추론 시 전부 per-column 상수로 붕괴하고, 그 죽은
# 상수가 CatBoost 를 miscalibrate 한다 (Task 1). 완전제거한 손빌드 제출본이 실전 1117.03
# (candidate B raw 1092.998 대비 +24.03, repo 최고). 이름 규칙: {metric}_{mean|std}_mean_{group}
# (8 metric × {mean,std} × {fastball,breaking,offspeed,other}). coarse pitchmix 4컬럼은
# 별개(투수 identity/season 없는 상황축 집계)라 제외 안 함.
TRACKMAN64_RE = re.compile(
    r"^(rel_speed|spin_rate|induced_vert_break|horz_break|extension|rel_height|rel_side|zone_speed)"
    r"_(mean|std)_mean_(fastball|breaking|offspeed|other)$"
)


def is_trackman64(col):
    return bool(TRACKMAN64_RE.match(col))


def apply_same_hand(df):
    """MLP 전용 라우팅 — 이 repo와 유담님 양쪽에서 동일하게 채택된 피처."""
    df = df.copy()
    df['same_hand'] = (df['pitcher_hand'] == df['batter_hand']).astype(np.int64)
    df['same_hand_advantage'] = df['pitcher_relative_success'] * df['same_hand']
    return df


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    DATA_DIR = "./open/data"
    target_col = 'control_success'
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")

    tr_final, match_cols, trackman_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[target_col]).reset_index(drop=True)

    # cutoff=7 스플릿 (= 유담님의 half_2024 fold와 정의 동일: 2019~2024/06 학습,
    # 2024/07~10 검증). CLAUDE.md "Train/eval split convention" 참고.
    train_mask = (train_df['season'] < 2024) | ((train_df['season'] == 2024) & (train_df['game_month'] < 7))
    val_mask = (train_df['season'] == 2024) & (train_df['game_month'] >= 7)

    league_success_mean = train_df.loc[train_mask, target_col].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    # coarse pitchmix (->CatBoost). holdout=2024로 val 시즌 자기 자신의 트랙맨은 제외.
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=2024)

    train_df = apply_same_hand(train_df)

    drop_cols = ['row_id', target_col]
    all_cols = [c for c in train_df.columns if c not in drop_cols]
    # 트랙맨64 는 컬럼 자체는 df 에 남기되(모델 입력에서만 제외) CatBoost/MLP 피처목록에서 뺀다.
    cat_feature_cols = [c for c in all_cols if c not in SAME_HAND_COLS and not is_trackman64(c)]
    num_cols = [c for c in all_cols
                if c not in CAT_COLS and c not in TE_RESIDUAL_COLS and not is_trackman64(c)]

    train_split = train_df.loc[train_mask, all_cols + [target_col]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, all_cols + [target_col]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)
    print(f"훈련 데이터 (F1 필터 적용): {len(train_split)}행 | 검증 데이터(cutoff7 val): {len(val_split)}행")

    te_prior = train_split[target_col].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)
    cat_feature_cols = cat_feature_cols + TE_RESIDUAL_COLS

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, target_col)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, target_col)
    y_val_np = val_proc[target_col].values

    device = get_device()
    print(f"[Device] {device}")

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    # 유담님 순정 레시피: quantile PLE 없음 (bin_edges=None -> mlp_model.py가 자동으로
    # raw-concat 경로로 폴백). 7-seed 앙상블.
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims, bin_edges=None,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
        seeds=YUDAM_ENSEMBLE_SEEDS, device=device,
    )
    mlp_bundle = make_bundle(
        members, CAT_COLS, num_cols, cat_dims, embed_dims,
        cat_encoder, num_imputer, num_scaler, bin_edges=None,
    )

    print(f"\n--- [CatBoost] 유담님 v2 재튜닝 하이퍼파라미터, {len(YUDAM_CATBOOST_SEEDS)}-seed 배깅 ---")
    import json
    with open("./teammate/yudam/model_py311/best_catboost_hparams_v2.json") as f:
        _tuned = json.load(f)["best_params"]
    yudam_catboost_params = dict(
        depth=_tuned["depth"], learning_rate=_tuned["learning_rate"], l2_leaf_reg=_tuned["l2_leaf_reg"],
        random_strength=_tuned["random_strength"], bagging_temperature=_tuned["bagging_temperature"],
        border_count=_tuned["border_count"], min_data_in_leaf=_tuned["min_data_in_leaf"],
        bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    )
    X_train_raw, y_train_raw = train_split[cat_feature_cols], train_split[target_col].values
    X_val_raw, y_val_raw = val_split[cat_feature_cols], val_split[target_col].values
    catboost_results = train_catboost_ensemble(
        X_train_raw, y_train_raw, X_val_raw, y_val_raw,
        seeds=YUDAM_CATBOOST_SEEDS, verbose=True, params=yudam_catboost_params,
    )
    catboost_models = [m for m, _ in catboost_results]
    catboost_best_iterations = [it for _, it in catboost_results]
    print(f"[CatBoost] {len(catboost_models)}-seed 학습 완료 (best_iterations={catboost_best_iterations})")

    mlp_val_preds = predict_bundle(mlp_bundle, val_split[all_cols], device=device)
    cat_val_preds = predict_catboost_ensemble(catboost_models, X_val_raw)

    # 유담님 메타모델 피팅 방식: val을 70/30으로 재분할, 70%로 fit. 30%eval과 val전체
    # 점수를 둘 다 로그로 남긴다(후자는 code/test.py의 기존 레퍼런스 비교값과 직접
    # 비교하기 위함 -- test.py는 latest_model.pkl을 val 전체로 재평가한다).
    rng = np.random.RandomState(42)
    n_val = len(y_val_raw)
    perm = rng.permutation(n_val)
    split = int(n_val * 0.7)
    fit_idx, eval_idx = perm[:split], perm[split:]
    clf = LogisticRegression()
    clf.fit(np.column_stack([cat_val_preds[fit_idx], mlp_val_preds[fit_idx]]), y_val_raw[fit_idx])
    w_cat, w_mlp = (float(c) for c in clf.coef_[0])
    intercept = float(clf.intercept_[0])
    meta_model = {"w_cat": w_cat, "w_mlp": w_mlp, "intercept": intercept}

    def _score(pred, y):
        r = y.mean(); brier = ((pred - y) ** 2).mean(); base = r * (1 - r)
        return max(0.0, 100000 * (1 - brier / base))

    blend_pred_eval30 = sigmoid(w_cat * cat_val_preds[eval_idx] + w_mlp * mlp_val_preds[eval_idx] + intercept)
    blend_pred_full = sigmoid(w_cat * cat_val_preds + w_mlp * mlp_val_preds + intercept)
    print(f"[Blend] 메타모델(70% fit) w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f}")
    print(f"[Blend] 30% eval 점수(유담님 방식)={_score(blend_pred_eval30, y_val_raw[eval_idx]):.2f} | "
          f"val 전체 점수(이 repo test.py 비교용)={_score(blend_pred_full, y_val_raw):.2f}")

    bundle = make_blend_bundle(catboost_models, mlp_bundle, meta_model, cat_feature_cols=cat_feature_cols)
    bundle["catboost_best_iteration"] = catboost_best_iterations[0]
    bundle["catboost_best_iterations"] = catboost_best_iterations

    os.makedirs("./open/temp", exist_ok=True)
    with open("./open/temp/latest_model.pkl", 'wb') as f:
        pickle.dump(bundle, f)
    print(f"✅ Model saved to ./open/temp/latest_model.pkl (mlp best_epoch_avg={mlp_bundle['best_epoch_']}, catboost best_iterations={catboost_best_iterations}, meta_model={meta_model})")


if __name__ == "__main__":
    main()
