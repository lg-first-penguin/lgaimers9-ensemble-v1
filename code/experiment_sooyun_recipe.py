# code/experiment_sooyun_recipe.py
"""손수연 팀원의 실제 제출 CatBoost 레시피(teammate/sooyun/lg-catboost-77feat-repo,
real 1007.52 솔로)를 최대한 충실하게 재현해 우리 자신의 cutoff7/season2023 검증
윈도우에서 CatBoost 솔로 + (우리 자신의 기존 MLP, 재학습 없이 재사용) 블렌드로
평가한다. 사용자 확인: "제 catboost에 창현님 mlp+std/gap피처 합쳐서 유담님
메타모델 방법으로 한거 맞습니다" — 즉 그의 CatBoost + 이 repo 자체 MLP(고정, 재학습
없음) + 2입력 로지스틱 메타모델(우리 §9와 같은 설계, `code/blend_model.py::fit_meta_model`
재사용)로 실전 1055.08.

**충실도 우선 원칙**: 그의 recipe를 최대한 그대로 재현한다 — F1 필터도 그의
script.py에 없으므로 여기서도 적용하지 않는다(이건 별도 확인된 우리 자신의 F1
필터 효과와 섞이지 않게, "그가 실제로 제출한 그대로"를 먼저 본다). count_diff 부호,
pitcher_consistency 공식(abs(prev1-prev5), 우리 std(prev1,prev3,prev5)와 다름),
league mean이 시즌별 갱신(prev_season_league_mean)인 점 등 사소해 보이는 차이도
그의 `script.py`(teammate/sooyun/lg-catboost-77feat-repo/repo_package/script.py)에서
그대로 복사해왔다.

CatBoost 하이퍼파라미터는 그의 실제 학습된 .cbm 파일(model_seed42.cbm)에서
`model.get_all_params()`로 직접 추출한 값(재탐색이 아니라 실측 확인):
depth=8/lr=0.03/l2_leaf_reg=3/random_strength=1/bagging_temperature=1/
border_count=128/min_data_in_leaf=1/iterations=549(고정, early stopping 없음)/
bootstrap_type=Bayesian/grow_policy=SymmetricTree/boosting_type=Plain. 카테고리
피처도 그가 선언한 8개(top_bottom/game_type/base_state/pitcher_hand/batter_hand/
pitcher_team_id/batter_team_id/hand_matchup, top_bottom도 T/B 문자열 그대로 유지)를
그대로 쓴다 — 우리는 top_bottom을 0/1로 미리 매핑하고 game_type/base_state 2개만
categorical로 선언하는 것과 다르다.

MLP는 재학습하지 않는다 — `open/reference/best_model.pkl`의 mlp_bundle을 그대로
불러와 val_split에 `predict_bundle`로 추론만 한다(순수 forward pass, 학습 없음 —
자원 절약을 위한 의도적 선택이자, 사용자가 확인해준 손수연님의 실제 방법과도 일치).

사용법: python -m code.experiment_sooyun_recipe
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.mlp_model import compute_bss, predict_bundle
from code.blend_model import fit_meta_model
from code.train import (
    build_season_end_lookup, apply_season_progression_features, apply_f1_filter,
)

DATA_DIR = "./open/data"
TARGET = "control_success"
SOOYUN_DIR = "./teammate/sooyun/lg-catboost-77feat-repo/repo_package"

MIN_AVAILABLE_MB = 800  # 임베딩 프로세스와 자원을 나눠 쓰는 상황 — 위험 수준이면 스스로 중단


def check_memory_or_abort(tag):
    """/proc/meminfo의 MemAvailable이 임계치 밑이면 즉시 중단한다(OS OOM killer가
    다른 프로세스를 죽이기 전에 우리 쪽에서 먼저 안전하게 실패하는 편이 낫다)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) / 1024
                print(f"[자원체크:{tag}] MemAvailable={available_mb:.0f}MB", flush=True)
                if available_mb < MIN_AVAILABLE_MB:
                    raise SystemExit(f"[중단] MemAvailable {available_mb:.0f}MB < {MIN_AVAILABLE_MB}MB 임계치 — 자원 부족으로 안전 중단")
                return

MATCH_COLS_6 = ["inning", "top_bottom", "balls_before", "strikes_before", "pitcher_hand", "batter_hand"]
STD_FEATURES = ["tm_rel_speed_std", "tm_spin_rate_std", "tm_ivb_std", "tm_hb_std", "tm_extension_std"]
GAP_FEATURES = ["speed_gap_fast_break", "spin_gap_fast_break", "ivb_gap_fast_break", "hbreak_gap_fast_break"]

SOOYUN_CATBOOST_PARAMS = dict(
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=3,
    random_strength=1,
    bagging_temperature=1,
    border_count=128,
    min_data_in_leaf=1,
    bootstrap_type="Bayesian",
    loss_function="Logloss",
    grow_policy="SymmetricTree",
    boosting_type="Plain",
    thread_count=4,  # 임베딩 프로세스와 CPU/메모리를 나눠 쓰는 상황 — 과도한 병렬화 자제
    verbose=False,
)
SOOYUN_ITERATIONS = 549

SOOYUN_CAT_COLS = [
    "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "base_state", "game_type", "top_bottom",
    "hand_matchup",
]


# --- 손수연 script.py의 피처 함수를 그대로 이식 (사소한 공식 차이까지 보존) ---

def add_league_features(df, league_mean_dict, time_col="season", target_col=TARGET):
    df = df.copy()
    df["prev_season_league_mean"] = df[time_col].map(lambda s: league_mean_dict.get(s - 1, np.nan))
    df["pitcher_relative_success"] = df["asof_pitcher_success_rate"] - df["prev_season_league_mean"]
    df["pitcher_recent1_gap"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["pitcher_recent3_gap"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["pitcher_recent5_gap"] = df["asof_pitcher_prev5_game_success_rate"] - df["asof_pitcher_success_rate"]
    return df


def add_interaction_features(df):
    df = df.copy()
    df["count_diff"] = df["balls_before"] - df["strikes_before"]  # 부호가 우리 train.py와 반대
    df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)
    df["pitcher_count_advantage_raw"] = df["asof_pitcher_success_rate"] * df["count_diff"]
    df["pitcher_count_advantage_rel"] = df["pitcher_relative_success"] * df["count_diff"]
    df["pitcher_trend"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]
    df["pitcher_consistency"] = (
        df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]
    ).abs()  # 우리 std(prev1,prev3,prev5)와 다른 공식
    df["matchup"] = df["asof_pitcher_success_rate"] - df["asof_batter_success_rate"]
    df["count_pressure"] = df["pitcher_relative_success"] * df["is_full_count"]  # 우리는 li 기반 압박신호와 다름
    return df


def add_same_hand_features(df):
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"].astype(str) == df["batter_hand"].astype(str)).astype(int)
    df["same_hand_advantage"] = df["pitcher_relative_success"] * df["same_hand"]
    return df


def add_merge_features(df, match_table, match_cols, time_col="season"):
    df = df.copy()
    n_before = len(df)
    df["match_season"] = df[time_col] - 1
    merged = df.merge(match_table, on=["match_season"] + match_cols, how="left")
    merged = merged.drop(columns=["match_season"])
    assert len(merged) == n_before, "merge로 행 개수가 변했습니다"
    return merged


def build_last_row_lookup(df, id_col, asof_n_col, asof_rate_col, time_col="season", target_col=TARGET):
    cols = ["row_id", id_col, time_col, asof_n_col, asof_rate_col, target_col]
    df_sorted = df[cols].sort_values("row_id")
    last_rows = df_sorted.drop_duplicates(subset=[id_col, time_col], keep="last").copy()
    last_rows = last_rows.drop(columns=["row_id"])
    last_rows = last_rows.rename(columns={
        asof_n_col: "last_asof_n", asof_rate_col: "last_asof_rate", target_col: "last_control_success",
    })
    return last_rows


def add_season_progress_features(df, id_col, asof_n_col, asof_rate_col, last_lookup, prefix, time_col="season"):
    df = df.copy()
    df["_cum_n"] = df[asof_n_col]
    df["_cum_success"] = (df[asof_n_col] * df[asof_rate_col]).round()

    lookup = last_lookup.copy()
    lookup["_pre_season"] = lookup[time_col] + 1
    lookup[f"_pre_{prefix}_n"] = lookup["last_asof_n"] + 1
    lookup[f"_pre_{prefix}_success"] = (
        (lookup["last_asof_n"] * lookup["last_asof_rate"]).round() + lookup["last_control_success"]
    )
    lookup = lookup[[id_col, "_pre_season", f"_pre_{prefix}_n", f"_pre_{prefix}_success"]]

    n_before = len(df)
    df = df.merge(lookup, left_on=[id_col, time_col], right_on=[id_col, "_pre_season"], how="left")
    assert len(df) == n_before, "merge로 행 개수가 변했습니다"
    df = df.drop(columns=["_pre_season"])

    df[f"_pre_{prefix}_n"] = df[f"_pre_{prefix}_n"].fillna(0)
    df[f"_pre_{prefix}_success"] = df[f"_pre_{prefix}_success"].fillna(0)

    df[f"{prefix}_season_n"] = (df["_cum_n"] - df[f"_pre_{prefix}_n"]).clip(lower=0)
    df[f"{prefix}_season_success_count"] = (df["_cum_success"] - df[f"_pre_{prefix}_success"]).clip(lower=0)

    n_col = df[f"{prefix}_season_n"]
    success_col = df[f"{prefix}_season_success_count"]
    rate = pd.Series(np.nan, index=df.index)
    valid_mask = n_col > 0
    rate.loc[valid_mask] = success_col.loc[valid_mask] / n_col.loc[valid_mask]
    df[f"{prefix}_season_success_rate"] = rate
    df[f"{prefix}_season_rate_gap"] = df[f"{prefix}_season_success_rate"] - df[asof_rate_col]

    df = df.drop(columns=["_cum_n", "_cum_success", f"_pre_{prefix}_n", f"_pre_{prefix}_success"])
    return df


def build_sooyun_features(df_all):
    """df_all(전체 train.csv, F1 필터 미적용 — 그의 recipe 그대로)에 그의 77피처
    엔지니어링을 전부 적용한다. season 진행분은 이 repo의 검증된 함수를 재사용
    (수식이 동일 — build_last_row_lookup/add_season_progress_features가 사실상
    build_season_end_lookup/apply_season_progression_features와 동일 계산)."""
    check_memory_or_abort("build_sooyun_features 시작")
    df = df_all.copy()
    df = df.drop(columns=[c for c in ["asof_pitcher_pitchmix_n"] if c in df.columns])

    league_mean_full = df.groupby("season")[TARGET].mean().to_dict()
    df = add_league_features(df, league_mean_full)
    df = add_interaction_features(df)
    df = add_same_hand_features(df)

    std_table = pd.read_csv(f"{SOOYUN_DIR}/model/trackman_match_table.csv")
    gap_table = pd.read_csv(f"{SOOYUN_DIR}/model/trackman_match_table_gap.csv")
    df = add_merge_features(df, std_table, MATCH_COLS_6)
    df = add_merge_features(df, gap_table, MATCH_COLS_6)

    pitcher_last = build_last_row_lookup(df, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate")
    batter_last = build_last_row_lookup(df, "batter_id", "asof_batter_n", "asof_batter_success_rate")
    df = add_season_progress_features(df, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", pitcher_last, "pitcher")
    df = add_season_progress_features(df, "batter_id", "asof_batter_n", "asof_batter_success_rate", batter_last, "batter")

    df["hand_matchup"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)

    df = df.drop(columns=["pitcher_id", "batter_id"])  # no_both
    return df


def run_regime(regime_name, train_mask, val_mask, df_raw, val_split_ours, mlp_bundle, device):
    print(f"\n{'='*70}\n=== 레짐: {regime_name} ===\n{'='*70}", flush=True)

    # --- 손수연 레시피 CatBoost (F1 필터 미적용, 그의 recipe 그대로) ---
    t0 = time.time()
    sooyun_full = build_sooyun_features(df_raw)
    train_sy = sooyun_full.loc[train_mask].reset_index(drop=True)
    val_sy = sooyun_full.loc[val_mask].reset_index(drop=True)
    del sooyun_full  # 슬라이스 완료 후 전체 df 즉시 해제 (메모리 절약)

    for c in SOOYUN_CAT_COLS:
        train_sy[c] = train_sy[c].astype(str)
        val_sy[c] = val_sy[c].astype(str)

    drop_cols = ["row_id", TARGET]
    feature_cols = [c for c in train_sy.columns if c not in drop_cols]
    print(f"[손수연 레시피] train={len(train_sy)} val={len(val_sy)} features={len(feature_cols)} ({time.time()-t0:.1f}s)", flush=True)

    train_pool = Pool(train_sy[feature_cols], train_sy[TARGET], cat_features=SOOYUN_CAT_COLS)
    val_pool = Pool(val_sy[feature_cols], val_sy[TARGET], cat_features=SOOYUN_CAT_COLS)

    check_memory_or_abort("CatBoost 학습 시작 전")
    t0 = time.time()
    model = CatBoostClassifier(**SOOYUN_CATBOOST_PARAMS, iterations=SOOYUN_ITERATIONS, random_seed=42)
    model.fit(train_pool)
    cat_preds = model.predict_proba(val_pool)[:, 1]
    y_val = val_sy[TARGET].values
    cat_brier, cat_bss, cat_score = compute_bss(cat_preds, y_val)
    print(f"[손수연 레시피] CatBoost 학습+예측 완료 ({time.time()-t0:.1f}s) | Val Score={cat_score:.2f}", flush=True)

    # --- 우리 자신의 MLP(고정, 재학습 없음)로 같은 val 행에 예측 ---
    mlp_preds = predict_bundle(mlp_bundle, val_split_ours, device=device)
    mlp_brier, mlp_bss, mlp_score = compute_bss(mlp_preds, y_val)
    print(f"[우리 MLP, 고정 재사용] Val Score={mlp_score:.2f}", flush=True)

    w_cat, w_mlp, intercept, blend_score, blend_brier = fit_meta_model(cat_preds, mlp_preds, y_val)
    print(f"[블렌드] w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f} | Val Score={blend_score:.2f}", flush=True)

    return {
        "regime": regime_name, "cat_score": cat_score, "mlp_score": mlp_score, "blend_score": blend_score,
    }


def main():
    print("[데이터 로드]", flush=True)
    t0 = time.time()
    df_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df_raw = df_raw.dropna(subset=[TARGET]).reset_index(drop=True)
    # 자원 절약: float64 컬럼을 float32로 다운캐스트 (임베딩 학습 프로세스와 메모리를
    # 나눠 쓰는 상황이라 메모리 사용량을 줄임 — CatBoost/BSS 계산 정밀도엔 영향 없음)
    for c in df_raw.select_dtypes(include="float64").columns:
        df_raw[c] = df_raw[c].astype(np.float32)
    for c in df_raw.select_dtypes(include="int64").columns:
        if c not in ("row_id",):
            df_raw[c] = df_raw[c].astype(np.int32)
    print(f"train.csv 로드 완료: {len(df_raw)}행 ({time.time()-t0:.1f}s), 메모리={df_raw.memory_usage(deep=True).sum()/1e6:.0f}MB", flush=True)
    check_memory_or_abort("train.csv 로드 직후")

    # 우리 자신의 파이프라인용 df (top_bottom 0/1 매핑 + F1필터 + 우리 엔지니어링) — MLP 추론용
    import pickle
    with open("./open/reference/best_model.pkl", "rb") as f:
        ref_bundle = pickle.load(f)
    mlp_bundle = ref_bundle["mlp_bundle"]
    from code.mlp_model import get_device
    device = get_device()

    df_ours = df_raw.copy()
    df_ours["top_bottom"] = df_ours["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    league_success_mean_full = df_ours[TARGET].mean()  # 근사(정확한 cutoff7 train_mask 평균은 레짐별로 재계산)

    from code.train import add_engineered_features, apply_same_hand

    results = []

    # === 레짐 1: cutoff7 (프로덕션 승격 기준) ===
    cutoff_train_mask = (df_raw["season"] < 2024) | ((df_raw["season"] == 2024) & (df_raw["game_month"] < 7))
    cutoff_val_mask = (df_raw["season"] == 2024) & (df_raw["game_month"] >= 7)

    league_mean_cutoff = df_ours.loc[cutoff_train_mask, TARGET].mean()
    df_ours_cutoff = add_engineered_features(df_ours.copy(), league_mean_cutoff)
    df_ours_cutoff = apply_same_hand(df_ours_cutoff)
    val_ours_cutoff = df_ours_cutoff.loc[cutoff_val_mask].reset_index(drop=True)

    results.append(run_regime("cutoff7", cutoff_train_mask, cutoff_val_mask, df_raw, val_ours_cutoff, mlp_bundle, device))

    print("\n" + "=" * 70)
    print(f"{'regime':<12}{'cat_solo':>12}{'mlp_solo':>12}{'blend':>12}")
    for r in results:
        print(f"{r['regime']:<12}{r['cat_score']:>12.2f}{r['mlp_score']:>12.2f}{r['blend_score']:>12.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
