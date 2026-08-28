# code/experiment_te_extra_axes.py
"""Track A 확장 실험 2탄: 팀원이 residual std 상위권으로 짚었던 나머지 미시도 후보 축들을
기존 6축에 하나씩 추가해 CatBoost-only 단일 홀드아웃으로 빠르게 스크리닝한다
(pitcher×count×batter_hand은 이미 테스트해 단일홀드아웃 플러스 -> rolling-origin foldcheck에서
기각 확인됨, code/experiment_te_pcnt_bhand.py 참고). 이번엔:

  - pitcher×runner×count   : ["pitcher_id", "num_runners_on", "balls_before", "strikes_before"]
  - pitcher×inning×count   : ["pitcher_id", "inning", "balls_before", "strikes_before"]
  - pitcher×opponent_team  : ["pitcher_id", "batter_team_id"]

여기서 통과(플러스)하는 축만 상대 세션의 rolling-origin foldcheck로 넘긴다 — 지난번
경험상 단일 홀드아웃 플러스가 fold check에서 뒤집힐 수 있으므로, 이건 최종 판단이 아니라
1차 필터일 뿐이다.

6축 baseline은 build_regime을 레짐당 1번만 만들어 재사용(각 축마다 매번 다시 안 만듦,
CPU 절약). thread_count=4(현재 다른 세션 유휴 상태 확인 후 완화).

사용법: python -m code.experiment_te_extra_axes
"""
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.train import (
    apply_f1_filter, add_engineered_features, TRACKMAN_TIER_FEED,
    TE_AXES, TE_MAIN_AXES, causal_smoothed_te_encode,
)
from code.mlp_model import compute_bss
from code.catboost_model import CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
THREAD_COUNT = 4

CANDIDATE_AXES = [
    ("te_p_run_cnt", ["pitcher_id", "num_runners_on", "balls_before", "strikes_before"], "p_main"),
    ("te_p_inn_cnt", ["pitcher_id", "inning", "balls_before", "strikes_before"], "p_main"),
    ("te_p_oppteam", ["pitcher_id", "batter_team_id"], "p_main"),
]


def apply_te_residuals(source_df, query_df, prior, axes):
    query_df = query_df.copy()
    mains = {}
    covered_any = np.zeros(len(query_df), dtype=np.int64)
    for name, group_cols in TE_MAIN_AXES:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        mains[name] = enc
        covered_any = np.maximum(covered_any, covered)
    res_cols = []
    for name, group_cols, main_key in axes:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        query_df[f"{name}_res"] = enc - mains[main_key]
        covered_any = np.maximum(covered_any, covered)
        res_cols.append(f"{name}_res")
    query_df["te_covered"] = covered_any
    return query_df, res_cols + ["te_covered"]


def load_base():
    t0 = time.time()
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    print(f"[load_base] 완료 (경과 {time.time()-t0:.1f}s)")
    return train_df, pitcher_map, df_trm, df_trm_clean


def build_regime(train_df, pitcher_map, df_trm, df_trm_clean, holdout, train_mask_fn, val_mask_fn):
    df = train_df.copy()
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]
    df = merge_coarse_pitchmix(df, df_trm, holdout=holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    train_mask = train_mask_fn(df)
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    features = [c for c in df.columns if c not in ["row_id", TARGET_COL]]
    cat_feature_cols = [c for c in features if c not in trk_mlp_cols]

    val_mask = val_mask_fn(df)
    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)
    return train_split, val_split, cat_feature_cols


def run_variant(name, train_split, val_split, cat_feature_cols, axes):
    te_prior = train_split[TARGET_COL].mean()
    tr, res_cols = apply_te_residuals(train_split, train_split, te_prior, axes)
    va, _ = apply_te_residuals(train_split, val_split, te_prior, axes)

    cols = cat_feature_cols + res_cols
    X_train, y_train = tr[cols], tr[TARGET_COL]
    X_val, y_val = va[cols], va[TARGET_COL]

    params = dict(CATBOOST_PARAMS)
    params.update(iterations=MAX_ITERATIONS, thread_count=THREAD_COUNT,
                   early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
    model = CatBoostClassifier(**params)
    train_pool = Pool(X_train, y_train, cat_features=["game_type", "base_state"])
    val_pool = Pool(X_val, y_val, cat_features=["game_type", "base_state"])
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    preds = model.predict_proba(X_val)[:, 1]
    _, bss, score = compute_bss(preds, y_val.values)
    print(f"  [{name}] n_features={len(cols)} best_iter={model.get_best_iteration()} score={score:.2f}")
    return score


def run_regime(regime_name, train_df, pitcher_map, df_trm, df_trm_clean, holdout, train_mask_fn, val_mask_fn):
    t0 = time.time()
    train_split, val_split, cat_feature_cols = build_regime(
        train_df, pitcher_map, df_trm, df_trm_clean, holdout, train_mask_fn, val_mask_fn
    )
    print(f"\n[{regime_name}] n_train={len(train_split)} n_val={len(val_split)} (build 경과 {time.time()-t0:.1f}s)")

    baseline = run_variant("6축(기존)", train_split, val_split, cat_feature_cols, TE_AXES)
    rows = [("6축(기존, baseline)", baseline, 0.0)]
    for axis in CANDIDATE_AXES:
        name = axis[0]
        score = run_variant(f"7축(+{name})", train_split, val_split, cat_feature_cols, TE_AXES + [axis])
        rows.append((name, score, score - baseline))
    print(f"[{regime_name}] 전체 경과 {time.time()-t0:.1f}s")
    return regime_name, rows


def main():
    train_df, pitcher_map, df_trm, df_trm_clean = load_base()
    all_results = []
    all_results.append(run_regime(
        "cutoff7", train_df, pitcher_map, df_trm, df_trm_clean, holdout=2024,
        train_mask_fn=lambda df: (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7)),
        val_mask_fn=lambda df: (df["season"] == 2024) & (df["game_month"] >= 7),
    ))
    all_results.append(run_regime(
        "season2023", train_df, pitcher_map, df_trm, df_trm_clean, holdout=2023,
        train_mask_fn=lambda df: df["season"] < 2023,
        val_mask_fn=lambda df: df["season"] == 2023,
    ))

    print("\n" + "=" * 70)
    print(f"{'축':<20}{'cutoff7':>12}{'season2023':>14}{'둘다 플러스?':>14}")
    axis_names = [a[0] for a in CANDIDATE_AXES]
    by_regime = {name: dict((r[0], r[2]) for r in rows) for name, rows in all_results}
    for name in axis_names:
        d1 = by_regime["cutoff7"][name]
        d2 = by_regime["season2023"][name]
        both = "예" if (d1 > 0 and d2 > 0) else "아니오"
        print(f"{name:<20}{d1:>+12.2f}{d2:>+14.2f}{both:>14}")
    print("=" * 70)


if __name__ == "__main__":
    main()
