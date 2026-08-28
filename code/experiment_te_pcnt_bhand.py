# code/experiment_te_pcnt_bhand.py
"""Track A(target-encoding 잔차, code/train.py의 TE_AXES) 확장 실험: 팀원이 residual std
최고치(0.0350)로 짚었던 pitcher x count x batter_hand 3-way 축(현재 TE_AXES는 pitcher x
count, pitcher x batter_hand을 각각 2-way로만 갖고 있음, 3-way 조합은 미시도)을 추가했을 때
기존 6축 대비 CatBoost가 더 개선되는지 확인.

CatBoost 전용(다른 세션이 code/train.py로 무거운 학습 중이라 MLP는 재학습 안 하고
CPU thread_count 캡), 단일 시드, 6축 vs 7축(6+신규) 비교, cutoff7/season2023 듀얼레짐.
code/train.py는 읽기(import)만 하고 수정하지 않는다(다른 세션이 활발히 쓰는 파일).

사용법: python -m code.experiment_te_pcnt_bhand
"""
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.train import (
    apply_f1_filter, add_engineered_features, TRACKMAN_TIER_FEED,
    TE_AXES, TE_MAIN_AXES, TE_K_SMOOTH, causal_smoothed_te_encode,
)
from code.mlp_model import compute_bss
from code.catboost_model import CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
THREAD_COUNT = 2

NEW_AXIS = ("te_p_cnt_bhand", ["pitcher_id", "balls_before", "strikes_before", "batter_hand"], "p_main")


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
    print(f"\n[{regime_name}] n_train={len(train_split)} n_val={len(val_split)}")

    score_6 = run_variant("6축(기존)", train_split, val_split, cat_feature_cols, TE_AXES)
    score_7 = run_variant("7축(+pitcher×count×batter_hand)", train_split, val_split, cat_feature_cols,
                           TE_AXES + [NEW_AXIS])
    print(f"[{regime_name}] delta(7축-6축) = {score_7-score_6:+.2f} | 경과 {time.time()-t0:.1f}s")
    return regime_name, score_6, score_7


def main():
    train_df, pitcher_map, df_trm, df_trm_clean = load_base()
    results = []
    results.append(run_regime(
        "cutoff7", train_df, pitcher_map, df_trm, df_trm_clean, holdout=2024,
        train_mask_fn=lambda df: (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7)),
        val_mask_fn=lambda df: (df["season"] == 2024) & (df["game_month"] >= 7),
    ))
    results.append(run_regime(
        "season2023", train_df, pitcher_map, df_trm, df_trm_clean, holdout=2023,
        train_mask_fn=lambda df: df["season"] < 2023,
        val_mask_fn=lambda df: df["season"] == 2023,
    ))

    print("\n" + "=" * 60)
    print(f"{'레짐':<12}{'6축':>12}{'7축(+신규)':>14}{'delta':>10}")
    for name, s6, s7 in results:
        print(f"{name:<12}{s6:>12.2f}{s7:>14.2f}{s7-s6:>+10.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
