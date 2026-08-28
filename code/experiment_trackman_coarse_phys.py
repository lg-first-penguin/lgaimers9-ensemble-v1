# code/experiment_trackman_coarse_phys.py
"""새 트랙맨 후보 스크리닝: coarse pitchmix와 동일한 (balls_before, strikes_before,
pitcher_hand, batter_hand) 축으로만 물리 지표(rel_speed, spin_rate, induced_vert_break,
horz_break, extension, rel_height, rel_side, zone_speed) 평균 8개를 집계한
coarse_phys_*(code/trackman_pitcher_features.py::merge_coarse_physmetrics)를 CatBoost에
추가했을 때 dual-regime(cutoff7 + season==2023)에서 이득이 있는지 확인한다.

tier A(투수 identity 크로스워크 x 구종군별 물리 지표)는 실전 -31.41로 기각됐지만
(PROJECT_HISTORY.md §41), 원인으로 유력한 크로스워크 커버리지 편향(핵심 교훈 #21)은
coarse_phys에는 구조적으로 없다 — coarse pitchmix가 성공한 것과 같은 non-identity 축을
물리 지표에 처음 적용해보는 신규 후보.

baseline은 현재 프로덕션 피처셋 그대로(tier A 없음, coarse pitchmix + season-progression
+ TE-residual 전부 포함, CatBoost-only), variant는 여기에 coarse_phys_* 8개만 추가.

사용법: python -m code.experiment_trackman_coarse_phys
"""
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.train import apply_f1_filter, add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS
from code.mlp_model import compute_bss
from code.catboost_model import CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.trackman_pitcher_features import (
    clean_trackman, merge_coarse_pitchmix, PITCHMIX_COLS,
    merge_coarse_physmetrics, COARSE_PHYS_COLS,
)

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
THREAD_COUNT = 4


def load_base():
    t0 = time.time()
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    print(f"[load_base] 완료 (경과 {time.time()-t0:.1f}s)")
    return train_df, df_trm, df_trm_clean


def build_regime(train_df, df_trm, df_trm_clean, holdout, train_mask_fn, val_mask_fn, with_phys):
    df = train_df.copy()
    df = merge_coarse_pitchmix(df, df_trm, holdout=holdout)
    trk_cat_cols = list(PITCHMIX_COLS)
    if with_phys:
        df = merge_coarse_physmetrics(df, df_trm_clean, holdout=holdout)
        trk_cat_cols = trk_cat_cols + COARSE_PHYS_COLS

    train_mask = train_mask_fn(df)
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    features = [c for c in df.columns if c not in ["row_id", TARGET_COL]]
    val_mask = val_mask_fn(df)
    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)
    cat_feature_cols = features + TE_RESIDUAL_COLS
    return train_split, val_split, cat_feature_cols


def run_variant(name, train_split, val_split, cat_feature_cols):
    X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val, y_val = val_split[cat_feature_cols], val_split[TARGET_COL].values

    params = dict(CATBOOST_PARAMS)
    params.update(iterations=MAX_ITERATIONS, thread_count=THREAD_COUNT,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
    model = CatBoostClassifier(**params)
    train_pool = Pool(X_train, y_train, cat_features=["game_type", "base_state"])
    val_pool = Pool(X_val, y_val, cat_features=["game_type", "base_state"])
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    preds = model.predict_proba(X_val)[:, 1]
    _, bss, score = compute_bss(preds, y_val)
    print(f"  [{name}] n_features={len(cat_feature_cols)} best_iter={model.get_best_iteration()} score={score:.2f}")
    return score


def run_regime(regime_name, train_df, df_trm, df_trm_clean, holdout, train_mask_fn, val_mask_fn):
    t0 = time.time()
    tr_base, va_base, cols_base = build_regime(
        train_df, df_trm, df_trm_clean, holdout, train_mask_fn, val_mask_fn, with_phys=False)
    tr_phys, va_phys, cols_phys = build_regime(
        train_df, df_trm, df_trm_clean, holdout, train_mask_fn, val_mask_fn, with_phys=True)
    print(f"\n[{regime_name}] n_train={len(tr_base)} n_val={len(va_base)} (build 경과 {time.time()-t0:.1f}s)")

    baseline = run_variant("baseline(pitchmix만)", tr_base, va_base, cols_base)
    variant = run_variant("+coarse_phys(8개)", tr_phys, va_phys, cols_phys)
    delta = variant - baseline
    print(f"[{regime_name}] delta={delta:+.2f}")
    return regime_name, baseline, variant, delta


def main():
    train_df, df_trm, df_trm_clean = load_base()
    results = []
    results.append(run_regime(
        "cutoff7", train_df, df_trm, df_trm_clean, holdout=2024,
        train_mask_fn=lambda df: (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7)),
        val_mask_fn=lambda df: (df["season"] == 2024) & (df["game_month"] >= 7),
    ))
    results.append(run_regime(
        "season2023", train_df, df_trm, df_trm_clean, holdout=2023,
        train_mask_fn=lambda df: df["season"] < 2023,
        val_mask_fn=lambda df: df["season"] == 2023,
    ))

    print("\n" + "=" * 70)
    print(f"{'regime':<14}{'baseline':>12}{'+coarse_phys':>14}{'delta':>10}")
    for name, base, var, delta in results:
        print(f"{name:<14}{base:>12.2f}{var:>14.2f}{delta:>+10.2f}")
    both_positive = all(d > 0 for _, _, _, d in results)
    print(f"둘 다 플러스? {'예' if both_positive else '아니오'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
