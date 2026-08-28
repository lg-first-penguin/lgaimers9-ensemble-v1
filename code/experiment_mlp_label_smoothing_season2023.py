# code/experiment_mlp_label_smoothing_season2023.py
"""code/experiment_mlp_label_smoothing.py의 cutoff7 스크리닝(재현성 확인된 버전)에서
eps=0.05가 +10.33으로 가장 좋았다. season==2023 재검증 — 이 프로젝트 관례상 승자
후보 하나만 두 번째 레짐으로 가져간다.

사용법: python -m code.experiment_mlp_label_smoothing_season2023
"""
import os
import time

import numpy as np
import pandas as pd
import torch

from code.train import apply_f1_filter, add_engineered_features
from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, apply_preprocessing, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, to_tensors,
)
from code.experiment_mlp_label_smoothing import run_variant, TARGET_COL

torch.use_deterministic_algorithms(True, warn_only=True)

DATA_DIR = "./open/data"
EPS_CANDIDATES = [None, 0.05]


def load_season2023():
    t0 = time.time()
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_mask = train_df['season'] < 2023
    val_mask = train_df['season'] == 2023

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    features = [c for c in train_df.columns if c not in ['row_id', TARGET_COL]]
    num_cols = [c for c in features if c not in CAT_COLS]

    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)
    print(f"[load_season2023] n_train={len(train_split)} n_val={len(val_split)} (경과 {time.time()-t0:.1f}s)")
    return train_split, val_split, num_cols


def main():
    train_split, val_split, num_cols = load_season2023()
    device = get_device()
    print(f"[Device] {device}")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_val = val_proc[TARGET_COL].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    results = {}
    for eps in EPS_CANDIDATES:
        name = "baseline(eps 없음)" if eps is None else f"label_smooth eps={eps}"
        score = run_variant(name, eps, X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                             X_val_cat, X_val_num, y_val, device)
        results[name] = score

    baseline_score = results["baseline(eps 없음)"]
    print("\n" + "=" * 60)
    print(f"{'variant':<24}{'score':>10}{'delta':>12}")
    for name, score in results.items():
        print(f"{name:<24}{score:>10.2f}{score-baseline_score:>+12.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
