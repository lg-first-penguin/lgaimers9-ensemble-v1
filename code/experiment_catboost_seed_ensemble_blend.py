# code/experiment_catboost_seed_ensemble_blend.py
"""experiment_catboost_seed_ensemble.py에서 CatBoost 단독 dual-regime 스크리닝이
견고한 플러스(cutoff7 +9.87, season==2023 +31.17, 5-seed)로 나왔다. 메타모델
재가중 착시(핵심 교훈 #23)는 CatBoost 자체 점수가 좋아진 것이므로 위험이 낮지만,
프로젝트 관례상 MLP까지 포함한 전체 블렌드로도 확인한다. cutoff7 레짐에서 MLP
3-seed 스크리닝 규모로 "CatBoost 단일시드(42) 블렌드" vs "CatBoost 5-seed 앙상블
블렌드"를 비교한다.

사용법:
  python -m code.experiment_catboost_seed_ensemble_blend --cutoff7
  python -m code.experiment_catboost_seed_ensemble_blend --holdout 2023
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from code.blend_model import fit_meta_model
from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.train import TRACKMAN_TIER_FEED, add_engineered_features, apply_f1_filter
from code.trackman_pitcher_features import PITCHMIX_COLS, add_all_tiers, clean_trackman, merge_coarse_pitchmix
from code.experiment_catboost_seed_ensemble import build_split as build_cat_split, train_one, SEED_POOL
from code.mlp_model import ENSEMBLE_SEEDS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
MLP_SEEDS_SCREEN = [42, 123, 7]


def build_mlp_data(holdout, cutoff7):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    if cutoff7:
        train_mask = (train_df["season"] < 2024) | ((train_df["season"] == 2024) & (train_df["game_month"] < 7))
        val_mask = (train_df["season"] == 2024) & (train_df["game_month"] >= 7)
    else:
        train_mask = train_df["season"] < holdout
        val_mask = train_df["season"] == holdout

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    train_df, trk_tier_cols = add_all_tiers(train_df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=holdout)
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in train_df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS and c not in trk_cat_cols]

    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_val_np = val_proc[TARGET_COL].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    return {
        "X_tr_cat": X_tr_cat, "X_tr_num": X_tr_num, "y_tr": y_tr,
        "X_val_cat": X_val_cat, "X_val_num": X_val_num, "y_val": y_val_np,
        "cat_dims": cat_dims, "embed_dims": embed_dims, "num_cols": num_cols,
    }


def run_regime(holdout, cutoff7, n_cat_seeds, mlp_seeds):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (CatBoost seed-ensemble + MLP 3-seed, 전체 블렌드) ===\n{'='*70}")

    print("[CatBoost] 데이터 구성")
    X_train, y_train, X_val, y_val = build_cat_split(holdout, cutoff7)
    seeds = SEED_POOL[:n_cat_seeds]
    cat_preds_list = []
    for seed in seeds:
        t0 = time.time()
        preds, best_iter = train_one(seed, X_train, y_train, X_val, y_val)
        print(f"  CatBoost seed={seed}: solo={compute_bss(preds, y_val)[2]:.2f} ({time.time()-t0:.1f}s)")
        cat_preds_list.append(preds)
    cat_pred_single = cat_preds_list[0]
    cat_pred_ens = np.mean(cat_preds_list, axis=0)
    print(f"  CatBoost 1-seed solo={compute_bss(cat_pred_single, y_val)[2]:.2f} | {n_cat_seeds}-seed ensemble solo={compute_bss(cat_pred_ens, y_val)[2]:.2f}")

    print(f"[MLP] {len(mlp_seeds)}-seed 앙상블 학습")
    device = get_device()
    data = build_mlp_data(holdout, cutoff7)
    bin_edges = fit_quantile_edges(data["X_tr_num"], n_bins=QUANTILE_N_BINS)
    t0 = time.time()
    members = train_ensemble(
        data["X_tr_cat"], data["X_tr_num"], data["y_tr"],
        cat_dims=data["cat_dims"], embed_dims=data["embed_dims"], bin_edges=bin_edges,
        X_val_cat=data["X_val_cat"], X_val_num=data["X_val_num"], y_val=data["y_val"],
        seeds=mlp_seeds, device=device,
    )
    mlp_pred = predict_ensemble(
        members, data["cat_dims"], len(data["num_cols"]), data["embed_dims"],
        data["X_val_cat"], data["X_val_num"], bin_edges=bin_edges, device=device,
    )
    print(f"  MLP {len(mlp_seeds)}-seed solo={compute_bss(mlp_pred, data['y_val'])[2]:.2f} ({time.time()-t0:.1f}s)")

    # y_val이 두 경로(카테고리 CatBoost df, MLP 전처리 df)에서 동일한 행 순서로
    # 만들어졌는지 확인 (둘 다 동일한 season/월 마스크로 원본 df를 그대로 슬라이싱)
    assert len(y_val) == len(data["y_val"]) and np.array_equal(y_val, data["y_val"]), "행 순서 불일치"

    for tag, cat_pred in [("단일시드(42) CatBoost", cat_pred_single), (f"{n_cat_seeds}-seed 앙상블 CatBoost", cat_pred_ens)]:
        w_cat, w_mlp, intercept, blend_score, blend_brier = fit_meta_model(cat_pred, mlp_pred, y_val)
        print(f"  [블렌드: {tag}] w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f} Blend Score={blend_score:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--n-cat-seeds", type=int, default=5)
    parser.add_argument("--full-mlp", action="store_true", help="MLP를 프로덕션 7-seed(ENSEMBLE_SEEDS)로 학습 (기본은 3-seed 스크리닝)")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    mlp_seeds = ENSEMBLE_SEEDS if args.full_mlp else MLP_SEEDS_SCREEN
    run_regime(holdout, args.cutoff7, args.n_cat_seeds, mlp_seeds)


if __name__ == "__main__":
    main()
