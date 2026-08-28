# code/experiment_trackB_mlp.py
"""Track B 경기단위 시계열 피처 6개(code/experiment_trackB_gamelevel_features.py 참고)를
MLP 단독으로 dual-regime(cutoff7 + season==2023) 스크리닝한다. 제안 문서가 실제로
검증했다고 주장한 대상은 CatBoost가 아니라 MLP였고("MLP 단독으로도 717.1 -> 813.5"),
CatBoost 단독 재검증(experiment_trackB_gamelevel_features.py)에서는 cutoff7 기준
-10.92로 이미 마이너스였다 — 프로덕션 피처셋에 이미 시즌진행분/TE-residual/pitchmix가
있어 신호가 흡수됐을 가능성. 이 스크립트는 그 baseline 흡수 가설과 무관하게 "MLP엔
그래도 도움이 되는가"를 우리 파이프라인으로 직접 확인한다.

사용법:
  python -m code.experiment_trackB_mlp --cutoff7
  python -m code.experiment_trackB_mlp --holdout 2023
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.train import TRACKMAN_TIER_FEED, add_engineered_features, apply_f1_filter
from code.trackman_pitcher_features import PITCHMIX_COLS, add_all_tiers, clean_trackman, merge_coarse_pitchmix
from code.experiment_trackB_gamelevel_features import add_trackB_features, TRACKB_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]


def build_mlp_data(holdout, cutoff7, use_trackB):
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
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS  # MLP에는 안 먹임(프로덕션과 동일)

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)
    if use_trackB:
        train_df = add_trackB_features(train_df)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in train_df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS and c not in trk_cat_cols]

    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)
    print(f"[data] 훈련 {len(train_split)}행 | 검증 {len(val_split)}행 | 수치형 {len(num_cols)}개 | 범주형 {len(CAT_COLS)}개")

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


def run_variant(data, seeds, device):
    bin_edges = fit_quantile_edges(data["X_tr_num"], n_bins=QUANTILE_N_BINS)
    t0 = time.time()
    members = train_ensemble(
        data["X_tr_cat"], data["X_tr_num"], data["y_tr"],
        cat_dims=data["cat_dims"], embed_dims=data["embed_dims"], bin_edges=bin_edges,
        X_val_cat=data["X_val_cat"], X_val_num=data["X_val_num"], y_val=data["y_val"],
        seeds=seeds, device=device,
    )
    preds = predict_ensemble(
        members, data["cat_dims"], len(data["num_cols"]), data["embed_dims"],
        data["X_val_cat"], data["X_val_num"], bin_edges=bin_edges, device=device,
    )
    score = compute_bss(preds, data["y_val"])[2]
    print(f"  {len(seeds)}-seed Val Score={score:.2f} ({time.time()-t0:.1f}s, n_num={len(data['num_cols'])})")
    return score


def run_regime(holdout, cutoff7):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    device = get_device()
    print(f"\n{'='*70}\n=== 레짐: {label} (MLP 단독, {SCREEN_SEEDS}) ===\n{'='*70}")

    print("[baseline]")
    data_base = build_mlp_data(holdout, cutoff7, use_trackB=False)
    base_score = run_variant(data_base, SCREEN_SEEDS, device)

    print("[+TrackB 6개]")
    data_trb = build_mlp_data(holdout, cutoff7, use_trackB=True)
    trb_score = run_variant(data_trb, SCREEN_SEEDS, device)

    print(f"\n--- {label} 요약: baseline={base_score:.2f} | +TrackB={trb_score:.2f} | delta={trb_score-base_score:+.2f} ---")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7)


if __name__ == "__main__":
    main()
