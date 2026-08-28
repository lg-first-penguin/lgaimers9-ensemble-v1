# code/experiment_nbins_finesweep.py
"""MLP 고도화 2단계(a): quantile/PLE 수치 임베딩의 n_bins를 현재 프로덕션 피처셋(cutoff7,
F1필터, season-progression, coarse pitchmix)으로 다시, 더 촘촘하게 스윕한다.

기존 n_bins 스윕(EXPERIMENTS.md §15, code/experiment_quantile_embed.py)은 {4,12,16,24,32}
5개 점만 봤고(간격 4~8), 그마저도 season-progression/TE-residual/pitchmix가 전부 추가되기
전의 구식 피처셋(`process_trackman_features_safe`, season==2024 전체 홀드아웃)으로 한
것이라 지금 재검증할 가치가 있다. 16과 24가 7/7 승리로 동률이었고 24가 근소 우위였는데,
그 사이(16~24)와 그 너머(24~32, 32 이상)를 더 세밀하게 본 적이 없다.

d(피처당 임베딩 차원, 현재 8 고정)는 이 스크립트의 대상이 아니다 — n_bins만 본다.
CatBoost는 건드리지 않는다(MLP 단독 스코어만 비교).

사용법:
  python -m code.experiment_nbins_finesweep --step screen   # 3-seed, 그리드 전체
  python -m code.experiment_nbins_finesweep --step reverify --bins 20   # 7-seed, 후보 하나 재검증
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from code.mlp_model import (
    CAT_COLS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, to_tensors, train_ensemble,
)
from code.train import TRACKMAN_TIER_FEED, add_engineered_features, apply_f1_filter
from code.trackman_pitcher_features import PITCHMIX_COLS, add_all_tiers, clean_trackman, merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]
FULL_SEEDS = [42, 123, 7, 2024, 99, 555, 31337]
FINE_GRID = [16, 18, 20, 22, 24, 26, 28, 30, 32]


def build_mlp_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_mask = (train_df["season"] < 2024) | ((train_df["season"] == 2024) & (train_df["game_month"] < 7))
    val_mask = (train_df["season"] == 2024) & (train_df["game_month"] >= 7)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    train_df, trk_tier_cols = add_all_tiers(train_df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=2024)
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=2024)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS  # MLP에는 안 먹임(프로덕션과 동일)

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

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


def run_nbins(data, n_bins, seeds, device):
    bin_edges = fit_quantile_edges(data["X_tr_num"], n_bins=n_bins)
    t0 = time.time()
    members = train_ensemble(
        data["X_tr_cat"], data["X_tr_num"], data["y_tr"],
        cat_dims=data["cat_dims"], embed_dims=data["embed_dims"], bin_edges=bin_edges,
        X_val_cat=data["X_val_cat"], X_val_num=data["X_val_num"], y_val=data["y_val"],
        seeds=seeds, device=device,
    )
    from code.mlp_model import predict_ensemble
    preds = predict_ensemble(
        members, data["cat_dims"], len(data["num_cols"]), data["embed_dims"],
        data["X_val_cat"], data["X_val_num"], bin_edges=bin_edges, device=device,
    )
    score = compute_bss(preds, data["y_val"])[2]
    print(f"[n_bins={n_bins}] {len(seeds)}-seed Val Score={score:.2f} ({time.time()-t0:.1f}s)")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["screen", "reverify"])
    parser.add_argument("--bins", type=int, default=24, help="reverify 스텝에서 볼 n_bins 하나")
    args = parser.parse_args()

    print("[data] 전처리된 텐서 준비 중...")
    data = build_mlp_data()
    device = get_device()

    if args.step == "screen":
        print(f"\n=== 3-seed 스크리닝, n_bins 그리드: {FINE_GRID} ===")
        results = {}
        for n_bins in FINE_GRID:
            results[n_bins] = run_nbins(data, n_bins, SCREEN_SEEDS, device)
        best = max(results, key=results.get)
        print(f"\n{'='*70}")
        for n_bins, score in sorted(results.items()):
            marker = " <== best" if n_bins == best else ""
            print(f"  n_bins={n_bins}: {score:.2f}{marker}")
        print(f"{'='*70}")
        return

    if args.step == "reverify":
        print(f"\n=== 7-seed 재검증, n_bins={args.bins} ===")
        score = run_nbins(data, args.bins, FULL_SEEDS, device)
        print(f"[reverify] n_bins={args.bins} 7-seed Val Score={score:.2f}")
        return


if __name__ == "__main__":
    main()
