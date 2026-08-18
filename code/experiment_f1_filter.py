# code/experiment_f1_filter.py
"""팀원 제보 — F1 필터(2022 이하 시즌의 game_type=='F' 행 제거) 검증 실험.

배경: game_type별 제구 성공률을 시즌별로 보면 2022년까지 F(퓨처스/2군)가 R(1군)보다
최대 +20.5%p 높았으나 2023년부터 오히려 -3.0%p로 역전됐다(직접 재현 확인 완료).
game_type은 CatBoost feature importance 1위(팀원 리포트 기준 25.4%)라 이 역전이 모델에
큰 혼란을 준다. 조치: 2022년 이하 시즌의 F 행을 학습에서만 제거한다.

    train = train[~((train["game_type"] == "F") & (train["season"] <= 2022))]

학습 데이터에만 적용하고 검증/추론 데이터는 원본을 유지한다 — 2023 이후 F는 새 관계가
유효하므로 남긴다(팀원 리포트: 전량 제거 시 2024 -105점 손해).

이 스크립트는 이번 세션에서 확정한 "트랙맨 피처 제거" 방향(experiment_trackman_ablation.py,
season==2024 기준 795.13) 위에 F1 필터를 얹어 순수하게 F1의 한계 기여도만 격리해서 본다.

사용법:
  python -m code.experiment_f1_filter --holdout 2024
  python -m code.experiment_f1_filter --holdout 2023
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from code.blend_model import fit_meta_model
from code.catboost_model import predict_catboost, train_catboost
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device,
    make_bundle, predict_bundle, to_tensors, train_ensemble,
)
from code.train import add_engineered_features

DATA_DIR = "./open/data"
TARGET_COL = "control_success"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024, choices=[2023, 2024])
    parser.add_argument("--apply-f1", dest="apply_f1", action="store_true", default=True)
    parser.add_argument("--no-f1", dest="apply_f1", action="store_false")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    train_mask = df["season"] < args.holdout
    val_mask = df["season"] == args.holdout

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS]

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)

    label = f"holdout={args.holdout} f1={'ON' if args.apply_f1 else 'OFF'}"
    if args.apply_f1:
        before = len(train_split)
        train_split = train_split[~((train_split["game_type"] == "F") & (train_split["season"] <= 2022))].reset_index(drop=True)
        print(f"[{label}] F1 필터 적용: {before} -> {len(train_split)}행 ({before - len(train_split)}행 제거)")
    else:
        print(f"[{label}] F1 필터 미적용")

    print(f"[{label}] 훈련: {len(train_split)}행 | 검증: {len(val_split)}행")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_val_np = val_proc[TARGET_COL].values

    device = get_device()
    print(f"[Device] {device}")

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
        seeds=ENSEMBLE_SEEDS, device=device,
    )
    print(f"[{label}] MLP 앙상블 학습 완료 ({time.time() - t0:.1f}s)")

    mlp_bundle = make_bundle(
        members, CAT_COLS, num_cols, cat_dims, embed_dims,
        cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges,
    )

    X_train_raw, y_train_raw = train_split[features], train_split[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[features], val_split[TARGET_COL].values
    t0 = time.time()
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    print(f"[{label}] CatBoost 학습 완료 (best_iteration={catboost_best_iteration}, {time.time() - t0:.1f}s)")

    mlp_val_preds = predict_bundle(mlp_bundle, X_val_raw, device=device)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)

    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    mlp_score = compute_bss(mlp_val_preds, y_val_raw)[2]
    w_cat, w_mlp, intercept, blend_score, blend_brier = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)

    print(f"\n[RESULT {label}] CatBoost={cat_score:.2f} | MLP={mlp_score:.2f} | Blend={blend_score:.2f}")
    print(f"  (w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f})")


if __name__ == "__main__":
    main()
