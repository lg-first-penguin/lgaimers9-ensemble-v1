# code/experiment_trackman_ablation.py
"""트랙맨 파생 피처를 아예 드롭했을 때 성능이 어떻게 되는지 검증하는 ablation 실험.

배경: `merge_trackman_features`의 season 등호 매칭은 실제 제출(test.csv, season==2025)에서
구조적으로 항상 실패한다. 학습 시점(season<2024)에는 트랙맨 커버리지(2019~2024) 안에서
season이 항상 정확히 매칭되므로, 모델은 "정밀한 season-exact 매칭" 분포로 트랙맨 피처를
학습한다. 반면 실제 추론 시점에는 "season 등호 실패 -> 9-key로 넓게 재매칭"된, 학습 때와는
분포가 다른(더 뭉뚱그려진) 값을 보게 된다 — 학습/서빙 분포 불일치(train-serve skew)다.

`code/experiment_trackman_season_fallback.py`는 기존 레퍼런스 번들(트랙맨 포함, season-exact로
학습됨)을 그대로 재사용해 이 skew가 있는 채로 fallback 매칭을 적용했을 때의 점수(752.42)를
측정했다 — broken(730.25, 전부 0)보다는 낫지만 as-is(880.38, 학습·평가 모두 트랙맨 정상)보다는
많이 못 미친다.

이 스크립트는 다른 질문에 답한다: "애초에 트랙맨 피처를 학습에 포함시키지 않았다면 어땠을까?"
트랙맨 피처를 완전히 제거한 채 CatBoost+MLP를 처음부터 재학습해서, 그 결과를 752.42(fallback로
유지)와 직접 비교한다. 이 ablation이 752.42보다 높으면 "fallback 패치보다 아예 드롭이 낫다"는
뜻이고, 낮으면 "skew가 있어도 트랙맨 신호가 여전히 순가치를 더한다"는 뜻이다.

사용법:
  python -m code.experiment_trackman_ablation
"""
import os
import pickle
import time

import numpy as np
import pandas as pd

from code.blend_model import fit_meta_model, predict_meta
from code.catboost_model import predict_catboost, train_catboost
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device,
    make_bundle, predict_bundle, to_tensors, train_ensemble,
)
from code.train import add_engineered_features

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
CACHE_PATH = "./open/temp/experiment_trackman_ablation_bundle.pkl"


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")

    train_mask = df["season"] < 2024
    val_mask = df["season"] == 2024
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()

    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS]
    print(f"트랙맨 피처 없이 학습 | 총 피처 수: {len(features)} (범주형 {len(CAT_COLS)}, 수치형 {len(num_cols)})")

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    y_val_np = val_split[TARGET_COL].values
    print(f"훈련 데이터: {len(train_split)}행 | 검증 데이터: {len(val_split)}행")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)

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
    print(f"[MLP 앙상블] 학습 완료 ({time.time() - t0:.1f}s)")

    mlp_bundle = make_bundle(
        members, CAT_COLS, num_cols, cat_dims, embed_dims,
        cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges,
    )

    print("\n--- [CatBoost] 트랙맨 없이 학습 ---")
    X_train_raw, y_train_raw = train_split[features], train_split[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[features], val_split[TARGET_COL].values
    t0 = time.time()
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    print(f"[CatBoost] 학습 완료 (best_iteration={catboost_best_iteration}, {time.time() - t0:.1f}s)")

    mlp_val_preds = predict_bundle(mlp_bundle, X_val_raw, device=device)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)

    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    mlp_score = compute_bss(mlp_val_preds, y_val_raw)[2]
    w_cat, w_mlp, intercept, blend_score, blend_brier = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)

    print(f"\n[트랙맨 완전 제거] CatBoost={cat_score:.2f} | MLP={mlp_score:.2f} | Blend={blend_score:.2f}")
    print(f"  (w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f})")

    print("\n--- 비교 (모두 동일 val: season==2024) ---")
    print(f"  broken   (트랙맨 포함, season skew로 전부 0)         : 730.25")
    print(f"  fallback (트랙맨 포함, season-drop 재매칭 패치)      : 752.42")
    print(f"  ablation (트랙맨 피처 자체를 학습에서 제거)          : {blend_score:.2f}")
    print(f"  as-is    (트랙맨 포함, 참고용 — 실제 배포에서 불가능): 880.38")
    print(f"\ndelta (ablation - fallback): {blend_score - 752.42:+.2f}")

    os.makedirs("./open/temp", exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "catboost_model": catboost_model, "mlp_bundle": mlp_bundle,
            "meta_model": {"w_cat": w_cat, "w_mlp": w_mlp, "intercept": intercept},
            "catboost_best_iteration": catboost_best_iteration,
            "cat_score": cat_score, "mlp_score": mlp_score, "blend_score": blend_score,
        }, f)
    print(f"\n번들 캐시 저장: {CACHE_PATH}")


if __name__ == "__main__":
    main()
