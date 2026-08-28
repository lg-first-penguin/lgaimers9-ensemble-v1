# code/experiment_trackman_conditioned_screen_v2.py
"""code/experiment_trackman_conditioned_screen.py의 재현판 -- 53.39% 대신 86.73%
row-level 매칭(code/trackman_conditioned_features_v2.py)으로 만든 조건화 피처를
cutoff7 3-seed로 재스크리닝. 전체 64컬럼 + 좁힌 2컬럼(induced_vert_break x breaking)
둘 다 확인.

사용법: python -m code.experiment_trackman_conditioned_screen_v2
"""
import time

import numpy as np
import pandas as pd

from code.blend_model import fit_meta_model
from code.experiment_thirdmodel_base import load_cache
from code.mlp_model import (
    CAT_COLS, QUANTILE_D, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.trackman_conditioned_features_v2 import merge_trackman_conditioned_v2

TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]


def run_variant(train_split_base, val_split_base, mlp_num_cols_base, cat_val_preds, mlp_val_preds, y_val,
                 meta, device, label, metrics=None, pitch_types=None):
    train_split = train_split_base.copy()
    val_split = val_split_base.copy()
    train_split["_split"] = "train"
    val_split["_split"] = "val"
    combined = pd.concat([train_split, val_split], ignore_index=True)

    t0 = time.time()
    combined, cond_cols = merge_trackman_conditioned_v2(combined, holdout=2024, metrics=metrics, pitch_types=pitch_types)
    print(f"[{label}] 병합 완료 ({time.time()-t0:.1f}s), 신규 피처 {len(cond_cols)}개")
    nz_frac = (combined[cond_cols] != 0.0).any(axis=1).mean()
    print(f"  0이 아닌 값을 가진 행 비율: {nz_frac:.2%}")

    train_split = combined[combined["_split"] == "train"].drop(columns="_split").reset_index(drop=True)
    val_split = combined[combined["_split"] == "val"].drop(columns="_split").reset_index(drop=True)

    cat_cols = CAT_COLS
    num_cols = mlp_num_cols_base + cond_cols

    train_enc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, cat_cols, num_cols)
    val_enc = apply_preprocessing(val_split, cat_cols, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_enc, cat_cols, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_enc, cat_cols, num_cols, TARGET_COL)
    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=val_enc[TARGET_COL].values,
        seeds=SCREEN_SEEDS, device=device, verbose=False,
    )
    print(f"[{label}] MLP {len(SCREEN_SEEDS)}-seed 학습 완료 ({time.time()-t0:.1f}s)")

    ens_pred = predict_ensemble(
        members, cat_dims, len(num_cols), embed_dims,
        X_val_cat, X_val_num, bin_edges=bin_edges, quantile_d=QUANTILE_D, device=device,
    )
    ens_score = compute_bss(ens_pred, y_val)[2]
    corr_cat = np.corrcoef(ens_pred, cat_val_preds)[0, 1]
    corr_mlp = np.corrcoef(ens_pred, mlp_val_preds)[0, 1]
    print(f"[{label}] {len(SCREEN_SEEDS)}-seed solo={ens_score:.2f} "
          f"(baseline MLP={meta['mlp_score']:.2f}, delta={ens_score - meta['mlp_score']:+.2f}) "
          f"corr(cat)={corr_cat:.4f} corr(mlp)={corr_mlp:.4f}")

    w_cat, w_mlp, intercept, blend2_new, _ = fit_meta_model(cat_val_preds, ens_pred, y_val)
    print(f"[{label}] 2-way(baseline)={meta['baseline_2way_score']:.2f} | "
          f"2-way(new)={blend2_new:.2f} (delta {blend2_new - meta['baseline_2way_score']:+.2f}) "
          f"weights cat={w_cat:.3f} mlp={w_mlp:.3f}")
    return ens_score, blend2_new


def main():
    train_split, val_split, cat_val_preds, mlp_val_preds, y_val, meta = load_cache()
    print(f"[baseline] CatBoost={meta['catboost_score']:.2f} | MLP(7-seed ref)={meta['mlp_score']:.2f} "
          f"| 2-way={meta['baseline_2way_score']:.2f}")

    device = get_device()
    print(f"[Device] {device}")

    print("\n=== 전체 64컬럼 버전 ===")
    solo_full, blend_full = run_variant(
        train_split, val_split, meta["mlp_num_cols"], cat_val_preds, mlp_val_preds, y_val, meta, device,
        label="전체(64컬럼)",
    )

    print("\n=== 좁힌 2컬럼 버전 (induced_vert_break x breaking) ===")
    solo_narrow, blend_narrow = run_variant(
        train_split, val_split, meta["mlp_num_cols"], cat_val_preds, mlp_val_preds, y_val, meta, device,
        label="좁힌(2컬럼)", metrics=["induced_vert_break"], pitch_types=["breaking"],
    )

    print("\n=== 요약 (구버전 53.39% 수치와 비교) ===")
    print(f"전체 64컬럼: solo delta={solo_full - meta['mlp_score']:+.2f} (구버전 -29.88) | blend delta={blend_full - meta['baseline_2way_score']:+.2f} (구버전 -8.16)")
    print(f"좁힌 2컬럼:  solo delta={solo_narrow - meta['mlp_score']:+.2f} (구버전 -39.99) | blend delta={blend_narrow - meta['baseline_2way_score']:+.2f} (구버전 -7.22)")


if __name__ == "__main__":
    main()
