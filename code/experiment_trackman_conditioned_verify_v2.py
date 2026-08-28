# code/experiment_trackman_conditioned_verify_v2.py
"""좁힌 2컬럼(induced_vert_break x breaking, 86.73% row-level 매칭 기반,
code/trackman_conditioned_features_v2.py) 변형의 cutoff7 3-seed 결과(solo -20.45,
blend +11.21, tier B와 같은 "solo 나쁜데 블렌드만 구제" 의심 패턴)를 season==2023
듀얼레짐 + 7-seed로 추가 검증한다. 부호가 뒤집히면 그 시점에서 기각.

사용법:
  python -m code.experiment_trackman_conditioned_verify_v2 --holdout 2023
  python -m code.experiment_trackman_conditioned_verify_v2 --holdout 2024 --ensemble
"""
import argparse

import numpy as np

from code.blend_model import fit_meta_model
from code.catboost_model import train_catboost, predict_catboost
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_D, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device,
    predict_ensemble, to_tensors, train_ensemble,
)
from code.thirdmodel_common import build_split, TARGET_COL
from code.trackman_conditioned_features_v2 import merge_trackman_conditioned_v2

SCREEN_SEEDS = [42, 123, 7]


def mlp_run(train_split, val_split, num_cols, seeds, device):
    tr_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    va_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(tr_proc, CAT_COLS, num_cols, TARGET_COL)
    X_va_cat, X_va_num, y_va = to_tensors(va_proc, CAT_COLS, num_cols, TARGET_COL)
    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_va_cat, X_val_num=X_va_num, y_val=y_va, seeds=seeds, device=device, verbose=False,
    )
    preds = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_va_cat, X_va_num,
                              bin_edges=bin_edges, quantile_d=QUANTILE_D, device=device)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--ensemble", action="store_true", default=False)
    args = parser.parse_args()

    cutoff7 = (args.holdout == 2024)
    seeds = ENSEMBLE_SEEDS if args.ensemble else SCREEN_SEEDS
    label = f"cutoff7({len(seeds)}-seed)" if cutoff7 else f"holdout=2023({len(seeds)}-seed)"
    print(f"\n{'='*70}\n=== {label} ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=args.holdout, apply_f1=True)
    y_val = val_split[TARGET_COL].values
    device = get_device()

    cat_model, _ = train_catboost(train_split[cat_feature_cols], train_split[TARGET_COL].values,
                                   val_split[cat_feature_cols], y_val, verbose=False)
    cat_val_preds = predict_catboost(cat_model, val_split[cat_feature_cols])
    cat_score = compute_bss(cat_val_preds, y_val)[2]
    print(f"[CatBoost 고정] {cat_score:.2f}")

    mlp_base_preds = mlp_run(train_split, val_split, mlp_num_cols, seeds, device)
    mlp_base_score = compute_bss(mlp_base_preds, y_val)[2]
    _, _, _, blend_base, _ = fit_meta_model(cat_val_preds, mlp_base_preds, y_val)
    print(f"[MLP baseline] solo={mlp_base_score:.2f} | blend={blend_base:.2f}")

    train_split["_split"] = "train"
    val_split["_split"] = "val"
    import pandas as pd
    combined = pd.concat([train_split, val_split], ignore_index=True)
    combined, cond_cols = merge_trackman_conditioned_v2(
        combined, holdout=args.holdout, metrics=["induced_vert_break"], pitch_types=["breaking"],
    )
    tr2 = combined[combined["_split"] == "train"].drop(columns="_split").reset_index(drop=True)
    va2 = combined[combined["_split"] == "val"].drop(columns="_split").reset_index(drop=True)
    nz_frac = (va2[cond_cols] != 0.0).any(axis=1).mean()
    print(f"[좁힌(2컬럼)] 신규 피처: {cond_cols}, val 0아닌 비율={nz_frac:.2%}")

    mlp_narrow_preds = mlp_run(tr2, va2, mlp_num_cols + cond_cols, seeds, device)
    mlp_narrow_score = compute_bss(mlp_narrow_preds, y_val)[2]
    corr_cat = np.corrcoef(mlp_narrow_preds, cat_val_preds)[0, 1]
    corr_mlp = np.corrcoef(mlp_narrow_preds, mlp_base_preds)[0, 1]
    _, _, _, blend_narrow, _ = fit_meta_model(cat_val_preds, mlp_narrow_preds, y_val)

    print(f"[좁힌(2컬럼)] solo={mlp_narrow_score:.2f} (delta={mlp_narrow_score - mlp_base_score:+.2f}) "
          f"corr(cat)={corr_cat:.4f} corr(mlp_base)={corr_mlp:.4f}")
    print(f"[좁힌(2컬럼)] blend={blend_narrow:.2f} (delta={blend_narrow - blend_base:+.2f})")

    print(f"\n--- {label} 요약 ---")
    print(f"solo delta={mlp_narrow_score - mlp_base_score:+.2f} | blend delta={blend_narrow - blend_base:+.2f}")


if __name__ == "__main__":
    main()
