# code/experiment_nam_tuned3way.py
"""NAM 튜닝판(ExU 유닛 + feature dropout + output penalty, Agarwal et al. 2021 NAM 논문
기법) + 3-way 스태킹 확인. `code/experiment_thirdmodel_base.py` 캐시(CatBoost/MLP 예측
재사용)로 3-way delta까지 확인한다.

사용법: python -m code.experiment_nam_tuned3way [--seeds 42 123 ...]
"""
import argparse
import os
import time

import numpy as np
import torch

from code.experiment_thirdmodel_base import load_cache
from code.experiment_3way_stack import fit_meta_model_n
from code.mlp_model import CAT_COLS, compute_bss, fit_preprocessing, apply_preprocessing, to_tensors, get_device
from code.nam_model import train_nam

TARGET_COL = "control_success"
NAM_CAT_COLS = CAT_COLS + ["pitcher_id", "batter_id"]
CACHE_DIR = "./open/temp/experiment_nam_tuned"


def run(seeds, hidden=32, feature_dropout=0.1, subnet_dropout=0.1, output_penalty=1e-3, patience=12):
    train_split, val_split, cat_val_preds, mlp_val_preds, y_val, meta = load_cache()
    nam_num_cols = [c for c in meta["mlp_num_cols"] if c not in ("pitcher_id", "batter_id")]

    print(f"[NAM-tuned] hidden={hidden} feature_dropout={feature_dropout} subnet_dropout={subnet_dropout} "
          f"output_penalty={output_penalty} patience={patience} seeds={seeds}")
    print(f"[baseline] CatBoost={meta['catboost_score']:.2f} | MLP={meta['mlp_score']:.2f} | 2-way={meta['baseline_2way_score']:.2f}")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        train_split[NAM_CAT_COLS + nam_num_cols + [TARGET_COL]], NAM_CAT_COLS, nam_num_cols,
    )
    val_proc = apply_preprocessing(val_split[NAM_CAT_COLS + nam_num_cols], NAM_CAT_COLS, nam_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr_t = to_tensors(train_proc, NAM_CAT_COLS, nam_num_cols, TARGET_COL)
    X_val_cat, X_val_num = to_tensors(val_proc, NAM_CAT_COLS, nam_num_cols)
    device = get_device()

    preds_list = []
    for seed in seeds:
        t0 = time.time()
        model, best_epoch = train_nam(
            X_tr_cat, X_tr_num, y_tr_t, cat_dims=cat_dims, hidden=hidden,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val.astype(np.float32),
            device=device, verbose=False, seed=seed, patience=patience,
            use_exu=True, feature_dropout=feature_dropout, subnet_dropout=subnet_dropout,
            output_penalty=output_penalty,
        )
        with torch.no_grad():
            pred = torch.sigmoid(model(X_val_cat.to(device), X_val_num.to(device))).cpu().numpy()
        preds_list.append(pred)
        seed_score = compute_bss(pred, y_val)[2]
        print(f"  seed={seed} solo={seed_score:.2f} (best_epoch={best_epoch}, {time.time()-t0:.1f}s)")

    nam_val_preds = np.mean(preds_list, axis=0)
    nam_score = compute_bss(nam_val_preds, y_val)[2]
    corr_cat = np.corrcoef(nam_val_preds, cat_val_preds)[0, 1]
    corr_mlp = np.corrcoef(nam_val_preds, mlp_val_preds)[0, 1]

    weights, intercept, score_3way, _ = fit_meta_model_n([cat_val_preds, mlp_val_preds, nam_val_preds], y_val)
    delta = score_3way - meta["baseline_2way_score"]

    print("\n" + "=" * 90)
    print(f"NAM({'seed' if len(seeds)==1 else f'{len(seeds)}-seed'})={nam_score:.2f} | corr(cat)={corr_cat:.4f} corr(mlp)={corr_mlp:.4f}")
    print(f"2-way={meta['baseline_2way_score']:.2f} | 3-way(+NAM)={score_3way:.2f} (delta {delta:+.2f}) "
          f"| weights cat={weights[0]:.3f} mlp={weights[1]:.3f} nam={weights[2]:.3f} intercept={intercept:.3f}")
    print("=" * 90)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(os.path.join(CACHE_DIR, "nam_val_preds.npz"), val_preds=nam_val_preds)
    print(f"[cache] saved val_preds -> {CACHE_DIR}/nam_val_preds.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7, 2024, 99, 555, 31337])
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--feature-dropout", type=float, default=0.1)
    parser.add_argument("--subnet-dropout", type=float, default=0.1)
    parser.add_argument("--output-penalty", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    run(args.seeds, hidden=args.hidden, feature_dropout=args.feature_dropout,
        subnet_dropout=args.subnet_dropout, output_penalty=args.output_penalty, patience=args.patience)


if __name__ == "__main__":
    main()
