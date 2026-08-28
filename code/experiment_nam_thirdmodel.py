# code/experiment_nam_thirdmodel.py
"""3rd-모델 탐색 4번 후보 스크리닝: NAM (code/nam_model.py). DeepFM/EBM과 동일하게
pitcher_id/batter_id를 포함한 전체 피처셋을 쓰되, NAM은 상호작용 항이 아예 없다 —
"에러 패턴이 다르려면 상호작용을 표현 못 하는 것도 하나의 답일 수 있는가"를 확인.

사용법: python -m code.experiment_nam_thirdmodel --cutoff7
"""
import argparse
import time

import numpy as np
import torch

from code.experiment_thirdmodel_common import build_full, load_prod_reference, TARGET_COL
from code.mlp_model import CAT_COLS, compute_bss, fit_preprocessing, apply_preprocessing, to_tensors, get_device
from code.nam_model import train_nam

NAM_CAT_COLS = CAT_COLS + ["pitcher_id", "batter_id"]


def run(holdout, cutoff7, hidden=32, seed=42):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} | 3rd 모델: NAM (hidden={hidden}, seed={seed}) ===\n{'='*70}")

    train_split, val_split, features, cat_feature_cols, num_cols_full = build_full(holdout, cutoff7)
    y_train = train_split[TARGET_COL].values
    y_val = val_split[TARGET_COL].values

    prod_pred, prod_score = load_prod_reference(val_split, features)
    print(f"[프로덕션 reference 블렌드] Val Score={prod_score:.2f} (참고용, 동일 val 재구성)")

    nam_num_cols = [c for c in num_cols_full if c not in ("pitcher_id", "batter_id")]

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        train_split[NAM_CAT_COLS + nam_num_cols + [TARGET_COL]], NAM_CAT_COLS, nam_num_cols,
    )
    val_proc = apply_preprocessing(val_split[NAM_CAT_COLS + nam_num_cols], NAM_CAT_COLS, nam_num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr_t = to_tensors(train_proc, NAM_CAT_COLS, nam_num_cols, TARGET_COL)
    X_val_cat, X_val_num = to_tensors(val_proc, NAM_CAT_COLS, nam_num_cols)
    y_val_t = np.asarray(y_val, dtype=np.float32)

    device = get_device()
    print(f"[Device] {device} | n_cat_fields={len(NAM_CAT_COLS)} (incl. pitcher_id/batter_id) | n_num_fields={len(nam_num_cols)}")

    t0 = time.time()
    model, best_epoch = train_nam(
        X_tr_cat, X_tr_num, y_tr_t, cat_dims=cat_dims, hidden=hidden,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_t,
        device=device, verbose=True, seed=seed,
    )
    elapsed = time.time() - t0

    with torch.no_grad():
        pred = torch.sigmoid(model(X_val_cat.to(device), X_val_num.to(device))).cpu().numpy()

    brier, bss, score = compute_bss(pred, y_val)
    corr = np.corrcoef(pred, prod_pred)[0, 1]
    print(f"[NAM] Val Score={score:.2f} (best_epoch={best_epoch}, {elapsed:.1f}s)")
    print(f"[NAM] 프로덕션 블렌드 예측과의 상관계수: {corr:.4f}")
    return score, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run(holdout, args.cutoff7, hidden=args.hidden, seed=args.seed)


if __name__ == "__main__":
    main()
