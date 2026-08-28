# code/experiment_fm_thirdmodel.py
"""3rd-모델 탐색 1번 후보: DeepFM (code/fm_model.py). 전체 피처(트랙맨 pitchmix, TE잔차,
season진행분 등 CatBoost가 보는 것과 동일한 cat_feature_cols 기준 num_cols)를 그대로 쓰되,
pitcher_id/batter_id를 FM 전용 임베딩 필드로 추가한다 — TabularMLP는 두 컬럼을 과적합 때문에
제외했지만, FM의 order-2 저랭크 상호작용이 raw 임베딩+dense 결합보다 더 규제되어 있어
"투수x타자 매치업" 신호를 안전하게 못 뽑아낼 이유가 없는지 확인하는 것이 핵심 가설.

사용법: python -m code.experiment_fm_thirdmodel --cutoff7
"""
import argparse
import time

import numpy as np

from code.experiment_thirdmodel_common import build_full, load_prod_reference, TARGET_COL
from code.mlp_model import CAT_COLS, compute_bss, fit_preprocessing, apply_preprocessing, fit_quantile_edges, to_tensors, get_device
from code.fm_model import train_deepfm, DeepFM

FM_CAT_COLS = CAT_COLS + ["pitcher_id", "batter_id"]


def run(holdout, cutoff7, k=8, seed=42):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} | 3rd 모델: DeepFM (k={k}, seed={seed}) ===\n{'='*70}")

    train_split, val_split, features, cat_feature_cols, num_cols_full = build_full(holdout, cutoff7)
    y_train = train_split[TARGET_COL].values
    y_val = val_split[TARGET_COL].values

    prod_pred, prod_score = load_prod_reference(val_split, features)
    print(f"[프로덕션 reference 블렌드] Val Score={prod_score:.2f} (참고용, 동일 val 재구성)")

    fm_num_cols = [c for c in num_cols_full if c not in ("pitcher_id", "batter_id")]

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        train_split[FM_CAT_COLS + fm_num_cols + [TARGET_COL]], FM_CAT_COLS, fm_num_cols,
    )
    val_proc = apply_preprocessing(val_split[FM_CAT_COLS + fm_num_cols], FM_CAT_COLS, fm_num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr_t = to_tensors(train_proc, FM_CAT_COLS, fm_num_cols, TARGET_COL)
    X_val_cat, X_val_num = to_tensors(val_proc, FM_CAT_COLS, fm_num_cols)
    y_val_t = np.asarray(y_val, dtype=np.float32)

    bin_edges = fit_quantile_edges(X_tr_num)
    device = get_device()
    print(f"[Device] {device} | n_cat_fields={len(FM_CAT_COLS)} (incl. pitcher_id/batter_id) | n_num_fields={len(fm_num_cols)}")

    t0 = time.time()
    model, best_epoch = train_deepfm(
        X_tr_cat, X_tr_num, y_tr_t, cat_dims=cat_dims, bin_edges=bin_edges, k=k,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_t,
        device=device, verbose=True, seed=seed,
    )
    elapsed = time.time() - t0

    import torch
    with torch.no_grad():
        pred = torch.sigmoid(model(X_val_cat.to(device), X_val_num.to(device))).cpu().numpy()

    brier, bss, score = compute_bss(pred, y_val)
    corr = np.corrcoef(pred, prod_pred)[0, 1]
    print(f"[DeepFM] Val Score={score:.2f} (best_epoch={best_epoch}, {elapsed:.1f}s)")
    print(f"[DeepFM] 프로덕션 블렌드 예측과의 상관계수: {corr:.4f}")
    return score, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run(holdout, args.cutoff7, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
