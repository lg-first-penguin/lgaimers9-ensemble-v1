# code/experiment_thirdmodel_rolling_deepfm.py
"""DeepFM 3-way delta를 2020/2021/2022/2023/2024 season-level rolling-origin으로
재검증한다(`code/experiment_thirdmodel_rolling_base.py`의 CatBoost/MLP fold 캐시 재사용).
§67.3 튜닝 설정 그대로(k=8, embed_dropout=0.2, high-card 10x weight_decay, patience=12),
fold당 3-seed(SCREEN_SEEDS, 프로젝트 rolling-origin 관례) 재학습.

사용법: python -m code.experiment_thirdmodel_rolling_deepfm
"""
import time

import numpy as np
import torch

from code.blend_model import fit_meta_model, predict_meta
from code.experiment_3way_stack import fit_meta_model_n
from code.experiment_thirdmodel_rolling_base import FOLD_SEASONS, SCREEN_SEEDS, get_fold_base, load_r_only
from code.fm_model import train_deepfm
from code.mlp_model import CAT_COLS, apply_preprocessing, compute_bss, fit_preprocessing, fit_quantile_edges, get_device, to_tensors

TARGET_COL = "control_success"
FM_K = 8
FM_EMBED_DROPOUT = 0.2
FM_HIGH_CARD_WD = 0.1
FM_PATIENCE = 12


def run_fm_fold(fold_data, device):
    train_split, val_split, num_cols = fold_data["train_split"], fold_data["val_split"], fold_data["num_cols"]
    y_val = fold_data["y_val"]

    fm_cat_cols = CAT_COLS + ["pitcher_id", "batter_id"]
    fm_num_cols = [c for c in num_cols if c not in ("pitcher_id", "batter_id")]
    high_card_idx = [fm_cat_cols.index("pitcher_id"), fm_cat_cols.index("batter_id")]

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        train_split[fm_cat_cols + fm_num_cols + [TARGET_COL]], fm_cat_cols, fm_num_cols,
    )
    val_proc = apply_preprocessing(val_split[fm_cat_cols + fm_num_cols], fm_cat_cols, fm_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, fm_cat_cols, fm_num_cols, TARGET_COL)
    X_val_cat, X_val_num = to_tensors(val_proc, fm_cat_cols, fm_num_cols)
    bin_edges = fit_quantile_edges(X_tr_num)

    preds_list = []
    for seed in SCREEN_SEEDS:
        model, best_epoch = train_deepfm(
            X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, bin_edges=bin_edges, k=FM_K,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val.astype(np.float32),
            device=device, verbose=False, seed=seed, patience=FM_PATIENCE,
            embed_dropout=FM_EMBED_DROPOUT, high_card_cat_idx=high_card_idx,
            high_card_weight_decay=FM_HIGH_CARD_WD,
        )
        with torch.no_grad():
            pred = torch.sigmoid(model(X_val_cat.to(device), X_val_num.to(device))).cpu().numpy()
        preds_list.append(pred)
    return np.mean(preds_list, axis=0)


def main():
    df_all, df_trm = load_r_only()
    device = get_device()

    fold_results = {}
    for val_season in FOLD_SEASONS:
        print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===")
        fold_data = get_fold_base(val_season, df_all, df_trm, device)
        t0 = time.time()
        fm_pred = run_fm_fold(fold_data, device)
        print(f"  [val={val_season}] DeepFM(3-seed) 완료 ({time.time()-t0:.1f}s)")
        fold_results[val_season] = (fold_data["cat_pred"], fold_data["mlp_pred"], fm_pred, fold_data["y_val"])

    print(f"\n{'='*90}\nfold별 2-way vs 3-way(+DeepFM)\n{'='*90}")
    deltas = []
    for val_season in FOLD_SEASONS:
        cat_p, mlp_p, fm_p, y = fold_results[val_season]
        w_cat, w_mlp, ic, _, _ = fit_meta_model(cat_p, mlp_p, y)
        score_2way = compute_bss(predict_meta(w_cat, w_mlp, ic, cat_p, mlp_p), y)[2]

        weights, intercept, score_3way, _ = fit_meta_model_n([cat_p, mlp_p, fm_p], y)
        delta = score_3way - score_2way
        deltas.append(delta)
        fm_score = compute_bss(fm_p, y)[2]
        corr_cat, corr_mlp = np.corrcoef(fm_p, cat_p)[0, 1], np.corrcoef(fm_p, mlp_p)[0, 1]
        print(f"[val={val_season}] DeepFM solo={fm_score:.2f} corr(cat)={corr_cat:.3f} corr(mlp)={corr_mlp:.3f} | "
              f"2-way={score_2way:.2f} | 3-way={score_3way:.2f} (delta {delta:+.2f})")

    wins = sum(d > 0 for d in deltas)
    print(f"\n{'='*90}\n{wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}, 표준편차: {np.std(deltas):.2f}\n{'='*90}")


if __name__ == "__main__":
    main()
