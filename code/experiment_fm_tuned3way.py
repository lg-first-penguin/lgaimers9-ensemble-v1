# code/experiment_fm_tuned3way.py
"""DeepFM 튜닝판 + 3-way 스태킹 확인. 1차 스크리닝(code/experiment_fm_thirdmodel.py,
solo 393.54)에서 train loss는 계속 내려가는데 val brier는 정체하는 패턴을 봤다 —
pitcher_id/batter_id 필드 쪽 과적합 진단. 문헌/실무 관례 기반 대책 3가지 적용:
  1. 고카디널리티 필드(pitcher_id/batter_id) 임베딩 전용 강한 L2(10x weight_decay)
  2. 필드 임베딩 드롭아웃(V 전체, DeepFM 구현 관례)
  3. patience 완화(7->12, 노이즈 큰 val brier가 조기 종료를 너무 빨리 트리거하지 않게)
`code/experiment_thirdmodel_base.py` 캐시(CatBoost/MLP 예측 재사용)로 3-way delta까지 확인.

사용법: python -m code.experiment_fm_tuned3way [--seeds 42 123 ...]
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch

from code.experiment_thirdmodel_base import load_cache
from code.experiment_3way_stack import fit_meta_model_n
from code.mlp_model import CAT_COLS, compute_bss, fit_preprocessing, apply_preprocessing, fit_quantile_edges, to_tensors, get_device
from code.fm_model import train_deepfm

CACHE_DIR = "./open/temp/experiment_fm_tuned"

TARGET_COL = "control_success"
FM_CAT_COLS = CAT_COLS + ["pitcher_id", "batter_id"]
HIGH_CARD_IDX = [FM_CAT_COLS.index("pitcher_id"), FM_CAT_COLS.index("batter_id")]


def run(seeds, k=8, embed_dropout=0.2, high_card_wd_mult=10.0, patience=12):
    train_split, val_split, cat_val_preds, mlp_val_preds, y_val, meta = load_cache()
    fm_num_cols = [c for c in meta["mlp_num_cols"] if c not in ("pitcher_id", "batter_id")]

    print(f"[FM-tuned] k={k} embed_dropout={embed_dropout} high_card_wd_mult={high_card_wd_mult} patience={patience} seeds={seeds}")
    print(f"[baseline] CatBoost={meta['catboost_score']:.2f} | MLP={meta['mlp_score']:.2f} | 2-way={meta['baseline_2way_score']:.2f}")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        train_split[FM_CAT_COLS + fm_num_cols + [TARGET_COL]], FM_CAT_COLS, fm_num_cols,
    )
    val_proc = apply_preprocessing(val_split[FM_CAT_COLS + fm_num_cols], FM_CAT_COLS, fm_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr_t = to_tensors(train_proc, FM_CAT_COLS, fm_num_cols, TARGET_COL)
    X_val_cat, X_val_num = to_tensors(val_proc, FM_CAT_COLS, fm_num_cols)
    bin_edges = fit_quantile_edges(X_tr_num)
    device = get_device()

    preds_list = []
    members = []
    for seed in seeds:
        t0 = time.time()
        model, best_epoch = train_deepfm(
            X_tr_cat, X_tr_num, y_tr_t, cat_dims=cat_dims, bin_edges=bin_edges, k=k,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val.astype(np.float32),
            device=device, verbose=False, seed=seed, patience=patience,
            embed_dropout=embed_dropout, high_card_cat_idx=HIGH_CARD_IDX,
            high_card_weight_decay=0.01 * high_card_wd_mult,
        )
        with torch.no_grad():
            pred = torch.sigmoid(model(X_val_cat.to(device), X_val_num.to(device))).cpu().numpy()
        preds_list.append(pred)
        members.append({
            "state_dict": {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()},
            "best_epoch": best_epoch, "seed": seed,
        })
        seed_score = compute_bss(pred, y_val)[2]
        print(f"  seed={seed} solo={seed_score:.2f} (best_epoch={best_epoch}, {time.time()-t0:.1f}s)")

    fm_val_preds = np.mean(preds_list, axis=0)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(os.path.join(CACHE_DIR, "fm_val_preds.npz"), fm_val_preds=fm_val_preds, seed_preds=np.array(preds_list))
    with open(os.path.join(CACHE_DIR, "fm_bundle.pkl"), "wb") as f:
        pickle.dump({
            "members": members, "cat_cols": FM_CAT_COLS, "num_cols": fm_num_cols,
            "cat_dims": cat_dims, "k": k, "bin_edges": bin_edges,
            "cat_encoder": cat_encoder, "num_imputer": num_imputer, "num_scaler": num_scaler,
            "high_card_cat_idx": HIGH_CARD_IDX, "embed_dropout": embed_dropout,
            "high_card_weight_decay": 0.01 * high_card_wd_mult,
        }, f)
    print(f"[cache] fm_val_preds + fm_bundle(모델가중치 포함) 저장 완료: {CACHE_DIR}")
    fm_score = compute_bss(fm_val_preds, y_val)[2]
    corr_cat = np.corrcoef(fm_val_preds, cat_val_preds)[0, 1]
    corr_mlp = np.corrcoef(fm_val_preds, mlp_val_preds)[0, 1]

    weights, intercept, score_3way, _ = fit_meta_model_n([cat_val_preds, mlp_val_preds, fm_val_preds], y_val)
    delta = score_3way - meta["baseline_2way_score"]

    print("\n" + "=" * 90)
    print(f"DeepFM({'seed' if len(seeds)==1 else f'{len(seeds)}-seed'})={fm_score:.2f} | corr(cat)={corr_cat:.4f} corr(mlp)={corr_mlp:.4f}")
    print(f"2-way={meta['baseline_2way_score']:.2f} | 3-way(+DeepFM)={score_3way:.2f} (delta {delta:+.2f}) "
          f"| weights cat={weights[0]:.3f} mlp={weights[1]:.3f} fm={weights[2]:.3f} intercept={intercept:.3f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--embed-dropout", type=float, default=0.2)
    parser.add_argument("--high-card-wd-mult", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    run(args.seeds, k=args.k, embed_dropout=args.embed_dropout, high_card_wd_mult=args.high_card_wd_mult, patience=args.patience)


if __name__ == "__main__":
    main()
