# code/experiment_ssl_pretrain.py
"""MLP 고도화 4단계: code/ssl_pretrain_mlp.py의 swap-noise denoising 사전학습으로
warm-start한 TabularMLP가 기존 무작위 초기화 대비 나은지 스크리닝한다.

3단계(attention block)가 cutoff7 한 레짐에서만(7-seed까지) 좋아 보이다 season==2023에서
완전히 뒤집힌 전례가 있어서, 이번엔 처음부터 두 레짐을 한 스크립트에서 같이 본다.
사전학습(인코더)은 시드 하나로 한 번만 돌리고(비지도라 라벨 없이 전체 train 행을 다
쓰며, 실제로도 7-seed 앙상블 전체가 같은 사전학습 인코더에서 시작하는 게 자연스러운
사용법이다), fine-tuning만 SCREEN_SEEDS로 반복한다.

이번 단계는 CatBoost/블렌드까지 가지 않고 MLP 단독 앙상블 점수만 비교하는 1차 스크리닝이다
— 두 레짐 다 뚜렷하게 긍정적일 때만 블렌드 확인으로 넘어간다.

사용법:
  python -m code.experiment_ssl_pretrain --regime cutoff7
  python -m code.experiment_ssl_pretrain --regime 2023
"""
import argparse
import time

import numpy as np

from code.experiment_nbins_finesweep import build_mlp_data
from code.experiment_season_progression import build_split
from code.mlp_model import (
    CAT_COLS, apply_preprocessing, compute_bss, embed_dim_for_cardinality, fit_preprocessing,
    fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble, QUANTILE_N_BINS,
)
from code.ssl_pretrain_mlp import build_warmstart_state_dict, finetune_mlp, pretrain_encoder
from code.train import apply_f1_filter

SCREEN_SEEDS = [42, 123, 7]


def build_data(regime):
    if regime == "cutoff7":
        return build_mlp_data()

    holdout = int(regime)
    df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols = build_split(holdout, cutoff7=False)
    drop_cols = ["row_id", "control_success"]
    features = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS and c not in trk_cat_cols]

    train_split = df.loc[train_mask, features + ["control_success"]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + ["control_success"]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)
    print(f"[data] 훈련 {len(train_split)}행 | 검증 {len(val_split)}행 | 수치형 {len(num_cols)}개")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, "control_success")
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, num_cols, "control_success")
    y_val = val_proc["control_success"].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    return {
        "X_tr_cat": X_tr_cat, "X_tr_num": X_tr_num, "y_tr": y_tr,
        "X_val_cat": X_val_cat, "X_val_num": X_val_num, "y_val": y_val,
        "cat_dims": cat_dims, "embed_dims": embed_dims, "num_cols": num_cols,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", required=True, choices=["cutoff7", "2023"])
    args = parser.parse_args()

    print(f"[data] {args.regime} 텐서 준비 중...")
    data = build_data(args.regime)
    device = get_device()
    bin_edges = fit_quantile_edges(data["X_tr_num"], n_bins=QUANTILE_N_BINS)

    print(f"\n=== baseline(무작위 초기화 TabularMLP) 3-seed ===")
    base_preds_list = []
    for seed in SCREEN_SEEDS:
        t0 = time.time()
        members = train_ensemble(
            data["X_tr_cat"], data["X_tr_num"], data["y_tr"], cat_dims=data["cat_dims"],
            embed_dims=data["embed_dims"], bin_edges=bin_edges,
            X_val_cat=data["X_val_cat"], X_val_num=data["X_val_num"], y_val=data["y_val"],
            seeds=[seed], device=device,
        )
        preds = predict_ensemble(
            members, data["cat_dims"], len(data["num_cols"]), data["embed_dims"],
            data["X_val_cat"], data["X_val_num"], bin_edges=bin_edges, device=device,
        )
        base_preds_list.append(preds)
        print(f"  [baseline] seed={seed} 개별 Val Score={compute_bss(preds, data['y_val'])[2]:.2f} ({time.time()-t0:.1f}s)")
    base_score = compute_bss(np.mean(base_preds_list, axis=0), data["y_val"])[2]
    print(f"[baseline] 3-seed 앙상블 점수: {base_score:.2f}")

    print(f"\n=== swap-noise denoising 사전학습 (라벨 없음, 1회) ===")
    t0 = time.time()
    encoder = pretrain_encoder(
        data["X_tr_cat"], data["X_tr_num"], cat_dims=data["cat_dims"], embed_dims=data["embed_dims"],
        bin_edges=bin_edges, device=device, verbose=True, seed=42,
    )
    warmstart_sd = build_warmstart_state_dict(encoder)
    print(f"[pretrain] 완료 ({time.time()-t0:.1f}s)")

    print(f"\n=== SSL warm-start TabularMLP 3-seed fine-tune ===")
    ssl_preds_list = []
    for seed in SCREEN_SEEDS:
        t0 = time.time()
        model, best_epoch = finetune_mlp(
            data["X_tr_cat"], data["X_tr_num"], data["y_tr"], cat_dims=data["cat_dims"],
            warmstart_state_dict=warmstart_sd,
            X_val_cat=data["X_val_cat"], X_val_num=data["X_val_num"], y_val=data["y_val"],
            embed_dims=data["embed_dims"], bin_edges=bin_edges, device=device, verbose=False, seed=seed,
        )
        preds = model(data["X_val_cat"].to(device), data["X_val_num"].to(device)).detach().cpu().numpy()
        ssl_preds_list.append(preds)
        print(f"  [ssl] seed={seed} best_epoch={best_epoch} 개별 Val Score={compute_bss(preds, data['y_val'])[2]:.2f} ({time.time()-t0:.1f}s)")
    ssl_score = compute_bss(np.mean(ssl_preds_list, axis=0), data["y_val"])[2]
    print(f"[ssl] 3-seed 앙상블 점수: {ssl_score:.2f}")

    print(f"\n{'='*70}")
    print(f"[{args.regime}] delta(ssl warm-start vs baseline): {ssl_score-base_score:+.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
