# code/experiment_mlp_cutmix.py
"""MLP 단독 점수 증강 4탄(마지막 후보): CutMix식 이산 feature-swap. mixup(§60)은 서로 다른
두 샘플의 임베딩 공간을 연속적으로 섞어서 실패했을 가능성이 있었다 — 범주형 원본 인덱스는
선형보간이 의미 없기 때문. 이번엔 보간 대신, 원본 입력(임베딩 이전) 단계에서 무작위로
고른 컬럼 일부를 배치 내 다른 샘플의 값으로 통째로 "교체"한다 — 범주형에도 그대로
자연스럽게 적용 가능한 이산적 연산이라 mixup의 잠재적 약점을 피해간다. 라벨은 섞지
않는다(원래 자기 샘플의 라벨 그대로) — CutMix를 표형 데이터에 적용할 때 흔히 쓰는 설계.

매 학습 스텝마다: 배치 순열 perm을 하나 뽑고, 컬럼별로 독립적인 Bernoulli(p) 마스크를
뽑아(범주형/수치형 각각) 마스크가 True인 컬럼만 x[perm]의 값으로 교체한다. 범주형은
인덱스를 그대로 스왑(임베딩 룩업 전이라 의미가 명확), 수치형은 표준화된 값을 그대로
스왑(quantile 인코딩 이전). 검증 시엔 스왑 미적용.

cutoff7, 3-seed, `torch.use_deterministic_algorithms(True, warn_only=True)` 적용(§64에서
확인한 재현성 수정). p 그리드: {없음(baseline), 0.1, 0.2, 0.3}.

사용법: python -m code.experiment_mlp_cutmix
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.train import apply_f1_filter, add_engineered_features
from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, TabularMLP, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device, to_tensors,
)

torch.use_deterministic_algorithms(True, warn_only=True)

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SEEDS = [42, 123, 7]
P_GRID = [None, 0.1, 0.2, 0.3]  # None = 스왑 없음(baseline)
BATCH_SIZE = 4096
LR = 0.003
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 60
PATIENCE = 7


def load_cutoff7():
    t0 = time.time()
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_mask = (train_df['season'] < 2024) | ((train_df['season'] == 2024) & (train_df['game_month'] < 7))
    val_mask = (train_df['season'] == 2024) & (train_df['game_month'] >= 7)

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    features = [c for c in train_df.columns if c not in ['row_id', TARGET_COL]]
    num_cols = [c for c in features if c not in CAT_COLS]

    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)
    print(f"[load_cutoff7] n_train={len(train_split)} n_val={len(val_split)} (경과 {time.time()-t0:.1f}s)")
    return train_split, val_split, num_cols


def train_mlp_cutmix(X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                      X_val_cat, X_val_num, y_val, seed, p, device):
    torch.manual_seed(seed)
    model = TabularMLP(cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    dataset = TensorDataset(X_tr_cat, X_tr_num, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    n_cat, n_num = X_tr_cat.shape[1], X_tr_num.shape[1]

    best_state, best_epoch, best_val_brier, epochs_no_improve = None, 0, float("inf"), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_cat, batch_num, batch_y in loader:
            batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
            if p is not None:
                bsz = batch_cat.shape[0]
                perm = torch.randperm(bsz, device=device)
                mask_cat = (torch.rand(n_cat, device=device) < p).unsqueeze(0)
                mask_num = (torch.rand(n_num, device=device) < p).unsqueeze(0)
                batch_cat = torch.where(mask_cat, batch_cat[perm], batch_cat)
                batch_num = torch.where(mask_num, batch_num[perm], batch_num)
            optimizer.zero_grad()
            preds = model(batch_cat, batch_num)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy()
        val_brier, _, val_score = compute_bss(val_preds, y_val)

        if val_brier < best_val_brier - 1e-9:
            best_val_brier, best_epoch = val_brier, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def run_variant(name, p, X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                 X_val_cat, X_val_num, y_val, device):
    t0 = time.time()
    preds_list = []
    best_epochs = []
    for seed in SEEDS:
        model, best_epoch = train_mlp_cutmix(
            X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
            X_val_cat, X_val_num, y_val, seed, p, device,
        )
        with torch.no_grad():
            preds_list.append(model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy())
        best_epochs.append(best_epoch)
    ens_preds = np.mean(preds_list, axis=0)
    _, _, score = compute_bss(ens_preds, y_val)
    print(f"  [{name}] 3-seed ensemble score={score:.2f} best_epochs={best_epochs} (경과 {time.time()-t0:.1f}s)")
    return score


def main():
    train_split, val_split, num_cols = load_cutoff7()
    device = get_device()
    print(f"[Device] {device} | deterministic_algorithms=True(warn_only)")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_val = val_proc[TARGET_COL].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    results = {}
    for p in P_GRID:
        name = "baseline(스왑 없음)" if p is None else f"cutmix p={p}"
        score = run_variant(name, p, X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                             X_val_cat, X_val_num, y_val, device)
        results[name] = score

    baseline_score = results["baseline(스왑 없음)"]
    print("\n" + "=" * 60)
    print(f"{'variant':<24}{'score':>10}{'delta':>12}")
    for name, score in results.items():
        print(f"{name:<24}{score:>10.2f}{score-baseline_score:>+12.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
