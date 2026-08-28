# code/experiment_mlp_label_smoothing.py
"""MLP 단독 점수 증강 3탄: 라벨 스무딩. §60(mixup)·§61-63(가우시안 노이즈) 둘 다 기각됐고,
특히 §63에서는 "완전히 동일한 시드·코드로 다시 돌려도 baseline 자체가 6~13점씩 흔들린다"는
재현성 문제 자체가 드러났다 — 이 정도 노이즈에서는 어떤 σ/eps 값을 골라도 신뢰할 수 없다.

원인으로 유력한 것: `nn.Embedding`의 CUDA backward(gradient accumulation에 쓰이는
index_add_/scatter_add_ 계열 연산)는 PyTorch 기본 설정에서 비결정적이다(공식 문서에
명시된 known nondeterminism). 이 스크립트는 라벨 스무딩 실험에 들어가기 전에
`torch.use_deterministic_algorithms(True, warn_only=True)`로 이 문제를 먼저 고치고,
baseline을 두 번 돌려 실제로 재현되는지부터 확인한다 — 재현이 안 되면 eps 그리드
결과 자체가 무의미하므로 먼저 검증하는 게 순서다.

라벨 스무딩: BCELoss의 타깃을 0/1 대신 [eps/2, 1-eps/2]로 살짝 무디게 만든다
(y_smooth = y*(1-eps) + 0.5*eps). 검증 시엔 원래 라벨(0/1)로 그대로 채점(표준 관례).
REL(캘리브레이션)이 블렌드 레벨에서 이미 거의 포화 상태(§56, 7.68pt)라는 진단과
겹쳐 이득이 작을 가능성이 있지만, 이건 MLP solo 레벨의 시도라 별개로 측정한다.

cutoff7, 3-seed, eps 그리드: {baseline_run1, baseline_run2(재현성 확인용), 0.02, 0.05, 0.1}.

사용법: python -m code.experiment_mlp_label_smoothing
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


def train_mlp_smooth(X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                      X_val_cat, X_val_num, y_val, seed, eps, device):
    torch.manual_seed(seed)
    model = TabularMLP(cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    dataset = TensorDataset(X_tr_cat, X_tr_num, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    best_state, best_epoch, best_val_brier, epochs_no_improve = None, 0, float("inf"), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_cat, batch_num, batch_y in loader:
            batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
            if eps is not None:
                batch_y = batch_y * (1 - eps) + 0.5 * eps
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


def run_variant(name, eps, X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                 X_val_cat, X_val_num, y_val, device):
    t0 = time.time()
    preds_list = []
    best_epochs = []
    for seed in SEEDS:
        model, best_epoch = train_mlp_smooth(
            X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
            X_val_cat, X_val_num, y_val, seed, eps, device,
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

    variants = [
        ("baseline_run1(eps 없음)", None),
        ("baseline_run2(재현성 확인)", None),
        ("label_smooth eps=0.02", 0.02),
        ("label_smooth eps=0.05", 0.05),
        ("label_smooth eps=0.1", 0.1),
    ]
    results = {}
    for name, eps in variants:
        score = run_variant(name, eps, X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                             X_val_cat, X_val_num, y_val, device)
        results[name] = score

    b1 = results["baseline_run1(eps 없음)"]
    b2 = results["baseline_run2(재현성 확인)"]
    print("\n" + "=" * 60)
    print(f"재현성 체크: run1={b1:.2f} run2={b2:.2f} diff={abs(b1-b2):.2f}")
    print(f"{'variant':<28}{'score':>10}{'delta(vs run1)':>16}")
    for name, score in results.items():
        print(f"{name:<28}{score:>10.2f}{score-b1:>+16.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
