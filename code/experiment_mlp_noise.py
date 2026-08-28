# code/experiment_mlp_noise.py
"""MLP 단독 점수 증강 2탄: 가우시안 노이즈 주입. §60(manifold mixup, 뚜렷한 신호 없이 기각)에
이어 사용자가 선택한 다음 후보 — mixup처럼 서로 다른 두 샘플의 임베딩 공간을 섞는 대신,
한 샘플 자신의 표준화된 수치형 입력에 작은 가우시안 노이즈만 더해 매 epoch마다 조금씩
다른 입력을 보여주는 가장 전통적인 형태의 tabular 증강이다. mixup이 "서로 다른 임베딩
공간을 섞어서 실패했을 가능성"을 배제하려는 목적으로, 이번엔 임베딩을 섞지 않고 원본
수치형(표준화 이후, quantile 인코딩 이전) 텐서에만 노이즈를 준다 — 모델 forward()를
그대로 쓸 수 있어 mixup보다 구현이 단순하다.

수치형은 이미 StandardScaler를 거쳐 (표준 스케일이면) 분산이 1에 가깝게 정규화돼 있으므로,
sigma는 "표준편차 대비 노이즈 비율"로 직접 해석 가능하다. 검증 시점엔 노이즈 미적용(표준
관례). 범주형 임베딩 입력은 건드리지 않는다(정수 인덱스라 노이즈 추가가 의미 없음).

cutoff7 레짐, 3-seed 스크리닝, MLP solo 점수 비교. sigma 그리드: {없음(baseline), 0.01, 0.02, 0.03, 0.04, 0.05}
— §61의 1차 스윕(0.05/0.1/0.2)에서 0.05가 봉우리였지만 season==2023에서 -2.28로 사라졌다.
사용자 요청으로 0.05 이하를 더 촘촘히 재검증한다.

사용법: python -m code.experiment_mlp_noise
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
# §63까지의 스윕은 nn.Embedding CUDA backward 비결정성 때문에 재현이 안 돼 무효였다(§64에서
# 원인 규명+torch.use_deterministic_algorithms로 수정). 재현성 확보 후 0.01 단위로 재스윕.
SIGMAS = [None, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
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


def train_mlp_noise(X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                     X_val_cat, X_val_num, y_val, seed, sigma, device):
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
            if sigma is not None:
                batch_num = batch_num + torch.randn_like(batch_num) * sigma
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


def run_variant(name, sigma, X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                 X_val_cat, X_val_num, y_val, device):
    t0 = time.time()
    preds_list = []
    best_epochs = []
    for seed in SEEDS:
        model, best_epoch = train_mlp_noise(
            X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
            X_val_cat, X_val_num, y_val, seed, sigma, device,
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
    print(f"[Device] {device}")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_val = val_proc[TARGET_COL].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    results = {}
    for sigma in SIGMAS:
        name = "baseline(노이즈 없음)" if sigma is None else f"noise sigma={sigma}"
        score = run_variant(name, sigma, X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                             X_val_cat, X_val_num, y_val, device)
        results[name] = score

    baseline_score = results["baseline(노이즈 없음)"]
    print("\n" + "=" * 60)
    print(f"{'variant':<24}{'score':>10}{'delta':>12}")
    for name, score in results.items():
        print(f"{name:<24}{score:>10.2f}{score-baseline_score:>+12.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
