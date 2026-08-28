# code/experiment_mlp_mixup.py
"""MLP 단독(solo) 점수를 늘리는 신규 시도: mixup(Zhang et al. 2018)을 TabularMLP 학습에
적용한다. 이 프로젝트에서 지금까지 시도된 MLP 강화 방향(capacity 확장 §40, n_bins
재스윕/attention/SSL denoising/메타모델 OOF §51)은 전부 기각됐지만, mixup은 그것들과
메커니즘이 다르다 — capacity를 늘리거나(과거 시도, 전부 기각) 표현을 사전학습하는(SSL
denoising, 기각) 게 아니라, 매 학습 배치에서 두 샘플을 선형보간해 결정 경계를 부드럽게
만드는 순수 정규화(regularization) 기법이라 아직 시도된 적이 없다.

범주형 원본 인덱스는 선형보간이 불가능하므로(임베딩 룩업 전이라 보간 의미가 없음),
manifold mixup 방식으로 구현한다 — TabularMLP의 임베딩+quantile 인코딩을 거쳐 만들어진
concat 벡터(트렁크 직전 입력) 단계에서 두 샘플을 lambda로 보간하고, 라벨도 같은 lambda로
보간해(BCELoss는 [0,1] 실수 타깃도 그대로 받아들임) 학습한다. 검증 시점엔 mixup을 쓰지
않는다(표준 관례).

TabularMLP 클래스 자체는 수정하지 않는다 — 이미 인스턴스화된 model의 하위 모듈
(model.embeddings/model.quantile/model.mlp/model.sigmoid)을 직접 호출해 인코딩 단계와
트렁크 단계를 분리한다(model.forward()를 그대로 쓰면 이 분리가 안 됨).

3-seed 스크리닝, cutoff7 레짐, MLP solo 점수만 비교(블렌드 아님 — 사용자 요청이 "MLP
단독 점수를 늘려보자"였음). alpha 그리드: baseline(mixup 없음) vs {0.2, 0.4, 1.0}.

사용법: python -m code.experiment_mlp_mixup
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

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SEEDS = [42, 123, 7]
ALPHAS = [None, 0.2, 0.4, 1.0]  # None = mixup 없음(baseline, 동일 커스텀 루프로 재확인)
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


def encode_features(model, x_cat, x_num):
    embeds = [emb(x_cat[:, i].long()) for i, emb in enumerate(model.embeddings)]
    x_embed = torch.cat(embeds, dim=1)
    x_num_enc = model.quantile(x_num) if model.quantile is not None else x_num
    return torch.cat([x_embed, x_num_enc], dim=1)


def forward_from_encoded(model, x_all):
    return model.sigmoid(model.mlp(x_all)).squeeze(-1)


def train_mlp_mixup(X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                     X_val_cat, X_val_num, y_val, seed, alpha, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
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
            optimizer.zero_grad()
            x_all = encode_features(model, batch_cat, batch_num)
            if alpha is not None:
                lam = float(rng.beta(alpha, alpha))
                perm = torch.randperm(x_all.shape[0], device=device)
                x_all = lam * x_all + (1 - lam) * x_all[perm]
                batch_y = lam * batch_y + (1 - lam) * batch_y[perm]
            preds = forward_from_encoded(model, x_all)
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


def run_variant(name, alpha, X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                 X_val_cat, X_val_num, y_val, device):
    t0 = time.time()
    preds_list = []
    best_epochs = []
    for seed in SEEDS:
        model, best_epoch = train_mlp_mixup(
            X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
            X_val_cat, X_val_num, y_val, seed, alpha, device,
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
    for alpha in ALPHAS:
        name = "baseline(mixup 없음)" if alpha is None else f"mixup alpha={alpha}"
        score = run_variant(name, alpha, X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                             X_val_cat, X_val_num, y_val, device)
        results[name] = score

    baseline_score = results["baseline(mixup 없음)"]
    print("\n" + "=" * 60)
    print(f"{'variant':<24}{'score':>10}{'delta':>12}")
    for name, score in results.items():
        print(f"{name:<24}{score:>10.2f}{score-baseline_score:>+12.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
