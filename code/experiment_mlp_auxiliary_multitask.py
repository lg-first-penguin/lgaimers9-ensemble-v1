# code/experiment_mlp_auxiliary_multitask.py
"""MLP 단독 점수 증강/정규화 새 메커니즘 — auxiliary multi-task 학습. 이 프로젝트에서
MLP-solo-boost 라인으로 시도된 것은 지금까지 캐패시티/n_bins/attention/SSL-사전학습/
증강(mixup/noise/label smoothing/cutmix) 뿐이었고, 보조 타겟을 동시에 예측하게 하는
멀티태스크 학습은 미시도였다.

**설계상 제약 명시**: 이 대회는 행별 독립 예측 규칙상 train.csv에 존재하는 유일한
"정답류" 컬럼이 control_success뿐이라(제구 실패 3유형을 구분하는 행별 레이블은 없음),
진짜 새로운 정보를 주는 보조 타겟을 만들 수 없다. 그래서 이미 입력으로 들어가는
asof_pitcher_middle_rate/ball_rate/strike_rate 3개(투수의 커리어 결과 유형 비율,
이미 num_cols에 포함된 입력 피처)를 그대로 보조 회귀 타겟으로 재사용한다 — 네트워크
입력을 그대로 베끼는 게 아니라, HIDDEN2(64차원) 병목을 통과한 표현에서 이 3개 값을
복원하도록 강제하는 구조라(aux head가 raw input이 아니라 압축된 공유 표현만 봄),
분류 헤드만 최적화할 때 손실될 수 있는 정보를 표현에 남겨두도록 압박하는 정규화
효과를 노린다. SSL 사전학습(전체 컬럼 swap-noise denoising, 별도 pretrain 단계,
-27.29로 기각)과는 메커니즘이 다르다 — 이건 처음부터 BCE와 공동 학습(joint)이고,
전체 컬럼이 아니라 일부러 고른 3개 컬럼만 재구성한다.

LAMBDA(보조 손실 가중치) 그리드로 cutoff7 3-seed 스크리닝 후, 방향성 있는 후보만
season==2023로 escalate하는 이 프로젝트의 증강 실험 컨벤션(code/experiment_mlp_noise.py)을
그대로 따른다.

사용법: python -m code.experiment_mlp_auxiliary_multitask
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
    CAT_COLS, QUANTILE_N_BINS, QuantileEmbedding, HIDDEN1, HIDDEN2, DROPOUT,
    apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, to_tensors,
)

torch.use_deterministic_algorithms(True, warn_only=True)

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SEEDS = [42, 123, 7]
LAMBDAS = [None, 0.1, 0.3, 1.0]
AUX_COLS = ["asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"]
BATCH_SIZE = 4096
LR = 0.003
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 60
PATIENCE = 7


class AuxTabularMLP(nn.Module):
    """production TabularMLP과 동일한 trunk(Linear(128)->BN->ReLU->Dropout->Linear(64)->BN->ReLU) +
    기존 main head(Linear(64,1)->Sigmoid) 뒤에, 같은 64차원 공유 표현에서 뻗어나가는
    aux head(Linear(64,n_aux), 활성화 없음 - 표준화된 값 회귀)를 추가한다."""

    def __init__(self, cat_dims, embed_dims, bin_edges, n_aux, quantile_d=8):
        super().__init__()
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        self.quantile = QuantileEmbedding(bin_edges, d_embed=quantile_d)
        numeric_out_dim = bin_edges.shape[0] * quantile_d
        total_input_dim = sum(self.embed_dims) + numeric_out_dim

        self.trunk = nn.Sequential(
            nn.Linear(total_input_dim, HIDDEN1),
            nn.BatchNorm1d(HIDDEN1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.BatchNorm1d(HIDDEN2),
            nn.ReLU(),
        )
        self.main_head = nn.Linear(HIDDEN2, 1)
        self.aux_head = nn.Linear(HIDDEN2, n_aux)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embeds, dim=1)
        x_num_enc = self.quantile(x_num)
        x_all = torch.cat([x_embed, x_num_enc], dim=1)
        shared = self.trunk(x_all)
        main_out = self.sigmoid(self.main_head(shared)).squeeze(-1)
        aux_out = self.aux_head(shared)
        return main_out, aux_out


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


def train_mlp_aux(X_tr_cat, X_tr_num, y_tr, aux_idx, cat_dims, embed_dims, bin_edges,
                   X_val_cat, X_val_num, y_val, seed, lam, device):
    torch.manual_seed(seed)
    model = AuxTabularMLP(cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges, n_aux=len(aux_idx)).to(device)
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    dataset = TensorDataset(X_tr_cat, X_tr_num, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    best_state, best_epoch, best_val_brier, epochs_no_improve = None, 0, float("inf"), 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_cat, batch_num, batch_y in loader:
            batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
            optimizer.zero_grad()
            main_out, aux_out = model(batch_cat, batch_num)
            loss = bce(main_out, batch_y)
            if lam is not None:
                aux_target = batch_num[:, aux_idx]
                loss = loss + lam * mse(aux_out, aux_target)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds, _ = model(X_val_cat.to(device), X_val_num.to(device))
            val_preds = val_preds.cpu().numpy()
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


def run_variant(name, lam, X_tr_cat, X_tr_num, y_tr, aux_idx, cat_dims, embed_dims, bin_edges,
                 X_val_cat, X_val_num, y_val, device):
    t0 = time.time()
    preds_list = []
    best_epochs = []
    for seed in SEEDS:
        model, best_epoch = train_mlp_aux(
            X_tr_cat, X_tr_num, y_tr, aux_idx, cat_dims, embed_dims, bin_edges,
            X_val_cat, X_val_num, y_val, seed, lam, device,
        )
        with torch.no_grad():
            main_out, _ = model(X_val_cat.to(device), X_val_num.to(device))
            preds_list.append(main_out.cpu().numpy())
        best_epochs.append(best_epoch)
    ens_preds = np.mean(preds_list, axis=0)
    _, _, score = compute_bss(ens_preds, y_val)
    print(f"  [{name}] 3-seed ensemble score={score:.2f} best_epochs={best_epochs} (경과 {time.time()-t0:.1f}s)")
    return score


def main():
    train_split, val_split, num_cols = load_cutoff7()
    device = get_device()
    print(f"[Device] {device}")

    aux_idx = [num_cols.index(c) for c in AUX_COLS]
    print(f"[Aux] 보조 타겟 컬럼: {AUX_COLS} -> num_cols 인덱스 {aux_idx}")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_val = val_proc[TARGET_COL].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    results = {}
    for lam in LAMBDAS:
        name = "baseline(보조태스크 없음)" if lam is None else f"lambda={lam}"
        score = run_variant(name, lam, X_tr_cat, X_tr_num, y_tr, aux_idx, cat_dims, embed_dims, bin_edges,
                             X_val_cat, X_val_num, y_val, device)
        results[name] = score

    baseline_score = results["baseline(보조태스크 없음)"]
    print("\n" + "=" * 60)
    print(f"{'variant':<28}{'score':>10}{'delta':>12}")
    for name, score in results.items():
        print(f"{name:<28}{score:>10.2f}{score-baseline_score:>+12.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
