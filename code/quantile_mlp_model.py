# code/quantile_mlp_model.py
"""수치형 피처용 quantile 기반 piecewise-linear 인코딩(PLE) 임베딩 실험
(`code/experiment_quantile_embed.py`).

Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep Learning"의
Q-LR(Quantile-based Piecewise Linear encoding + Linear + ReLU) 임베딩을 이 프로젝트의
`TabularMLP`(code/mlp_model.py)에 적용합니다. `code/periodic_mlp_model.py`(sin/cos
주파수 기반 PLR)와 뒷단 구조(피처별 독립 Linear + ReLU)는 동일하고, 앞단 인코딩만
"주파수"에서 "학습 데이터 분포의 quantile 구간"으로 바뀝니다 — periodic이 노이즈
수준으로 판정된 것(EXPERIMENTS.md §15)과 다른 결과가 나오는지 확인하기 위한 대조 실험.

피처 x가 구간 j(경계 [e_j, e_{j+1}])에 속할 때 인코딩 벡터 v는:
  - j'< j (왼쪽 구간들): v_j' = 1
  - j'== j (자기 구간): v_j = (x - e_j) / (e_{j+1} - e_j) (0~1로 clamp)
  - j'> j (오른쪽 구간들): v_j' = 0
구간 경계(quantile)는 트레인 스플릿에서만 계산합니다(리크 방지, `fit_preprocessing`과
동일한 컨벤션) — `code/mlp_model.py::fit_preprocessing`이 이미 표준화한 값에 대해
quantile을 잡습니다. StandardScaler는 순위를 보존하는 단조변환이라 bin 소속 자체는
원본 값 기준과 동일합니다.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.mlp_model import embed_dim_for_cardinality, compute_bss, get_device, HIDDEN1, HIDDEN2, DROPOUT

LR = 0.003
WEIGHT_DECAY = 0.01
BATCH_SIZE = 4096
MAX_EPOCHS = 60
PATIENCE = 7


def fit_quantile_edges(X_num, n_bins=8):
    """(N, num_numeric) 학습 텐서/배열에서 피처별 quantile 경계 (num_numeric, n_bins+1)를 계산합니다.
    중복 경계(카디널리티가 낮은 피처)는 단조 증가를 보장하도록 아주 작은 epsilon으로 밀어냅니다."""
    X_np = X_num.numpy() if torch.is_tensor(X_num) else np.asarray(X_num)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(X_np, qs, axis=0).T  # (num_numeric, n_bins+1)
    edges = np.maximum.accumulate(edges, axis=1)
    for i in range(1, edges.shape[1]):
        tie = edges[:, i] <= edges[:, i - 1]
        edges[tie, i] = edges[tie, i - 1] + 1e-6
    return torch.tensor(edges, dtype=torch.float32)


class QuantileEmbedding(nn.Module):
    """수치형 피처별 quantile 기반 piecewise-linear 인코딩(PLE) + 피처별 독립 Linear + ReLU."""

    def __init__(self, bin_edges, d_embed=8, use_relu=True):
        super().__init__()
        self.register_buffer("edges", bin_edges)  # (num_numeric, n_bins+1), 학습 안 함
        num_numeric, n_bins_plus1 = bin_edges.shape
        n_bins = n_bins_plus1 - 1
        self.num_numeric = num_numeric
        self.n_bins = n_bins
        self.use_relu = use_relu
        self.weight = nn.Parameter(torch.empty(num_numeric, n_bins, d_embed))
        self.bias = nn.Parameter(torch.zeros(num_numeric, d_embed))
        bound = 1.0 / math.sqrt(n_bins)
        nn.init.uniform_(self.weight, -bound, bound)

    def encode(self, x_num):
        left = self.edges[:, :-1].unsqueeze(0)   # (1, num_numeric, n_bins)
        right = self.edges[:, 1:].unsqueeze(0)    # (1, num_numeric, n_bins)
        x = x_num.unsqueeze(-1)                   # (batch, num_numeric, 1)
        width = (right - left).clamp_min(1e-6)
        frac = (x - left) / width
        return frac.clamp(0.0, 1.0)                # (batch, num_numeric, n_bins)

    def forward(self, x_num):
        p = self.encode(x_num)
        e = torch.einsum("bnf,nfd->bnd", p, self.weight) + self.bias
        if self.use_relu:
            e = torch.relu(e)
        return e.reshape(e.shape[0], -1)


class TabularMLPQuantile(nn.Module):
    """`code/mlp_model.py::TabularMLP`와 동일한 구조(128->64->1)이되, 수치형 입력을
    표준화 값 그대로 concat하는 대신 QuantileEmbedding(PLE)을 거칩니다."""

    def __init__(self, bin_edges, cat_dims, embed_dims=None, quantile_d=8, quantile_relu=True):
        super().__init__()
        num_numeric_feats = bin_edges.shape[0]
        if embed_dims is None:
            embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        self.quantile = QuantileEmbedding(bin_edges, d_embed=quantile_d, use_relu=quantile_relu)
        total_input_dim = sum(self.embed_dims) + num_numeric_feats * quantile_d

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, HIDDEN1),
            nn.BatchNorm1d(HIDDEN1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.BatchNorm1d(HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embeds, dim=1)
        x_quantile = self.quantile(x_num)
        x_all = torch.cat([x_embed, x_quantile], dim=1)
        return self.sigmoid(self.mlp(x_all)).squeeze(-1)


def train_mlp_quantile(
    X_tr_cat, X_tr_num, y_tr, cat_dims, bin_edges,
    X_val_cat=None, X_val_num=None, y_val=None,
    embed_dims=None, quantile_d=8, quantile_relu=True,
    max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
    lr=LR, weight_decay=WEIGHT_DECAY, device=None, verbose=True, seed=42,
):
    """`code/mlp_model.py::train_mlp`/`code/periodic_mlp_model.py::train_mlp_periodic`와
    동일한 early-stopping 학습 루프이되, TabularMLPQuantile을 사용합니다."""
    torch.manual_seed(seed)
    device = device or get_device()
    model = TabularMLPQuantile(
        bin_edges=bin_edges.to(device), cat_dims=cat_dims, embed_dims=embed_dims,
        quantile_d=quantile_d, quantile_relu=quantile_relu,
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(X_tr_cat, X_tr_num, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    has_val = X_val_cat is not None and y_val is not None and len(y_val) > 0

    best_state = None
    best_epoch = 0
    best_val_brier = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_cat, batch_num, batch_y in loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_cat, batch_num)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(loader), 1)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy()
            val_brier, val_bss, val_score = compute_bss(val_preds, y_val)
            if verbose:
                print(f"Epoch {epoch}/{max_epochs} | Train Loss: {avg_loss:.5f} | Val Brier: {val_brier:.6f} | Val Score: {val_score:.2f}")

            if val_brier < best_val_brier - 1e-9:
                best_val_brier = val_brier
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"[EarlyStopping] Val Brier 개선 없음 {patience} epoch 지속 -> epoch {epoch}에서 종료 (best epoch: {best_epoch})")
                    break
        else:
            if verbose:
                print(f"Epoch {epoch}/{max_epochs} | Train Loss: {avg_loss:.5f}")
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def predict_quantile(model, X_cat, X_num, device=None):
    device = device or get_device()
    model.eval()
    with torch.no_grad():
        return model(X_cat.to(device), X_num.to(device)).cpu().numpy()
