# code/periodic_mlp_model.py
"""수치형 피처용 주기함수(periodic, sin/cos) 임베딩 실험 (`code/experiment_periodic_embed.py`).

Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep Learning"의
PLR(Periodic-Linear-ReLU) 임베딩을 이 프로젝트의 `TabularMLP`(code/mlp_model.py)에
적용해봅니다. 기존 MLP는 수치형 44개 컬럼을 표준화만 해서 그대로 concat했는데,
그 대신 피처마다 학습 가능한 주파수로 sin/cos 특징을 만들고 피처별 독립 Linear를
얹어 범주형 임베딩과 같은 형태의 학습된 표현으로 바꿔봅니다.

PROJECT_HISTORY.md/EXPERIMENTS.md §14.4에서 "여전히 미시도"로 남겨둔 항목이며,
`code/experiment_3way_stack.py`와 동일하게 `dopip.py` 메인 파이프라인에는 반영되지
않은 실험 단계 모듈입니다.
"""
import math

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


class PeriodicEmbedding(nn.Module):
    """수치형 피처별 주기함수 임베딩(PLR). 피처마다 학습 가능한 주파수
    c_j ~ N(0, sigma^2)로 [sin(2*pi*c_j*x), cos(2*pi*c_j*x)]를 만들고, 피처별
    독립 Linear(비공유 가중치)로 d_embed 차원 임베딩을 얻습니다. sigma(주파수
    초기화 스케일)가 논문에서도 가장 성능을 좌우하는 하이퍼파라미터로 지목됨 —
    이 실험에서도 그리드로 스윕합니다."""

    def __init__(self, num_numeric, k=8, d_embed=8, sigma=0.1, use_relu=True):
        super().__init__()
        self.num_numeric = num_numeric
        self.k = k
        self.d_embed = d_embed
        self.use_relu = use_relu
        self.freq = nn.Parameter(torch.randn(num_numeric, k) * sigma)
        self.weight = nn.Parameter(torch.empty(num_numeric, 2 * k, d_embed))
        self.bias = nn.Parameter(torch.zeros(num_numeric, d_embed))
        bound = 1.0 / math.sqrt(2 * k)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x_num):
        # x_num: (batch, num_numeric) — 이미 표준화(평균0, 표준편차1)된 값
        v = 2 * math.pi * self.freq.unsqueeze(0) * x_num.unsqueeze(-1)  # (batch, num_numeric, k)
        p = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (batch, num_numeric, 2k)
        e = torch.einsum("bnf,nfd->bnd", p, self.weight) + self.bias  # (batch, num_numeric, d_embed)
        if self.use_relu:
            e = torch.relu(e)
        return e.reshape(e.shape[0], -1)  # (batch, num_numeric * d_embed)


class TabularMLPPeriodic(nn.Module):
    """`code/mlp_model.py::TabularMLP`와 동일한 구조(128->64->1, BatchNorm/Dropout)이되,
    수치형 입력을 표준화 값 그대로 concat하는 대신 PeriodicEmbedding을 거칩니다."""

    def __init__(self, num_numeric_feats, cat_dims, embed_dims=None,
                 periodic_k=8, periodic_d=8, periodic_sigma=0.1, periodic_relu=True):
        super().__init__()
        if embed_dims is None:
            embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        self.periodic = PeriodicEmbedding(
            num_numeric_feats, k=periodic_k, d_embed=periodic_d, sigma=periodic_sigma, use_relu=periodic_relu,
        )
        total_input_dim = sum(self.embed_dims) + num_numeric_feats * periodic_d

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
        x_periodic = self.periodic(x_num)
        x_all = torch.cat([x_embed, x_periodic], dim=1)
        return self.sigmoid(self.mlp(x_all)).squeeze(-1)


def train_mlp_periodic(
    X_tr_cat, X_tr_num, y_tr, cat_dims, num_numeric_feats,
    X_val_cat=None, X_val_num=None, y_val=None,
    embed_dims=None, periodic_k=8, periodic_d=8, periodic_sigma=0.1, periodic_relu=True,
    max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
    lr=LR, weight_decay=WEIGHT_DECAY, device=None, verbose=True, seed=42,
):
    """`code/mlp_model.py::train_mlp`와 동일한 early-stopping 학습 루프이되,
    TabularMLPPeriodic을 사용합니다."""
    torch.manual_seed(seed)
    device = device or get_device()
    model = TabularMLPPeriodic(
        num_numeric_feats=num_numeric_feats, cat_dims=cat_dims, embed_dims=embed_dims,
        periodic_k=periodic_k, periodic_d=periodic_d, periodic_sigma=periodic_sigma, periodic_relu=periodic_relu,
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


def predict_periodic(model, X_cat, X_num, device=None):
    device = device or get_device()
    model.eval()
    with torch.no_grad():
        return model(X_cat.to(device), X_num.to(device)).cpu().numpy()
