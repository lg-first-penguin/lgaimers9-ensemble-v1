# code/attention_mlp_model.py
"""MLP 고도화 3단계: 7-seed 앙상블의 "멤버 자체" 아키텍처에 경량 self-attention 블록을
끼워 넣은 실험용 변형. code/mlp_model.py::TabularMLP는 범주형 임베딩 + 수치형 PLE 임베딩을
그냥 concat해서 Linear(128)로 넘기는데, 여기서는 concat 대신 각 피처(범주형/수치형 각각)를
"토큰"으로 보고 얕은 self-attention 레이어 하나를 거친 뒤 flatten해서 동일한 head
(Linear(128)->BN->ReLU->Dropout->Linear(64)->BN->ReLU->Linear(1))에 넣는다.

이 프로젝트에서 폭/깊이 확장(핵심 교훈 #27)과 하이퍼파라미터 재튠은 반복 기각됐지만,
attention은 "더 크게"가 아니라 "피처 간 상호작용을 모델이 직접 학습"하는 다른 종류의
용량이라 별개로 검증한다. 3rd 모델(ExcelFormer 등)로 병렬 스태킹하는 것과 달리, 이건
7-seed 앙상블의 실제 멤버 아키텍처 교체 실험이다.

토큰 차원(d_token)은 QuantileEmbedding의 d_embed와 동일하게 맞춰 수치형 토큰은 추가
프로젝션 없이 바로 쓰고, 범주형 임베딩(임베딩 차원이 컬럼마다 다름)만 컬럼별 Linear로
d_token에 투영한다.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.mlp_model import (
    DROPOUT, HIDDEN1, HIDDEN2, QuantileEmbedding, compute_bss, get_device,
)

MAX_EPOCHS = 60
PATIENCE = 7
BATCH_SIZE = 4096
LR = 0.003
WEIGHT_DECAY = 0.01
SEED = 42


def quantile_tokens(quantile_module, x_num):
    """QuantileEmbedding.forward()와 동일한 계산이지만 flatten하지 않고
    (batch, num_numeric, d_embed) 토큰 형태 그대로 반환한다."""
    p = quantile_module.encode(x_num)
    e = torch.einsum("bnf,nfd->bnd", p, quantile_module.weight) + quantile_module.bias
    if quantile_module.use_relu:
        e = torch.relu(e)
    return e


class AttentionTabularMLP(nn.Module):
    def __init__(self, cat_dims, embed_dims, bin_edges, d_token=8, n_heads=2, ffn_dim=16, attn_dropout=0.1):
        super().__init__()
        assert bin_edges.shape[1] - 1 > 0
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        self.cat_proj = nn.ModuleList([nn.Linear(edim, d_token) for edim in self.embed_dims])

        self.quantile = QuantileEmbedding(bin_edges, d_embed=d_token)
        self.n_cat = len(cat_dims)
        self.num_numeric = bin_edges.shape[0]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=attn_dropout, batch_first=True, activation="relu",
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=1)

        total_input_dim = (self.n_cat + self.num_numeric) * d_token
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
        cat_tokens = [proj(emb(x_cat[:, i].long())) for i, (emb, proj) in enumerate(zip(self.embeddings, self.cat_proj))]
        cat_tokens = torch.stack(cat_tokens, dim=1)  # (batch, n_cat, d_token)
        num_tokens = quantile_tokens(self.quantile, x_num)  # (batch, n_num, d_token)
        tokens = torch.cat([cat_tokens, num_tokens], dim=1)
        attended = self.attn(tokens)
        flat = attended.reshape(attended.shape[0], -1)
        return self.sigmoid(self.mlp(flat)).squeeze(-1)


def train_mlp_attention(
    X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
    X_val_cat=None, X_val_num=None, y_val=None,
    d_token=8, n_heads=2, ffn_dim=16, attn_dropout=0.1,
    max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
    lr=LR, weight_decay=WEIGHT_DECAY, device=None, verbose=True, seed=SEED,
):
    """code/mlp_model.py::train_mlp와 동일한 학습 루프(BCELoss/AdamW/Val Brier early
    stopping)를 AttentionTabularMLP에 대해 반복한다."""
    torch.manual_seed(seed)
    device = device or get_device()
    model = AttentionTabularMLP(
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        d_token=d_token, n_heads=n_heads, ffn_dim=ffn_dim, attn_dropout=attn_dropout,
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


def predict_attention(model, X_cat, X_num, device=None):
    device = device or get_device()
    model.eval()
    with torch.no_grad():
        return model(X_cat.to(device), X_num.to(device)).cpu().numpy()
