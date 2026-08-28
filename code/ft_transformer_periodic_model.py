# code/ft_transformer_periodic_model.py
"""FT-Transformer(`code/ft_transformer_model.py`)의 수치형 토큰화를 quantile-PLE
대신 주기함수(periodic, sin/cos) 임베딩으로 바꾼 변형 — 2026-08-21 세션, §50에서
종결됐던 "3rd-모델" 라인 재개 중 "quantile 말고 sin/cos도" 요청에 대응.

`code/periodic_mlp_model.py::PeriodicEmbedding`과 같은 공식(Gorishniy et al.의
PLR: c_j~N(0,sigma^2)로 [sin(2*pi*c*x), cos(2*pi*c*x)] 생성 후 피처별 독립
Linear+ReLU)이지만, 그 파일은 평탄화된 MLP-concat용으로 마지막에
`.reshape(batch, -1)`을 하는 반면, 여기서는 FT-Transformer의 토큰 시퀀스
형태(batch, num_numeric, d_token)를 유지해야 attention이 피처 단위로 걸린다 —
그래서 새로 작성(`PeriodicTokenizer`), `ft_transformer_model.py::NumericTokenizer`와
자리만 바꿔 끼우는 구조.

나머지(CategoricalTokenizer/CLS/TransformerEncoder/batched_forward/앙상블/번들)는
`ft_transformer_model.py`와 완전히 동일 — 코드 중복을 감수하고 별도 파일로 둔 이유는
같은 이유(§50 이전 실험들의 재현성 보존을 위해 기존 ft_transformer_model.py를
건드리지 않음, CLAUDE.md의 periodic_mlp_model.py/quantile_mlp_model.py 병렬 보관
관례와 동일)."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.mlp_model import compute_bss, get_device
from code.ft_transformer_model import CategoricalTokenizer, batched_forward

D_TOKEN = 32
N_LAYERS = 3
N_HEADS = 8
FFN_MULT = 2
DROPOUT = 0.1
LR = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 2048
MAX_EPOCHS = 60
PATIENCE = 7
SEED = 42
PERIODIC_K = 24
PERIODIC_SIGMA = 0.1

ENSEMBLE_SEEDS = [42, 123, 7, 2024, 99, 555, 31337]


class PeriodicTokenizer(nn.Module):
    """수치형 피처별 주기함수(PLR) 토큰화. (batch, num_numeric, d_token) 유지(flatten 없음)."""

    def __init__(self, num_numeric, k=PERIODIC_K, d_token=D_TOKEN, sigma=PERIODIC_SIGMA):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(num_numeric, k) * sigma)
        self.weight = nn.Parameter(torch.empty(num_numeric, 2 * k, d_token))
        self.bias = nn.Parameter(torch.zeros(num_numeric, d_token))
        bound = 1.0 / math.sqrt(2 * k)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x_num):
        v = 2 * math.pi * self.freq.unsqueeze(0) * x_num.unsqueeze(-1)  # (batch, num_numeric, k)
        p = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (batch, num_numeric, 2k)
        e = torch.einsum("bnf,nfd->bnd", p, self.weight) + self.bias
        return torch.relu(e)  # (batch, num_numeric, d_token)


class FTTransformerPeriodic(nn.Module):
    def __init__(self, cat_dims, num_numeric, d_token=D_TOKEN, n_layers=N_LAYERS,
                 n_heads=N_HEADS, ffn_mult=FFN_MULT, dropout=DROPOUT,
                 periodic_k=PERIODIC_K, periodic_sigma=PERIODIC_SIGMA):
        super().__init__()
        self.cat_tokenizer = CategoricalTokenizer(cat_dims, d_token)
        self.num_tokenizer = PeriodicTokenizer(num_numeric, k=periodic_k, d_token=d_token, sigma=periodic_sigma)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * ffn_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        cat_tokens = self.cat_tokenizer(x_cat)
        num_tokens = self.num_tokenizer(x_num)
        tokens = torch.cat([cat_tokens, num_tokens], dim=1)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        cls_out = self.norm(encoded[:, 0])
        return self.sigmoid(self.head(cls_out)).squeeze(-1)


def train_ft_periodic(
    X_tr_cat, X_tr_num, y_tr, cat_dims, num_numeric,
    X_val_cat=None, X_val_num=None, y_val=None,
    d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_mult=FFN_MULT, dropout=DROPOUT,
    periodic_k=PERIODIC_K, periodic_sigma=PERIODIC_SIGMA,
    max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
    lr=LR, weight_decay=WEIGHT_DECAY, device=None, verbose=True, seed=SEED,
):
    torch.manual_seed(seed)
    device = device or get_device()
    model = FTTransformerPeriodic(
        cat_dims=cat_dims, num_numeric=num_numeric, d_token=d_token, n_layers=n_layers,
        n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
        periodic_k=periodic_k, periodic_sigma=periodic_sigma,
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(X_tr_cat, X_tr_num, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    has_val = X_val_cat is not None and y_val is not None and len(y_val) > 0

    best_state, best_epoch, best_val_brier, epochs_no_improve = None, 0, float("inf"), 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_cat, batch_num, batch_y in loader:
            batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_cat, batch_num)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(loader), 1)

        if has_val:
            model.eval()
            val_preds = batched_forward(model, X_val_cat, X_val_num, device=device)
            val_brier, val_bss, val_score = compute_bss(val_preds, y_val)
            if verbose:
                print(f"Epoch {epoch}/{max_epochs} | Train Loss: {avg_loss:.5f} | Val Brier: {val_brier:.6f} | Val Score: {val_score:.2f}")
            if val_brier < best_val_brier - 1e-9:
                best_val_brier, best_epoch = val_brier, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"[EarlyStopping] epoch {epoch}에서 종료 (best epoch: {best_epoch})")
                    break
        else:
            if verbose:
                print(f"Epoch {epoch}/{max_epochs} | Train Loss: {avg_loss:.5f}")
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def train_ft_periodic_ensemble(
    X_tr_cat, X_tr_num, y_tr, cat_dims, num_numeric,
    X_val_cat=None, X_val_num=None, y_val=None, seeds=ENSEMBLE_SEEDS, device=None, verbose=True, **kwargs,
):
    device = device or get_device()
    members = []
    for seed in seeds:
        model, best_epoch = train_ft_periodic(
            X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, num_numeric=num_numeric,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
            device=device, verbose=verbose, seed=seed, **kwargs,
        )
        members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": best_epoch, "seed": seed,
        })
        if verbose:
            print(f"[Ensemble] seed={seed} 학습 완료 (best_epoch={best_epoch})")
    return members


def predict_ft_periodic_ensemble(members, cat_dims, num_numeric, X_cat, X_num, d_token=D_TOKEN,
                                  n_layers=N_LAYERS, n_heads=N_HEADS, ffn_mult=FFN_MULT, dropout=DROPOUT,
                                  periodic_k=PERIODIC_K, periodic_sigma=PERIODIC_SIGMA, device=None):
    device = device or get_device()
    preds_list = []
    for member in members:
        model = FTTransformerPeriodic(
            cat_dims=cat_dims, num_numeric=num_numeric, d_token=d_token, n_layers=n_layers,
            n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
            periodic_k=periodic_k, periodic_sigma=periodic_sigma,
        ).to(device)
        model.load_state_dict(member["state_dict"])
        model.eval()
        preds_list.append(batched_forward(model, X_cat, X_num, device=device))
    return np.mean(preds_list, axis=0)
