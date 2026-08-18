# code/ft_transformer_model.py
"""FT-Transformer(Feature Tokenizer + Transformer, Gorishniy et al. 2021) 공용 모델/전처리 유틸리티.

3번째 앙상블 후보 모델 — CatBoost(GBDT) + Tabular MLP(concat-and-MLP)와 구조적으로
다른 계열(피처-간 self-attention)을 넣어 스태킹 앙상블의 독립성을 높이려는 실험
(`code/experiment_attention.py`에서 사용, PROJECT_HISTORY.md "다음으로 시도해볼 만한
방향"의 "세 번째 모델 계열 추가" 항목).

전처리(fit_preprocessing/apply_preprocessing/to_tensors/get_device/compute_bss/
fit_quantile_edges)는 `code/mlp_model.py`의 것을 그대로 재사용합니다 — 이 유틸리티들은
TabularMLP 전용이 아니라 순수 sklearn 전처리기/텐서 변환이라 모델 계열과 무관합니다.

TabularMLP와의 핵심 차이: FT-Transformer는 카테고리/수치형 피처를 모두 "같은 차원(d_token)의
토큰"으로 만들어야 self-attention으로 묶을 수 있습니다. 그래서
  - 범주형: `nn.Embedding(cardinality+2, d_token)` (TabularMLP처럼 컬럼별로 다른
    embed_dim을 쓰는 fastai 휴리스틱은 여기서는 쓸 수 없음 — 전부 d_token으로 통일).
  - 수치형: `code/mlp_model.py::QuantileEmbedding`과 동일한 PLE(quantile piecewise-linear)
    로직이지만, 마지막에 전부 이어붙여 flatten하지 않고 (batch, num_numeric, d_token)
    형태를 그대로 유지해 토큰 시퀀스로 씁니다(`NumericTokenizer`).
  - 학습 가능한 [CLS] 토큰을 시퀀스 맨 앞에 붙이고, 표준 Transformer 인코더(pre-norm,
    GELU FFN)를 통과시킨 뒤 [CLS] 최종 표현을 `Linear(d_token, 1) + Sigmoid`로 사영합니다.

번들 형식은 `code/mlp_model.py::make_bundle`과 동일한 관례(순수 dict + state_dict 리스트,
프로젝트 전용 클래스 인스턴스를 pickle하지 않음)를 따릅니다.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.mlp_model import compute_bss, get_device

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

ENSEMBLE_SEEDS = [42, 123, 7, 2024, 99, 555, 31337]


class NumericTokenizer(nn.Module):
    """수치형 피처별 PLE(quantile piecewise-linear) 인코딩. `code/mlp_model.py::QuantileEmbedding`과
    동일한 로직이지만 (batch, num_numeric, d_token)으로 유지해 flatten하지 않습니다."""

    def __init__(self, bin_edges, d_token=D_TOKEN):
        super().__init__()
        self.register_buffer("edges", bin_edges)  # (num_numeric, n_bins+1)
        num_numeric, n_bins_plus1 = bin_edges.shape
        n_bins = n_bins_plus1 - 1
        self.weight = nn.Parameter(torch.empty(num_numeric, n_bins, d_token))
        self.bias = nn.Parameter(torch.zeros(num_numeric, d_token))
        bound = 1.0 / math.sqrt(n_bins)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x_num):
        left = self.edges[:, :-1].unsqueeze(0)
        right = self.edges[:, 1:].unsqueeze(0)
        x = x_num.unsqueeze(-1)
        width = (right - left).clamp_min(1e-6)
        frac = ((x - left) / width).clamp(0.0, 1.0)
        e = torch.einsum("bnf,nfd->bnd", frac, self.weight) + self.bias
        return torch.relu(e)  # (batch, num_numeric, d_token)


class CategoricalTokenizer(nn.Module):
    def __init__(self, cat_dims, d_token=D_TOKEN):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim + 2, d_token) for dim in cat_dims])

    def forward(self, x_cat):
        toks = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        return torch.stack(toks, dim=1)  # (batch, num_cat, d_token)


def batched_forward(model, x_cat, x_num, batch_size=8192, device=None):
    """어텐션 기반 모델은 forward 메모리가 O(batch*heads*seq^2)로 스케일합니다.
    검증/추론에서 전체 배치를 한 번에 넣으면(예: 검증 25만행, 실전 24.6만행)
    커스텀 attn_mask가 있는 모델(ExcelFormer)에서는 PyTorch가 fused/flash
    커널 대신 느린 경로로 빠지며 OOM이 납니다(253507행 기준 약 27GB 시도,
    실측 확인됨). FT-Transformer(마스크 없음)는 이 GPU에서 fused 경로로
    우연히 통과했지만, 실전 평가 서버(L4)에서 같은 경로를 탄다는 보장이 없어
    두 모델 모두 항상 배치로 나눠 추론합니다."""
    device = device or get_device()
    preds = []
    with torch.no_grad():
        for start in range(0, x_cat.shape[0], batch_size):
            end = start + batch_size
            out = model(x_cat[start:end].to(device), x_num[start:end].to(device))
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


class FTTransformer(nn.Module):
    def __init__(self, cat_dims, bin_edges, d_token=D_TOKEN, n_layers=N_LAYERS,
                 n_heads=N_HEADS, ffn_mult=FFN_MULT, dropout=DROPOUT):
        super().__init__()
        self.cat_tokenizer = CategoricalTokenizer(cat_dims, d_token)
        self.num_tokenizer = NumericTokenizer(bin_edges, d_token)
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


def train_ft(
    X_tr_cat, X_tr_num, y_tr, cat_dims, bin_edges,
    X_val_cat=None, X_val_num=None, y_val=None,
    d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_mult=FFN_MULT, dropout=DROPOUT,
    max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
    lr=LR, weight_decay=WEIGHT_DECAY, device=None, verbose=True, seed=SEED,
):
    """TabularMLP의 `train_mlp`와 동일한 early-stopping 패턴. 검증 텐서가 있으면 Val Brier
    기준 patience만큼 개선 없을 때 멈추고 best epoch 가중치로 복원, 없으면 max_epochs 전부."""
    torch.manual_seed(seed)
    device = device or get_device()
    model = FTTransformer(
        cat_dims=cat_dims, bin_edges=bin_edges, d_token=d_token, n_layers=n_layers,
        n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
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
            val_preds = batched_forward(model, X_val_cat, X_val_num, device=device)
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


def train_ft_ensemble(
    X_tr_cat, X_tr_num, y_tr, cat_dims, bin_edges,
    X_val_cat=None, X_val_num=None, y_val=None,
    seeds=ENSEMBLE_SEEDS, d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS,
    ffn_mult=FFN_MULT, dropout=DROPOUT, max_epochs=MAX_EPOCHS, patience=PATIENCE,
    batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, device=None, verbose=True,
):
    device = device or get_device()
    members = []
    for seed in seeds:
        model, best_epoch = train_ft(
            X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, bin_edges=bin_edges,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
            d_token=d_token, n_layers=n_layers, n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
            max_epochs=max_epochs, patience=patience, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay, device=device, verbose=verbose, seed=seed,
        )
        members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": best_epoch,
            "seed": seed,
        })
        if verbose:
            print(f"[Ensemble] seed={seed} 학습 완료 (best_epoch={best_epoch})")
    return members


def predict_ft_ensemble(members, cat_dims, bin_edges, X_cat, X_num, d_token=D_TOKEN,
                         n_layers=N_LAYERS, n_heads=N_HEADS, ffn_mult=FFN_MULT, dropout=DROPOUT, device=None):
    device = device or get_device()
    preds_list = []
    for member in members:
        model = FTTransformer(
            cat_dims=cat_dims, bin_edges=bin_edges, d_token=d_token, n_layers=n_layers,
            n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
        ).to(device)
        model.load_state_dict(member["state_dict"])
        model.eval()
        preds_list.append(batched_forward(model, X_cat, X_num, device=device))
    return np.mean(preds_list, axis=0)


def make_ft_bundle(members, cat_cols, num_cols, cat_dims, bin_edges, cat_encoder, num_imputer, num_scaler,
                    d_token=D_TOKEN, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_mult=FFN_MULT, dropout=DROPOUT):
    avg_best_epoch = int(round(np.mean([m["best_epoch"] for m in members]))) if members else 0
    return {
        "members": members,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_dims": cat_dims,
        "bin_edges": bin_edges,
        "cat_encoder": cat_encoder,
        "num_imputer": num_imputer,
        "num_scaler": num_scaler,
        "d_token": d_token,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "ffn_mult": ffn_mult,
        "dropout": dropout,
        "best_epoch_": avg_best_epoch,
    }


def predict_ft_bundle(bundle, df, device=None):
    from code.mlp_model import apply_preprocessing, to_tensors
    device = device or get_device()
    df_proc = apply_preprocessing(
        df, bundle["cat_cols"], bundle["num_cols"],
        bundle["cat_encoder"], bundle["num_imputer"], bundle["num_scaler"],
    )
    X_cat, X_num = to_tensors(df_proc, bundle["cat_cols"], bundle["num_cols"])
    return predict_ft_ensemble(
        bundle["members"], bundle["cat_dims"], bundle["bin_edges"], X_cat, X_num,
        d_token=bundle.get("d_token", D_TOKEN), n_layers=bundle.get("n_layers", N_LAYERS),
        n_heads=bundle.get("n_heads", N_HEADS), ffn_mult=bundle.get("ffn_mult", FFN_MULT),
        dropout=bundle.get("dropout", DROPOUT), device=device,
    )
