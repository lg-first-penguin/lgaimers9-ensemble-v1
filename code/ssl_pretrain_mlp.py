# code/ssl_pretrain_mlp.py
"""MLP 고도화 4단계: self-supervised pretraining. 라벨(control_success) 없이, "swap
noise"(VIME/SubTab류 tabular SSL 표준 기법 — 배치 안에서 컬럼별로 무작위로 다른 행의
값과 바꿔치기)로 피처를 오염시킨 뒤, 원래 값을 복원하도록 인코더(범주형 임베딩 + PLE
수치 임베딩 + trunk)를 먼저 학습시키고, 그 가중치로 code/mlp_model.py::TabularMLP를
warm-start해서 기존과 동일한 지도학습 fine-tuning(BCE, early stopping)을 돌린다.

라벨 데이터가 120만 행으로 이미 충분해 기대 효과가 낮다고 판단해 처음엔 후순위로
미뤘던 방향이지만("자기지도학습은 라벨이 부족할 때 이득이 크다"는 일반론), swap-noise
denoising pretraining은 라벨 희소성과 무관하게 "더 나은 초기화/표현"을 주는 정규화
효과가 보고된 바 있어(VIME 등) 데이터 양과 별개로 시도해볼 가치가 있다는 사용자 판단에
따라 진행한다.

인코더 구조는 TabularMLP.mlp의 앞부분(Linear(128)->BN->ReLU->Dropout->Linear(64)->BN->ReLU,
즉 마지막 Linear(64,1) 직전까지)과 완전히 동일하게 맞춰서, pretrain 후 그대로
transplant할 수 있게 한다 — 마지막 분류 head(Linear(64,1))만 무작위 초기화로 남긴다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.mlp_model import (
    BATCH_SIZE, DROPOUT, HIDDEN1, HIDDEN2, LR, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY,
    QuantileEmbedding, TabularMLP, compute_bss, get_device,
)

SWAP_PROB = 0.15
PRETRAIN_EPOCHS = 15
PRETRAIN_BATCH_SIZE = 4096
PRETRAIN_LR = 0.003
PRETRAIN_WEIGHT_DECAY = 0.01


class DenoisingEncoder(nn.Module):
    """TabularMLP와 동일한 임베딩+trunk(마지막 분류 head 직전까지)에 재구성 head를 얹은
    pretraining 전용 네트워크."""

    def __init__(self, cat_dims, embed_dims, bin_edges, quantile_d=8):
        super().__init__()
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        self.quantile = QuantileEmbedding(bin_edges, d_embed=quantile_d)
        numeric_out_dim = bin_edges.shape[0] * quantile_d
        total_input_dim = sum(self.embed_dims) + numeric_out_dim

        # TabularMLP.mlp와 인덱스가 1:1로 대응(0~6) — transplant 시 그대로 state_dict 복사.
        self.trunk = nn.Sequential(
            nn.Linear(total_input_dim, HIDDEN1),
            nn.BatchNorm1d(HIDDEN1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.BatchNorm1d(HIDDEN2),
            nn.ReLU(),
        )
        self.cat_dims = list(cat_dims)
        self.cat_heads = nn.ModuleList([nn.Linear(HIDDEN2, dim + 2) for dim in cat_dims])
        self.num_head = nn.Linear(HIDDEN2, bin_edges.shape[0])

    def encode(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embeds, dim=1)
        x_num_enc = self.quantile(x_num)
        x_all = torch.cat([x_embed, x_num_enc], dim=1)
        return self.trunk(x_all)

    def forward(self, x_cat, x_num):
        h = self.encode(x_cat, x_num)
        cat_logits = [head(h) for head in self.cat_heads]
        num_pred = self.num_head(h)
        return cat_logits, num_pred


def swap_noise(x_cat, x_num, p=SWAP_PROB):
    """배치 내에서 컬럼별로 독립적으로, 확률 p의 위치를 같은 컬럼의 다른(무작위 순열) 행
    값으로 바꿔치기한다. 오염된 위치를 가리키는 (cat_mask, num_mask)도 함께 반환한다."""
    n = x_cat.shape[0]
    x_cat_c = x_cat.clone()
    x_num_c = x_num.clone()
    cat_mask = torch.rand(x_cat.shape, device=x_cat.device) < p
    num_mask = torch.rand(x_num.shape, device=x_num.device) < p
    for j in range(x_cat.shape[1]):
        perm = torch.randperm(n, device=x_cat.device)
        col_mask = cat_mask[:, j]
        x_cat_c[col_mask, j] = x_cat[perm][col_mask, j]
    for j in range(x_num.shape[1]):
        perm = torch.randperm(n, device=x_num.device)
        col_mask = num_mask[:, j]
        x_num_c[col_mask, j] = x_num[perm][col_mask, j]
    return x_cat_c, x_num_c, cat_mask, num_mask


def pretrain_encoder(
    X_tr_cat, X_tr_num, cat_dims, embed_dims, bin_edges, quantile_d=8,
    epochs=PRETRAIN_EPOCHS, batch_size=PRETRAIN_BATCH_SIZE,
    lr=PRETRAIN_LR, weight_decay=PRETRAIN_WEIGHT_DECAY, device=None, verbose=True, seed=42,
):
    """swap-noise denoising 재구성 목표로 인코더를 라벨 없이 사전학습한다. 오염된 위치에서만
    손실을 계산한다(VIME 관례) — 오염 안 된 위치까지 넣으면 항등함수로 수렴해버린다."""
    torch.manual_seed(seed)
    device = device or get_device()
    encoder = DenoisingEncoder(cat_dims, embed_dims, bin_edges, quantile_d=quantile_d).to(device)
    optimizer = optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(X_tr_cat, X_tr_num)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(1, epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        for batch_cat, batch_num in loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            x_cat_c, x_num_c, cat_mask, num_mask = swap_noise(batch_cat, batch_num)

            optimizer.zero_grad()
            cat_logits, num_pred = encoder(x_cat_c, x_num_c)

            loss = 0.0
            n_terms = 0
            for j, logits in enumerate(cat_logits):
                m = cat_mask[:, j]
                if m.any():
                    loss = loss + F.cross_entropy(logits[m], batch_cat[m, j].long())
                    n_terms += 1
            num_sq_err = (num_pred - batch_num) ** 2
            if num_mask.any():
                loss = loss + (num_sq_err * num_mask).sum() / num_mask.sum().clamp_min(1)
                n_terms += 1
            loss = loss / max(n_terms, 1)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(loader), 1)
        if verbose:
            print(f"[pretrain] epoch {epoch}/{epochs} | 재구성 손실(평균): {avg_loss:.5f}")

    return encoder


def finetune_mlp(
    X_tr_cat, X_tr_num, y_tr, cat_dims, warmstart_state_dict=None,
    X_val_cat=None, X_val_num=None, y_val=None,
    embed_dims=None, bin_edges=None, quantile_d=8,
    max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
    device=None, verbose=True, seed=42,
):
    """code/mlp_model.py::train_mlp와 완전히 동일한 학습 루프이되, 모델 생성 직후
    warmstart_state_dict(build_warmstart_state_dict 결과)를 strict=False로 로드해
    임베딩+trunk를 pretrain된 가중치로 시작한다(마지막 분류 head는 무작위 초기화 유지)."""
    torch.manual_seed(seed)
    device = device or get_device()
    model = TabularMLP(
        num_numeric_feats=bin_edges.shape[0] if bin_edges is not None else None,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges, quantile_d=quantile_d,
    ).to(device)
    if warmstart_state_dict is not None:
        missing, unexpected = model.load_state_dict(warmstart_state_dict, strict=False)
        if verbose:
            print(f"[finetune] warm-start 로드 완료 (missing={len(missing)}개, unexpected={len(unexpected)}개)")

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


def build_warmstart_state_dict(encoder):
    """DenoisingEncoder -> TabularMLP로 옮길 수 있는 state_dict 조각(임베딩+quantile+
    mlp.0~6)만 추린다. TabularMLP.load_state_dict(..., strict=False)에 바로 쓸 수 있다."""
    sd = {}
    enc_sd = encoder.state_dict()
    for k, v in enc_sd.items():
        if k.startswith("embeddings.") or k.startswith("quantile."):
            sd[k] = v
        elif k.startswith("trunk."):
            idx = k.split(".", 2)[1]  # trunk.<idx>.weight -> idx
            sd[f"mlp.{idx}.{k.split('.', 2)[2]}"] = v
    return sd
