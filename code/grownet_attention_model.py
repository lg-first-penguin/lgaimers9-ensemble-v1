# code/grownet_attention_model.py
"""GrowNet(Badirli et al. 2020, "Gradient Boosting Neural Networks: GrowNet") 스타일
boosting을 경량 attention weak learner에 적용 — 2026-08-21 세션, §50에서 종결된
"3rd-모델" 라인 재개 중 "bagging 말고 boosting으로 attention 모델을 할 수 있나"
요청에 대응. 이 프로젝트가 지금까지 시도한 앙상블 다양화(seed bagging - MLP/CatBoost
둘 다, CatBoost feature-subset bagging)는 전부 "독립적으로 학습 후 평균"이었고,
이전 스테이지의 residual에 순차적으로 fit하는 boosting은 미시도 — 메커니즘 자체가
다르므로 §50의 "새로운 메커니즘 없이 3rd-모델 재제안 금지" 원칙에 대한 예외로 취급.

**구조**: 매 스테이지 t는 소형 1-layer self-attention 블록(WeakLearner)으로, 원본
피처 토큰 시퀀스에 이전 스테이지의 penultimate 표현(CLS 출력)을 추가 토큰으로
이어붙여(GrowNet 논문의 "penultimate feature augmentation") 원본 F.forward를 만든다.
학습 가능한 boosting rate(eta, 스테이지별 스칼라)로 누적 로짓에 더해진다:
F_t = F_{t-1} + eta_t * f_t(x, h_{t-1}). 매 스테이지는 그리디하게(이전 스테이지는
detach) BCE(sigmoid(F_{t-1}+eta_t*f_t), y)를 직접 최소화 — GrowNet 논문의 "정식
negative-gradient 회귀"보다 단순화된 버전(실무적으로 흔히 쓰는 방식: 그리디 BCE
직접 최적화)이다. 전체 스테이지를 다 키운 뒤, 논문이 성능에 필수라고 지목한
**corrective step**(모든 스테이지 파라미터 + eta를 합쳐 몇 epoch 공동 미세조정)을
마지막에 수행한다 — 이게 없으면 greedy boosting만으로는 성능이 떨어진다는 게
원 논문의 핵심 주장.

수치형 토큰화는 `code/ft_transformer_model.py::NumericTokenizer`(quantile-PLE, 이미
검증된 방식)를 그대로 재사용 — 이 실험이 검증하려는 축은 "학습 절차(boosting vs
bagging/단일학습)"이지 임베딩 종류가 아니므로, 임베딩은 이미 신뢰된 것으로 고정해
변수를 하나로 유지한다(주기함수 임베딩은 `code/ft_transformer_periodic_model.py`가
별도로 담당).

`batched_forward`(어텐션 forward는 O(batch*heads*seq^2)라 전체 검증셋을 한 번에
넣으면 OOM — `code/ft_transformer_model.py` 문서 참고)와 동일한 이유로, 이 파일의
추론 함수도 항상 청크 단위로 순전파한다.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.ft_transformer_model import CategoricalTokenizer, NumericTokenizer
from code.mlp_model import compute_bss, get_device

D_TOKEN = 16
N_HEADS = 2
FFN_MULT = 2
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 2048
N_STAGES = 5
STAGE_MAX_EPOCHS = 15
STAGE_PATIENCE = 3
CORRECTIVE_EPOCHS = 3
CHUNK_SIZE = 8192


class WeakLearner(nn.Module):
    """스테이지 하나. has_prev=True면 이전 스테이지 CLS 표현(h_prev, d_token차원)을
    프로젝션해 추가 토큰으로 시퀀스에 이어붙인다(GrowNet의 penultimate feature
    augmentation). head는 sigmoid 없는 raw logit 증분을 반환."""

    def __init__(self, cat_dims, bin_edges, d_token=D_TOKEN, n_heads=N_HEADS,
                 ffn_mult=FFN_MULT, dropout=DROPOUT, has_prev=True):
        super().__init__()
        self.has_prev = has_prev
        self.cat_tokenizer = CategoricalTokenizer(cat_dims, d_token)
        self.num_tokenizer = NumericTokenizer(bin_edges, d_token)
        if has_prev:
            self.prev_proj = nn.Linear(d_token, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * ffn_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x_cat, x_num, h_prev=None):
        cat_tokens = self.cat_tokenizer(x_cat)
        num_tokens = self.num_tokenizer(x_num)
        tokens = torch.cat([cat_tokens, num_tokens], dim=1)
        if self.has_prev:
            prev_token = self.prev_proj(h_prev).unsqueeze(1)
            tokens = torch.cat([prev_token, tokens], dim=1)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        cls_out = self.norm(encoded[:, 0])
        logit_delta = self.head(cls_out).squeeze(-1)
        return logit_delta, cls_out


def _chunked_stage_forward(model, x_cat, x_num, h_prev, device, chunk_size=CHUNK_SIZE):
    """추론 전용(no_grad) 청크 순전파. h_prev가 있으면 같은 청크 슬라이스로 나눠 넣는다."""
    logit_deltas, cls_outs = [], []
    n = x_cat.shape[0]
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = start + chunk_size
            hp = h_prev[start:end].to(device) if h_prev is not None else None
            ld, co = model(x_cat[start:end].to(device), x_num[start:end].to(device), hp)
            logit_deltas.append(ld.cpu())
            cls_outs.append(co.cpu())
    return torch.cat(logit_deltas), torch.cat(cls_outs)


def train_grownet(
    X_tr_cat, X_tr_num, y_tr_np, cat_dims, bin_edges,
    X_val_cat, X_val_num, y_val_np,
    n_stages=N_STAGES, stage_max_epochs=STAGE_MAX_EPOCHS, stage_patience=STAGE_PATIENCE,
    corrective_epochs=CORRECTIVE_EPOCHS, d_token=D_TOKEN, n_heads=N_HEADS, ffn_mult=FFN_MULT,
    dropout=DROPOUT, lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
    device=None, verbose=True, seed=42,
):
    """스테이지별 그리디 학습 -> corrective 공동 미세조정. 반환: (stages, etas) — 둘 다
    `predict_grownet`에 그대로 넘기면 됨."""
    torch.manual_seed(seed)
    device = device or get_device()
    y_tr = torch.tensor(y_tr_np, dtype=torch.float32)

    stages, etas = [], []
    F_tr = torch.zeros(len(y_tr_np))
    F_val = torch.zeros(len(y_val_np))
    h_prev_tr, h_prev_val = None, None

    for stage_idx in range(n_stages):
        has_prev = stage_idx > 0
        model = WeakLearner(cat_dims, bin_edges, d_token=d_token, n_heads=n_heads,
                             ffn_mult=ffn_mult, dropout=dropout, has_prev=has_prev).to(device)
        eta = nn.Parameter(torch.tensor(1.0, device=device))
        criterion = nn.BCELoss()
        params = list(model.parameters()) + [eta]
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        idx_dataset = TensorDataset(torch.arange(len(y_tr_np)))
        loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        F_tr_fixed = F_tr.detach().clone()
        best_state, best_eta, best_epoch, best_val_brier, no_improve = None, None, 0, float("inf"), 0

        for epoch in range(1, stage_max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            for (batch_idx,) in loader:
                bc = X_tr_cat[batch_idx].to(device)
                bn = X_tr_num[batch_idx].to(device)
                by = y_tr[batch_idx].to(device)
                bhp = h_prev_tr[batch_idx].to(device) if has_prev else None
                bf = F_tr_fixed[batch_idx].to(device)

                optimizer.zero_grad()
                logit_delta, _ = model(bc, bn, bhp)
                cum_logit = bf + eta * logit_delta
                preds = torch.sigmoid(cum_logit)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / max(len(loader), 1)

            model.eval()
            val_ld, _ = _chunked_stage_forward(model, X_val_cat, X_val_num, h_prev_val, device)
            with torch.no_grad():
                val_cum = F_val.detach() + eta.detach().cpu() * val_ld
                val_preds = torch.sigmoid(val_cum).numpy()
            val_brier, _, val_score = compute_bss(val_preds, y_val_np)
            if verbose:
                print(f"  [stage {stage_idx+1}/{n_stages}] epoch {epoch}/{stage_max_epochs} "
                      f"loss={avg_loss:.5f} val_brier={val_brier:.6f} val_score={val_score:.2f} eta={eta.item():.3f}")

            if val_brier < best_val_brier - 1e-9:
                best_val_brier, best_epoch = val_brier, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_eta = eta.detach().clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= stage_patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        eta_fixed = best_eta.to(device)

        tr_ld, tr_cls = _chunked_stage_forward(model, X_tr_cat, X_tr_num, h_prev_tr, device)
        val_ld, val_cls = _chunked_stage_forward(model, X_val_cat, X_val_num, h_prev_val, device)
        with torch.no_grad():
            F_tr = F_tr + eta_fixed.cpu() * tr_ld
            F_val = F_val + eta_fixed.cpu() * val_ld
        h_prev_tr, h_prev_val = tr_cls, val_cls

        stage_score = compute_bss(torch.sigmoid(F_val).numpy(), y_val_np)[2]
        if verbose:
            print(f"[stage {stage_idx+1}/{n_stages} 완료] best_epoch={best_epoch} eta={eta_fixed.item():.3f} "
                  f"누적 Val Score={stage_score:.2f}")

        stages.append(model)
        etas.append(eta_fixed)

    if corrective_epochs > 0:
        if verbose:
            print(f"\n[corrective step] {corrective_epochs} epoch 동안 {n_stages}개 스테이지 공동 미세조정")
        for s in stages:
            s.train()
        eta_params = [nn.Parameter(e.clone()) for e in etas]
        all_params = [p for s in stages for p in s.parameters()] + eta_params
        optimizer = optim.AdamW(all_params, lr=lr * 0.3, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        idx_dataset = TensorDataset(torch.arange(len(y_tr_np)))
        loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in range(1, corrective_epochs + 1):
            epoch_loss = 0.0
            for (batch_idx,) in loader:
                bc = X_tr_cat[batch_idx].to(device)
                bn = X_tr_num[batch_idx].to(device)
                by = y_tr[batch_idx].to(device)
                optimizer.zero_grad()
                cum_logit = torch.zeros(len(batch_idx), device=device)
                h_prev = None
                for s, e in zip(stages, eta_params):
                    ld, cls_out = s(bc, bn, h_prev)
                    cum_logit = cum_logit + e * ld
                    h_prev = cls_out
                loss = criterion(torch.sigmoid(cum_logit), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            for s in stages:
                s.eval()
            etas = [e.detach().clone() for e in eta_params]
            val_preds = predict_grownet(stages, etas, cat_dims, bin_edges, X_val_cat, X_val_num, device=device)
            val_score = compute_bss(val_preds, y_val_np)[2]
            if verbose:
                print(f"  [corrective] epoch {epoch}/{corrective_epochs} loss={epoch_loss/max(len(loader),1):.5f} "
                      f"val_score={val_score:.2f} etas={[f'{e.item():.3f}' for e in etas]}")
            for s in stages:
                s.train()

        for s in stages:
            s.eval()

    return stages, etas


def predict_grownet(stages, etas, cat_dims, bin_edges, X_cat, X_num, device=None, chunk_size=CHUNK_SIZE):
    """스테이지를 순서대로 청크 단위 순전파, h_prev를 체이닝하며 최종 sigmoid(cum_logit) 반환."""
    device = device or get_device()
    n = X_cat.shape[0]
    cum_logit = np.zeros(n, dtype=np.float64)
    h_prev = None
    for stage_idx, (model, eta) in enumerate(zip(stages, etas)):
        model.eval()
        ld, cls_out = _chunked_stage_forward(model, X_cat, X_num, h_prev, device, chunk_size=chunk_size)
        cum_logit += eta.item() * ld.numpy()
        h_prev = cls_out
    return 1.0 / (1.0 + np.exp(-cum_logit))
