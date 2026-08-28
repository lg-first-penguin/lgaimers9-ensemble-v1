# code/fm_model.py
"""DeepFM 계열 3rd-모델 후보 (code/mlp_model.py의 TabularMLP와 나란히 두고 비교하기 위한
독립 모듈). 목적: CatBoost(그리디 트리 분할)와 MLP(quantile-embedding + dense 비선형 결합)
둘 다 "범용 함수근사기"라서 잘 튜닝할수록 서로 수렴하는 문제(§thirdmodel_2026_reopen 참고)를
피하기 위해, 명시적으로 다른 함수형태(order-2 인수분해 상호작용)를 쓰는 모델을 시도한다.

핵심 아이디어: 모든 피처(카테고리+수치)를 "필드"로 보고 필드별 k차원 잠재벡터를 얻은 뒤,
  - order-1(선형) 항: 필드별 스칼라 가중치
  - order-2(FM) 항: 모든 필드 쌍의 내적 합 (sum-square-minus-square-sum 트릭으로 O(nk) 계산)
  - deep 항: 필드 잠재벡터를 concat해 통과시키는 dense MLP (DeepFM 관례)
을 더해 시그모이드. pitcher_id/batter_id를 필드로 포함시킨 것이 이 모델의 핵심 차별점 —
TabularMLP는 두 컬럼을 과적합 때문에 아예 제외했지만(mlp_model.py CAT_COLS 주석), FM의
저랭크(k=8) 상호작용 항은 원본 MLP의 큰 임베딩+비선형결합보다 구조적으로 더 규제되어 있어
"투수x타자 매치업 고유 상호작용"을 과적합 없이 포착할 수 있는지 확인하는 것이 실험의 요지."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.mlp_model import QuantileEmbedding, compute_bss, get_device

FM_HIDDEN1 = 128
FM_HIDDEN2 = 64
FM_DROPOUT = 0.3
FM_LR = 0.003
FM_WEIGHT_DECAY = 0.01
FM_BATCH_SIZE = 4096
FM_MAX_EPOCHS = 60
FM_PATIENCE = 7


class DeepFM(nn.Module):
    def __init__(self, cat_dims, num_numeric, bin_edges, k=8, deep_hidden=(FM_HIDDEN1, FM_HIDDEN2), dropout=FM_DROPOUT,
                 embed_dropout=0.0, high_card_cat_idx=()):
        super().__init__()
        self.k = k
        self.n_cat = len(cat_dims)
        self.n_num = num_numeric
        self.high_card_cat_idx = set(high_card_cat_idx)
        # 튜닝판: pitcher_id/batter_id처럼 카디널리티가 매우 높은 필드는 나머지 필드와
        # 똑같이 규제하면 1차 스크리닝에서 확인된 과적합(train loss는 계속 내려가는데
        # val brier는 정체)이 재현된다 — 프로덕션 RecSys 관례(고카디널리티 sparse 필드
        # 전용 L2)를 따라 train_deepfm에서 이 인덱스들의 임베딩 파라미터에만 별도로
        # 강한 weight_decay를 건다. embed_dropout은 DeepFM 구현에서 흔히 쓰는 필드
        # 임베딩 드롭아웃(V 전체에 적용) — 특정 필드에 대한 의존을 줄여 일반화를 돕는다.
        self.embed_dropout = nn.Dropout(embed_dropout) if embed_dropout > 0 else None

        # order-2용 k차원 필드 임베딩: 카테고리는 룩업, 수치는 QuantileEmbedding(피처별
        # 독립 비선형 표현) 재사용 — 둘 다 (batch, n_fields, k) 형태로 맞춘다.
        # 필드 수(~65개)가 많아 order-2 항(모든 쌍의 내적 합)이 기본 nn.Embedding 초기화
        # 분산(N(0,1))로는 순식간에 발산한다(1차 시도에서 확인, logit이 폭주해 BCELoss가
        # saturate) — 작은 표준편차로 초기화해 학습 초반 스케일을 억제한다.
        self.cat_embed_k = nn.ModuleList([nn.Embedding(dim + 2, k) for dim in cat_dims])
        for emb in self.cat_embed_k:
            nn.init.normal_(emb.weight, std=0.01)
        self.num_embed_k = QuantileEmbedding(bin_edges, d_embed=k, use_relu=False) if num_numeric > 0 else None
        if self.num_embed_k is not None:
            nn.init.normal_(self.num_embed_k.weight, std=0.01)

        # order-1(선형)용 스칼라 가중치: 카테고리는 임베딩 dim=1, 수치는 값 자체에 곱하는 스칼라.
        self.cat_linear = nn.ModuleList([nn.Embedding(dim + 2, 1) for dim in cat_dims])
        for emb in self.cat_linear:
            nn.init.zeros_(emb.weight)
        self.num_linear = nn.Parameter(torch.zeros(num_numeric)) if num_numeric > 0 else None
        self.bias = nn.Parameter(torch.zeros(1))
        # order-2/선형 항 모두 필드 수에 비례해 스케일이 흔들리므로 BatchNorm으로 정규화
        # 후 합산 — DeepFM 구현에서 흔히 쓰는 안정화 트릭.
        self.fm_bn = nn.BatchNorm1d(1)
        self.linear_bn = nn.BatchNorm1d(1)

        n_fields = self.n_cat + self.n_num
        deep_in = n_fields * k
        layers = []
        prev = deep_in
        for h in deep_hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            if h == deep_hidden[0]:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        cat_k = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.cat_embed_k)]  # 각 (batch, k)
        fields_k = list(cat_k)
        if self.num_embed_k is not None:
            num_k = self.num_embed_k(x_num)  # (batch, n_num*k)
            num_k = num_k.view(x_num.shape[0], self.n_num, self.k)
            fields_k += [num_k[:, i, :] for i in range(self.n_num)]
        V = torch.stack(fields_k, dim=1)  # (batch, n_fields, k)
        if self.embed_dropout is not None:
            V = self.embed_dropout(V)

        sum_sq = V.sum(dim=1) ** 2
        sq_sum = (V ** 2).sum(dim=1)
        fm_order2 = 0.5 * (sum_sq - sq_sum).sum(dim=1, keepdim=True)  # (batch, 1)
        fm_order2 = self.fm_bn(fm_order2)

        lin_terms = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.cat_linear)]
        linear = torch.cat(lin_terms, dim=1).sum(dim=1, keepdim=True) if lin_terms else torch.zeros_like(fm_order2)
        if self.num_linear is not None:
            linear = linear + (x_num * self.num_linear).sum(dim=1, keepdim=True)
        linear = self.linear_bn(linear) + self.bias

        deep_out = self.deep(V.reshape(V.shape[0], -1))

        logit = (linear + fm_order2 + deep_out).squeeze(-1)
        return logit


def train_deepfm(
    X_tr_cat, X_tr_num, y_tr, cat_dims, bin_edges, k=8,
    X_val_cat=None, X_val_num=None, y_val=None,
    max_epochs=FM_MAX_EPOCHS, patience=FM_PATIENCE, batch_size=FM_BATCH_SIZE,
    lr=FM_LR, weight_decay=FM_WEIGHT_DECAY, device=None, verbose=True, seed=42,
    embed_dropout=0.0, high_card_cat_idx=(), high_card_weight_decay=None,
):
    """`high_card_cat_idx`(예: pitcher_id/batter_id 필드 인덱스)를 주면 그 필드들의
    임베딩(order-2용 + order-1 선형용) 파라미터에만 `high_card_weight_decay`(기본:
    `weight_decay`의 10배)를 걸고 나머지는 `weight_decay`를 쓴다 — 고카디널리티
    sparse 필드 전용 강한 L2는 프로덕션 RecSys에서 흔한 과적합 대책."""
    torch.manual_seed(seed)
    device = device or get_device()
    model = DeepFM(
        cat_dims=cat_dims, num_numeric=X_tr_num.shape[1], bin_edges=bin_edges, k=k,
        embed_dropout=embed_dropout, high_card_cat_idx=high_card_cat_idx,
    ).to(device)
    # order-2 항의 saturate 위험 때문에 sigmoid+BCELoss 대신 logit 기준 BCEWithLogitsLoss를
    # 쓴다(1차 시도에서 확인된 발산 원인 중 하나 — pred가 0.0/1.0으로 saturate되면
    # BCELoss가 log(0)에 걸린다). gradient clipping도 안전장치로 추가.
    criterion = nn.BCEWithLogitsLoss()

    high_card_weight_decay = high_card_weight_decay if high_card_weight_decay is not None else weight_decay * 10
    high_card_idx = set(high_card_cat_idx)
    high_card_params, normal_params = [], []
    for i, emb in enumerate(model.cat_embed_k):
        (high_card_params if i in high_card_idx else normal_params).extend(emb.parameters())
    for i, emb in enumerate(model.cat_linear):
        (high_card_params if i in high_card_idx else normal_params).extend(emb.parameters())
    covered = {id(p) for p in high_card_params} | {id(p) for p in normal_params}
    normal_params += [p for p in model.parameters() if id(p) not in covered]
    param_groups = [{"params": normal_params, "weight_decay": weight_decay}]
    if high_card_params:
        param_groups.append({"params": high_card_params, "weight_decay": high_card_weight_decay})
    optimizer = optim.AdamW(param_groups, lr=lr)

    loader = DataLoader(TensorDataset(X_tr_cat, X_tr_num, y_tr), batch_size=batch_size, shuffle=True, num_workers=0)
    has_val = X_val_cat is not None and y_val is not None and len(y_val) > 0

    best_state, best_epoch, best_val_brier, no_improve = None, 0, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for bc, bn, by in loader:
            bc, bn, by = bc.to(device), bn.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bc, bn)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(loader), 1)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_preds = torch.sigmoid(model(X_val_cat.to(device), X_val_num.to(device))).cpu().numpy()
            val_brier, val_bss, val_score = compute_bss(val_preds, y_val)
            if verbose:
                print(f"Epoch {epoch}/{max_epochs} | Train Loss: {avg_loss:.5f} | Val Brier: {val_brier:.6f} | Val Score: {val_score:.2f}")
            if val_brier < best_val_brier - 1e-9:
                best_val_brier, best_epoch, no_improve = val_brier, epoch, 0
                best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"[EarlyStopping] epoch {epoch}에서 종료 (best epoch: {best_epoch})")
                    break
        else:
            best_epoch = epoch
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch
