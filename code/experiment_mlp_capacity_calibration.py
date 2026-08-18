# code/experiment_mlp_capacity_calibration.py
"""현재 프로덕션 MLP 아키텍처 기록 (code/mlp_model.py, 2026-08-17 기준, tier A + coarse
pitchmix 트랙맨 피처가 반영된 지금의 code/train.py 구성 그대로):
  - 구조: Linear(in->128) -> BN -> ReLU -> Dropout(0.3) -> Linear(128->64) -> BN -> ReLU
          -> Linear(64->1) -> Sigmoid  (HIDDEN1=128, HIDDEN2=64, 2-hidden-layer)
  - DROPOUT=0.3 (첫 블록에만 적용, 두 번째 블록엔 없음), LR=0.003, WEIGHT_DECAY=0.01,
    BATCH_SIZE=4096, ENSEMBLE_SEEDS=7개(스크리닝은 3개: 42/123/7)
  - 수치형은 QuantileEmbedding(PLE, n_bins=24, d=8)로 인코딩 후 concat.
  - 폭(width) 256/128 스윕은 이미 시도해 이득 없음 확인됨(777.92 vs 780.92, 정체) — 미채택.
  - 깊이(depth)는 지금까지 스윕한 적 없음 — 이 스크립트가 처음 다룬다.

팀원이 공유한 calibration 논문(Guo et al. 류 계열, capacity가 커질수록 raw error는
줄어들지만 ECE(calibration error)가 커진다는 실험 결과, BatchNorm도 calibration을 낮춘다는
결과, weight decay가 클수록 calibration이 좋아진다는 결과)의 주장을 이 프로젝트의 실제
데이터/지표(BSS)에서 검증한다. BSS는 본질적으로 calibration(reliability)에 민감한 지표이므로
"raw discrimination은 오르지만 calibration이 나빠진다"는 논문의 주장이 맞다면, 단순히
레이어를 늘리는 것은 BSS를 오히려 깎아먹을 수 있다 — 이걸 직접 스크리닝한다.

1차 라운드 결과(cutoff7, 3-seed 스크리닝): baseline MLP=732.22/ECE=0.0048/Blend=744.44,
deeper(3층) MLP=716.09(-16.13)/ECE=0.0054(악화)/Blend=762.51(+18.07), deeper_wd(3층+wd=0.05)
MLP=708.09(-24.13)/ECE=0.0079(더 악화)/Blend=747.85(+3.41). weight_decay를 키워도 ECE가
오히려 더 나빠져 논문의 "wd↑→calibration↑" 주장이 그대로 재현되지 않았다. 또한 deeper의
Blend만 큰 폭으로 좋아진 패턴은 이 프로젝트가 이미 겪은 "핵심 교훈 #23"(서브모델 자체는
나빠졌는데 메타모델 재가중으로 블렌드만 좋아 보이는 가짜 개선)과 형태가 같아 곧바로
신뢰하지 않는다 — season==2023 교차검증으로 재확인 중.

2라운드는 두 축을 추가한다:
  - depth 축: deep4 = hidden=[128, 96, 64, 32] (4층, wd=0.01)까지만 스윕한다. Gorishniy et
    al., "Revisiting Deep Learning Models for Tabular Data"(2021)의 강한 MLP baseline도
    3층(각 256)이 전부이고, 그 이상 깊이를 쌓으려면 skip connection이 있는 ResNet 블록을
    별도로 제안했을 정도로 "skip connection 없는 plain MLP는 4층을 넘기면 학습이 잘 안 된다"는
    것이 정설이다. deeper(3층)에서 이미 early-stop이 2~8 epoch로 매우 이르게 걸린 것도 이
    징후로 보여, skip connection 없이 5층 이상 가는 실험은 하지 않는다.
  - seed 축(weight_decay 대신): deeper(3층, wd=0.01) 구성을 SCREEN_SEEDS(3개) 대신
    ENSEMBLE_SEEDS(프로덕션 7개)로 돌려, capacity 증가로 나빠진 calibration을 정규화가 아니라
    앙상블 크기로 회복시킬 수 있는지 본다. Lakshminarayanan et al., "Simple and Scalable
    Predictive Uncertainty Estimation using Deep Ensembles"(2017)에 따르면 서로 다른 랜덤
    초기화로 학습한 모델들의 평균은 "gold standard" 수준으로 calibration을 개선한다고
    알려져 있다 — weight decay(개별 모델을 덜 자신만만하게 만듦)와는 메커니즘이 다르지만
    (서로 다른 local optimum의 overconfident한 예측들이 평균으로 상쇄됨), 결과적으로 더
    낮은 ECE를 내는지는 같은 잣대(ECE)로 비교 가능하다.

각 변형에 대해 BSS(MLP 단독/블렌드)뿐 아니라 ECE(15-bin reliability)도 함께 리포트해
"raw error는 오르는데 calibration이 나빠지는" 논문의 메커니즘이 실제로 관찰되는지 직접
확인한다. CatBoost는 변형 간 비교의 기준을 고정하기 위해 한 번만 학습해서 재사용한다.

train_split/val_split 구성(트랙맨 tier A+pitchmix, F1 필터, cutoff7 옵션)은
code/experiment_residual_correction_9_10.py::build_split을 그대로 재사용한다(현재
프로덕션 code/train.py와 동일 파이프라인).

사용법:
  python -m code.experiment_mlp_capacity_calibration --cutoff7   # 프로덕션 레짐(권장 1차)
  python -m code.experiment_mlp_capacity_calibration --holdout 2023  # 사전-ABS 레짐 교차검증
  python -m code.experiment_mlp_capacity_calibration --cutoff7 --round2  # deep4 + 7-seed deeper
  python -m code.experiment_mlp_capacity_calibration --cutoff7 --resnet  # ResNet 블록(2/4/6개) 스윕
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.blend_model import fit_meta_model
from code.catboost_model import predict_catboost, train_catboost
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, QuantileEmbedding, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device, to_tensors,
)
from code.experiment_residual_correction_9_10 import SCREEN_SEEDS, build_split

HIDDEN_CONFIGS = {
    "baseline":  {"model_type": "flexible", "hidden_dims": [128, 64],     "weight_decay": 0.01, "seeds": SCREEN_SEEDS},  # 현재 프로덕션
    "deeper":    {"model_type": "flexible", "hidden_dims": [128, 64, 32], "weight_decay": 0.01, "seeds": SCREEN_SEEDS},  # capacity만 증가
    "deeper_wd": {"model_type": "flexible", "hidden_dims": [128, 64, 32], "weight_decay": 0.05, "seeds": SCREEN_SEEDS},  # capacity + 정규화 강화
}
# --round2: depth 축(deep4)과 seed 축(deeper를 7-seed로) 추가 스윕.
# deep4는 Gorishniy et al.(2021)의 강한 MLP baseline이 3층(각 256)까지만 쓰고 그 이상은
# skip connection 있는 ResNet 블록으로 넘어간다는 점을 근거로 4층에서 멈춘다(스크립트
# 상단 docstring 참고). deeper_7seed는 weight_decay 대신 앙상블 크기로 calibration을
# 회복시킬 수 있는지 보는 대조군(Lakshminarayanan et al. 2017의 deep ensembles 근거).
ROUND2_CONFIGS = {
    "deep4": {"model_type": "flexible", "hidden_dims": [128, 96, 64, 32], "weight_decay": 0.01, "seeds": SCREEN_SEEDS},
    "deeper_7seed": {"model_type": "flexible", "hidden_dims": [128, 64, 32], "weight_decay": 0.01, "seeds": ENSEMBLE_SEEDS},
}
# --resnet: round1/2에서 plain 3~4층이 손해였던 게 skip connection 부재 때문인지 직접 확인.
# Gorishniy et al.(2021)의 ResNet-for-tabular 블록(ResBlock, d=128 고정, hidden_factor=2)을
# 2/4/6개 스윕한다 — skip connection이 있으면 plain보다 더 깊이 가도 손해를 안 보는지 본다.
RESNET_CONFIGS = {
    "resnet2": {"model_type": "resnet", "d": 128, "n_blocks": 2, "hidden_factor": 2, "weight_decay": 0.01, "seeds": SCREEN_SEEDS},
    "resnet4": {"model_type": "resnet", "d": 128, "n_blocks": 4, "hidden_factor": 2, "weight_decay": 0.01, "seeds": SCREEN_SEEDS},
    "resnet6": {"model_type": "resnet", "d": 128, "n_blocks": 6, "hidden_factor": 2, "weight_decay": 0.01, "seeds": SCREEN_SEEDS},
}
LR = 0.003
BATCH_SIZE = 4096
MAX_EPOCHS = 60
PATIENCE = 7
DROPOUT = 0.3


class FlexibleTabularMLP(nn.Module):
    """production TabularMLP과 동일한 임베딩/QuantileEmbedding 경로를 쓰되, hidden_dims
    리스트로 층 수를 자유롭게 바꿀 수 있게 한 실험 전용 변형. dropout은 production과 동일하게
    첫 번째 hidden 블록에만 적용한다(두 번째/세 번째 블록엔 없음 -> capacity 증가 자체의
    효과만 보기 위해 다른 조건은 그대로 둔다)."""

    def __init__(self, cat_dims, embed_dims, bin_edges, hidden_dims, quantile_d=8, dropout=DROPOUT):
        super().__init__()
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        self.quantile = QuantileEmbedding(bin_edges, d_embed=quantile_d)
        numeric_out_dim = bin_edges.shape[0] * quantile_d
        in_dim = sum(self.embed_dims) + numeric_out_dim

        layers = []
        prev = in_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if i == 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embeds, dim=1)
        x_num_enc = self.quantile(x_num)
        x_all = torch.cat([x_embed, x_num_enc], dim=1)
        return self.sigmoid(self.mlp(x_all)).squeeze(-1)


class ResBlock(nn.Module):
    """Gorishniy et al.(2021)의 tabular ResNet 블록: BN -> Linear(확장) -> ReLU -> Dropout
    -> Linear(축소) -> Dropout -> residual add. plain deeper MLP(FlexibleTabularMLP)와 달리
    입력을 그대로 더해주는 skip connection이 있어 층을 더 쌓아도 gradient 경로가 짧게 유지된다."""

    def __init__(self, d, hidden_factor=2, dropout=DROPOUT):
        super().__init__()
        d_hidden = d * hidden_factor
        self.bn = nn.BatchNorm1d(d)
        self.lin1 = nn.Linear(d, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.bn(x)
        h = torch.relu(self.lin1(h))
        h = self.dropout(h)
        h = self.lin2(h)
        h = self.dropout(h)
        return x + h


class ResNetTabularMLP(nn.Module):
    """임베딩/QuantileEmbedding 경로는 FlexibleTabularMLP와 동일하고, 이후를 입력
    projection(Linear) + ResBlock 스택 + 출력 head(BN->ReLU->Linear)로 구성한다."""

    def __init__(self, cat_dims, embed_dims, bin_edges, d=128, n_blocks=2, hidden_factor=2,
                 quantile_d=8, dropout=DROPOUT):
        super().__init__()
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        self.quantile = QuantileEmbedding(bin_edges, d_embed=quantile_d)
        numeric_out_dim = bin_edges.shape[0] * quantile_d
        in_dim = sum(self.embed_dims) + numeric_out_dim

        self.input_proj = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([ResBlock(d, hidden_factor, dropout) for _ in range(n_blocks)])
        self.out_bn = nn.BatchNorm1d(d)
        self.out_linear = nn.Linear(d, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embeds, dim=1)
        x_num_enc = self.quantile(x_num)
        x_all = torch.cat([x_embed, x_num_enc], dim=1)
        z = self.input_proj(x_all)
        for block in self.blocks:
            z = block(z)
        z = torch.relu(self.out_bn(z))
        return self.sigmoid(self.out_linear(z)).squeeze(-1)


def compute_ece(preds, y, n_bins=15):
    """Equal-width bin ECE: 각 bin의 |평균 예측확률 - 실제 성공률|을 bin 표본비율로 가중평균."""
    preds = np.asarray(preds, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(preds)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (preds >= lo) & (preds < hi) if i < n_bins - 1 else (preds >= lo) & (preds <= hi)
        cnt = mask.sum()
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(preds[mask].mean() - y[mask].mean())
    return ece


def build_model(cfg, cat_dims, embed_dims, bin_edges):
    if cfg["model_type"] == "resnet":
        return ResNetTabularMLP(
            cat_dims, embed_dims, bin_edges,
            d=cfg["d"], n_blocks=cfg["n_blocks"], hidden_factor=cfg["hidden_factor"],
        )
    return FlexibleTabularMLP(cat_dims, embed_dims, bin_edges, cfg["hidden_dims"])


def train_flexible_mlp(X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                        X_val_cat, X_val_num, y_val, cfg, seed, device):
    torch.manual_seed(seed)
    model = build_model(cfg, cat_dims, embed_dims, bin_edges).to(device)
    weight_decay = cfg["weight_decay"]
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(X_tr_cat, X_tr_num, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    best_state, best_epoch, best_val_brier, no_improve = None, 0, float("inf"), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for bc, bn, by in loader:
            bc, bn, by = bc.to(device), bn.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bc, bn), by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy()
        val_brier = compute_bss(val_preds, y_val)[0]
        if val_brier < best_val_brier - 1e-9:
            best_val_brier, best_epoch, no_improve = val_brier, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def run_variant(name, cfg, X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
                 X_val_cat, X_val_num, y_val, device):
    seeds = cfg["seeds"]
    preds_list, best_epochs = [], []
    t0 = time.time()
    for seed in seeds:
        model, best_epoch = train_flexible_mlp(
            X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
            X_val_cat, X_val_num, y_val, cfg, seed, device,
        )
        with torch.no_grad():
            preds = model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy()
        preds_list.append(preds)
        best_epochs.append(best_epoch)
    mlp_preds = np.mean(preds_list, axis=0)
    brier, bss, score = compute_bss(mlp_preds, y_val)
    ece = compute_ece(mlp_preds, y_val)
    if cfg["model_type"] == "resnet":
        arch_desc = f"resnet d={cfg['d']} n_blocks={cfg['n_blocks']} hf={cfg['hidden_factor']}"
    else:
        arch_desc = f"hidden={cfg['hidden_dims']}"
    print(f"[{name}] {arch_desc} wd={cfg['weight_decay']} seeds={len(seeds)} | "
          f"MLP solo Score={score:.2f} ECE={ece:.4f} best_epochs={best_epochs} ({time.time()-t0:.1f}s)")
    return mlp_preds, score, ece


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--round2", action="store_true", help="baseline/deeper/deeper_wd 대신 deep4 + deeper_7seed를 돌린다")
    parser.add_argument("--resnet", action="store_true", help="baseline + ResNet 스타일 skip-connection 블록(2/4/6개)을 돌린다")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    label = "cutoff7" if args.cutoff7 else f"holdout={holdout}"
    print(f"=== MLP capacity/calibration 스크리닝 ({label}) ===")

    train_split, val_split, features, cat_features, mlp_num_cols = build_split(holdout, args.cutoff7)

    device = get_device()
    print(f"[Device] {device}")
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, "control_success")
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, "control_success")
    y_val_np = val_proc["control_success"].values

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    X_train_raw, y_train_raw = train_split[cat_features], train_split["control_success"].values
    X_val_raw, y_val_raw = val_split[cat_features], val_split["control_success"].values
    t0 = time.time()
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    print(f"[CatBoost(고정, 변형 간 공통)] Val Score={cat_score:.2f} (best_iteration={catboost_best_iteration}, {time.time()-t0:.1f}s)\n")

    if args.resnet:
        configs = {"baseline": HIDDEN_CONFIGS["baseline"], **RESNET_CONFIGS}
    elif args.round2:
        configs = {"baseline": HIDDEN_CONFIGS["baseline"], **ROUND2_CONFIGS}
    else:
        configs = HIDDEN_CONFIGS

    results = {}
    for name, cfg in configs.items():
        mlp_preds, mlp_score, ece = run_variant(
            name, cfg, X_tr_cat, X_tr_num, y_tr, cat_dims, embed_dims, bin_edges,
            X_val_cat, X_val_num, y_val_np, device,
        )
        w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_val_preds, mlp_preds, y_val_raw)
        results[name] = (mlp_score, ece, blend_score)
        print(f"    -> Blend Score={blend_score:.2f} (w_cat={w_cat:.3f} w_mlp={w_mlp:.3f})\n")

    base_mlp, base_ece, base_blend = results["baseline"]
    print(f"{'variant':<14}{'MLP solo':>10}{'ECE':>10}{'Blend':>10}{'dMLP':>10}{'dBlend':>10}")
    for name, (mlp_score, ece, blend_score) in results.items():
        print(f"{name:<14}{mlp_score:>10.2f}{ece:>10.4f}{blend_score:>10.2f}{mlp_score-base_mlp:>+10.2f}{blend_score-base_blend:>+10.2f}")


if __name__ == "__main__":
    main()
