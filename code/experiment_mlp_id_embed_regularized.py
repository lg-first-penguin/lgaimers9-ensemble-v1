# code/experiment_mlp_id_embed_regularized.py
"""3rd-모델 스태킹 라인(DeepFM/EBM/BART/NAM, capacity_lever_recheck_2026_08_22)을 4번째로
닫은 직후, 사용자가 제안한 새 방향: `pitcher_id`/`batter_id`를 MLP에 임베딩으로 추가하되
정규화로 과적합을 통제해보자는 아이디어.

**왜 다시 시도해볼 가치가 있나 (과거 실패와 다른 점)**: `code/mlp_model.py`의 CAT_COLS
주석은 "pitcher_id/batter_id 임베딩 -> 660.61에서 304로 급락"을 근거로 이 둘을 제외한다고
적혀 있다. 하지만 PROJECT_HISTORY.md §6.2 원문을 다시 보면, 그 실험은 pitcher_id/batter_id를
`season`과 **동시에** 추가했고, "이 둘(pitcher_id/batter_id)만 빼도 351.53으로 여전히
낮다"는 결과로 "season이 더 큰 원인"이라고 결론 내렸을 뿐 — **season을 뺀 상태에서
pitcher_id/batter_id만 단독으로 넣는 조합은 한 번도 격리 테스트된 적이 없다.** 게다가 그
실험은 quantile embedding(PLE) 도입 이전, cutoff7 스플릿 도입 이전, TE-residual/season-
progression 피처 도입 이전의 훨씬 오래된 파이프라인 기준이었다. 그리고 DeepFM 3rd-모델
튜닝에서 정확히 이 두 컬럼에 10x weight_decay + 임베딩 드롭아웃 0.2를 적용해 고카디널리티
임베딩을 어느 정도 안정화한 전례가 있다(capacity_lever_recheck_2026_08_22.md).

**이번 실험 설계**: `code/experiment_thirdmodel_base.py`가 이미 만들어 둔 cutoff7 캐시
(train_split/val_split, mlp_num_cols에 pitcher_id/batter_id가 이미 원본 숫자로 포함돼
있음, baseline MLP=738.55)를 재사용한다. CAT_COLS(7개)에 pitcher_id/batter_id를 추가하되:
  - 임베딩 차원을 fastai 휴리스틱(최대 50)이 아니라 8로 고정(용량 제한)
  - 이 두 임베딩 테이블에만 별도 파라미터 그룹으로 10x weight_decay(0.01->0.1) 적용
  - 이 두 임베딩 출력에만 별도 dropout(0.2) 적용(다른 카테고리/수치 경로는 그대로)
나머지(quantile embedding, 나머지 7개 카테고리, 아키텍처, lr, patience)는 프로덕션과 동일.

3-seed(42/123/7) 스크리닝: baseline MLP=738.55 대비 solo 점수, 그리고 캐시된 cat_val_preds로
2-way 블렌드(baseline 753.37) 대비 delta까지 확인한다.

사용법: python -m code.experiment_mlp_id_embed_regularized
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from code.blend_model import fit_meta_model
from code.experiment_thirdmodel_base import load_cache
from code.mlp_model import (
    CAT_COLS, QUANTILE_D, TabularMLP, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device, to_tensors,
)

torch.use_deterministic_algorithms(True, warn_only=True)

TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]
BATCH_SIZE = 4096
LR = 0.003
WEIGHT_DECAY = 0.01
ID_WEIGHT_DECAY = 0.1  # DeepFM 전례(high_card_wd_mult=10.0)와 동일 배율
ID_EMBED_DIM = 8
ID_EMBED_DROPOUT = 0.2
MAX_EPOCHS = 60
PATIENCE = 7
ID_COLS = ["pitcher_id", "batter_id"]


class TabularMLPHighCardDropout(TabularMLP):
    """pitcher_id/batter_id 임베딩(마지막 두 카테고리 컬럼으로 가정)에만 별도 dropout을 준다."""

    def __init__(self, *args, high_card_idx=(), high_card_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.high_card_idx = set(high_card_idx)
        self.embed_dropout = nn.Dropout(high_card_dropout) if high_card_dropout > 0 else None

    def forward(self, x_cat, x_num):
        embeds = []
        for i, emb in enumerate(self.embeddings):
            e = emb(x_cat[:, i].long())
            if i in self.high_card_idx and self.embed_dropout is not None:
                e = self.embed_dropout(e)
            embeds.append(e)
        x_embed = torch.cat(embeds, dim=1)
        x_num_enc = self.quantile(x_num) if self.quantile is not None else x_num
        x_all = torch.cat([x_embed, x_num_enc], dim=1)
        return self.sigmoid(self.mlp(x_all)).squeeze(-1)


def train_regularized(
    X_tr_cat, X_tr_num, y_tr, cat_dims, high_card_idx, bin_edges,
    X_val_cat, X_val_num, y_val, seed, device,
):
    torch.manual_seed(seed)
    embed_dims = [
        ID_EMBED_DIM if i in high_card_idx else embed_dim_for_cardinality(d)
        for i, d in enumerate(cat_dims)
    ]
    model = TabularMLPHighCardDropout(
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges, quantile_d=QUANTILE_D,
        high_card_idx=high_card_idx, high_card_dropout=ID_EMBED_DROPOUT,
    ).to(device)

    id_params, rest_params = [], []
    for i, emb in enumerate(model.embeddings):
        (id_params if i in high_card_idx else rest_params).extend(emb.parameters())
    rest_params.extend(model.quantile.parameters())
    rest_params.extend(model.mlp.parameters())

    optimizer = optim.AdamW([
        {"params": rest_params, "weight_decay": WEIGHT_DECAY},
        {"params": id_params, "weight_decay": ID_WEIGHT_DECAY},
    ], lr=LR)
    criterion = nn.BCELoss()

    dataset = TensorDataset(X_tr_cat, X_tr_num, y_tr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

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
            val_pred = model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy()
        val_brier = ((val_pred - y_val.numpy()) ** 2).mean()
        if val_brier < best_val_brier:
            best_val_brier, best_epoch, no_improve = val_brier, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy()
    return val_pred, best_epoch


def main():
    train_split, val_split, cat_val_preds, mlp_val_preds, y_val, meta = load_cache()
    print(f"[baseline] CatBoost={meta['catboost_score']:.2f} | MLP(7-seed ref)={meta['mlp_score']:.2f} "
          f"| 2-way={meta['baseline_2way_score']:.2f}")

    cat_cols = CAT_COLS + ID_COLS
    num_cols = [c for c in meta["mlp_num_cols"] if c not in ID_COLS]
    high_card_idx = {cat_cols.index(c) for c in ID_COLS}

    train_enc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, cat_cols, num_cols)
    val_enc = apply_preprocessing(val_split, cat_cols, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_enc, cat_cols, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_enc, cat_cols, num_cols, TARGET_COL)
    bin_edges = fit_quantile_edges(X_tr_num)

    device = get_device()
    preds = []
    for seed in SCREEN_SEEDS:
        t0 = time.time()
        val_pred, best_epoch = train_regularized(
            X_tr_cat, X_tr_num, y_tr, cat_dims, high_card_idx, bin_edges,
            X_val_cat, X_val_num, y_val_t, seed, device,
        )
        score = compute_bss(val_pred, y_val)[2]
        print(f"  seed={seed} solo={score:.2f} (best_epoch={best_epoch}, {time.time()-t0:.1f}s)")
        preds.append(val_pred)

    ens_pred = np.mean(preds, axis=0)
    ens_score = compute_bss(ens_pred, y_val)[2]
    corr_cat = np.corrcoef(ens_pred, cat_val_preds)[0, 1]
    corr_mlp = np.corrcoef(ens_pred, mlp_val_preds)[0, 1]
    print(f"\n[id-embed-reg] {len(SCREEN_SEEDS)}-seed solo={ens_score:.2f} "
          f"(baseline MLP={meta['mlp_score']:.2f}, delta={ens_score - meta['mlp_score']:+.2f}) "
          f"corr(cat)={corr_cat:.4f} corr(mlp)={corr_mlp:.4f}")

    w_cat, w_mlp, intercept, blend2_new, _ = fit_meta_model(cat_val_preds, ens_pred, y_val)
    print(f"2-way(baseline mlp)={meta['baseline_2way_score']:.2f} | "
          f"2-way(id-embed-reg mlp)={blend2_new:.2f} (delta {blend2_new - meta['baseline_2way_score']:+.2f}) "
          f"weights cat={w_cat:.3f} mlp={w_mlp:.3f} intercept={intercept:.3f}")


if __name__ == "__main__":
    main()
