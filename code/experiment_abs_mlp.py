# code/experiment_abs_mlp.py
"""`code/experiment_abs_regime.py`(CatBoost 단독, ABS 레짐 가설)를 MLP+블렌드까지
확장 검증. CatBoost만으로는 weight=5 근처에서 2024년 9~10월 검증 기준 baseline(269.18)
대비 +66.78의 큰 개선을 봤는데, 이게 MLP+CatBoost 블렌드(실제 프로덕션이 최적화하는
지표)에서도 재현되는지 확인한다.

Config A(baseline)는 재학습 없이 `open/reference/best_model.pkl`(2019~2023만 학습된
현재 프로덕션)의 예측을 캐시에서 그대로 가져와 9~10월로 마스킹한다 — 7-seed 앙상블.
Config C(2024 head 가중치)는 시간 비용 때문에 3-seed로 스크리닝한다(공정 비교가
아니라 "적은 시드 핸디캡을 안고도 이기는지" 보수적으로 보는 것 — 학습 데이터셋 크기
문제로 sample_weight를 CatBoost처럼 못 쓰는 대신, 2024 head 행에 WeightedRandomSampler로
가중 샘플링을 적용한다(물리적 행 복제 대신 — 데이터셋 크기를 부풀리지 않아 epoch 비용이
그대로 유지됨).

사용법:
  python -m code.experiment_abs_mlp --step run --weight 5.0
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from code.experiment_abs_regime import build_splits, train_and_score
from code.experiment_meta_nonlinear import load_cache
from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, TabularMLP, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, apply_preprocessing, fit_quantile_edges, get_device, to_tensors,
)
from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, predict_catboost
from catboost import CatBoostClassifier, Pool
from code.blend_model import fit_meta_model, predict_meta

SCREEN_SEEDS = [42, 123, 7]  # 3-seed 스크리닝 (프로덕션 7-seed 대비 핸디캡을 안고 검증)


def train_mlp_weighted(X_tr_cat, X_tr_num, y_tr, sample_weight, cat_dims, embed_dims, bin_edges,
                        X_val_cat, X_val_num, y_val, max_epochs=60, patience=7,
                        batch_size=4096, lr=0.003, weight_decay=0.01, device=None, seed=42, verbose=False):
    """code/mlp_model.py::train_mlp 과 동일한 구조이지만, DataLoader에 shuffle 대신
    WeightedRandomSampler(sample_weight)를 사용해 2024 head 행을 더 자주 뽑는다.
    (물리적 행 복제가 아니므로 epoch당 배치 수/비용은 그대로.)"""
    torch.manual_seed(seed)
    device = device or get_device()
    model = TabularMLP(cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(X_tr_cat, X_tr_num, y_tr)
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    best_state, best_epoch, best_val_brier, no_improve = None, 0, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_cat, batch_num, batch_y in loader:
            batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_cat, batch_num), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_cat.to(device), X_val_num.to(device)).cpu().numpy()
        val_brier, _, val_score = compute_bss(val_preds, y_val)
        if verbose:
            print(f"  epoch {epoch} val_score={val_score:.2f}")
        if val_brier < best_val_brier - 1e-9:
            best_val_brier, best_epoch = val_brier, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def predict_mlp(model, X_cat, X_num, device):
    model.eval()
    with torch.no_grad():
        return model(X_cat.to(device), X_num.to(device)).cpu().numpy()


def step_run(weight_2024, val_month_cutoff=9):
    pre2024, head2024, tail2024, features = build_splits(val_month_cutoff=val_month_cutoff)
    num_cols = [c for c in features if c not in CAT_COLS]
    device = get_device()

    # ---- Config A: 재학습 없이 기존 reference(2019~2023만 학습) 예측을 tail로 마스킹 ----
    cat_val_preds, mlp_val_preds, y_val, game_month, meta = load_cache()
    tail_mask = game_month >= val_month_cutoff
    cat_a, mlp_a, y_a = cat_val_preds[tail_mask], mlp_val_preds[tail_mask], y_val[tail_mask]
    assert len(y_a) == len(tail2024), f"tail 행수 불일치: 캐시={len(y_a)} vs 재구성={len(tail2024)}"
    w_cat_a, w_mlp_a, ic_a, score_a, _ = fit_meta_model(cat_a, mlp_a, y_a)
    print(f"[A] CatBoost(7s prod)={compute_bss(cat_a, y_a)[2]:.2f} | MLP(7-seed prod)={compute_bss(mlp_a, y_a)[2]:.2f} "
          f"| 블렌드={score_a:.2f}")

    # ---- Config C: pre2024 + head2024(가중 샘플링), 3-seed 스크리닝 ----
    train_c = pd.concat([pre2024, head2024], ignore_index=True)
    X_train_raw, y_train_raw = train_c[features], train_c["control_success"].values
    X_val_raw, y_val_raw = tail2024[features], tail2024["control_success"].values

    # CatBoost (weight=5 등, sample_weight로 실제 지원됨)
    sw = np.ones(len(train_c))
    sw[len(pre2024):] = weight_2024
    params = dict(CATBOOST_PARAMS)
    params.update(iterations=1500, early_stopping_rounds=50, verbose=False)
    train_pool = Pool(data=X_train_raw, label=y_train_raw, cat_features=CAT_FEATURES, weight=sw)
    val_pool = Pool(data=X_val_raw, label=y_val_raw, cat_features=CAT_FEATURES)
    cb_model = CatBoostClassifier(**params)
    cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    cat_c = predict_catboost(cb_model, X_val_raw)
    print(f"[C] CatBoost(weight={weight_2024}) best_iter={cb_model.get_best_iteration()} | solo={compute_bss(cat_c, y_val_raw)[2]:.2f}")

    # MLP (WeightedRandomSampler, 3-seed)
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_c, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(tail2024, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, "control_success")
    X_va_cat, X_va_num, y_va = to_tensors(val_proc, CAT_COLS, num_cols, "control_success")
    y_va_np = val_proc["control_success"].values

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)
    sample_weight_mlp = np.ones(len(train_c))
    sample_weight_mlp[len(pre2024):] = weight_2024

    mlp_preds_list = []
    for seed in SCREEN_SEEDS:
        model, best_epoch = train_mlp_weighted(
            X_tr_cat, X_tr_num, y_tr, sample_weight_mlp, cat_dims, embed_dims, bin_edges,
            X_va_cat, X_va_num, y_va_np, seed=seed, device=device,
        )
        preds = predict_mlp(model, X_va_cat, X_va_num, device)
        solo_score = compute_bss(preds, y_va_np)[2]
        print(f"[C] MLP seed={seed} best_epoch={best_epoch} solo={solo_score:.2f}")
        mlp_preds_list.append(preds)
    mlp_c = np.mean(mlp_preds_list, axis=0)
    print(f"[C] MLP(3-seed 평균) solo={compute_bss(mlp_c, y_va_np)[2]:.2f}")

    w_cat_c, w_mlp_c, ic_c, score_c, _ = fit_meta_model(cat_c, mlp_c, y_va_np)
    print(f"[C] 블렌드={score_c:.2f}")

    print(f"\n=== 결과 요약 (검증: 2024년 {val_month_cutoff}~10월, weight={weight_2024}, n={len(y_a)}) ===")
    print(f"{'설정':<45} {'블렌드 score':>12}")
    print(f"{'A (2019~2023만, 프로덕션 7-seed 그대로)':<45} {score_a:>12.2f}")
    print(f"{'C (+2024 head weight='+str(weight_2024)+', 3-seed 스크리닝)':<45} {score_c:>12.2f}")
    print(f"delta: {score_c - score_a:+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["run"])
    parser.add_argument("--weight", type=float, default=5.0)
    parser.add_argument("--val-cutoff", type=int, default=9, help="검증 시작 월 (예: 7 -> 2024년 7~10월 검증)")
    args = parser.parse_args()
    step_run(weight_2024=args.weight, val_month_cutoff=args.val_cutoff)


if __name__ == "__main__":
    main()
