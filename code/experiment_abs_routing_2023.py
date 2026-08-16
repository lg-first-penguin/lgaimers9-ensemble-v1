# code/experiment_abs_routing_2023.py
"""§35.6에서 기각한 "9~10월엔 weight=5 모델, 나머지는 weight=1 모델로 라우팅/게이팅"
아이디어를 사용자가 재검토 요청: "w=5가 9~10월 검증에서 (같은 cutoff 기준) 항상
w=1을 이긴다" — 실제로 2024 cutoff=9/10 블렌드에서는 맞는 관찰이다
(cutoff9: w1=408.75 vs w5=430.28, cutoff10: w1=257.37 vs w5=318.21).
다만 두 표본 모두 검증행수가 작아(34,976 / 1,671) 노이즈일 가능성을 배제 못 했다.

이 스크립트는 같은 비교를 **2023년**에도 재현해, "9~10월엔 w=5가 유리하다"는 게
2024만의 우연(작은 표본 노이즈)인지 아니면 두 시즌에 걸쳐 재현되는 진짜 패턴인지를
CatBoost+MLP 블렌드(3-seed 스크리닝) 기준으로 판정한다.

§35.5에서 확인한 F1 필터 트랩(season<2023만 학습하면 F1 필터가 F행을 100% 제거)을
피하기 위해 game_type='R'만 사용한다(F/R 이슈를 완전히 배제하는 가장 깨끗한 방법,
§35.5의 "R전용" 방식과 동일). 2023 R전용 월별 breakdown(§35.5)에서 9월(391.74)이
연중 최저, 10월(407.49)도 낮아 2024와 마찬가지로 9~10월이 약한 구간이다.

사용법:
  python -m code.experiment_abs_routing_2023 --val-cutoff 9
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool

from code.train import add_engineered_features
from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, predict_catboost
from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, apply_preprocessing, fit_quantile_edges, get_device, to_tensors,
)
from code.experiment_abs_mlp import train_mlp_weighted, predict_mlp, SCREEN_SEEDS
from code.blend_model import fit_meta_model

DATA_DIR = "./open/data"
TARGET_COL = "control_success"


def build_splits_r_only(val_month_cutoff):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    full_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    full_df = full_df[full_df["game_type"] == "R"].reset_index(drop=True)  # F/R 이슈 완전 배제

    pre_mask = full_df["season"] < 2023
    head_mask = (full_df["season"] == 2023) & (full_df["game_month"] < val_month_cutoff)
    tail_mask = (full_df["season"] == 2023) & (full_df["game_month"] >= val_month_cutoff)

    league_mean = full_df.loc[pre_mask, TARGET_COL].mean()
    fdf = add_engineered_features(full_df, league_mean)
    for c in CAT_FEATURES:
        fdf[c] = fdf[c].astype(str)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in fdf.columns if c not in drop_cols]

    pre = fdf.loc[pre_mask, features + [TARGET_COL]].reset_index(drop=True)
    head = fdf.loc[head_mask, features + [TARGET_COL]].reset_index(drop=True)
    tail = fdf.loc[tail_mask, features + [TARGET_COL]].reset_index(drop=True)

    print(f"[split, R전용] pre-2023: {len(pre)}행 | 2023 {val_month_cutoff-3 if val_month_cutoff>3 else 3}~{val_month_cutoff-1}월(head): {len(head)}행 "
          f"| 2023 {val_month_cutoff}~10월(검증, tail): {len(tail)}행")
    return pre, head, tail, features


def train_blend(pre, head, tail, features, weight, seeds=SCREEN_SEEDS):
    num_cols = [c for c in features if c not in CAT_COLS]
    device = get_device()

    train_df = pd.concat([pre, head], ignore_index=True)
    X_train_raw, y_train_raw = train_df[features], train_df[TARGET_COL].values
    X_val_raw, y_val_raw = tail[features], tail[TARGET_COL].values

    sw = np.ones(len(train_df))
    sw[len(pre):] = weight
    params = dict(CATBOOST_PARAMS)
    params.update(iterations=1500, early_stopping_rounds=50, verbose=False)
    train_pool = Pool(data=X_train_raw, label=y_train_raw, cat_features=CAT_FEATURES, weight=sw)
    val_pool = Pool(data=X_val_raw, label=y_val_raw, cat_features=CAT_FEATURES)
    cb_model = CatBoostClassifier(**params)
    cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    cat_preds = predict_catboost(cb_model, X_val_raw)
    print(f"  [CatBoost weight={weight}] best_iter={cb_model.get_best_iteration()} | solo={compute_bss(cat_preds, y_val_raw)[2]:.2f}")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_df, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(tail, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_va_cat, X_va_num, _ = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_va_np = val_proc[TARGET_COL].values

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)
    sample_weight_mlp = np.ones(len(train_df))
    sample_weight_mlp[len(pre):] = weight

    mlp_preds_list = []
    for seed in seeds:
        model, best_epoch = train_mlp_weighted(
            X_tr_cat, X_tr_num, y_tr, sample_weight_mlp, cat_dims, embed_dims, bin_edges,
            X_va_cat, X_va_num, y_va_np, seed=seed, device=device,
        )
        preds = predict_mlp(model, X_va_cat, X_va_num, device)
        print(f"  [MLP weight={weight} seed={seed}] best_epoch={best_epoch} | solo={compute_bss(preds, y_va_np)[2]:.2f}")
        mlp_preds_list.append(preds)
    mlp_preds = np.mean(mlp_preds_list, axis=0)
    print(f"  [MLP weight={weight} 3-seed 평균] solo={compute_bss(mlp_preds, y_va_np)[2]:.2f}")

    w_cat, w_mlp, ic, score, _ = fit_meta_model(cat_preds, mlp_preds, y_va_np)
    print(f"  [블렌드 weight={weight}] score={score:.2f}")
    return score


def step_run(val_month_cutoff):
    pre, head, tail, features = build_splits_r_only(val_month_cutoff)

    print(f"\n--- weight=1 ---")
    score_w1 = train_blend(pre, head, tail, features, weight=1.0)
    print(f"\n--- weight=5 ---")
    score_w5 = train_blend(pre, head, tail, features, weight=5.0)

    print(f"\n=== 결과 요약 (2023년 {val_month_cutoff}~10월 검증, R전용, n={len(tail)}) ===")
    print(f"{'설정':<20} {'블렌드 score':>12}")
    print(f"{'weight=1':<20} {score_w1:>12.2f}")
    print(f"{'weight=5':<20} {score_w5:>12.2f}")
    print(f"delta(w5-w1): {score_w5 - score_w1:+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-cutoff", type=int, default=9)
    args = parser.parse_args()
    step_run(val_month_cutoff=args.val_cutoff)


if __name__ == "__main__":
    main()
