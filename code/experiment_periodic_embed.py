# code/experiment_periodic_embed.py
"""수치형 피처 주기함수(periodic sin/cos) 임베딩 실험: `code/mlp_model.py::TabularMLP`가
수치형 44개 컬럼을 표준화만 해서 그대로 concat하는 대신, `code/periodic_mlp_model.py`의
PeriodicEmbedding(PLR)으로 바꾸면 MLP 단독 성능이 개선되는지 검증합니다.

PROJECT_HISTORY.md/EXPERIMENTS.md §14.4에서 "여전히 미시도"로 남겨둔 방향이며,
`code/experiment_3way_stack.py`와 동일하게 `dopip.py` 메인 파이프라인에는 아직
반영되지 않은 실험 단계 스크립트입니다. 유의미한 이득이 확인되면 그때
`code/mlp_model.py`/`code/train.py`/`submit/script.py`에 정식으로 반영합니다.

§14.3(loss function 스크리닝)과 동일한 관례로, 7-seed 풀 앙상블을 매번 돌리는 대신
3-seed 스크리닝으로 방향성만 먼저 확인합니다 — 노이즈 수준(시드 간 변동폭)보다 델타가
뚜렷하게 커야 7-seed 재검증으로 넘어갈 가치가 있다고 판단합니다.

`code/experiment_3way_stack.py --step base`가 이미 만들어 둔
`./open/temp/experiment_stack3/{train_split,val_split}.pkl` 캐시(트랙맨 결합 +
파생 피처까지 끝난 상태)를 재사용합니다 — 없으면 새로 빌드합니다.

사용법:
  python -m code.experiment_periodic_embed --step baseline
  python -m code.experiment_periodic_embed --step periodic --sigma 0.1 --k 8 --d 8
  python -m code.experiment_periodic_embed --step sweep   # sigma 그리드 한 번에
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    to_tensors, train_mlp, compute_bss, get_device,
)
from code.periodic_mlp_model import train_mlp_periodic, predict_periodic

CACHE_DIR = "./open/temp/experiment_stack3"
DATA_DIR = "./open/data"
TARGET_COL = "control_success"

SCREEN_SEEDS = [42, 123, 7]  # §14.3과 동일한 3-seed 스크리닝 세트
SIGMA_GRID = [0.01, 0.1, 1.0]


def build_features():
    from code.train import process_trackman_features_safe, add_engineered_features

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_mask = train_df["season"] < 2024
    val_mask = train_df["season"] == 2024

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in train_df.columns if c not in drop_cols]

    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    return train_split, val_split, features


def load_splits():
    cache_train = os.path.join(CACHE_DIR, "train_split.pkl")
    cache_val = os.path.join(CACHE_DIR, "val_split.pkl")
    if os.path.exists(cache_train) and os.path.exists(cache_val):
        print(f"[data] 캐시 재사용: {cache_train}, {cache_val}")
        train_split = pd.read_pickle(cache_train)
        val_split = pd.read_pickle(cache_val)
        features = [c for c in train_split.columns if c != TARGET_COL]
        return train_split, val_split, features

    print("[data] 캐시 없음 — 트랙맨 결합 + 파생 피처부터 새로 빌드합니다 (시간이 걸립니다)...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_split, val_split, features = build_features()
    train_split.to_pickle(cache_train)
    val_split.to_pickle(cache_val)
    return train_split, val_split, features


def prepare_tensors():
    train_split, val_split, features = load_splits()
    num_cols = [c for c in features if c not in CAT_COLS]

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_val_np = val_proc[TARGET_COL].values

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    return dict(
        X_tr_cat=X_tr_cat, X_tr_num=X_tr_num, y_tr=y_tr,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
        cat_dims=cat_dims, num_cols=num_cols, embed_dims=embed_dims,
    )


def run_baseline(data, seeds=SCREEN_SEEDS, verbose=False):
    device = get_device()
    scores = []
    for seed in seeds:
        t0 = time.time()
        model, best_epoch = train_mlp(
            data["X_tr_cat"], data["X_tr_num"], data["y_tr"],
            cat_dims=data["cat_dims"], num_numeric_feats=len(data["num_cols"]), embed_dims=data["embed_dims"],
            X_val_cat=data["X_val_cat"], X_val_num=data["X_val_num"], y_val=data["y_val"],
            device=device, verbose=verbose, seed=seed,
        )
        with torch.no_grad():
            preds = model(data["X_val_cat"].to(device), data["X_val_num"].to(device)).cpu().numpy()
        score = compute_bss(preds, data["y_val"])[2]
        scores.append(score)
        print(f"[baseline] seed={seed} best_epoch={best_epoch} Val Score={score:.2f} ({time.time()-t0:.1f}s)")
    print(f"[baseline] {len(seeds)}-seed 평균: {np.mean(scores):.2f} (std {np.std(scores):.2f})")
    return scores


def run_periodic(data, k, d, sigma, use_relu=True, seeds=SCREEN_SEEDS, verbose=False):
    device = get_device()
    scores = []
    for seed in seeds:
        t0 = time.time()
        model, best_epoch = train_mlp_periodic(
            data["X_tr_cat"], data["X_tr_num"], data["y_tr"],
            cat_dims=data["cat_dims"], num_numeric_feats=len(data["num_cols"]), embed_dims=data["embed_dims"],
            X_val_cat=data["X_val_cat"], X_val_num=data["X_val_num"], y_val=data["y_val"],
            periodic_k=k, periodic_d=d, periodic_sigma=sigma, periodic_relu=use_relu,
            device=device, verbose=verbose, seed=seed,
        )
        preds = predict_periodic(model, data["X_val_cat"], data["X_val_num"], device=device)
        score = compute_bss(preds, data["y_val"])[2]
        scores.append(score)
        print(f"[periodic k={k} d={d} sigma={sigma}] seed={seed} best_epoch={best_epoch} Val Score={score:.2f} ({time.time()-t0:.1f}s)")
    print(f"[periodic k={k} d={d} sigma={sigma}] {len(seeds)}-seed 평균: {np.mean(scores):.2f} (std {np.std(scores):.2f})")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["baseline", "periodic", "sweep"])
    parser.add_argument("--k", type=int, default=8, help="피처당 주파수 개수")
    parser.add_argument("--d", type=int, default=8, help="피처당 임베딩 차원")
    parser.add_argument("--sigma", type=float, default=0.1, help="주파수 초기화 스케일 (핵심 하이퍼파라미터)")
    parser.add_argument("--no-relu", action="store_true", help="PLR 대신 PL(ReLU 없이)로 실행")
    parser.add_argument("--sigmas", type=str, default=None,
                         help="sweep 스텝에서 SIGMA_GRID 대신 쓸 콤마구분 sigma 목록, 예: 0.001,0.003,0.02,0.05")
    parser.add_argument("--baseline-mean", type=float, default=None,
                         help="sweep 스텝에서 baseline을 재학습하지 않고 이 값을 baseline 평균으로 사용")
    parser.add_argument("--seeds", type=str, default=None,
                         help="SCREEN_SEEDS 대신 쓸 콤마구분 시드 목록, 예: 2024,99,555,31337")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SCREEN_SEEDS

    print("[data] 전처리된 텐서 준비 중...")
    data = prepare_tensors()
    print(f"[data] 수치형 {len(data['num_cols'])}개 | 범주형 {len(CAT_COLS)}개 (cat_dims={data['cat_dims']})")

    if args.step == "baseline":
        run_baseline(data, seeds=seeds)
    elif args.step == "periodic":
        run_periodic(data, k=args.k, d=args.d, sigma=args.sigma, use_relu=not args.no_relu, seeds=seeds)
    elif args.step == "sweep":
        if args.baseline_mean is not None:
            baseline_mean = args.baseline_mean
            print(f"[baseline] 재학습 생략, 지정된 평균 사용: {baseline_mean:.2f}")
        else:
            baseline_scores = run_baseline(data)
            baseline_mean = np.mean(baseline_scores)

        sigma_grid = [float(s) for s in args.sigmas.split(",")] if args.sigmas else SIGMA_GRID
        print(f"\n=== sigma 그리드 스윕 (k={args.k}, d={args.d}) — baseline 평균 {baseline_mean:.2f} ===")
        for sigma in sigma_grid:
            scores = run_periodic(data, k=args.k, d=args.d, sigma=sigma)
            delta = np.mean(scores) - baseline_mean
            print(f"[sweep] sigma={sigma} 평균 {np.mean(scores):.2f} (delta {delta:+.2f})\n")


if __name__ == "__main__":
    main()
