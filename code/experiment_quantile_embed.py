# code/experiment_quantile_embed.py
"""수치형 피처 quantile 기반 piecewise-linear 인코딩(PLE) 임베딩 실험.

`code/experiment_periodic_embed.py`(sin/cos 주파수 기반 PLR)와 대조 실험 —
같은 baseline, 같은 데이터 캐시, 같은 스크리닝 절차를 쓰되 앞단 인코딩만
quantile 구간 기반으로 바꿉니다. periodic 실험(EXPERIMENTS.md §15)이
3-seed에서는 좋아 보였다가 7-seed에서 노이즈로 판정된 교훈에 따라, 이번엔
처음부터 "3-seed로 방향 확인 -> 괜찮으면 바로 7-seed 재검증"까지 한 번에 갑니다.

`prepare_tensors`/`run_baseline`은 `code/experiment_periodic_embed.py`와 완전히
동일한 로직(모델 자체가 baseline이라 quantile과 무관)이라 그대로 재사용합니다.

사용법:
  python -m code.experiment_quantile_embed --step quantile --bins 8 --d 8 --seeds 42,123,7
  python -m code.experiment_quantile_embed --step sweep --binlist 4,8,16,32
"""
import argparse
import time

import numpy as np
import torch

from code.experiment_periodic_embed import prepare_tensors, run_baseline, CAT_COLS, SCREEN_SEEDS
from code.mlp_model import compute_bss, get_device
from code.quantile_mlp_model import fit_quantile_edges, train_mlp_quantile, predict_quantile

BIN_GRID = [4, 8, 16, 32]


def run_quantile(data, bin_edges, d, use_relu=True, seeds=SCREEN_SEEDS, verbose=False, label=None):
    device = get_device()
    label = label or f"n_bins={bin_edges.shape[1]-1} d={d}"
    scores = []
    for seed in seeds:
        t0 = time.time()
        model, best_epoch = train_mlp_quantile(
            data["X_tr_cat"], data["X_tr_num"], data["y_tr"],
            cat_dims=data["cat_dims"], bin_edges=bin_edges, embed_dims=data["embed_dims"],
            X_val_cat=data["X_val_cat"], X_val_num=data["X_val_num"], y_val=data["y_val"],
            quantile_d=d, quantile_relu=use_relu,
            device=device, verbose=verbose, seed=seed,
        )
        preds = predict_quantile(model, data["X_val_cat"], data["X_val_num"], device=device)
        score = compute_bss(preds, data["y_val"])[2]
        scores.append(score)
        print(f"[quantile {label}] seed={seed} best_epoch={best_epoch} Val Score={score:.2f} ({time.time()-t0:.1f}s)")
    print(f"[quantile {label}] {len(seeds)}-seed 평균: {np.mean(scores):.2f} (std {np.std(scores):.2f})")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["baseline", "quantile", "sweep"])
    parser.add_argument("--bins", type=int, default=8, help="피처당 quantile 구간 개수")
    parser.add_argument("--d", type=int, default=8, help="피처당 임베딩 차원")
    parser.add_argument("--no-relu", action="store_true", help="Q-LR 대신 Q-L(ReLU 없이)로 실행")
    parser.add_argument("--binlist", type=str, default=None,
                         help="sweep 스텝에서 BIN_GRID 대신 쓸 콤마구분 n_bins 목록, 예: 4,8,16,32")
    parser.add_argument("--baseline-mean", type=float, default=None,
                         help="baseline을 재학습하지 않고 이 값을 baseline 평균으로 사용 (기존 periodic 실험값 재사용 가능: 677.06)")
    parser.add_argument("--seeds", type=str, default=None,
                         help="SCREEN_SEEDS 대신 쓸 콤마구분 시드 목록, 예: 42,123,7 또는 2024,99,555,31337")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SCREEN_SEEDS

    print("[data] 전처리된 텐서 준비 중...")
    data = prepare_tensors()
    print(f"[data] 수치형 {len(data['num_cols'])}개 | 범주형 {len(CAT_COLS)}개 (cat_dims={data['cat_dims']})")

    if args.step == "baseline":
        run_baseline(data, seeds=seeds)
        return

    if args.step == "quantile":
        bin_edges = fit_quantile_edges(data["X_tr_num"], n_bins=args.bins)
        run_quantile(data, bin_edges, d=args.d, use_relu=not args.no_relu, seeds=seeds)
        return

    if args.step == "sweep":
        if args.baseline_mean is not None:
            baseline_mean = args.baseline_mean
            print(f"[baseline] 재학습 생략, 지정된 평균 사용: {baseline_mean:.2f}")
        else:
            baseline_scores = run_baseline(data, seeds=seeds)
            baseline_mean = np.mean(baseline_scores)

        bin_grid = [int(b) for b in args.binlist.split(",")] if args.binlist else BIN_GRID
        print(f"\n=== n_bins 그리드 스윕 (d={args.d}) — baseline 평균 {baseline_mean:.2f} ===")
        for n_bins in bin_grid:
            bin_edges = fit_quantile_edges(data["X_tr_num"], n_bins=n_bins)
            scores = run_quantile(data, bin_edges, d=args.d, seeds=seeds, label=f"n_bins={n_bins}")
            delta = np.mean(scores) - baseline_mean
            print(f"[sweep] n_bins={n_bins} 평균 {np.mean(scores):.2f} (delta {delta:+.2f})\n")


if __name__ == "__main__":
    main()
