# code/experiment_yudam_re_career_rolling.py
"""[재검증 #6 - rolling-origin] career-trend(CatBoost 전용, 4컬럼)를 시즌레벨
expand-window rolling-origin 2021/2022/2023 3-fold 로 검증. CatBoost-solo 만 측정
(MLP 스킵 -> 빠름). dual-regime(cutoff7 +6.30 / season2023 +2.71) 통과 후 진짜 관문.

원조 career_trajectory 는 여기서 1/3 승·평균 +1.37 로 기각됐다
(memory: career_trajectory_rejected_rolling_origin). 3/3 또는 평균 명확히 양수여야 후보.

각 fold Y: train = season<Y (+F1필터), val = season==Y. CatBoost = 유담 v2 HP 3-seed 배깅.

사용법:
  python -m code.experiment_yudam_re_career_rolling
  python -m code.experiment_yudam_re_career_rolling --seeds 3 --folds 2021,2022,2023
"""
import argparse
import gc

import numpy as np

from code.experiment_yudam_common import build_split, _yudam_catboost_params, TARGET
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.mlp_model import compute_bss
from code.train import YUDAM_CATBOOST_SEEDS
from code.experiment_yudam_re_career import make_adder


def cb_solo(train_split, val_split, cat_feature_cols, seeds):
    res = train_catboost_ensemble(
        train_split[cat_feature_cols], train_split[TARGET].values,
        val_split[cat_feature_cols], val_split[TARGET].values,
        seeds=seeds, verbose=False, params=_yudam_catboost_params(),
    )
    models = [m for m, _ in res]
    preds = predict_catboost_ensemble(models, val_split[cat_feature_cols])
    return compute_bss(preds, val_split[TARGET].values)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=str, default="2021,2022,2023")
    args = ap.parse_args()
    seeds = YUDAM_CATBOOST_SEEDS[:args.seeds]
    folds = args.folds.split(",")
    adder = make_adder(accel=False, cat_only=True)

    rows = []
    for fold in folds:
        print(f"\n{'='*60}\n=== rolling-origin fold: val=season=={fold} ===\n{'='*60}", flush=True)
        tr_b, va_b, _, cat_b, _ = build_split(regime=fold)
        base = cb_solo(tr_b, va_b, cat_b, seeds)
        del tr_b, va_b; gc.collect()

        tr_c, va_c, _, cat_c, _ = build_split(regime=fold, add_features_fn=adder)
        cand = cb_solo(tr_c, va_c, cat_c, seeds)
        del tr_c, va_c; gc.collect()

        d = cand - base
        rows.append((fold, base, cand, d))
        print(f"[fold {fold}] CatBoost solo  base={base:.2f}  +career-trend={cand:.2f}  Δ={d:+.2f}", flush=True)

    print(f"\n{'='*60}\n=== rolling-origin 요약 (career-trend -> CatBoost 전용) ===\n{'='*60}", flush=True)
    deltas = [d for _, _, _, d in rows]
    for fold, base, cand, d in rows:
        print(f"  fold {fold}:  base={base:8.2f}  new={cand:8.2f}  Δ={d:+7.2f}", flush=True)
    wins = sum(1 for d in deltas if d > 0)
    print(f"\n  {wins}/{len(deltas)} fold 승 | 평균 Δ={np.mean(deltas):+.2f} | std={np.std(deltas):.2f} | "
          f"min={min(deltas):+.2f} max={max(deltas):+.2f}", flush=True)
    print(f"  (원조 career_trajectory rolling-origin: 1/3, 평균 +1.37 -> 기각)", flush=True)


if __name__ == "__main__":
    main()
