# code/experiment_yudam_reverse_rate_prune.py
"""[Pruning 실험 2] reverse_rate 시즌 진행분(2컬럼) 완전 제거.

배경: 유담님 레시피 전면교체(2026-08-27)가 `apply_rate_progression_features` 를 끌고
들어왔다 -> `pitcher_reverse_season_rate`, `pitcher_reverse_season_rate_gap` 2컬럼이
현재 프로덕션(1117.03)에서 CatBoost+MLP 양쪽에 먹이고 있다. 이 repo 자체 이력:
  - 단일시드 기각: MLP/blend 부호반전 (cutoff7 -13.01 / season2023 +35.36)
    (memory: asof_reverse_rate_season_progression_rejected)
  - 5-seed 재검증으로 "재오픈, rolling-origin 필요, 미확정" (blend +0.43 / +6.49,
    둘 다 노이즈밴드 안)
즉 트랙맨64와 같은 상황: 포트가 끌고 들어왔고 / 이 repo 로컬 근거는 약함·부호반전 /
현행 1117 베이스라인에서 개별 remove-test 안 됨.

트랙맨64와 다른 점: `build_rate_end_lookup` 이 season+1 조인이라 2025 test 행은
2024 시즌말 baseline 을 정상적으로 받는다 -> 트랙맨64식 "2025 추론때 상수붕괴 ->
CatBoost miscalibrate" 는 없다. 그래서 collapse-sim 불필요, 그냥 PRUNE vs BASE 를
4레짐 × 3-seed 로 비교한다.

검증 (레짐당, 같은 build_split):
  BASE   : 현행 프로덕션 (reverse_rate 진행분 2컬럼 포함), val 정상 채점
  PRUNE  : 2컬럼을 num_cols / cat_feature_cols 양쪽에서 제거하고 학습, val 정상 채점

결정 지표 : PRUNE.blend - BASE.blend   (>0 이면 제거가 이득)
레짐: cutoff7 / 2023 / 2022 / 2021.  MLP 3-seed + CatBoost 3-seed 기본.

사용법:
  python -m code.experiment_yudam_reverse_rate_prune --seeds 3 --regimes cutoff7,2023,2022,2021
"""
import argparse
import gc

import numpy as np

from code.experiment_yudam_common import build_split, run_experiment
from code.mlp_model import get_device
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS

# code/train.py::RATE_PROGRESSION_SPECS name="pitcher_reverse" ->
# apply_rate_progression_features 가 만드는 2컬럼.
REVERSE_RATE_COLS = ["pitcher_reverse_season_rate", "pitcher_reverse_season_rate_gap"]


def _drop(cols):
    return [c for c in cols if c in REVERSE_RATE_COLS]


def run_regime(regime, mlp_seeds, cb_seeds):
    print(f"\n{'='*70}\n=== regime={regime} ===\n{'='*70}", flush=True)

    base = run_experiment(*build_split(regime), mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label="BASE")
    gc.collect()
    prune = run_experiment(*build_split(regime, drop_cols=_drop),
                           mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label="PRUNE")
    gc.collect()

    d_cat = prune["cat_solo"] - base["cat_solo"]
    d_mlp = prune["mlp_solo"] - base["mlp_solo"]
    d_bl = prune["blend"] - base["blend"]
    print(f"\n--- regime={regime} 요약 (mlp{len(mlp_seeds)}seed / cb{len(cb_seeds)}seed) ---", flush=True)
    print(f"  BASE   CatBoost {base['cat_solo']:8.2f} | MLP {base['mlp_solo']:8.2f} | blend {base['blend']:8.2f}", flush=True)
    print(f"  PRUNE  CatBoost {prune['cat_solo']:8.2f} | MLP {prune['mlp_solo']:8.2f} | blend {prune['blend']:8.2f}", flush=True)
    print(f"  >>> Δ (PRUNE - BASE)  CatBoost {d_cat:+.2f} | MLP {d_mlp:+.2f} | blend {d_bl:+.2f}", flush=True)
    return dict(regime=regime, base=base, prune=prune, d_cat=d_cat, d_mlp=d_mlp, d_blend=d_bl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--regimes", type=str, default="cutoff7,2023,2022,2021")
    args = ap.parse_args()
    mlp_seeds = list(YUDAM_ENSEMBLE_SEEDS[:args.seeds])
    cb_seeds = list(YUDAM_CATBOOST_SEEDS[:args.seeds])
    get_device()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    rows = []
    for rg in regimes:
        rows.append(run_regime(rg, mlp_seeds, cb_seeds))
        gc.collect()

    print(f"\n{'='*78}\n=== reverse_rate 진행분 완전제거 pruning — 종합 (mlp{args.seeds}seed/cb{args.seeds}seed) ===\n{'='*78}", flush=True)
    print(f"{'regime':>9} | {'BASE blend':>10} | {'PRUNE blend':>11} | {'Δ cat':>8} | {'Δ mlp':>8} | {'Δ blend':>8}", flush=True)
    for r in rows:
        print(f"{r['regime']:>9} | {r['base']['blend']:10.2f} | {r['prune']['blend']:11.2f} | "
              f"{r['d_cat']:+8.2f} | {r['d_mlp']:+8.2f} | {r['d_blend']:+8.2f}", flush=True)
    dd = [r["d_blend"] for r in rows]
    print(f"\nΔ blend (PRUNE - BASE): mean {np.mean(dd):+.2f}  std {np.std(dd):.2f}  "
          f"min {np.min(dd):+.2f}  max {np.max(dd):+.2f}", flush=True)
    print("판정 가이드: 모든 레짐 Δblend >= ~0 (+ 크면 명확) 이면 실전 제출 후보. "
          "부호가 레짐마다 뒤집히면 regime-flip = 제거 안 함(현행 유지).", flush=True)


if __name__ == "__main__":
    main()
