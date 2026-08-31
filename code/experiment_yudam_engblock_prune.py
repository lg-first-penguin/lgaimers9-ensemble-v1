# code/experiment_yudam_engblock_prune.py
"""[Pruning 실험 3] 유담 레시피가 끌고 들어온 엔지니어링 피처 블록 그룹단위 제거.

배경: `code/train.py::add_engineered_features` 는 유담 레시피에서 통째로 딸려왔고
이 repo 에서 개별 ablation 을 한 적이 없다. 현재 전부 CatBoost+MLP 양쪽에 먹인다.
일부는 과거 기각 계열(career-trajectory/acceleration/month-trend, 손수연식 interaction,
pair-matchup)과 형태가 닮았다.

그룹 (전부 both-model 제거):
  G_trend    : pitcher_trend, pitcher_consistency, pitcher_recent{1,3,5}_gap  (추세축 분해 5개)
  G_interact : pitcher_count_advantage_raw, pitcher_count_advantage_rel, count_pressure  (interaction 3개)
  G_matchup  : matchup  (asof_pitcher_success_rate - asof_batter_success_rate, 1개)
  G_count    : count_diff, is_full_count  (원시 카운트 재표현 2개)
  G_all      : 위 전부 (11개)

주의: pitcher_relative_success 는 제거 대상 아님 (same_hand_advantage / count_pressure /
count_advantage_rel 의 입력이라 build_split 안에서 이미 계산됨 — drop_cols 는 출력 컬럼만
피처목록에서 뺄 뿐 계산은 그대로 돈다).

Phase 1 (기본): cutoff7 single-seed 로 BASE 1회 + 각 그룹 PRUNE. blend Δ 로 coarse 필터.
Phase 2 : 생존분(blend Δ ≳ +5)만 --groups 로 지정 + --regimes cutoff7,2023,2022,2021
          --seeds 3 으로 escalate.

사용법:
  python -m code.experiment_yudam_engblock_prune                              # phase1: 전그룹 cutoff7 1-seed
  python -m code.experiment_yudam_engblock_prune --groups G_trend,G_matchup --regimes cutoff7,2023,2022,2021 --seeds 3
"""
import argparse
import gc

import numpy as np

from code.experiment_yudam_common import build_split, run_experiment
from code.mlp_model import get_device
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS

GROUPS = {
    "G_trend": ["pitcher_trend", "pitcher_consistency",
                "pitcher_recent1_gap", "pitcher_recent3_gap", "pitcher_recent5_gap"],
    "G_interact": ["pitcher_count_advantage_raw", "pitcher_count_advantage_rel", "count_pressure"],
    "G_matchup": ["matchup"],
    "G_count": ["count_diff", "is_full_count"],
}
GROUPS["G_all"] = [c for g in ("G_trend", "G_interact", "G_matchup", "G_count") for c in GROUPS[g]]


def _drop_fn(colset):
    s = set(colset)
    return lambda cols: [c for c in cols if c in s]


def run_regime(regime, group_names, mlp_seeds, cb_seeds):
    print(f"\n{'='*70}\n=== regime={regime} ===\n{'='*70}", flush=True)
    base = run_experiment(*build_split(regime), mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label="BASE")
    gc.collect()
    rows = []
    for gname in group_names:
        cols = GROUPS[gname]
        pr = run_experiment(*build_split(regime, drop_cols=_drop_fn(cols)),
                            mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label=f"PRUNE:{gname}")
        gc.collect()
        d_cat = pr["cat_solo"] - base["cat_solo"]
        d_mlp = pr["mlp_solo"] - base["mlp_solo"]
        d_bl = pr["blend"] - base["blend"]
        print(f"  >>> Δ {gname:11s} (PRUNE-BASE)  CatBoost {d_cat:+.2f} | MLP {d_mlp:+.2f} | blend {d_bl:+.2f}", flush=True)
        rows.append(dict(regime=regime, group=gname, d_cat=d_cat, d_mlp=d_mlp, d_blend=d_bl,
                         base_blend=base["blend"], prune_blend=pr["blend"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=str, default="G_trend,G_interact,G_matchup,G_count,G_all")
    ap.add_argument("--regimes", type=str, default="cutoff7")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    group_names = [g.strip() for g in args.groups.split(",") if g.strip()]
    for g in group_names:
        if g not in GROUPS:
            raise SystemExit(f"unknown group {g}; valid: {list(GROUPS)}")
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    mlp_seeds = list(YUDAM_ENSEMBLE_SEEDS[:args.seeds])
    cb_seeds = list(YUDAM_CATBOOST_SEEDS[:args.seeds])
    get_device()

    all_rows = []
    for rg in regimes:
        all_rows.extend(run_regime(rg, group_names, mlp_seeds, cb_seeds))
        gc.collect()

    print(f"\n{'='*82}\n=== 엔지니어링 블록 그룹제거 — 종합 (mlp{args.seeds}seed/cb{args.seeds}seed) ===\n{'='*82}", flush=True)
    print(f"{'regime':>9} | {'group':>11} | {'BASE bl':>9} | {'PRUNE bl':>9} | {'Δ cat':>8} | {'Δ mlp':>8} | {'Δ blend':>8}", flush=True)
    for r in all_rows:
        print(f"{r['regime']:>9} | {r['group']:>11} | {r['base_blend']:9.2f} | {r['prune_blend']:9.2f} | "
              f"{r['d_cat']:+8.2f} | {r['d_mlp']:+8.2f} | {r['d_blend']:+8.2f}", flush=True)
    print("\n판정: single-seed cutoff7 는 coarse 필터(노이즈 ~19pt). blend Δ ≳ +5 인 그룹만 "
          "4레짐 3-seed 로 escalate. 4레짐 전부 Δblend ≥ ~0 이어야 실전 후보.", flush=True)


if __name__ == "__main__":
    main()
