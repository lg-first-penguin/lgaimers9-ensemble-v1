# code/experiment_yudam_engblock_escalate.py
"""[Pruning 실험 3-2] 엔지니어링 블록 제거 — 4레짐 3-seed escalation.

Phase 1 (`code/experiment_yudam_engblock_prune.py`, cutoff7 1-seed): 5그룹 전부 blend
Δ +15~+29, CatBoost solo Δ +49~+86 (G_count 같은 2컬럼짜리 자명한 그룹도 +71 → single-seed
CatBoost 분산이 큼, magnitude 불신, 방향만). G_all ≈ G_trend ≈ G_interact (안 쌓임).

이 스크립트가 답하는 것:
  (a) 큰 CatBoost 이득이 rolling-origin(2023/2022/2021)에서도 유지되나, cutoff7만의 신기루인가
  (b) 양쪽 제거 vs CatBoost만 제거(=MLP-only feed) — 사용자 질문. Phase 1 은 MLP 가
      이 피처들에 무관심(+0.5~+28)이라 두 옵션이 비슷할 것으로 예측.

arm (레짐당):
  BASE          : 현행 (엔지니어링 11컬럼 CatBoost+MLP 양쪽)
  PRUNE_both    : 11컬럼 num_cols + cat_feature_cols 양쪽 제거
  PRUNE_catonly : 11컬럼 cat_feature_cols 에서만 제거 (MLP 는 유지 = MLP-only feed)

결정지표: PRUNE_both.blend - BASE.blend, PRUNE_catonly.blend - BASE.blend  (>0 이면 이득)
레짐 전부 Δblend ≥ ~0 이어야 실전 후보. cutoff7 만 + 이고 나머지 - 면 regime-flip → 현행 유지.

사용법:
  python -m code.experiment_yudam_engblock_escalate --seeds 3 --regimes cutoff7,2023,2022,2021
"""
import argparse
import gc

import numpy as np

from code.experiment_yudam_common import build_split, run_experiment
from code.experiment_yudam_engblock_prune import GROUPS
from code.mlp_model import get_device
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS

ENG11 = GROUPS["G_all"]  # 11 cols


def _drop_both(cols):
    return [c for c in cols if c in set(ENG11)]


def _exclude_catonly(df, holdout):
    # 컬럼 추가 없음. CatBoost 에서만 ENG11 제외, MLP(num_cols)엔 유지.
    return df, set(ENG11), set()


def run_regime(regime, mlp_seeds, cb_seeds):
    print(f"\n{'='*70}\n=== regime={regime} ===\n{'='*70}", flush=True)

    base = run_experiment(*build_split(regime), mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label="BASE")
    gc.collect()
    pboth = run_experiment(*build_split(regime, drop_cols=_drop_both),
                           mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label="PRUNE_both")
    gc.collect()
    pcat = run_experiment(*build_split(regime, add_features_fn=_exclude_catonly),
                          mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label="PRUNE_catonly")
    gc.collect()

    def _d(x):
        return (x["cat_solo"] - base["cat_solo"], x["mlp_solo"] - base["mlp_solo"], x["blend"] - base["blend"])

    db = _d(pboth)
    dc = _d(pcat)
    print(f"\n--- regime={regime} 요약 (mlp{len(mlp_seeds)}seed / cb{len(cb_seeds)}seed) ---", flush=True)
    print(f"  BASE           CatBoost {base['cat_solo']:8.2f} | MLP {base['mlp_solo']:8.2f} | blend {base['blend']:8.2f}", flush=True)
    print(f"  PRUNE_both     Δcat {db[0]:+7.2f} | Δmlp {db[1]:+7.2f} | Δblend {db[2]:+7.2f}", flush=True)
    print(f"  PRUNE_catonly  Δcat {dc[0]:+7.2f} | Δmlp {dc[1]:+7.2f} | Δblend {dc[2]:+7.2f}", flush=True)
    return dict(regime=regime, base=base, d_both=db, d_catonly=dc,
                both_blend=pboth["blend"], catonly_blend=pcat["blend"])


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

    print(f"\n{'='*86}\n=== 엔지니어링 블록 11컬럼 제거 — 4레짐 종합 (mlp{args.seeds}seed/cb{args.seeds}seed) ===\n{'='*86}", flush=True)
    print(f"{'regime':>9} | {'BASE bl':>9} | {'both Δcat':>9} {'Δmlp':>7} {'Δblend':>8} | {'catonly Δcat':>12} {'Δmlp':>7} {'Δblend':>8}", flush=True)
    for r in rows:
        b, c = r["d_both"], r["d_catonly"]
        print(f"{r['regime']:>9} | {r['base']['blend']:9.2f} | {b[0]:+9.2f} {b[1]:+7.2f} {b[2]:+8.2f} | "
              f"{c[0]:+12.2f} {c[1]:+7.2f} {c[2]:+8.2f}", flush=True)
    bb = [r["d_both"][2] for r in rows]
    cc = [r["d_catonly"][2] for r in rows]
    print(f"\nΔblend PRUNE_both    : mean {np.mean(bb):+.2f} std {np.std(bb):.2f} min {np.min(bb):+.2f} max {np.max(bb):+.2f}", flush=True)
    print(f"Δblend PRUNE_catonly : mean {np.mean(cc):+.2f} std {np.std(cc):.2f} min {np.min(cc):+.2f} max {np.max(cc):+.2f}", flush=True)
    print("판정: 두 arm 다 4레짐 전부 Δblend ≥ ~0 이면 실전 후보. cutoff7만 +면 regime-flip=현행유지.", flush=True)


if __name__ == "__main__":
    main()
