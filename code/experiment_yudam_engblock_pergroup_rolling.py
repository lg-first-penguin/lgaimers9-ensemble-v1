# code/experiment_yudam_engblock_pergroup_rolling.py
"""[Pruning 실험 3-3] 엔지니어링 블록 — 그룹 하나씩 4레짐 3-seed 롤링 검증.

Phase 1(`experiment_yudam_engblock_prune.py`, cutoff7 1-seed)에서 G_trend/G_interact/
G_matchup/G_count 각각 제거가 blend Δ +15~+29 로 보였다. 3-2(`experiment_yudam_engblock_
escalate.py`)는 G_all(11컬럼 통째)만 4레짐 3-seed 로 escalate 해 기각했다(both regime-flip
mean -3.73 / catonly 2/4승 noise). 사용자 요청: G_all 대신 **그룹을 하나씩** 똑같은
프로토콜(4레짐 = cutoff7/2023/2022/2021, MLP 3-seed + CatBoost 3-seed, arm = both /
catonly)로 돌려서 G_all 이 희석한 단일 그룹 신호가 있는지 확인.

레짐당: BASE 1회 + (그룹수 × arm수) PRUNE.
  both    : 그룹 컬럼을 num_cols + cat_feature_cols 양쪽 제거
  catonly : 그룹 컬럼을 cat_feature_cols 에서만 제거 (MLP 유지 = MLP-only feed)

결정: 한 그룹이 both 또는 catonly 로 4레짐 전부 Δblend ≥ ~0 이면 실전 후보.
cutoff7만 + 이고 나머지 - 면 regime-flip = 현행 유지 (지금까지의 모든 케이스).

사용법 (기본값이 요청 그대로):
  python -m code.experiment_yudam_engblock_pergroup_rolling
  python -m code.experiment_yudam_engblock_pergroup_rolling --groups G_trend,G_matchup --seeds 3
"""
import argparse
import gc

import numpy as np

from code.experiment_yudam_common import build_split, run_experiment
from code.experiment_yudam_engblock_prune import GROUPS
from code.mlp_model import get_device
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS


def _drop_fn(cols):
    s = set(cols)
    return lambda allcols: [c for c in allcols if c in s]


def _catonly_fn(cols):
    s = set(cols)
    return lambda df, holdout: (df, s, set())


def run_regime(regime, group_names, arms, mlp_seeds, cb_seeds):
    print(f"\n{'='*72}\n=== regime={regime} ===\n{'='*72}", flush=True)
    base = run_experiment(*build_split(regime), mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label="BASE")
    gc.collect()
    out = []
    for g in group_names:
        cols = GROUPS[g]
        for arm in arms:
            if arm == "both":
                sp = build_split(regime, drop_cols=_drop_fn(cols))
            else:  # catonly
                sp = build_split(regime, add_features_fn=_catonly_fn(cols))
            r = run_experiment(*sp, mlp_seeds=mlp_seeds, cb_seeds=cb_seeds, label=f"{g}:{arm}")
            gc.collect()
            d_cat = r["cat_solo"] - base["cat_solo"]
            d_mlp = r["mlp_solo"] - base["mlp_solo"]
            d_bl = r["blend"] - base["blend"]
            print(f"  >>> Δ {g:11s} {arm:8s}  CatBoost {d_cat:+7.2f} | MLP {d_mlp:+7.2f} | blend {d_bl:+7.2f}", flush=True)
            out.append(dict(regime=regime, group=g, arm=arm, d_cat=d_cat, d_mlp=d_mlp, d_blend=d_bl,
                            base_blend=base["blend"], prune_blend=r["blend"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=str, default="G_trend,G_interact,G_matchup,G_count")
    ap.add_argument("--regimes", type=str, default="cutoff7,2023,2022,2021")
    ap.add_argument("--arms", type=str, default="both,catonly")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    group_names = [g.strip() for g in args.groups.split(",") if g.strip()]
    for g in group_names:
        if g not in GROUPS:
            raise SystemExit(f"unknown group {g}; valid: {list(GROUPS)}")
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    mlp_seeds = list(YUDAM_ENSEMBLE_SEEDS[:args.seeds])
    cb_seeds = list(YUDAM_CATBOOST_SEEDS[:args.seeds])
    get_device()
    print(f"groups={group_names} regimes={regimes} arms={arms} "
          f"seeds mlp={mlp_seeds} cb={cb_seeds}", flush=True)

    rows = []
    for rg in regimes:
        rows.extend(run_regime(rg, group_names, arms, mlp_seeds, cb_seeds))
        gc.collect()

    # ---- 종합: arm별 그룹별 4레짐 Δblend 매트릭스 ----
    print(f"\n{'='*90}\n=== 엔지니어링 블록 그룹별 4레짐 롤링 — 종합 (mlp{args.seeds}seed/cb{args.seeds}seed) ===\n{'='*90}", flush=True)
    for arm in arms:
        print(f"\n--- arm = {arm} ---", flush=True)
        print(f"{'group':>11} | " + " | ".join(f"{rg:>9}" for rg in regimes) + " | "
              f"{'mean':>7} {'wins':>5}", flush=True)
        for g in group_names:
            per = {r["regime"]: r["d_blend"] for r in rows if r["group"] == g and r["arm"] == arm}
            vals = [per[rg] for rg in regimes]
            wins = sum(1 for v in vals if v > 0)
            print(f"{g:>11} | " + " | ".join(f"{per[rg]:+9.2f}" for rg in regimes) +
                  f" | {np.mean(vals):+7.2f} {wins}/{len(vals)}", flush=True)
    # CatBoost/MLP solo Δ도 같이
    for arm in arms:
        print(f"\n--- arm = {arm}  (CatBoost solo Δ / MLP solo Δ) ---", flush=True)
        for g in group_names:
            gr = [r for r in rows if r["group"] == g and r["arm"] == arm]
            cs = ", ".join(f"{r['regime']}:{r['d_cat']:+.1f}" for r in gr)
            ms = ", ".join(f"{r['regime']}:{r['d_mlp']:+.1f}" for r in gr)
            print(f"  {g:11s} cat[{cs}]  mlp[{ms}]", flush=True)
    print("\n판정: 어떤 그룹이 both 또는 catonly 로 4레짐 전부 Δblend ≥ ~0 (mean 확실히 +) 이면 "
          "실전 후보. cutoff7만 +면 regime-flip = 현행 유지.", flush=True)


if __name__ == "__main__":
    main()
