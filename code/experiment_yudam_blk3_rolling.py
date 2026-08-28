# code/experiment_yudam_blk3_rolling.py
"""[Pruning 실험 2 - 3단계] Cohen's d Phase B blk3 3컬럼 제거의 rolling-origin 다중시드 재검증.

Task 2-2(experiment_yudam_cohend_prune_ablate)에서 Phase B 6블록 중 유일하게
두 레짐(cutoff7 +3.01 / 2023 +19.76, 3-seed) 모두 blend Δ 양수였던 블록:

    blk3 = ['run_top_before', 'run_total_before', 'runner_on_1b']

이 3컬럼은 run_bot_before / base_state / num_runners_on 과 정보가 겹치는 중복 컬럼.
cutoff7 +3.01 은 노이즈 수준이라, 채택 판단 전에
  (1) rolling-origin 4-fold (cutoff7 / 2023 / 2022 / 2021)
  (2) 단일/3-seed 가 아니라 SEED_N(기본 5) 시드
로 다시 본다. drop 방식은 ablate 와 동일 — df 에서 물리 제거하지 않고 build_split 이
돌려준 num_cols / cat_feature_cols 리스트에서만 뺀다(TE-residual group key KeyError 방지).

출력: scratchpad/blk3_rolling.log + scratchpad/blk3_rolling_result.json

사용법: python -m code.experiment_yudam_blk3_rolling            # 5-seed, 4 regime
        python -m code.experiment_yudam_blk3_rolling --seeds 7
        python -m code.experiment_yudam_blk3_rolling --regimes cutoff7,2023
"""
import argparse
import gc
import json

import numpy as np

from code.experiment_yudam_common import build_split, run_experiment

OUT_JSON = "/tmp/claude-1000/-home-user-contest-mlp-lgaimers9/64d8ecb5-0a28-4eef-9550-ce16083f96e8/scratchpad/blk3_rolling_result.json"

BLK3 = ["run_top_before", "run_total_before", "runner_on_1b"]
DEFAULT_REGIMES = ["cutoff7", "2023", "2022", "2021"]


def _seeds(n):
    return list(range(1, n + 1))


def evaluate(regime, drop_list, tag, seed_n):
    parts = list(build_split(regime, verbose=True))
    if drop_list:
        s = set(drop_list)
        before = (len(parts[2]), len(parts[3]))
        parts[2] = [c for c in parts[2] if c not in s]      # num_cols (MLP)
        parts[3] = [c for c in parts[3] if c not in s]      # cat_feature_cols (CatBoost)
        print(f"[prune {tag}] num_cols {before[0]}->{len(parts[2])} | "
              f"cat_feature_cols {before[1]}->{len(parts[3])}", flush=True)
    res = run_experiment(*parts, mlp_seeds=_seeds(seed_n), cb_seeds=_seeds(seed_n),
                         label=f"{regime}:{tag}")
    del parts
    gc.collect()
    return dict(cat=res["cat_solo"], mlp=res["mlp_solo"], blend=res["blend"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--regimes", type=str, default=",".join(DEFAULT_REGIMES))
    args = ap.parse_args()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    seed_n = args.seeds

    print(f"[blk3-rolling] drop={BLK3} | seeds={seed_n} | regimes={regimes}", flush=True)
    result = {"drop": BLK3, "seeds": seed_n, "regimes": {}}

    for rg in regimes:
        base = evaluate(rg, None, "baseline", seed_n)
        cand = evaluate(rg, BLK3, "drop_blk3", seed_n)
        d = {k: cand[k] - base[k] for k in ("cat", "mlp", "blend")}
        result["regimes"][rg] = {"baseline": base, "candidate": cand, "delta": d}
        print(f"  [Δ blk3 | {rg}]  cat {d['cat']:+.2f} | mlp {d['mlp']:+.2f} | "
              f"blend {d['blend']:+.2f}", flush=True)
        with open(OUT_JSON, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # ---- 요약 ----
    print("\n" + "=" * 66, flush=True)
    print(f"=== blk3 제거 rolling-origin ({seed_n}-seed) blend Δ ===", flush=True)
    print(f"  {'regime':10s} | {'cat Δ':>9s} | {'mlp Δ':>9s} | {'blend Δ':>9s}", flush=True)
    bl = []
    for rg in regimes:
        d = result["regimes"][rg]["delta"]
        bl.append(d["blend"])
        print(f"  {rg:10s} | {d['cat']:+9.2f} | {d['mlp']:+9.2f} | {d['blend']:+9.2f}", flush=True)
    bl = np.array(bl)
    print(f"\n  blend Δ  mean {bl.mean():+.2f}  std {bl.std():.2f}  "
          f"min {bl.min():+.2f}  wins {int((bl > 0).sum())}/{len(bl)}", flush=True)
    verdict = ("채택후보 (전 레짐 양수 & mean>noise)"
               if bl.min() > 0 and bl.mean() > bl.std()
               else "노이즈/regime-flip -> 기각")
    print(f"  판정: {verdict}", flush=True)
    print(f"\n-> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
