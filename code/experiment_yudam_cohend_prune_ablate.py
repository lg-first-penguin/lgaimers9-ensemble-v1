# code/experiment_yudam_cohend_prune_ablate.py
"""[Pruning 실험 2 - 2단계] Cohen's d 후보 셋의 실제 제거·재검증 + 상호작용 격리.

1단계(experiment_yudam_cohend_prune.py)가 고른 프루닝 후보 셋을
`scratchpad/cohend_prune_sets.json` 에서 읽어, candidate B(유담) raw 피처셋 위에서
실제로 빼고 cutoff7 + 2023 두 레짐, 3-seed 로 재검증한다.

Phase A (통째 제거):
  baseline(제거 없음) vs
    drop S_both(18)                     — d<0.10 AND 트리 하위25%, 가장 안전
    drop S_d_negligible(25)             — |d|<0.02
    drop (S_both ∪ S_d_negligible)      — 합집합
  트랙맨64 는 [실험1] 소관이라 여기선 건드리지 않음(전부 유지).

Phase B (자동, Phase A 후보가 어느 레짐에서든 blend 를 REGRESS_TOL 이상 깎을 때만):
  그 후보 셋을 cb_imp 오름차순 정렬 후 3개씩 블록으로 쪼개, 각 블록만 baseline 위에서
  제거해보고(= "2~3개씩 넣다 뺐다" 상호작용 체크), Δblend >= -KEEP_TOL 인 블록만
  "안전" 으로 모아 합집합을 다시 통째로 재검증한다. 해로운 블록을 격리해서 보고.

출력: scratchpad/cohend_ablate.log (표) + scratchpad/cohend_ablate_result.json.

사용법: python -m code.experiment_yudam_cohend_prune_ablate
"""
import gc
import json
import re

import numpy as np

from code.experiment_yudam_common import build_split, run_experiment

SETS_JSON = "/tmp/claude-1000/-home-user-contest-mlp-lgaimers9/64d8ecb5-0a28-4eef-9550-ce16083f96e8/scratchpad/cohend_prune_sets.json"
OUT_JSON = "/tmp/claude-1000/-home-user-contest-mlp-lgaimers9/64d8ecb5-0a28-4eef-9550-ce16083f96e8/scratchpad/cohend_ablate_result.json"

REGIMES = ["cutoff7", "2023"]
SEEDS = (3, 3)             # (mlp_seed_count, cb_seed_count) -> [1,2,3] each
REGRESS_TOL = 3.0         # blend 이만큼 이상 깎이면 Phase B 격리 발동
KEEP_TOL = 1.5            # Phase B: 블록 제거 Δblend >= -이 값이면 "안전"


def _seeds(n):
    return list(range(1, n + 1))


def evaluate(regime, drop_list, tag):
    """baseline 대비가 아니라 절대 점수만 반환. drop_list=None 이면 baseline.

    피처 제거는 df 에서 물리적으로 빼지 않는다(그러면 TE-residual 이 group key 로 쓰는
    num_runners_on/strikes_before 등이 KeyError). build_split 이 돌려준 num_cols /
    cat_feature_cols 리스트에서만 빼서 두 모델 입력에서 제외한다 — 프루닝 의미론상 이게
    맞고, TE_RESIDUAL_COLS(te_covered 등)까지 확실히 제외된다."""
    parts = list(build_split(regime, verbose=True))
    if drop_list:
        s = set(drop_list)
        before = (len(parts[2]), len(parts[3]))
        parts[2] = [c for c in parts[2] if c not in s]      # num_cols (MLP)
        parts[3] = [c for c in parts[3] if c not in s]      # cat_feature_cols (CatBoost)
        print(f"[prune {tag}] num_cols {before[0]}->{len(parts[2])} | "
              f"cat_feature_cols {before[1]}->{len(parts[3])}", flush=True)
    res = run_experiment(*parts, mlp_seeds=_seeds(SEEDS[0]), cb_seeds=_seeds(SEEDS[1]),
                         label=f"{regime}:{tag}")
    del parts
    gc.collect()
    return dict(cat=res["cat_solo"], mlp=res["mlp_solo"], blend=res["blend"])


def main():
    with open(SETS_JSON) as f:
        sets = json.load(f)
    S_both = sets["S_both"]
    S_neg = sets["S_d_negligible"]
    S_union = sorted(set(S_both) | set(S_neg))
    print(f"[sets] S_both={len(S_both)} S_d_negligible={len(S_neg)} union={len(S_union)}", flush=True)

    phaseA = {
        "S_both": S_both,
        "S_d_negligible": S_neg,
        "S_both_UNION_neg": S_union,
    }

    result = {"seeds": SEEDS, "phaseA": {}, "phaseB": {}}
    baseline = {}
    for rg in REGIMES:
        baseline[rg] = evaluate(rg, None, "baseline")
        result["phaseA"].setdefault("_baseline", {})[rg] = baseline[rg]

    for name, dl in phaseA.items():
        result["phaseA"][name] = {}
        for rg in REGIMES:
            sc = evaluate(rg, dl, f"drop_{name}")
            d = {k: sc[k] - baseline[rg][k] for k in ("cat", "mlp", "blend")}
            result["phaseA"][name][rg] = {"score": sc, "delta": d}
            print(f"  [Δ {name} | {rg}]  cat {d['cat']:+.2f} | mlp {d['mlp']:+.2f} | "
                  f"blend {d['blend']:+.2f}", flush=True)

    # ---- Phase B: 어느 레짐에서든 REGRESS_TOL 이상 깎은 후보만 3개씩 블록 격리 ----
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    for name, dl in phaseA.items():
        worst = min(result["phaseA"][name][rg]["delta"]["blend"] for rg in REGIMES)
        if worst >= -REGRESS_TOL:
            print(f"[Phase B] {name}: 최악 레짐 blend Δ {worst:+.2f} >= -{REGRESS_TOL} "
                  f"-> 격리 불필요 (통째 제거 안전)", flush=True)
            continue
        print(f"\n[Phase B] {name}: 최악 blend Δ {worst:+.2f} < -{REGRESS_TOL} -> 3개씩 블록 격리", flush=True)
        # cb_imp 오름차순 정렬용: cohend_prune_sets.json 엔 imp 가 없으니 이름 정렬로 안정화만.
        feats = sorted(dl)
        blocks = [feats[i:i + 3] for i in range(0, len(feats), 3)]
        block_report = []
        safe = []
        for bi, blk in enumerate(blocks):
            row = {"block": blk}
            for rg in REGIMES:
                sc = evaluate(rg, blk, f"{name}_blk{bi}")
                dbl = sc["blend"] - baseline[rg]["blend"]
                row[rg] = dbl
                print(f"    blk{bi} {blk} | {rg} blend Δ {dbl:+.2f}", flush=True)
            row["min_delta"] = min(row[rg] for rg in REGIMES)
            row["safe"] = row["min_delta"] >= -KEEP_TOL
            if row["safe"]:
                safe.extend(blk)
            block_report.append(row)
        # 안전 블록 합집합 재검증
        recheck = {}
        if safe:
            for rg in REGIMES:
                sc = evaluate(rg, safe, f"{name}_safeunion")
                recheck[rg] = {"score": sc,
                               "delta_blend": sc["blend"] - baseline[rg]["blend"]}
        result["phaseB"][name] = {"blocks": block_report, "safe_union": safe, "recheck": recheck}
        with open(OUT_JSON, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # ---- 최종 요약 ----
    print("\n" + "=" * 70, flush=True)
    print("=== Phase A 요약 (blend Δ, baseline 대비) ===", flush=True)
    print(f"  {'set':22s} | {'cutoff7':>9s} | {'2023':>9s}", flush=True)
    for name in phaseA:
        c = result["phaseA"][name]["cutoff7"]["delta"]["blend"]
        e = result["phaseA"][name]["2023"]["delta"]["blend"]
        verdict = "제거후보" if min(c, e) >= -REGRESS_TOL else "regime-flip/손해"
        print(f"  {name:22s} | {c:+9.2f} | {e:+9.2f}   {verdict}", flush=True)
    for name, pb in result["phaseB"].items():
        print(f"\n=== Phase B {name}: 안전 블록 합집합 = {pb['safe_union']}", flush=True)
        for rg, rc in pb["recheck"].items():
            print(f"    {rg} safe-union blend Δ {rc['delta_blend']:+.2f}", flush=True)
    print(f"\n-> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
