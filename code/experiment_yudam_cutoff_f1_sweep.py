# code/experiment_yudam_cutoff_f1_sweep.py
"""[Step 2] train cutoff month + F1 필터 경계년도 재스윕.

구 cutoff 스윕(cutoff∈{4..10}, weight∈{1,5} -> cutoff=7 채택)은 옛 모델·옛 피처셋
(트랙맨64 없음 / reverse_rate 없음 / 이 repo 자체 CatBoost HP)에서 한 것이다.
유담 레시피(트랙맨64 복원 + reverse_rate + v2 CatBoost HP)에서 최적 cutoff/F1경계가
다를 수 있어 재확인한다. 저비용 빚 갚기.

두 개의 1-D 스윕 (교차 안 함):

  A) cutoff month 스윕 — val 창을 2024/08~10 로 **고정**해 점수 비교가능하게.
     train = season<2024 | (season==2024 & month < C),  C ∈ {4,5,6,7,8}
     F1 경계 = 2022 고정.

  B) F1 경계 스윕 — cutoff=7 / val=2024/07~10 (프로덕션 관례) 고정.
     f1_boundary ∈ {None(off), 2021, 2022, 2023}

각 config: raw-concat MLP + v2 CatBoost, 기본 3-seed씩, 메타 val 전체 fit/채점.
CatBoost solo 는 결정론적이라 config 간 직접 비교 가능. blend 3-seed 노이즈밴드 ±~7 감안.

사용법:
  python -m code.experiment_yudam_cutoff_f1_sweep --which A --seeds 3
  python -m code.experiment_yudam_cutoff_f1_sweep --which B --seeds 3
  python -m code.experiment_yudam_cutoff_f1_sweep --which both --seeds 3
"""
import argparse
import gc

from code.experiment_yudam_common import build_split, run_experiment


def one(label, **split_kw):
    ts, vs, num_cols, catf, allc = build_split(regime="cutoff7", **split_kw)
    r = run_experiment(ts, vs, num_cols, catf, allc,
                       mlp_seeds=SEEDS, cb_seeds=SEEDS, label=label)
    del ts, vs
    gc.collect()
    return r


def sweep_A():
    print(f"\n{'#'*70}\n# 스윕 A: cutoff month (val=2024/08~10 고정, F1=2022)\n{'#'*70}", flush=True)
    rows = []
    for C in [5, 6, 7, 8]:   # {4}는 구 스윕이 커버한 저구간이라 제외
        r = one(f"A:cutoff={C}", cutoff_month=C, fixed_val_month=8, f1_boundary=2022)
        rows.append((C, r))
    print(f"\n=== 스윕 A 요약 (val 고정 2024/08~10) ===", flush=True)
    print(f"{'cutoff':>7} | {'CatBoost':>9} | {'MLP':>8} | {'blend':>8}", flush=True)
    for C, r in rows:
        print(f"{C:>7} | {r['cat_solo']:9.2f} | {r['mlp_solo']:8.2f} | {r['blend']:8.2f}", flush=True)
    best = max(rows, key=lambda x: x[1]["blend"])
    prod_blend = [r["blend"] for C, r in rows if C == 7][0]
    print(f"  -> blend 최고: cutoff={best[0]} (blend={best[1]['blend']:.2f}). "
          f"프로덕션 cutoff=7 blend={prod_blend:.2f} (Δ {best[1]['blend'] - prod_blend:+.2f})", flush=True)


def sweep_B():
    print(f"\n{'#'*70}\n# 스윕 B: F1 경계년도 (cutoff=7, val=2024/07~10 고정)\n{'#'*70}", flush=True)
    rows = []
    for b in [None, 2021, 2022, 2023]:
        r = one(f"B:f1={b}", cutoff_month=7, fixed_val_month=7, f1_boundary=b)
        rows.append((b, r))
    print(f"\n=== 스윕 B 요약 ===", flush=True)
    print(f"{'f1_bnd':>7} | {'CatBoost':>9} | {'MLP':>8} | {'blend':>8}", flush=True)
    for b, r in rows:
        print(f"{str(b):>7} | {r['cat_solo']:9.2f} | {r['mlp_solo']:8.2f} | {r['blend']:8.2f}", flush=True)
    prod = [r for b, r in rows if b == 2022][0]
    best = max(rows, key=lambda x: x[1]["blend"])
    print(f"  -> blend 최고: f1_boundary={best[0]} (blend={best[1]['blend']:.2f}). "
          f"프로덕션 f1=2022 blend={prod['blend']:.2f}", flush=True)


def main():
    global SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["A", "B", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    from code.train import YUDAM_ENSEMBLE_SEEDS
    SEEDS = list(YUDAM_ENSEMBLE_SEEDS[:args.seeds])
    print(f"seeds={SEEDS}", flush=True)
    if args.which in ("A", "both"):
        sweep_A()
    if args.which in ("B", "both"):
        sweep_B()


if __name__ == "__main__":
    main()
