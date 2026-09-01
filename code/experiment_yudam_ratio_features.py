# code/experiment_yudam_ratio_features.py
"""[신규 국면] "기존 피처를 빼기/나눗셈으로 조합" 재검증 (다른 팀 제보).

우리가 이미 한 것:
  - 빼기(차분/gap/residual): season_rate_gap, TE-residual(*_res), pitcher_recent{1,3,5}_gap,
    pitcher_relative_success, count_diff, pitcher_trend, matchup, career_trend(기각) 등 —
    프로덕션 핵심이 여기 있음.
  - 나눗셈: coarse pitchmix(구종횟수/전체)와 rate 내부계산뿐. "rate A / rate B" 형태의
    명시적 비율 조합은 체계적으로 안 해봤다.  <-- 이 스크립트가 그걸 메꾼다.
  - 곱셈(교차항): G1(middle/reverse × count/pressure) 전부 기각(2026-08-29).

후보 (전부 공식 asof_* rate 컬럼 산술 · 2025 상수붕괴 위험 0):
  나눗셈 Q_EPS=0.02 additive smoothing, a/(b+eps), 콜드스타트 NaN 은 imputer 처리.
  ── 실패구성비 (실패①가운데 / 실패②큰이탈(볼) / 실패③반대) ──
   q_mid_ball   = middle_rate  / (ball_rate + e)      실패할때 가운데몰림 vs 큰이탈
   q_rev_mid    = reverse_rate / (middle_rate + e)    반대방향 vs 가운데
   q_strike_ball= strike_rate  / (ball_rate + e)      고전 제구지표(스트라이크/볼)
   q_mid_succ   = middle_rate  / (success_rate + e)   성공 대비 위험코스 비중
  ── 최근/통산 비율 (기존엔 gap=차분만) ──
   q_prev1_career   = prev1_game_success_rate / (success_rate + e)
   q_prev5_career   = prev5_game_success_rate / (success_rate + e)
   q_prev1mid_career= prev1_game_middle_rate  / (middle_rate + e)
  ── 투수/타자 대칭 비율 (기존 matchup=차분만) ──
   q_pb_succ = asof_pitcher_success_rate / (asof_batter_success_rate + e)
   q_pb_mid  = asof_pitcher_middle_rate  / (asof_batter_middle_rate + e)
  ── 구종 비율 ──
   q_fb_br      = fastball_rate / (breaking_rate + e)
  ── 신규 차분 (add_engineered_features 에 없는 축) ──
   d_rev_mid    = reverse_rate - middle_rate
   d_ball_strike= ball_rate    - strike_rate
   d_prev1mid   = prev1_game_middle_rate - middle_rate   (recent_gap 의 middle 축 버전)

Phase 1 (이 스크립트 기본): 그룹 단위 1-seed cutoff7 스크리닝.
  q_misscomp / q_form / q_matchup / q_mix / q_all(나눗셈전부) / d_novel(신규차분)
통과분(blend Δ >= +2)만 Phase 2 에서 개별 분해 + 3-seed 4-regime + catonly.

사용법:
  python -m code.experiment_yudam_ratio_features                       # phase1 1-seed cutoff7
  python -m code.experiment_yudam_ratio_features --seeds 3 --regime 2023
  python -m code.experiment_yudam_ratio_features --configs q_mid_ball,q_rev_mid
  python -m code.experiment_yudam_ratio_features --configs q_misscomp --catonly
"""
import argparse
import gc
import json
import os

import numpy as np

from code.experiment_yudam_common import build_split, run_experiment

OUT_JSON = os.path.join(os.path.dirname(__file__), os.pardir,
                        "scratchpad", "ratio_features_result.json")
Q_EPS = 0.02


def _q(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / (b + Q_EPS)


# name -> callable(df) -> np.ndarray
def _pool(df):
    p_s = df["asof_pitcher_success_rate"].values
    p_r = df["asof_pitcher_reverse_rate"].values
    p_m = df["asof_pitcher_middle_rate"].values
    p_b = df["asof_pitcher_ball_rate"].values
    p_k = df["asof_pitcher_strike_rate"].values
    b_s = df["asof_batter_success_rate"].values
    b_m = df["asof_batter_middle_rate"].values
    pv1s = df["asof_pitcher_prev1_game_success_rate"].values
    pv5s = df["asof_pitcher_prev5_game_success_rate"].values
    pv1m = df["asof_pitcher_prev1_game_middle_rate"].values
    fb = df["asof_pitcher_fastball_rate"].values
    br = df["asof_pitcher_breaking_rate"].values
    return {
        "q_mid_ball": _q(p_m, p_b),
        "q_rev_mid": _q(p_r, p_m),
        "q_strike_ball": _q(p_k, p_b),
        "q_mid_succ": _q(p_m, p_s),
        "q_prev1_career": _q(pv1s, p_s),
        "q_prev5_career": _q(pv5s, p_s),
        "q_prev1mid_career": _q(pv1m, p_m),
        "q_pb_succ": _q(p_s, b_s),
        "q_pb_mid": _q(p_m, b_m),
        "q_fb_br": _q(fb, br),
        "d_rev_mid": p_r - p_m,
        "d_ball_strike": p_b - p_k,
        "d_prev1mid": pv1m - p_m,
    }


GROUPS = {
    "q_misscomp": ["q_mid_ball", "q_rev_mid", "q_strike_ball", "q_mid_succ"],
    "q_form": ["q_prev1_career", "q_prev5_career", "q_prev1mid_career"],
    "q_matchup": ["q_pb_succ", "q_pb_mid"],
    "q_mix": ["q_fb_br"],
    "q_all": ["q_mid_ball", "q_rev_mid", "q_strike_ball", "q_mid_succ",
              "q_prev1_career", "q_prev5_career", "q_prev1mid_career",
              "q_pb_succ", "q_pb_mid", "q_fb_br"],
    "d_novel": ["d_rev_mid", "d_ball_strike", "d_prev1mid"],
}


def _resolve(names):
    """config 이름 리스트를 {config_name: [feature_col,...]} 로 편다.
    GROUPS 키면 그대로, 개별 피처명이면 단독 config 로."""
    out = {}
    for nm in names:
        if nm in GROUPS:
            out[nm] = GROUPS[nm]
        else:
            out[nm] = [nm]
    return out


def make_add_fn(cols, routing="both"):
    """routing: 'both' | 'catonly' (exclude_from_mlp=cols) | 'mlponly' (exclude_from_cat=cols)."""
    def fn(df, holdout):
        df = df.copy()
        pool = _pool(df)
        for c in cols:
            if c not in pool:
                raise KeyError(f"unknown ratio feature {c!r}")
            df[c] = pool[c]
        exc_cat = list(cols) if routing == "mlponly" else []
        exc_mlp = list(cols) if routing == "catonly" else []
        return df, exc_cat, exc_mlp
    return fn


def _run(regime, seeds, add_fn, label):
    parts = build_split(regime, add_features_fn=add_fn, verbose=True)
    res = run_experiment(*parts, mlp_seeds=list(range(1, seeds + 1)),
                         cb_seeds=list(range(1, seeds + 1)), label=label)
    del parts
    gc.collect()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--regime", default="cutoff7")
    ap.add_argument("--configs", default=None,
                    help="쉼표구분: GROUPS 키 또는 개별 피처명. 기본 = 모든 GROUPS")
    ap.add_argument("--catonly", action="store_true", help="추가피처를 CatBoost 전용으로")
    ap.add_argument("--mlponly", action="store_true", help="추가피처를 MLP 전용으로")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    out_json = os.path.abspath(args.out)
    if args.catonly and args.mlponly:
        ap.error("--catonly 와 --mlponly 는 동시 사용 불가")
    routing = "catonly" if args.catonly else ("mlponly" if args.mlponly else "both")

    if args.configs:
        names = args.configs.split(",")
    else:
        names = list(GROUPS)
    configs = _resolve(names)

    tag = f"{args.regime}{'' if routing == 'both' else '/' + routing}"
    result = {"regime": args.regime, "seeds": args.seeds, "routing": routing,
              "q_eps": Q_EPS, "baseline": None, "configs": {}}
    base = _run(args.regime, args.seeds, None, f"{tag}:baseline")
    result["baseline"] = dict(cat=base["cat_solo"], mlp=base["mlp_solo"], blend=base["blend"])
    with open(out_json, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    for nm, cols in configs.items():
        try:
            res = _run(args.regime, args.seeds, make_add_fn(cols, routing=routing), f"{tag}:{nm}")
            dd = dict(cat=res["cat_solo"] - base["cat_solo"],
                      mlp=res["mlp_solo"] - base["mlp_solo"],
                      blend=res["blend"] - base["blend"])
            result["configs"][nm] = dict(
                cols=cols,
                score=dict(cat=res["cat_solo"], mlp=res["mlp_solo"], blend=res["blend"]),
                delta=dd)
            print(f"  [Δ {nm} | {tag}]  cat {dd['cat']:+.2f} | mlp {dd['mlp']:+.2f} | "
                  f"blend {dd['blend']:+.2f}", flush=True)
        except Exception as e:
            result["configs"][nm] = {"error": repr(e)}
            print(f"  [ERR {nm}] {e!r}", flush=True)
        with open(out_json, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"=== 요약 ({tag}, {args.seeds}-seed, blend Δ 내림차순) ===", flush=True)
    rows = [(nm, d["delta"]["blend"], d["delta"]["cat"], d["delta"]["mlp"])
            for nm, d in result["configs"].items() if "delta" in d]
    for nm, b, c, m in sorted(rows, key=lambda x: -x[1]):
        flag = "  <- 통과후보" if b >= 2.0 else ""
        print(f"  {nm:20s} blend {b:+7.2f} | cat {c:+7.2f} | mlp {m:+7.2f}{flag}", flush=True)
    for nm, d in result["configs"].items():
        if "error" in d:
            print(f"  {nm:20s} ERROR: {d['error']}", flush=True)
    print(f"\n-> {out_json}", flush=True)


if __name__ == "__main__":
    main()
