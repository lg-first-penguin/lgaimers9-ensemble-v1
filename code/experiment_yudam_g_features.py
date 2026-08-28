# code/experiment_yudam_g_features.py
"""[Task 3] 근본 고찰(scratchpad/FEATURE_THESIS.md)에서 나온 신규 피쳐군 G1~G4 검증.

전부 순수 공식컬럼 산술 (2025 추론 시 상수붕괴 위험 0). candidate B(유담) raw
피처셋 위에 add_features_fn / drop_cols 훅으로 얹어 재검증한다.

G1 실패-직결 교차항 (기존 교차항은 전부 success_rate 기반; middle=실패①, reverse=실패③ 기반은 없음)
  g1_mxc = asof_pitcher_middle_rate  * count_diff
  g1_rxp = asof_pitcher_reverse_rate * pressure_signal  (= li*((s>=2)|(b>=3)), 프로덕션 동일식)
  g1_mxp = asof_pitcher_middle_rate  * pressure_signal
  g1_rxc = asof_pitcher_reverse_rate * count_diff

G2 타자쪽 대칭항 (matchup/count_advantage 가 전부 투수쪽)
  g2_brel = asof_batter_success_rate - league_mean(<holdout)
  g2_bca  = g2_brel * count_diff
  g2_bmxc = asof_batter_middle_rate * count_diff
  g2_pbm  = asof_pitcher_middle_rate - asof_batter_middle_rate

G3 identity-free 상황축 TE-residual (기존 TE축은 전부 id 기반) — CatBoost 전용
  g3_bso = (balls, strikes, outs) 별 control_success 잔차
  g3_bo  = (base_state, outs_before) 별
  g3_li  = li 5분위 별
  누수차단: causal_smoothed_te_encode (season as-of, exact=False) = 프로덕션 TE 동일 패턴.

G4 ABS-era realignment (사용자 제안): 시즌진행분 앵커를 1~2년 뒤로 + SWAP
  ① = asof_{p,b}_success_rate  ② = {p,b}_season_* (시즌진행분 계열)
  V-A g4_{role}_lastseason_rate = 직전 완결시즌 그 시즌만의 성공률 (cum(s-1)-cum(s-2))
  V-B g4_{role}_recent2_rate    = 2년전 시즌말 앵커 진행분 (asof_now - cum(s-2))
  g4_1 +VA / g4_2 +VB / g4_3 +VA+VB / g4_4 ②+VA(①제거 SWAP) /
  g4_5 SWAP+①fallback(VA결측시만) / g4_6 ①+VA(②제거)

Phase 1 = 전 config 1-seed cutoff7 스크리닝(이 스크립트). 통과분만 이후 3-seed
dual-regime + rolling 2021/2022 + 2·3개씩 교차 (별도 실행).

사용법: python -m code.experiment_yudam_g_features
        python -m code.experiment_yudam_g_features --only g4 --seeds 3 --regime 2023
        python -m code.experiment_yudam_g_features --configs g1_mxc,g4_4_swap
"""
import argparse
import gc
import json

import numpy as np
import pandas as pd

from code.experiment_yudam_common import build_split, run_experiment, TARGET
from code.train import causal_smoothed_te_encode, TE_K_SMOOTH

OUT_JSON = "/tmp/claude-1000/-home-user-contest-mlp-lgaimers9/64d8ecb5-0a28-4eef-9550-ce16083f96e8/scratchpad/g_features_result.json"

CAREER_SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
]
SEASON_PROG_PREFIXES = ("pitcher_season_", "batter_season_")


def _pressure_signal(df):
    return df["li"].values * (((df["strikes_before"] >= 2) | (df["balls_before"] >= 3)).astype(np.int64).values)


# ------- G1 -------
def make_g1(cols):
    def fn(df, holdout):
        df = df.copy()
        ps = _pressure_signal(df)
        m = df["asof_pitcher_middle_rate"].values
        r = df["asof_pitcher_reverse_rate"].values
        cd = df["count_diff"].values
        pool = {"g1_mxc": m * cd, "g1_rxp": r * ps, "g1_mxp": m * ps, "g1_rxc": r * cd}
        for c in cols:
            df[c] = pool[c]
        return df, [], []
    return fn


# ------- G2 -------
def make_g2(cols):
    def fn(df, holdout):
        df = df.copy()
        lg = df.loc[df["season"] < holdout, TARGET].mean()
        cd = df["count_diff"].values
        brel = df["asof_batter_success_rate"].values - lg
        pool = {
            "g2_brel": brel,
            "g2_bca": brel * cd,
            "g2_bmxc": df["asof_batter_middle_rate"].values * cd,
            "g2_pbm": df["asof_pitcher_middle_rate"].values - df["asof_batter_middle_rate"].values,
        }
        for c in cols:
            df[c] = pool[c]
        return df, [], []
    return fn


# ------- G3 (CatBoost 전용) -------
G3_AXES = {
    "g3_bso": ["balls_before", "strikes_before", "outs_before"],
    "g3_bo": ["base_state", "outs_before"],
    "g3_li": ["_li_bucket"],
}


def make_g3(cols):
    def fn(df, holdout):
        df = df.copy()
        prior = df.loc[df["season"] < holdout, TARGET].mean()
        if "g3_li" in cols:
            df["_li_bucket"] = pd.qcut(df["li"].rank(method="first"), 5, labels=False).astype(np.int64)
        added = []
        for c in cols:
            enc, _cov = causal_smoothed_te_encode(df, df, G3_AXES[c], prior, k=TE_K_SMOOTH)
            df[c] = enc - prior
            added.append(c)
        if "_li_bucket" in df.columns:
            df = df.drop(columns=["_li_bucket"])
        return df, [], added   # exclude_from_mlp = added -> CatBoost 전용
    return fn


# ------- G4 -------
def _career_end_table(df, id_col, n_col, rate_col):
    idx = df.groupby([id_col, "season"])[n_col].idxmax()
    t = df.loc[idx, [id_col, "season", n_col, rate_col]].copy()
    t.columns = ["id", "s", "end_n", "end_rate"]
    t["end_succ"] = (t["end_n"] * t["end_rate"]).round()
    return t[["id", "s", "end_n", "end_succ", "end_rate"]]


def _lookup_offset(df, id_col, table, k):
    q = df[[id_col, "season"]].copy()
    q["s"] = q["season"] - k
    m = q.merge(table, left_on=[id_col, "s"], right_on=["id", "s"], how="left")
    return m["end_n"].values, m["end_succ"].values, m["end_rate"].values


def make_g4(add, career_fallback=False):
    def fn(df, holdout):
        df = df.copy()
        for role, id_col, n_col, rate_col in CAREER_SPECS:
            t = _career_end_table(df, id_col, n_col, rate_col)
            n1, sc1, _er1 = _lookup_offset(df, id_col, t, 1)
            n2, sc2, _er2 = _lookup_offset(df, id_col, t, 2)
            now_n = df[n_col].values.astype(np.float64)
            now_succ = np.round(now_n * df[rate_col].values)

            last_n = n1 - np.where(np.isnan(n2), 0.0, n2)
            last_sc = sc1 - np.where(np.isnan(sc2), 0.0, sc2)
            with np.errstate(invalid="ignore", divide="ignore"):
                va = np.where(last_n > 0, last_sc / last_n, np.nan)
            va = np.where(np.isnan(n1), np.nan, va)

            rec_n = now_n - n2
            rec_sc = now_succ - sc2
            with np.errstate(invalid="ignore", divide="ignore"):
                vb = np.where(rec_n > 0, rec_sc / rec_n, np.nan)
            vb = np.where(np.isnan(n2), np.nan, vb)

            if "VA" in add:
                if career_fallback:
                    va = np.where(np.isnan(va), df[rate_col].values, va)
                df[f"g4_{role}_lastseason_rate"] = va
            if "VB" in add:
                df[f"g4_{role}_recent2_rate"] = vb
        return df, [], []
    return fn


CAREER_ONE = ["asof_pitcher_success_rate", "asof_batter_success_rate"]


def make_g4_mlp_swap(mode="median"):
    """G4 SWAP 을 MLP 에만 적용: CatBoost 는 candidate B raw 그대로(① 유지, VA 없음),
    MLP 만 ①(asof_{p,b}_success_rate) 를 빼고 VA(lastseason_rate) 로 교체.
    Phase 1 에서 g4_4/5 가 CatBoost solo −18~−23 인데 MLP solo +10/+5 였던 걸 분리.

    직전 완결시즌 없는 선수(신인/공백) VA 결측 처리 (mode):
      "median" : NaN → MLP SimpleImputer(median) 가 train-median(≈리그평균 0.53)으로 채움.
                 신인이 리그평균 베테랑으로 위장됨 — 정보손실.
      "live"   : NaN → 그 행의 asof_{p,b}_success_rate(라이브 통산률)로 대체. ①을 뺀 이유가
                 ABS오염인데 결측행엔 그 값을 다시 넣는 셈 — graceful 하지만 자기모순적.
      "flag"   : NaN → median + 동반 이진컬럼 g4_{role}_lastseason_missing (1=직전시즌없음).
                 numeric 채널 오염 없이 "콜드스타트니 이 피처 할인해" 를 학습 가능
                 (sklearn add_indicator 패턴). 0 sentinel 은 StandardScaler 에서 ≈−3σ 로
                 뭉개져 "역대최악 제구" 로 오독되므로 쓰지 않음.
    """
    base = make_g4(["VA"], career_fallback=(mode == "live"))

    def fn(df, holdout):
        df2, _, _ = base(df, holdout)
        va_cols = ["g4_pitcher_lastseason_rate", "g4_batter_lastseason_rate"]
        add_cat_excl = list(va_cols)
        if mode == "flag":
            for role in ("pitcher", "batter"):
                col = f"g4_{role}_lastseason_rate"
                df2[f"g4_{role}_lastseason_missing"] = df2[col].isna().astype(np.int64).values
                add_cat_excl.append(f"g4_{role}_lastseason_missing")
        # exclude_from_cat = VA(+flag) (CatBoost 안 씀) / exclude_from_mlp = ① (MLP 안 씀)
        return df2, add_cat_excl, list(CAREER_ONE)
    return fn


# ------- config 레지스트리: name -> (add_fn, drop_spec) -------
# drop_spec: None | list[str] | "CAREER" | "SEASON" | "CAREER+... "
def build_registry():
    return {
        "g1_mxc": (make_g1(["g1_mxc"]), None),
        "g1_rxp": (make_g1(["g1_rxp"]), None),
        "g1_mxc_rxp": (make_g1(["g1_mxc", "g1_rxp"]), None),
        "g1_mxp_rxc": (make_g1(["g1_mxp", "g1_rxc"]), None),
        "g1_all": (make_g1(["g1_mxc", "g1_rxp", "g1_mxp", "g1_rxc"]), None),

        "g2_brel_bca": (make_g2(["g2_brel", "g2_bca"]), None),
        "g2_bmxc": (make_g2(["g2_bmxc"]), None),
        "g2_pbm": (make_g2(["g2_pbm"]), None),
        "g2_all": (make_g2(["g2_brel", "g2_bca", "g2_bmxc", "g2_pbm"]), None),

        "g3_bso": (make_g3(["g3_bso"]), None),
        "g3_bo": (make_g3(["g3_bo"]), None),
        "g3_li": (make_g3(["g3_li"]), None),
        "g3_all": (make_g3(["g3_bso", "g3_bo", "g3_li"]), None),

        "g4_1_VA": (make_g4(["VA"]), None),
        "g4_2_VB": (make_g4(["VB"]), None),
        "g4_3_VAVB": (make_g4(["VA", "VB"]), None),
        "g4_4_swap": (make_g4(["VA"]), "CAREER"),
        "g4_5_swap_fb": (make_g4(["VA"], career_fallback=True), "CAREER"),
        "g4_6_dropseason": (make_g4(["VA"]), "SEASON"),

        # follow-up: SWAP 을 MLP 에만 (CatBoost 는 raw 유지). 결측처리 3종.
        "g4swap_mlp": (make_g4_mlp_swap("median"), None),
        "g4swap_mlp_fb": (make_g4_mlp_swap("live"), None),
        "g4swap_mlp_flag": (make_g4_mlp_swap("flag"), None),
    }


def _make_drop_cols(drop_spec):
    if drop_spec is None:
        return None
    concrete = set()
    season = False
    if drop_spec == "CAREER":
        concrete = {"asof_pitcher_success_rate", "asof_batter_success_rate"}
    elif drop_spec == "SEASON":
        season = True
    elif isinstance(drop_spec, (list, tuple, set)):
        concrete = set(drop_spec)

    def dl(cols):
        out = [c for c in cols if c in concrete]
        if season:
            out += [c for c in cols if c.startswith(SEASON_PROG_PREFIXES)]
        return out
    return dl


def _run(regime, seeds, add_fn, drop_spec, label):
    parts = build_split(regime, add_features_fn=add_fn, drop_cols=_make_drop_cols(drop_spec), verbose=True)
    res = run_experiment(*parts, mlp_seeds=list(range(1, seeds + 1)),
                         cb_seeds=list(range(1, seeds + 1)), label=label)
    del parts
    gc.collect()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--regime", default="cutoff7")
    ap.add_argument("--only", default=None, help="g1|g2|g3|g4 접두어")
    ap.add_argument("--configs", default=None, help="쉼표구분 명시 config")
    ap.add_argument("--out", default=OUT_JSON, help="결과 JSON 경로")
    args = ap.parse_args()
    out_json = args.out

    reg = build_registry()
    names = list(reg)
    if args.only:
        names = [n for n in names if n.startswith(args.only)]
    if args.configs:
        want = set(args.configs.split(","))
        names = [n for n in names if n in want]

    result = {"regime": args.regime, "seeds": args.seeds, "baseline": None, "configs": {}}
    base = _run(args.regime, args.seeds, None, None, f"{args.regime}:baseline")
    result["baseline"] = dict(cat=base["cat_solo"], mlp=base["mlp_solo"], blend=base["blend"])
    with open(out_json, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    for nm in names:
        add_fn, drop_spec = reg[nm]
        try:
            res = _run(args.regime, args.seeds, add_fn, drop_spec, f"{args.regime}:{nm}")
            dd = dict(cat=res["cat_solo"] - base["cat_solo"],
                      mlp=res["mlp_solo"] - base["mlp_solo"],
                      blend=res["blend"] - base["blend"])
            result["configs"][nm] = dict(
                score=dict(cat=res["cat_solo"], mlp=res["mlp_solo"], blend=res["blend"]), delta=dd)
            print(f"  [Δ {nm} | {args.regime}]  cat {dd['cat']:+.2f} | mlp {dd['mlp']:+.2f} | "
                  f"blend {dd['blend']:+.2f}", flush=True)
        except Exception as e:
            result["configs"][nm] = {"error": repr(e)}
            print(f"  [ERR {nm}] {e!r}", flush=True)
        with open(out_json, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"=== Phase1 요약 ({args.regime}, {args.seeds}-seed, blend Δ 내림차순) ===", flush=True)
    rows = [(nm, d["delta"]["blend"], d["delta"]["cat"], d["delta"]["mlp"])
            for nm, d in result["configs"].items() if "delta" in d]
    for nm, b, c, m in sorted(rows, key=lambda x: -x[1]):
        flag = "  <- 통과후보" if b >= 2.0 else ""
        print(f"  {nm:18s} blend {b:+7.2f} | cat {c:+7.2f} | mlp {m:+7.2f}{flag}", flush=True)
    for nm, d in result["configs"].items():
        if "error" in d:
            print(f"  {nm:18s} ERROR: {d['error']}", flush=True)
    print(f"\n-> {out_json}", flush=True)


if __name__ == "__main__":
    main()
