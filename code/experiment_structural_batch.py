# code/experiment_structural_batch.py
"""구조적 재검증 배치 — cat_team(실전 1126.77) 프로덕션 베이스라인 위에서.

configs:
  d1_career_catdrop : asof_{pitcher,batter}_success_rate 를 CatBoost 피처에서만 제거
      (MLP 는 유지). 근거: career-누적 성공률은 2025 베테랑의 경우 pre-ABS(2019-2023)
      투구가 지배 -> stale·레짐오염. season-progression *_season_rate 가 라이브 신호로
      이미 있음. G4-SWAP 은 MLP전용+cutoff7만이라 이 축(CatBoost전용+rolling)은 미시도.
  d2_coldstart_flags : is_{pitcher,batter}_lowinfo = (asof_*_n < 200) 이진플래그 2개를
      양쪽 모델에 추가. 2025 신인/저표본 선수를 "리그평균"으로 뭉개지 않게 별도 레짐 표시.
  b_monotone : 방향이 명확한 rate 피처 14개에 CatBoost monotone_constraints
      (+1: 성공률류, -1: middle/reverse류). HP 프리서치가 아니라 도메인 사전지식 =
      용량을 줄이는 구조적 정규화(과적합 위험 구조적으로 낮음).

레짐: cutoff7 / 2023 / 2022 / 2021 (expand-window rolling-origin). 기본 3-seed.
레짐별로 baseline 1회 + 선택된 config 들. --out JSON 에 레짐 단위로 머지(부분진행 누적).

사용법:
  python -m code.experiment_structural_batch --seeds 3 --regimes cutoff7 --configs baseline,d1_career_catdrop
  python -m code.experiment_structural_batch --seeds 3 --regimes 2023 --configs baseline,d1_career_catdrop,d2_coldstart_flags,b_monotone
"""
import argparse
import gc
import json
import os

from code.experiment_yudam_common import build_split, run_experiment

OUT_JSON = os.path.join(os.path.dirname(__file__), os.pardir,
                        "scratchpad", "structural_batch_result.json")

CATTEAM_CAT = ["game_type", "base_state", "pitcher_team_id", "batter_team_id"]

# ---- D1 ----
D1_CAT_DROP = ["asof_pitcher_success_rate", "asof_batter_success_rate"]


def d1_add_fn(df, holdout):
    # 새 컬럼 없음. CatBoost 피처목록에서만 제외(exclude_from_cat), MLP(exclude_from_mlp) 는 비움.
    return df, list(D1_CAT_DROP), []


# ---- D2 ----
D2_THRESH = 200.0


def d2_add_fn(df, holdout):
    df = df.copy()
    pn = df["asof_pitcher_n"].fillna(0).to_numpy()
    bn = df["asof_batter_n"].fillna(0).to_numpy()
    df["is_pitcher_lowinfo"] = (pn < D2_THRESH).astype("float64")
    df["is_batter_lowinfo"] = (bn < D2_THRESH).astype("float64")
    return df, [], []


# ---- season 제거 (2025 추론때 season=2025 는 학습(2019-2024)에 없는 값 ->
#      MLP 표준화가 학습범위 밖 z-score 로 밀어서 모든 2025 행에 계통편향.
#      CatBoost 는 >2024 leaf 로 떨어져 무해. season 임베딩은 과거 기각됐지만
#      num_cols 에서 제거는 미시도.) ----
def mlp_drop_season_fn(df, holdout):
    return df, [], ["season"]          # MLP 에서만 제외, CatBoost 는 유지


def cat_drop_season_fn(df, holdout):
    return df, ["season"], ["season"]  # 양쪽 다 제외 (완전 제거 대조군)


# ---- B ----
B_MONO_POS = [
    "asof_pitcher_success_rate", "asof_batter_success_rate",
    "pitcher_season_success_rate", "batter_season_success_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]
B_MONO_NEG = [
    "asof_pitcher_middle_rate", "asof_batter_middle_rate", "asof_pitcher_reverse_rate",
    "pitcher_reverse_season_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]


def b_mono_constraints(cat_feature_cols):
    cols = set(cat_feature_cols)
    d = {}
    for c in B_MONO_POS:
        if c in cols:
            d[c] = 1
    for c in B_MONO_NEG:
        if c in cols:
            d[c] = -1
    missing = [c for c in (B_MONO_POS + B_MONO_NEG) if c not in cols]
    return d, missing


CONFIGS = {
    "baseline": dict(add_fn=None, mono=False),
    "d1_career_catdrop": dict(add_fn=d1_add_fn, mono=False),
    "d2_coldstart_flags": dict(add_fn=d2_add_fn, mono=False),
    "b_monotone": dict(add_fn=None, mono=True),
    "mlp_drop_season": dict(add_fn=mlp_drop_season_fn, mono=False),
    "cat_drop_season": dict(add_fn=cat_drop_season_fn, mono=False),
}


def _run_one(regime, seeds, cfg_name):
    cfg = CONFIGS[cfg_name]
    parts = build_split(regime, add_features_fn=cfg["add_fn"], verbose=True)
    _, _, _, cat_feature_cols, _ = parts
    extra = None
    if cfg["mono"]:
        mono, missing = b_mono_constraints(cat_feature_cols)
        extra = {"monotone_constraints": mono}
        print(f"[b_monotone] {len(mono)} 제약 적용: {mono}", flush=True)
        if missing:
            print(f"[b_monotone] ⚠️ 목록에 없어 건너뛴 컬럼: {missing}", flush=True)
        if not mono:
            print("[b_monotone] ⚠️⚠️ 제약 0개 — baseline 과 동일하게 돌아감", flush=True)
            extra = None
    res = run_experiment(
        *parts,
        mlp_seeds=list(range(1, seeds + 1)), cb_seeds=list(range(1, seeds + 1)),
        label=f"{regime}:{cfg_name}",
        cb_cat_features=CATTEAM_CAT, cb_params_extra=extra,
    )
    del parts
    gc.collect()
    return res


def _load(out_json):
    if os.path.exists(out_json):
        with open(out_json) as f:
            return json.load(f)
    return {"seeds": None, "catteam_baseline": True, "regimes": {}}


def _save(out_json, d):
    with open(out_json, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--regimes", default="cutoff7,2023,2022,2021")
    ap.add_argument("--configs", default="baseline,d1_career_catdrop,d2_coldstart_flags,b_monotone")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    out_json = os.path.abspath(args.out)

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    cfg_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in cfg_names:
        if c not in CONFIGS:
            ap.error(f"unknown config {c!r}; 가능: {list(CONFIGS)}")
    if "baseline" not in cfg_names:
        cfg_names = ["baseline"] + cfg_names

    d = _load(out_json)
    d["seeds"] = args.seeds

    for regime in regimes:
        print(f"\n{'#' * 30} regime={regime} {'#' * 30}", flush=True)
        d["regimes"].setdefault(regime, {})
        for cfg_name in cfg_names:
            try:
                res = _run_one(regime, args.seeds, cfg_name)
                d["regimes"][regime][cfg_name] = res
            except Exception as e:
                d["regimes"][regime][cfg_name] = {"error": repr(e)}
                print(f"  [ERR {regime}:{cfg_name}] {e!r}", flush=True)
            _save(out_json, d)

        base = d["regimes"][regime].get("baseline", {})
        b_bl, b_c, b_m = base.get("blend"), base.get("cat_solo"), base.get("mlp_solo")
        print(f"\n----- [{regime}] 요약 (baseline blend={b_bl}) -----", flush=True)
        for cfg_name in cfg_names:
            if cfg_name == "baseline":
                continue
            r = d["regimes"][regime].get(cfg_name, {})
            if "error" in r:
                print(f"  {cfg_name:22s} ERROR {r['error']}", flush=True)
                continue
            if None in (b_bl, b_c, b_m):
                continue
            print(f"  {cfg_name:22s} blend {r['blend'] - b_bl:+7.2f} | "
                  f"cat {r['cat_solo'] - b_c:+7.2f} | mlp {r['mlp_solo'] - b_m:+7.2f}", flush=True)

    print(f"\n-> {out_json}", flush=True)


if __name__ == "__main__":
    main()
