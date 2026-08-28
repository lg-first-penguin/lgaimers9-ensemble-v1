# code/experiment_yudam_re_career.py
"""[재검증 #6] "커리어 다년 궤적" 피처를 raw-concat MLP 베이스라인 위에서 다시 측정.
축 자체가 다름 — 시즌진행분(이번 시즌 vs 직전 시즌 말)이나 TE-residual(상황축 교차)이
못 잡는 "여러 완결 시즌 사이의 연도별 추세 + 경력 연차".

4컬럼 (pitcher/batter 각 2):
  {role}_career_trend_1v2         = rate(S-1) − rate(S-2)   (각 시즌 "그 시즌만의" 성공률)
  {role}_career_experience_seasons = 현재 시즌 이전 관측 시즌 수 (0=신인)

배경: CatBoost 단독 dual-regime 스크리닝에서 cutoff7 +0.59 / season2023 +32.06 ("한쪽만
튀는" 노이즈 모양) → rolling-origin 3-fold 1/3승·평균 +1.37 로 기각
(memory: career_trajectory_rejected_rolling_origin). 원본은 CatBoost 전용이었음 — 여기서는
raw-concat MLP 도입 후 라우팅 both 로 MLP/blend 델타까지 본다.

lookup 은 원본과 동일하게 스플릿 이전 전체 df(F1 필터 미적용)에서 계산 — 완결된 과거
시즌 경계만 참조하므로 미래 누출 없음.

helper 는 code/experiment_career_trajectory.py 에서 복제(원본 import 시 구식
thirdmodel_common 체인이 끌려옴).

사용법:
  python -m code.experiment_yudam_re_career --regime cutoff7
  python -m code.experiment_yudam_re_career --regime cutoff7 --accel   # trend 대신 2차(가속도)
"""
import argparse

import numpy as np

from code.experiment_yudam_common import compare
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS

TARGET = "control_success"
SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
]


def build_season_only_table(raw, id_col, n_col, rate_col):
    idx = raw.groupby([id_col, "season"])[n_col].idxmax()
    se = raw.loc[idx, [id_col, "season", n_col, rate_col, TARGET]].copy()
    se.columns = [id_col, "season", "end_n", "end_rate", "end_success"]
    se["cum_n"] = se["end_n"] + 1
    se["cum_success"] = (se["end_n"] * se["end_rate"]).round() + se["end_success"]
    se = se.sort_values([id_col, "season"]).reset_index(drop=True)
    g = se.groupby(id_col)
    prev_n = g["cum_n"].shift(1).fillna(0)
    prev_s = g["cum_success"].shift(1).fillna(0)
    s_n = (se["cum_n"] - prev_n).clip(lower=0)
    s_s = (se["cum_success"] - prev_s).clip(lower=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        se["season_rate_"] = np.where(s_n > 0, s_s / s_n, np.nan)
    se["experience_seasons"] = g.cumcount()
    return se[[id_col, "season", "season_rate_", "experience_seasons"]]


def build_trajectory_tables(raw):
    return {role: build_season_only_table(raw, idc, nc, rc) for role, idc, nc, rc in SPECS}


def apply_trajectory_features(df, tables):
    df = df.copy()
    new_cols = []
    for role, id_col, n_col, rate_col in SPECS:
        t = tables[role]
        base = df[[id_col, "season"]].copy()
        base["season_m1"] = base["season"] - 1
        base["season_m2"] = base["season"] - 2
        m1 = base.merge(t[[id_col, "season", "season_rate_"]].rename(columns={"season": "season_m1", "season_rate_": "r1"}),
                        on=[id_col, "season_m1"], how="left")
        m2 = base.merge(t[[id_col, "season", "season_rate_"]].rename(columns={"season": "season_m2", "season_rate_": "r2"}),
                        on=[id_col, "season_m2"], how="left")
        exp = base.merge(t[[id_col, "season", "experience_seasons"]], on=[id_col, "season"], how="left")
        df[f"{role}_career_trend_1v2"] = m1["r1"].values - m2["r2"].values
        df[f"{role}_career_experience_seasons"] = exp["experience_seasons"].fillna(0).values
        new_cols += [f"{role}_career_trend_1v2", f"{role}_career_experience_seasons"]
    return df, new_cols


def apply_acceleration_features(df, tables):
    df = df.copy()
    new_cols = []
    for role, id_col, n_col, rate_col in SPECS:
        t = tables[role][[id_col, "season", "season_rate_"]]
        base = df[[id_col, "season"]].copy()
        for k in (1, 2, 3):
            base[f"season_m{k}"] = base["season"] - k
        mm = {}
        for k in (1, 2, 3):
            mm[k] = base.merge(t.rename(columns={"season": f"season_m{k}", "season_rate_": f"r{k}"}),
                               on=[id_col, f"season_m{k}"], how="left")[f"r{k}"].values
        df[f"{role}_career_accel"] = mm[1] - 2 * mm[2] + mm[3]
        new_cols.append(f"{role}_career_accel")
    return df, new_cols


def make_adder(accel, cat_only=False):
    fn = apply_acceleration_features if accel else apply_trajectory_features

    def add_career(df_full, holdout):
        tables = build_trajectory_tables(df_full)
        df_out, new_cols = fn(df_full, tables)
        route = "CatBoost 전용" if cat_only else "CatBoost+MLP 양쪽"
        print(f"[career{'-accel' if accel else '-trend'}] {len(new_cols)}개 컬럼 추가 -> {route} ({new_cols})", flush=True)
        exclude_from_mlp = set(new_cols) if cat_only else set()
        return df_out, set(), exclude_from_mlp
    return add_career


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="cutoff7", choices=["cutoff7", "2023"])
    ap.add_argument("--accel", action="store_true")
    ap.add_argument("--cat-only", dest="cat_only", action="store_true", help="career 컬럼을 CatBoost에만 (MLP 제외)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    n = 1 if args.quick else args.seeds
    route = "cat" if args.cat_only else "both"
    compare(regime=args.regime, add_features_fn=make_adder(args.accel, args.cat_only),
            mlp_seeds=YUDAM_ENSEMBLE_SEEDS[:n], cb_seeds=YUDAM_CATBOOST_SEEDS[:n],
            label=f"career{'accel' if args.accel else 'trend'}->{route}")


if __name__ == "__main__":
    main()
