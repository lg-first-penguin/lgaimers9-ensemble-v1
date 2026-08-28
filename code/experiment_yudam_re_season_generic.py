# code/experiment_yudam_re_season_generic.py
"""[재검증 #5] 시즌진행분 메커니즘(전 시즌 끝 누적치를 빼서 "이번 시즌만의" 값 + 커리어
대비 격차)을 asof_pitcher_success_rate / asof_pitcher_reverse_rate (둘 다 이미 프로덕션)
외의 7개 asof_* 비율 컬럼에 일반화한 피처를, raw-concat MLP 베이스라인 위에서 다시 측정.

7종 (각 4컬럼: _season_n/_season_count/_season_rate/_season_rate_gap):
  pitcher_middle / pitcher_ball / pitcher_strike / batter_middle /
  pitcher_fastball / pitcher_breaking / pitcher_offspeed

배경: PLE 시절 cutoff7 에서 7종 전부 노이즈 또는 마이너스로 승격 후보가 없었다
(memory: asof_season_progression_generalization_mixed). 원조 시즌진행분(success_rate)과
달리 원본 이벤트 플래그가 없어 pre_count=round(end_n*end_rate) 근사만 쓴다.
라우팅: CatBoost + MLP 양쪽(원본 실험과 동일).

helper 로직(GENERIC_SPECS / build_generic_season_end_lookup / apply_generic_season_progression)
은 code/experiment_asof_season_progression_all.py 에서 그대로 복제 — 원본을 import 하면
구식 code/thirdmodel_common(TRACKMAN_TIER_FEED 참조, 새 train.py 와 비호환)까지 끌려온다.

lookup 은 "F1 필터 적용된 학습 파티션"에서만 만든다(val 시즌 제외 + control_success
미사용이라 누출 없음).

사용법:
  python -m code.experiment_yudam_re_season_generic --regime cutoff7                 # 7종 합쳐서
  python -m code.experiment_yudam_re_season_generic --regime cutoff7 --only pitcher_strike
  python -m code.experiment_yudam_re_season_generic --regime cutoff7 --each          # 7종 개별 순회
"""
import argparse

import numpy as np
import pandas as pd

from code.experiment_yudam_common import compare, REGIMES
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS, apply_f1_filter

# --- experiment_asof_season_progression_all.py 에서 복제 (import 체인 회피) ---
GENERIC_SPECS = [
    ("pitcher_middle", "pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate"),
    ("pitcher_ball", "pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate"),
    ("pitcher_strike", "pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate"),
    ("batter_middle", "batter_id", "asof_batter_n", "asof_batter_middle_rate"),
    ("pitcher_fastball", "pitcher_id", "asof_pitcher_n", "asof_pitcher_fastball_rate"),
    ("pitcher_breaking", "pitcher_id", "asof_pitcher_n", "asof_pitcher_breaking_rate"),
    ("pitcher_offspeed", "pitcher_id", "asof_pitcher_n", "asof_pitcher_offspeed_rate"),
]


def build_generic_season_end_lookup(df, specs):
    tables = []
    for name, id_col, n_col, rate_col in specs:
        idx = df.groupby([id_col, "season"])[n_col].idxmax()
        season_end = df.loc[idx, [id_col, "season", n_col, rate_col]].copy()
        season_end.columns = ["id", "season", "end_n", "end_rate"]
        season_end["season"] = season_end["season"] + 1
        season_end.insert(0, "name", name)
        tables.append(season_end)
    return pd.concat(tables, ignore_index=True)


def apply_generic_season_progression(df, lookup, specs):
    df = df.copy()
    new_cols = []
    for name, id_col, n_col, rate_col in specs:
        lut = lookup.loc[lookup["name"] == name, ["id", "season", "end_n", "end_rate"]]
        merged = df[[id_col, "season"]].merge(
            lut, left_on=[id_col, "season"], right_on=["id", "season"], how="left",
        )
        pre_n = merged["end_n"].fillna(0).values
        pre_count = (merged["end_n"] * merged["end_rate"]).round().fillna(0).values

        cum_n = df[n_col].values
        cum_count = np.round(df[n_col].values * df[rate_col].values)
        season_n = np.maximum(cum_n - pre_n, 0)
        season_count = np.maximum(cum_count - pre_count, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            season_rate = np.where(season_n > 0, season_count / season_n, np.nan)

        c_n, c_cnt, c_rate, c_gap = (f"{name}_season_n", f"{name}_season_count",
                                      f"{name}_season_rate", f"{name}_season_rate_gap")
        df[c_n] = season_n
        df[c_cnt] = season_count
        df[c_rate] = season_rate
        df[c_gap] = season_rate - df[rate_col].values
        new_cols += [c_n, c_cnt, c_rate, c_gap]
    return df, new_cols
# --- 복제 끝 ---


def make_adder(specs):
    def add_generic(df_full, holdout):
        regime = "cutoff7" if holdout == 2024 else "2023"
        train_mask_fn, _ = REGIMES[regime]
        src = df_full[train_mask_fn(df_full)]
        src = apply_f1_filter(src)                       # 원본과 동일: F1 필터된 학습 파티션에서 lookup
        lookup = build_generic_season_end_lookup(src, specs)
        df_out, new_cols = apply_generic_season_progression(df_full, lookup, specs)
        print(f"[season-generic] {len(new_cols)}개 컬럼 추가 -> CatBoost+MLP 양쪽 "
              f"({', '.join(s[0] for s in specs)})", flush=True)
        return df_out, set(), set()
    return add_generic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="cutoff7", choices=["cutoff7", "2023"])
    ap.add_argument("--only", type=str, default=None, help="콤마구분 spec name")
    ap.add_argument("--each", action="store_true", help="7종 개별 순회")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    n = 1 if args.quick else args.seeds
    ms, cs = YUDAM_ENSEMBLE_SEEDS[:n], YUDAM_CATBOOST_SEEDS[:n]

    if args.each:
        for spec in GENERIC_SPECS:
            compare(regime=args.regime, add_features_fn=make_adder([spec]),
                    mlp_seeds=ms, cb_seeds=cs, label=f"seasongen:{spec[0]}")
    else:
        specs = GENERIC_SPECS if not args.only else [s for s in GENERIC_SPECS if s[0] in set(args.only.split(","))]
        compare(regime=args.regime, add_features_fn=make_adder(specs),
                mlp_seeds=ms, cb_seeds=cs,
                label="seasongen:" + ("ALL7" if not args.only else args.only))


if __name__ == "__main__":
    main()
