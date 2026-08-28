# code/experiment_career_trajectory.py
"""새 피처 후보: "커리어 다년 궤적"(multi-year trajectory).

지금까지 asof_* 성공률을 분해하는 시도는 전부 "이번 시즌 vs 커리어 누적"(시즌진행분,
채택) 또는 "같은 시즌 내 다른 rate 컬럼"(TE-residual 축 확장, reverse/ball/strike/
pitchmix 시즌분해 등, 대부분 기각) 축이었다 — 전부 "이 시즌 하나"를 기준점으로 삼는다.
이번 후보는 축 자체가 다르다: **여러 완결된 과거 시즌들 사이의 변화(연도별 추세)**를
본다 — "이 선수가 최근 몇 년간 좋아지고 있는가/나빠지고 있는가"와 "몇 시즌째인가(경력
연차)"는 시즌진행분(이번 시즌 vs 직전 시즌 말 스냅샷 1개)도 TE-residual(상황축 교차)도
포착하지 못하는 정보다.

두 컬럼(투수/타자 각각):
  - `{role}_career_trend_1v2`: (시즌 S-1의 "그 시즌만의" 성공률) − (시즌 S-2의 "그
    시즌만의" 성공률). 시즌진행분과 동일한 "그 시즌만의" 분해를 모든 과거 시즌에 대해
    미리 계산해두고, 현재 행의 시즌 기준으로 두 시즌 전 값을 그대로 뺀다. 두 시즌 중
    하나라도 없으면(신인, 공백 시즌 등) NaN — CatBoost는 NaN 네이티브 처리, MLP는 중앙값
    대체.
  - `{role}_career_experience_seasons`: 현재 시즌 이전에 관측된 서로 다른 시즌 수(0=
    신인/첫 시즌). asof_pitcher_n(누적 투구 수)과 달리 "연차"를 직접 나타내는 컬럼은
    지금까지 없었다.

lookup은 시즌진행분과 동일하게 스플릿 이전 전체 df(train.csv 전량, F1 필터 미적용)에서
계산한다 — 완결된 과거 시즌 경계만 참조하므로 행별 미래 정보 누출이 아니다(경력 연차는
cumcount로, 미래 시즌을 보지 않는다).

1단계: CatBoost 단독 dual-regime(cutoff7 + season==2023) 스크리닝. 프로덕션 전체
피처셋(시즌진행분+TE-residual+coarse pitchmix, tier A 없음)에 이 4컬럼만 추가해 비교.

사용법:
  python -m code.experiment_career_trajectory --cutoff7
  python -m code.experiment_career_trajectory --holdout 2023
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, DATA_DIR, TARGET_COL

SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
]

TRAJ_COLS = [f"{role}_career_trend_1v2" for role, *_ in SPECS] + \
            [f"{role}_career_experience_seasons" for role, *_ in SPECS]


def build_season_only_table(raw, id_col, n_col, rate_col):
    """(id, season)별 "그 시즌만의" 성공률 + 그 시즌 이전에 관측된 시즌 수(경력 연차).
    apply_season_progression_features와 동일한 "+1 자기 결과 포함" 보정을 쓴다."""
    idx = raw.groupby([id_col, "season"])[n_col].idxmax()
    season_end = raw.loc[idx, [id_col, "season", n_col, rate_col, TARGET_COL]].copy()
    season_end.columns = [id_col, "season", "end_n", "end_rate", "end_success"]
    season_end["cum_n"] = season_end["end_n"] + 1
    season_end["cum_success"] = (season_end["end_n"] * season_end["end_rate"]).round() + season_end["end_success"]
    season_end = season_end.sort_values([id_col, "season"]).reset_index(drop=True)

    g = season_end.groupby(id_col)
    prev_cum_n = g["cum_n"].shift(1).fillna(0)
    prev_cum_success = g["cum_success"].shift(1).fillna(0)
    season_n = (season_end["cum_n"] - prev_cum_n).clip(lower=0)
    season_success = (season_end["cum_success"] - prev_cum_success).clip(lower=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        season_end["season_rate_"] = np.where(season_n > 0, season_success / season_n, np.nan)
    season_end["experience_seasons"] = g.cumcount()
    return season_end[[id_col, "season", "season_rate_", "experience_seasons"]]


def build_trajectory_tables(raw):
    return {role: build_season_only_table(raw, id_col, n_col, rate_col) for role, id_col, n_col, rate_col in SPECS}


def apply_trajectory_features(df, tables):
    df = df.copy()
    for role, id_col, n_col, rate_col in SPECS:
        t = tables[role]
        base = df[[id_col, "season"]].copy()
        base["season_m1"] = base["season"] - 1
        base["season_m2"] = base["season"] - 2

        m1 = base.merge(t[[id_col, "season", "season_rate_"]].rename(columns={"season": "season_m1", "season_rate_": "rate_lag1"}),
                         on=[id_col, "season_m1"], how="left")
        m2 = base.merge(t[[id_col, "season", "season_rate_"]].rename(columns={"season": "season_m2", "season_rate_": "rate_lag2"}),
                         on=[id_col, "season_m2"], how="left")
        exp = base.merge(t[[id_col, "season", "experience_seasons"]], on=[id_col, "season"], how="left")

        df[f"{role}_career_trend_1v2"] = m1["rate_lag1"].values - m2["rate_lag2"].values
        df[f"{role}_career_experience_seasons"] = exp["experience_seasons"].fillna(0).values
    return df


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout)

    raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                       usecols=["pitcher_id", "batter_id", "season", "asof_pitcher_n", "asof_pitcher_success_rate",
                                "asof_batter_n", "asof_batter_success_rate", TARGET_COL])
    raw = raw.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    tables = build_trajectory_tables(raw)

    results = {}
    for tag, use_new in [("baseline(현재 프로덕션)", False), ("+career_trajectory(4)", True)]:
        if use_new:
            ts = apply_trajectory_features(train_split, tables)
            vs = apply_trajectory_features(val_split, tables)
            cat_cols = cat_feature_cols + TRAJ_COLS
        else:
            ts, vs, cat_cols = train_split, val_split, cat_feature_cols

        X_train, y_train = ts[cat_cols], ts[TARGET_COL].values
        X_val, y_val = vs[cat_cols], vs[TARGET_COL].values

        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"[{tag}] Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_cols)})")
        results[tag] = score

    base, new = results["baseline(현재 프로덕션)"], results["+career_trajectory(4)"]
    print(f"\n--- {label} 요약 --- baseline={base:.2f} new={new:.2f} delta={new-base:+.2f}")
    return base, new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(args.cutoff7, holdout)


if __name__ == "__main__":
    main()
