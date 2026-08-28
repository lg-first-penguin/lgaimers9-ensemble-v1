# code/experiment_month_yoy_trend.py
"""새 피처 후보: "월별로 촘촘한" 1차 추세 — 같은 달력월의 연도 간(YoY) 비교.

career_trend_1v2(시즌 단위 1차 추세)와 career_accel(시즌 단위 2차 추세)이 모두 기각된
뒤([[career-trajectory-rejected-rolling-origin]], [[career-acceleration-rejected-sign-flip]]),
사용자가 "더 촘촘한 quantile(월별)로 추세를 보자"는 후속 아이디어를 제안했다. 원안은
"작년 X월 vs 올해 X월", "재작년 X월 vs 올해 X월", 연도 방향 선형추세, 작년 내 월-to-월
bin 추세, 재작년 내 월-to-월 bin 추세, 그 둘의 차분까지 포함하는 복합 스펙이었다.

이 스크립트는 그 중 **causally 가장 안전하고 구현이 가장 싼 조각 하나만** 먼저 검증한다:
"올해"(진행 중인 달) 텀은 asof 누적치를 월 경계에서 다시 빼야 하고, 그러면 월초 행은
분모(그 달 누적 투구 수)가 한 자릿수로 떨어져 시즌 단위보다 구조적으로 더 노이즈일
위험이 크다. 그래서 "진행 중인 달"은 빼고, **완결된 두 과거 구간만** 비교한다:

  month_yoy_trend = month_rate_(S-1, 이 행의 달) - month_rate_(S-2, 이 행의 달)

즉 "이 선수가 이 달력월(예: 4월)에 작년 대비 재작년에 얼마나 좋았는가"를 재작년-작년
사이의 변화로 본다. career_trend_1v2와 같은 뺄셈 구조지만 시즌 전체가 아니라 같은
달력월끼리만 비교한다는 점이 다르다.

시즌 경계 대신 (id, season, game_month) 경계로 누적치를 다시 미분해 "그 달만의" 성공률
테이블(month_rate_)을 만든다 — season-progression/career_trajectory와 동일한 "+1
자기결과 포함" 보정 트릭. 완결된 과거 두 구간만 참조하므로 미래 정보 누출 없음.

1단계: CatBoost 단독 dual-regime(cutoff7 + season==2023) 스크리닝.

사용법:
  python -m code.experiment_month_yoy_trend --cutoff7
  python -m code.experiment_month_yoy_trend --holdout 2023
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_career_trajectory import SPECS
from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, DATA_DIR, TARGET_COL

MONTH_YOY_COLS = [f"{role}_month_yoy_trend" for role, *_ in SPECS]


def build_month_only_table(raw, id_col, n_col, rate_col):
    """(id, season, game_month)별 "그 달만의" 성공률. build_season_only_table과 동일한
    +1 자기결과 보정을 월 경계에 적용."""
    idx = raw.groupby([id_col, "season", "game_month"])[n_col].idxmax()
    month_end = raw.loc[idx, [id_col, "season", "game_month", n_col, rate_col, TARGET_COL]].copy()
    month_end.columns = [id_col, "season", "game_month", "end_n", "end_rate", "end_success"]
    month_end["cum_n"] = month_end["end_n"] + 1
    month_end["cum_success"] = (month_end["end_n"] * month_end["end_rate"]).round() + month_end["end_success"]
    month_end = month_end.sort_values([id_col, "season", "game_month"]).reset_index(drop=True)

    g = month_end.groupby(id_col)
    prev_cum_n = g["cum_n"].shift(1).fillna(0)
    prev_cum_success = g["cum_success"].shift(1).fillna(0)
    month_n = (month_end["cum_n"] - prev_cum_n).clip(lower=0)
    month_success = (month_end["cum_success"] - prev_cum_success).clip(lower=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        month_end["month_rate_"] = np.where(month_n > 0, month_success / month_n, np.nan)
    return month_end[[id_col, "season", "game_month", "month_rate_"]]


def build_month_tables(raw):
    return {role: build_month_only_table(raw, id_col, n_col, rate_col) for role, id_col, n_col, rate_col in SPECS}


def apply_month_yoy_trend(df, tables):
    df = df.copy()
    for role, id_col, n_col, rate_col in SPECS:
        t = tables[role]
        base = df[[id_col, "season", "game_month"]].copy()
        base["season_m1"] = base["season"] - 1
        base["season_m2"] = base["season"] - 2

        m1 = base.merge(
            t.rename(columns={"season": "season_m1", "month_rate_": "rate_m1"}),
            on=[id_col, "season_m1", "game_month"], how="left")
        m2 = base.merge(
            t.rename(columns={"season": "season_m2", "month_rate_": "rate_m2"}),
            on=[id_col, "season_m2", "game_month"], how="left")

        df[f"{role}_month_yoy_trend"] = m1["rate_m1"].values - m2["rate_m2"].values
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
                       usecols=["pitcher_id", "batter_id", "season", "game_month",
                                "asof_pitcher_n", "asof_pitcher_success_rate",
                                "asof_batter_n", "asof_batter_success_rate", TARGET_COL])
    raw = raw.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    tables = build_month_tables(raw)

    results = {}
    for tag, use_new in [("baseline(현재 프로덕션)", False), ("+month_yoy_trend(2)", True)]:
        if use_new:
            ts = apply_month_yoy_trend(train_split, tables)
            vs = apply_month_yoy_trend(val_split, tables)
            cat_cols = cat_feature_cols + MONTH_YOY_COLS
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

    base, new = results["baseline(현재 프로덕션)"], results["+month_yoy_trend(2)"]
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
