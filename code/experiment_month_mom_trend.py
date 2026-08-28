# code/experiment_month_mom_trend.py
"""새 피처 후보: 과거 두 시즌 각각의 "월-to-월" 변화율(month-over-month), 그리고 그
변화율 자체의 연도 간(YoY) 차분.

month_yoy_trend(같은 달력월을 연도 간 직접 비교, `code/experiment_month_yoy_trend.py`)이
cutoff7 -5.42 / season2023 +16.66으로 부호가 갈려 기각된 뒤, 사용자가 설계를 바꿔
제안했다: "25년 실전에서는 행간 조작 불가하니 완결된 과거 구간끼리만" —

  - 재작년(S-2) 안에서: 이 행의 달(M)이 그 전 달(M-1)보다 얼마나 좋았는가
      s2_mom = month_rate_(S-2, M) - month_rate_(S-2, M-1)
  - 작년(S-1) 안에서: 이 행의 달(M)이 그 전 달(M-1)보다 얼마나 좋았는가
      s1_mom = month_rate_(S-1, M) - month_rate_(S-1, M-1)
  - 그 두 "월간 변화율"이 재작년 대비 작년에 얼마나 달라졌는가
      yoy_diff_of_mom = s1_mom - s2_mom

세 값 모두 강한/약한 정도(크기)와 방향(부호)을 하나의 연속값에 함께 담는다 — 이
프로젝트의 기존 관례(career_trend_1v2, season progression 등)와 동일하게 부호/크기를
별도 카테고리 컬럼으로 분리하지 않고 raw 차분값 하나로 CatBoost/MLP가 직접 학습하게
둔다. 세 텀 모두 S-1/S-2(둘 다 현재 시즌보다 앞선, 완결된 과거 시즌)의 월 데이터만
참조하므로 "진행 중인 이번 달" 문제가 없다 — month_yoy_trend와 달리 causal 재구성
자체는 이미 완결된 두 구간 뺄셈이라 추가 위험이 없다(월별 표본 자체가 작다는 문제는
동일하게 남아있음).

월별 성공률 테이블(month_rate_)은 `experiment_month_yoy_trend.py::build_month_only_table`
그대로 재사용한다.

1단계: CatBoost 단독 dual-regime(cutoff7 + season==2023) 스크리닝.

사용법:
  python -m code.experiment_month_mom_trend --cutoff7
  python -m code.experiment_month_mom_trend --holdout 2023
"""
import argparse
import os
import time

import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_career_trajectory import SPECS
from code.experiment_month_yoy_trend import build_month_tables
from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, DATA_DIR, TARGET_COL

MOM_COLS = []
for role, *_ in SPECS:
    MOM_COLS += [f"{role}_month_mom_s1", f"{role}_month_mom_s2", f"{role}_month_mom_yoy_diff"]


def apply_month_mom_trend(df, tables):
    df = df.copy()
    for role, id_col, n_col, rate_col in SPECS:
        t = tables[role]
        base = df[[id_col, "season", "game_month"]].copy()
        base["season_m1"] = base["season"] - 1
        base["season_m2"] = base["season"] - 2
        base["month_prev"] = base["game_month"] - 1

        def lookup(season_col, month_col, out_name):
            b = base[[id_col, season_col, month_col]].rename(columns={season_col: "season", month_col: "game_month"})
            m = b.merge(t, on=[id_col, "season", "game_month"], how="left")
            return m["month_rate_"].values

        s1_cur = lookup("season_m1", "game_month", "s1_cur")
        s1_prev = lookup("season_m1", "month_prev", "s1_prev")
        s2_cur = lookup("season_m2", "game_month", "s2_cur")
        s2_prev = lookup("season_m2", "month_prev", "s2_prev")

        s1_mom = s1_cur - s1_prev
        s2_mom = s2_cur - s2_prev

        df[f"{role}_month_mom_s1"] = s1_mom
        df[f"{role}_month_mom_s2"] = s2_mom
        df[f"{role}_month_mom_yoy_diff"] = s1_mom - s2_mom
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
    for tag, use_new in [("baseline(현재 프로덕션)", False), ("+month_mom_trend(6)", True)]:
        if use_new:
            ts = apply_month_mom_trend(train_split, tables)
            vs = apply_month_mom_trend(val_split, tables)
            cat_cols = cat_feature_cols + MOM_COLS
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

    base, new = results["baseline(현재 프로덕션)"], results["+month_mom_trend(6)"]
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
