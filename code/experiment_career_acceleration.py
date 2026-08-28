# code/experiment_career_acceleration.py
"""새 피처 후보: "커리어 추세의 추세"(2차 변화량, 가속도).

career_trajectory(`code/experiment_career_trajectory.py`)의 1차 추세
`{role}_career_trend_1v2` = rate(S-1) - rate(S-2) 는 dual-regime에서 cutoff7 +0.59
(거의 평평) / season==2023 +32.06(큰 플러스)라는, 이 프로젝트에서 반복적으로 노이즈로
판명난 "한쪽만 튀는" 모양을 보였고 rolling-origin 3-fold에서 1/3·평균+1.37로 기각됐다
([[career-trajectory-rejected-rolling-origin]]).

이번 후보는 그 1차 추세를 한 단계 더 미분한다: "최근 추세(S-1 대비 S-2) 자체가
그 전 추세(S-2 대비 S-3)보다 더 좋아지고 있는가/나빠지고 있는가" — 가속도.

  accel = trend(S-1,S-2) - trend(S-2,S-3)
        = (rate(S-1) - rate(S-2)) - (rate(S-2) - rate(S-3))
        = rate(S-1) - 2*rate(S-2) + rate(S-3)

3개 과거 시즌이 모두 있어야 계산되므로(1차 추세보다 결측이 더 많음 - 신인/2년차/공백
시즌은 전부 NaN) 표본이 1차 추세보다도 훨씬 희소하다. 1차 추세조차 노이즈로 판명났으니
이 2차 추세는 사전 확률이 더 낮다고 봐야 하지만, 사용자 요청에 따라 동일한 검증 단계
(dual-regime -> 필요시 rolling-origin)를 그대로 거쳐 확인한다.

lookup은 career_trajectory와 동일하게 스플릿 이전 전체 df(F1 필터 미적용)에서 계산 —
완결된 과거 시즌 경계만 참조하므로 미래 정보 누출 없음.

1단계: CatBoost 단독 dual-regime(cutoff7 + season==2023) 스크리닝. 프로덕션 전체
피처셋(시즌진행분+TE-residual+coarse pitchmix, tier A/career_trajectory 없음)에
이 2컬럼(pitcher/batter 각 1개)만 추가해 비교.

사용법:
  python -m code.experiment_career_acceleration --cutoff7
  python -m code.experiment_career_acceleration --holdout 2023
"""
import argparse
import os
import time

import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_career_trajectory import SPECS, build_trajectory_tables
from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, DATA_DIR, TARGET_COL

ACCEL_COLS = [f"{role}_career_accel" for role, *_ in SPECS]


def apply_acceleration_features(df, tables):
    df = df.copy()
    for role, id_col, n_col, rate_col in SPECS:
        t = tables[role][[id_col, "season", "season_rate_"]]
        base = df[[id_col, "season"]].copy()
        base["season_m1"] = base["season"] - 1
        base["season_m2"] = base["season"] - 2
        base["season_m3"] = base["season"] - 3

        m1 = base.merge(t.rename(columns={"season": "season_m1", "season_rate_": "rate_lag1"}),
                         on=[id_col, "season_m1"], how="left")
        m2 = base.merge(t.rename(columns={"season": "season_m2", "season_rate_": "rate_lag2"}),
                         on=[id_col, "season_m2"], how="left")
        m3 = base.merge(t.rename(columns={"season": "season_m3", "season_rate_": "rate_lag3"}),
                         on=[id_col, "season_m3"], how="left")

        df[f"{role}_career_accel"] = (
            m1["rate_lag1"].values - 2 * m2["rate_lag2"].values + m3["rate_lag3"].values
        )
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
    for tag, use_new in [("baseline(현재 프로덕션)", False), ("+career_accel(2)", True)]:
        if use_new:
            ts = apply_acceleration_features(train_split, tables)
            vs = apply_acceleration_features(val_split, tables)
            cat_cols = cat_feature_cols + ACCEL_COLS
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

    base, new = results["baseline(현재 프로덕션)"], results["+career_accel(2)"]
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
