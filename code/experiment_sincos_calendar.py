# code/experiment_sincos_calendar.py
"""새 피처 후보: 달력 순환(cyclical) 인코딩 — game_month/game_dayofweek를 sin/cos로.

[[month-level-trend-rejected-finer-granularity]]에서 사용자가 "이런 복합적인 추세
변화에 주기성이 있지 않을까, sin/cos로 표현하는 기법은?"이라고 물었던 것의 실제 실험.
월/추세 파생 피처가 아니라 가장 기본적인 형태 — raw 달력 컬럼(game_month,
game_dayofweek) 자체를 순환 인코딩하는 표준 기법(`sin(2*pi*x/period)`,
`cos(2*pi*x/period)`)을 테스트한다. asof_* 분해가 아니라 원본 컬럼의 표현 방식만
바꾸는 것이므로 다른 행 참조 없이 완전히 row-local, 리크 위험이 전혀 없다.

주의: 이 프로젝트는 이미 관련된 걸 한 번 시도한 적이 있다 — MLP 수치형 임베딩 방식
자체를 quantile(PLE) 대신 sin/cos "periodic" 임베딩으로 바꾸는 프로젝트 전역 실험
(`code/periodic_mlp_model.py`)이 7-seed 재검증에서 노이즈로 판명나 기각됐다
(CLAUDE.md MLP 섹션). 그건 "모든 수치형 피처의 인코딩 방식"을 바꾸는 것이었고, 이번은
`game_month`/`game_dayofweek` 딱 2개 컬럼에 대해서만 파생 컬럼 4개를 **추가**하는
좁은 실험이라 성격이 다르다 — 기존 raw game_month/game_dayofweek 컬럼은 그대로 두고
sin/cos 컬럼을 얹기만 한다(둘 다 CatBoost cat_feature_cols/MLP mlp_num_cols에 이미
포함돼 있음).

  month_sin = sin(2*pi*game_month/12), month_cos = cos(2*pi*game_month/12)
  dow_sin   = sin(2*pi*game_dayofweek/7), dow_cos = cos(2*pi*game_dayofweek/7)

KBO 시즌은 대략 3~10월만 진행되므로(12월/1월 경기 없음) month의 "연말-연초 인접성"은
실질적으로 거의 안 걸리는 케이스라 효과가 작을 수 있음 — dayofweek(월요일-일요일
순환, 매 경기마다 발생)이 더 유의미할 가능성.

1단계: CatBoost 단독 dual-regime(cutoff7 + season==2023) 스크리닝.

사용법:
  python -m code.experiment_sincos_calendar --cutoff7
  python -m code.experiment_sincos_calendar --holdout 2023
"""
import argparse
import time

import numpy as np
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, TARGET_COL

SINCOS_COLS = ["month_sin", "month_cos", "dow_sin", "dow_cos"]


def apply_sincos_features(df):
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["game_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["game_month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["game_dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["game_dayofweek"] / 7)
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

    results = {}
    for tag, use_new in [("baseline(현재 프로덕션)", False), ("+sincos_calendar(4)", True)]:
        if use_new:
            ts = apply_sincos_features(train_split)
            vs = apply_sincos_features(val_split)
            cat_cols = cat_feature_cols + SINCOS_COLS
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

    base, new = results["baseline(현재 프로덕션)"], results["+sincos_calendar(4)"]
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
