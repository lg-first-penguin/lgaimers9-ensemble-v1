# code/experiment_season_progression_middle.py
"""시즌진행분(성공률 분해, code/train.py::apply_season_progression_features) 방식을
asof_{pitcher,batter}_middle_rate(가운데/위험 코스 비율)에도 확장 적용해본다.

성공률 분해와의 차이: control_success는 행 단위 라벨이 train.csv에 있어서 시즌
마지막 행 자신의 결과를 "+1"로 정확히 더할 수 있었지만, middle_rate는 그 행이
가운데였는지 여부의 행 단위 라벨이 없다(팀원 제보 실험기록도 동일하게 지적:
"실패 유형별 지표의 타자 버전은 만들 수 없다 — 투구별 실패 유형이 데이터에 없음").
asof_pitcher_n * asof_pitcher_middle_rate가 거의 정수(반올림 오차만, 실측
max frac error 0.0014)이므로 분모는 성공률과 동일함을 확인했다 — 대신 "+1
보정"만 생략하고(시즌 마지막 행 자신의 몫만큼 최대 1구 오차), 시즌 표본수
분모는 이미 계산된 pitcher_season_n/batter_season_n을 그대로 재사용한다.

사용법:
  python -m code.experiment_season_progression_middle --cutoff7
  python -m code.experiment_season_progression_middle --holdout 2023
"""
import argparse
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_season_progression import SEASON_PROGRESSION_COLS, TARGET_COL, build_split
from code.mlp_model import compute_bss
from code.train import apply_f1_filter

# (role, id_col, n_col, rate_col, tag)
EXTRA_RATE_SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate", "middle"),
    ("batter", "batter_id", "asof_batter_n", "asof_batter_middle_rate", "middle"),
]


def build_extra_rate_lookup(df, specs):
    tables = []
    for role, id_col, n_col, rate_col, tag in specs:
        idx = df.groupby([id_col, "season"])[n_col].idxmax()
        season_end = df.loc[idx, [id_col, "season", n_col, rate_col]].copy()
        season_end.columns = ["id", "season", "end_n", "end_rate"]
        season_end["season"] = season_end["season"] + 1  # 다음 시즌에 적용되는 키로 이동
        season_end.insert(0, "role_tag", f"{role}_{tag}")
        tables.append(season_end)
    return pd.concat(tables, ignore_index=True)


def apply_extra_rate_features(df, lookup, specs):
    """`+1` 보정 없이(라벨 없음) 시즌 진행분을 근사 계산한다. 분모(시즌 내 표본수)는
    성공률 분해에서 이미 계산된 `{role}_season_n`을 그대로 재사용 — 같은 투구 수이므로
    두 번 계산할 이유가 없고, `end_n`(+1 없는 버전) 기준 정의도 성공률 쪽과 최대 1구
    차이라 무시 가능."""
    df = df.copy()
    for role, id_col, n_col, rate_col, tag in specs:
        key = f"{role}_{tag}"
        lut = lookup.loc[lookup["role_tag"] == key, ["id", "season", "end_n", "end_rate"]]
        merged = df[[id_col, "season"]].merge(
            lut, left_on=[id_col, "season"], right_on=["id", "season"], how="left",
        )
        pre_n = merged["end_n"].fillna(0).values
        pre_count = (merged["end_n"] * merged["end_rate"]).round().fillna(0).values

        cum_count = np.round(df[n_col].values * df[rate_col].values)
        season_n = df[f"{role}_season_n"].values
        season_count = np.maximum(cum_count - pre_count, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            season_rate = np.where(season_n > 0, season_count / season_n, np.nan)

        df[f"{key}_season_rate"] = season_rate
        df[f"{key}_season_gap"] = season_rate - df[rate_col].values
    return df


EXTRA_COLS = [c for role, *_ , tag in EXTRA_RATE_SPECS for c in (f"{role}_{tag}_season_rate", f"{role}_{tag}_season_gap")]
PITCHER_MID_COLS = [c for c in EXTRA_COLS if c.startswith("pitcher_")]

VARIANTS = [
    ("baseline (full8만)", []),
    ("+pitcher_middle(2)", PITCHER_MID_COLS),
    ("+pitcher_middle+batter_middle(4)", EXTRA_COLS),
]


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def run_regime(holdout, cutoff7):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols = build_split(holdout, cutoff7)
    lookup = build_extra_rate_lookup(df, EXTRA_RATE_SPECS)
    df = apply_extra_rate_features(df, lookup, EXTRA_RATE_SPECS)

    drop_cols = ["row_id", TARGET_COL]
    results = {}
    for tag, keep_cols in VARIANTS:
        exclude_cols = [c for c in EXTRA_COLS if c not in keep_cols]
        base_features = [
            c for c in df.columns
            if c not in drop_cols and c not in trk_mlp_cols and c not in trk_cat_cols and c not in exclude_cols
        ]
        cat_features = base_features + trk_cat_cols

        train_split = df.loc[train_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        val_split = df.loc[val_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        train_split = apply_f1_filter(train_split)

        X_train, y_train = train_split[cat_features], train_split[TARGET_COL].values
        X_val, y_val = val_split[cat_features], val_split[TARGET_COL].values

        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"[{tag}] Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_features)})")
        results[tag] = score

    base = results["baseline (full8만)"]
    print(f"\n--- {label} 요약 (vs baseline) ---")
    for tag, _ in VARIANTS:
        print(f"  {tag}: {results[tag]:.2f} ({results[tag]-base:+.2f})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7)


if __name__ == "__main__":
    main()
