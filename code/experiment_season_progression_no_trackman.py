# code/experiment_season_progression_no_trackman.py
"""§45 시즌진행분 검증은 pitchmix-only(coarse pitchmix 유지) 배경에서만 이뤄졌다 —
"트랙맨을 아예 안 쓰는 순정 982.22 베이스"에서도 같은 이득이 나는지, coarse pitchmix와
상쇄/중복되지는 않는지 별도로 확인하기 위한 2x2 분해 체크 (트랙맨 유무 x 시즌진행분 유무).
CatBoost 단독, 빠른 확인용.

사용법:
  python -m code.experiment_season_progression_no_trackman --cutoff7
  python -m code.experiment_season_progression_no_trackman --holdout 2023
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import compute_bss
from code.train import add_engineered_features, apply_f1_filter
from code.experiment_season_progression import SEASON_PROGRESSION_COLS, add_season_progression_features

DATA_DIR = "./open/data"
TARGET_COL = "control_success"


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def build_split_no_trackman(holdout, cutoff7):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    if cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
    else:
        train_mask = df["season"] < holdout
        val_mask = df["season"] == holdout

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)  # 시즌진행분 포함(이미 코드 반영됨)
    return df, train_mask, val_mask


def run_regime(holdout, cutoff7):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (트랙맨 완전 미사용) ===\n{'='*70}")

    df, train_mask, val_mask = build_split_no_trackman(holdout, cutoff7)

    for tag, use_new in [("시즌진행분 없음(순정 982.22 베이스)", False), ("+시즌진행분 8개", True)]:
        dfx = df.copy()
        if not use_new:
            dfx = dfx.drop(columns=SEASON_PROGRESSION_COLS)

        drop_cols = ["row_id", TARGET_COL]
        cat_features = [c for c in dfx.columns if c not in drop_cols]

        train_split = dfx.loc[train_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        val_split = dfx.loc[val_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        train_split = apply_f1_filter(train_split)

        X_train, y_train = train_split[cat_features], train_split[TARGET_COL].values
        X_val, y_val = val_split[cat_features], val_split[TARGET_COL].values

        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"[{tag}] Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_features)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7)


if __name__ == "__main__":
    main()
