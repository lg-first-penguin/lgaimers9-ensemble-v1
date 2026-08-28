# code/experiment_reverse_season_foldcheck.py
"""Group B의 reverse_season(asof_pitcher_reverse_rate 시즌분해, +2컬럼) dual-regime
결과(cutoff7/season==2023 스크리닝, code/experiment_asof_rate_decomposition.py
--group b)가 borderline이었던 데 대한 rolling-origin 재검증이다.

code/experiment_te_residual_foldcheck.py와 동일한 패턴: game_type=='R'만 써서 F1
필터가 이른 cutoff에서 학습 구간의 F행을 통째로 지워버리는 함정(핵심 교훈 #20/#28)을
피하고, CatBoost 단독(동일 시드)으로 2021/2022/2023을 각각 val로 삼아 train<val
방식 rolling-origin 검증한다.

사용법:
  python -m code.experiment_reverse_season_foldcheck
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_asof_rate_decomposition import (
    GROUP_B_SPECS,
    apply_rate_features,
    build_rate_lookup,
    cols_for,
)
from code.mlp_model import compute_bss
from code.train import add_engineered_features
from code.trackman_pitcher_features import PITCHMIX_COLS, merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2021, 2022, 2023, 2024]
REVERSE_SPECS = [s for s in GROUP_B_SPECS if s[4] == "reverse"]
REVERSE_COLS = cols_for(GROUP_B_SPECS, "reverse")


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def load_r_only():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df[df["game_type"] == "R"].reset_index(drop=True)  # F1 트랩 회피

    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    return df, df_trm


def run_fold(df_all, df_trm, val_season):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values

    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)

    lookup = build_rate_lookup(df, REVERSE_SPECS)
    df = apply_rate_features(df, lookup, REVERSE_SPECS)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in PITCHMIX_COLS and c not in REVERSE_COLS]

    all_cols = base_features + PITCHMIX_COLS + REVERSE_COLS + [TARGET_COL]
    train_split = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    y_val = val_split[TARGET_COL].values

    results = {}
    for tag, extra in [("baseline", []), ("+reverse_season(2)", REVERSE_COLS)]:
        cat_feature_cols = base_features + PITCHMIX_COLS + extra
        X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
        X_val = val_split[cat_feature_cols]
        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"  [val={val_season}][{tag}] Val Score={score:.2f} (best_iter={best_iter}, {time.time()-t0:.1f}s, n_train={len(train_split)}, n_val={len(val_split)})")
        results[tag] = score
    return results


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행")

    deltas = []
    for val_season in FOLD_SEASONS:
        print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===")
        r = run_fold(df_all, df_trm, val_season)
        delta = r["+reverse_season(2)"] - r["baseline"]
        deltas.append(delta)
        print(f"  delta: {delta:+.2f}")

    wins = sum(d > 0 for d in deltas)
    print(f"\n{'='*70}\n{wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}\n{'='*70}")


if __name__ == "__main__":
    main()
