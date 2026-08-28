# code/experiment_catboost_seed_ensemble_foldcheck.py
"""experiment_catboost_seed_ensemble.py의 dual-regime 결과(cutoff7 CatBoost 단독
+9.87 / season==2023 +31.17, 5-seed)가 검증 윈도우 특유의 우연인지 여러 시점에
걸쳐 일반화되는 신호인지, TE-residual 채택 때 쓴 것과 동일한 rolling-origin
3-fold 방법(code/experiment_te_residual_foldcheck.py)으로 재확인한다.

§35/§46과 동일하게 game_type=='R'만 써서 F1 필터가 이른 cutoff(train<2021 등)에서
학습 구간의 F행을 통째로 지우는 함정(핵심 교훈 #20/#28)을 피한다. 프로덕션
피처셋(시즌진행분+TE-residual+coarse pitchmix)을 그대로 재현하고, CatBoost
단독(random_seed=42 단일 vs 5-seed 앙상블 평균)만 비교한다 — TE-residual fold
check와 동일한 이유로 신호 자체의 시간적 일반화만 보는 게 목적이라 MLP/블렌드는
생략한다.

사용법:
  python -m code.experiment_catboost_seed_ensemble_foldcheck
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.mlp_model import compute_bss
from code.train import add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS
from code.trackman_pitcher_features import PITCHMIX_COLS, merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2021, 2022, 2023]
SEED_POOL = [42, 123, 7, 2024, 99]


def load_r_only():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df[df["game_type"] == "R"].reset_index(drop=True)  # F1 트랩 회피
    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    return df, df_trm


def train_one(seed, X_train, y_train, X_val, y_val):
    params = dict(CATBOOST_PARAMS)
    params["random_seed"] = seed
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1], int(model.get_best_iteration())


def run_fold(df_all, df_trm, val_season, n_seeds):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values

    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols]

    all_cols = base_features + [TARGET_COL]
    train_split = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    y_val = val_split[TARGET_COL].values

    prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, prior)
    val_split = apply_te_residual_features(te_source, val_split, prior)
    cat_feature_cols = base_features + TE_RESIDUAL_COLS

    X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val = val_split[cat_feature_cols]

    seeds = SEED_POOL[:n_seeds]
    preds_list = []
    for seed in seeds:
        t0 = time.time()
        preds, best_iter = train_one(seed, X_train, y_train, X_val, y_val)
        score = compute_bss(preds, y_val)[2]
        print(f"  [val={val_season}] seed={seed}: solo={score:.2f} (best_iter={best_iter}, {time.time()-t0:.1f}s)")
        preds_list.append(preds)

    single_score = compute_bss(preds_list[0], y_val)[2]
    ens_score = compute_bss(np.mean(preds_list, axis=0), y_val)[2]
    print(f"  [val={val_season}] 단일(42)={single_score:.2f} | {n_seeds}-seed 앙상블={ens_score:.2f} | delta={ens_score-single_score:+.2f} (n_train={len(train_split)}, n_val={len(val_split)})")
    return ens_score - single_score


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행")

    deltas = []
    for val_season in FOLD_SEASONS:
        print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===")
        delta = run_fold(df_all, df_trm, val_season, n_seeds=5)
        deltas.append(delta)

    wins = sum(d > 0 for d in deltas)
    print(f"\n{'='*70}\n{wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}\n{'='*70}")


if __name__ == "__main__":
    main()
