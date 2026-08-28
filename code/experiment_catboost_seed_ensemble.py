# code/experiment_catboost_seed_ensemble.py
"""CatBoost는 지금까지 이 프로젝트에서 항상 random_seed=42 고정 단일 모델로만
학습됐다 (code/catboost_model.py::CATBOOST_PARAMS). MLP는 7-seed 앙상블(평균)로
분산을 줄여 큰 이득을 봤고(690->789), CatBoost도 thread_count/GPU 비결정성만으로
런마다 ~7-19점씩 흔들린다는 게 이미 문서화돼 있다(teammate_catboost_mlp_track_983
메모 "side finding"). CatBoost도 동일한 아이디어(여러 random_seed로 학습 후 확률
평균)로 그 노이즈를 줄일 수 있는지 dual-regime(cutoff7 + season==2023)으로
스크리닝한다. 프로덕션 피처셋(시즌진행분+TE-residual+coarse pitchmix, 트랙맨 tier
전부 미사용)을 그대로 재현한다.

사용법:
  python -m code.experiment_catboost_seed_ensemble --cutoff7
  python -m code.experiment_catboost_seed_ensemble --holdout 2023
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.mlp_model import compute_bss
from code.train import (
    TRACKMAN_TIER_FEED, add_engineered_features, apply_f1_filter,
    apply_te_residual_features, TE_RESIDUAL_COLS,
)
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"

SEED_POOL = [42, 123, 7, 2024, 99, 31337, 8]


def build_split(holdout, cutoff7):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm_full = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm_full)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=holdout)
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]

    df = merge_coarse_pitchmix(df, df_trm_full, holdout=holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    if cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
    else:
        train_mask = df["season"] < holdout
        val_mask = df["season"] == holdout

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in trk_mlp_cols]
    cat_features = base_features  # trk_cat_cols는 이미 df 컬럼에 포함, base_features가 커버

    train_split = df.loc[train_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)
    cat_features_full = cat_features + TE_RESIDUAL_COLS

    X_train, y_train = train_split[cat_features_full], train_split[TARGET_COL].values
    X_val, y_val = val_split[cat_features_full], val_split[TARGET_COL].values
    return X_train, y_train, X_val, y_val


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


def run_regime(holdout, cutoff7, n_seeds):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (CatBoost seed-ensemble, n_seeds={n_seeds}) ===\n{'='*70}")

    X_train, y_train, X_val, y_val = build_split(holdout, cutoff7)
    seeds = SEED_POOL[:n_seeds]

    preds_list = []
    for seed in seeds:
        t0 = time.time()
        preds, best_iter = train_one(seed, X_train, y_train, X_val, y_val)
        score = compute_bss(preds, y_val)[2]
        print(f"  seed={seed}: Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s)")
        preds_list.append(preds)

    baseline_score = compute_bss(preds_list[0], y_val)[2]
    for k in range(2, len(preds_list) + 1):
        ens_pred = np.mean(preds_list[:k], axis=0)
        ens_score = compute_bss(ens_pred, y_val)[2]
        print(f"  [{k}-seed 평균] Val Score={ens_score:.2f} (vs seed=42 단독 {baseline_score:.2f}, delta={ens_score-baseline_score:+.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7, args.n_seeds)


if __name__ == "__main__":
    main()
