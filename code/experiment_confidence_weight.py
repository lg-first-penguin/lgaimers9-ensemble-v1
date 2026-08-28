# code/experiment_confidence_weight.py
"""신뢰도(confidence) 기반 샘플 가중치 — 두 번째로 시도해보는 미시도 방향
(첫 번째인 CatBoost 시드 앙상블은 rolling-origin 3-fold에서 노이즈로 확정 기각,
code/experiment_catboost_seed_ensemble_foldcheck.py 참고).

asof_pitcher_n(해당 투수의 누적 투구 이력 수)이 작은 행일수록 asof_pitcher_success_rate
같은 이력 기반 피처가 소표본 추정치라 노이즈가 크다. 이 프로젝트가 지금까지 시도한
가중치는 시즌-recency(§35.4-6, 기각)뿐이고 "행 자체의 피처 신뢰도"로 가중치를 주는
시도는 없었다. weight = clip(asof_pitcher_n / N0, floor, 1.0)로 콜드스타트 행을
다운웨이팅해 CatBoost 단독으로 dual-regime(cutoff7 + season==2023) 스크리닝한다.

N0=500(대략 15~20 백분위수), floor=0.2(완전 배제는 F1 필터급 리스크라 피함)로 시작.

사용법:
  python -m code.experiment_confidence_weight --cutoff7
  python -m code.experiment_confidence_weight --holdout 2023
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
N0 = 500.0
FLOOR = 0.2


def build_split(holdout, cutoff7):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm_full = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm_full)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]

    df = merge_coarse_pitchmix(df, df_trm_full, holdout=holdout)

    if cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
    else:
        train_mask = df["season"] < holdout
        val_mask = df["season"] == holdout

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)
    df["_conf_weight"] = np.clip(df["asof_pitcher_n"].values / N0, FLOOR, 1.0)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in trk_mlp_cols and c != "_conf_weight"]

    train_split = df.loc[train_mask, base_features + ["_conf_weight", TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, base_features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)
    cat_features_full = base_features + TE_RESIDUAL_COLS

    X_train, y_train = train_split[cat_features_full], train_split[TARGET_COL].values
    w_train = train_split["_conf_weight"].values
    X_val, y_val = val_split[cat_features_full], val_split[TARGET_COL].values
    return X_train, y_train, w_train, X_val, y_val


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features, weight=None):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features, weight=weight)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def run_regime(holdout, cutoff7):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (confidence weight, N0={N0}, floor={FLOOR}) ===\n{'='*70}")

    X_train, y_train, w_train, X_val, y_val = build_split(holdout, cutoff7)
    print(f"  weight 분포: min={w_train.min():.3f} mean={w_train.mean():.3f} max={w_train.max():.3f} (weight<1인 비율={ (w_train<1.0).mean():.1%})")

    for tag, weight in [("baseline(weight=1)", None), ("confidence-weight", w_train)]:
        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES, weight=weight)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"  [{tag}] Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7)


if __name__ == "__main__":
    main()
