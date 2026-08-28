# code/experiment_li_zero_filter.py
"""li==0(상황 중요도 0) 행 학습 제외 — 팀원(969 시절)의 데이터 정합성 점검 로그에서
나온 미시도 후보. li==0인 행(전체의 1.86%, 27,506행)은 이닝 7~9에 집중되어 있고
성공률이 0.5010으로 전체(0.5238) 대비 -2.3%p 낮다 — 승부가 이미 갈린 상황에서
추격조/패전처리 투수가 등판하는 실제 현상으로, apply_f1_filter와 같은 논리
(오염된/이질적인 소수 구간을 학습에서만 제거)로 시도해볼 만한 후보.

apply_f1_filter와 동일한 컨벤션: 학습 데이터에서만 li==0 행을 제거하고,
검증/추론 데이터는 건드리지 않는다 (li==0 행도 실제 평가 대상이므로).

1단계: CatBoost 단독 dual-regime(cutoff7 + season==2023) 스크리닝.

사용법:
  python -m code.experiment_li_zero_filter --cutoff7
  python -m code.experiment_li_zero_filter --holdout 2023
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


def apply_li_zero_filter(df):
    """li==0 행을 학습에서만 제거. apply_f1_filter와 동일한 방식(행 필터, 학습에만 적용)."""
    before = len(df)
    filtered = df[df["li"] != 0].reset_index(drop=True)
    print(f"  [li==0 필터] li==0 제거: {before} -> {len(filtered)}행 ({before - len(filtered)}행 제거)")
    return filtered


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

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in trk_mlp_cols]

    train_split = df.loc[train_mask, base_features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, base_features + [TARGET_COL]].reset_index(drop=True)
    train_split_f1 = apply_f1_filter(train_split)

    train_split_baseline = train_split_f1
    train_split_lizero = apply_li_zero_filter(train_split_f1)

    variants = {}
    for tag, tsplit in [("baseline", train_split_baseline), ("li_zero_filtered", train_split_lizero)]:
        te_prior = tsplit[TARGET_COL].mean()
        te_source = tsplit
        tsplit_te = apply_te_residual_features(te_source, tsplit, te_prior)
        vsplit_te = apply_te_residual_features(te_source, val_split, te_prior)
        cat_features_full = base_features + TE_RESIDUAL_COLS
        X_train, y_train = tsplit_te[cat_features_full], tsplit_te[TARGET_COL].values
        X_val, y_val = vsplit_te[cat_features_full], vsplit_te[TARGET_COL].values
        variants[tag] = (X_train, y_train, X_val, y_val, cat_features_full)
    return variants


def train_catboost_custom(X_train, y_train, X_val, y_val):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def run_regime(holdout, cutoff7):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (li==0 필터) ===\n{'='*70}")

    variants = build_split(holdout, cutoff7)
    scores = {}
    for tag, (X_train, y_train, X_val, y_val, _feature_cols) in variants.items():
        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        scores[tag] = score
        print(f"  [{tag}] Val Score={score:.2f} (best_iteration={best_iter}, n_train={len(X_train)}, {time.time()-t0:.1f}s)")
    print(f"  Delta (li_zero_filtered - baseline) = {scores['li_zero_filtered'] - scores['baseline']:+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7)


if __name__ == "__main__":
    main()
