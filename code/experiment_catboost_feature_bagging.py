# code/experiment_catboost_feature_bagging.py
"""CatBoost 피처-서브셋 배깅(Random Subspace) 앙상블. seed-ensemble(다양성 소스 =
난수 시드, code/experiment_catboost_seed_ensemble.py)은 dual-regime 스크리닝에서
좋아 보였지만(+9.87/+31.17) rolling-origin 3-fold 재검증에서 2/3·평균 +3.43로
노이즈임이 확인되어 기각됐다(teammate_catboost_mlp_track_983 메모). 다양성 소스를
바꿔, 각 멤버가 전체 피처 중 서로 다른 랜덤 서브셋(컬럼 배깅, RandomForest의
max_features와 같은 아이디어를 부스팅에 적용)만 보고 학습하게 하면 seed 노이즈와는
다른 종류의 다양성을 얻을 수 있는지 dual-regime으로 스크리닝한다.

프로덕션 피처셋(시즌진행분+TE-residual+coarse pitchmix, 트랙맨 tier 전부 미사용)을
그대로 재현한다. 각 멤버: 전체 피처 컬럼 중 FEATURE_FRAC 비율만 무작위(복원 없이)
샘플링해 학습, 예측도 같은 서브셋으로. 범주형(CAT_FEATURES)이 서브셋에서 빠지면
Pool의 cat_features 인자에서도 제외한다.

사용법:
  python -m code.experiment_catboost_feature_bagging --cutoff7
  python -m code.experiment_catboost_feature_bagging --holdout 2023
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

FEATURE_FRAC = 0.8
BAG_SEEDS = [1, 2, 3, 4, 5]  # 컬럼 샘플링용 시드 (모델 random_seed는 CATBOOST_PARAMS 고정값 그대로 사용)


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
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)
    cat_features_full = base_features + TE_RESIDUAL_COLS

    X_train, y_train = train_split[cat_features_full], train_split[TARGET_COL].values
    X_val, y_val = val_split[cat_features_full], val_split[TARGET_COL].values
    return X_train, y_train, X_val, y_val, cat_features_full


def sample_columns(all_cols, frac, seed):
    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(all_cols) * frac)))
    idx = rng.choice(len(all_cols), size=n, replace=False)
    return [all_cols[i] for i in sorted(idx)]


def train_one(cols, X_train, y_train, X_val, y_val):
    cat_features_here = [c for c in CAT_FEATURES if c in cols]
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train[cols], label=y_train, cat_features=cat_features_here)
    val_pool = Pool(data=X_val[cols], label=y_val, cat_features=cat_features_here)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val[cols])[:, 1], int(model.get_best_iteration())


def run_regime(holdout, cutoff7, n_members):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (CatBoost feature-subset bagging, frac={FEATURE_FRAC}, n_members={n_members}) ===\n{'='*70}")

    X_train, y_train, X_val, y_val, all_cols = build_split(holdout, cutoff7)
    print(f"  전체 피처 수: {len(all_cols)}, 서브셋 크기: {int(round(len(all_cols)*FEATURE_FRAC))}")

    t0 = time.time()
    full_preds, full_best_iter = train_one(all_cols, X_train, y_train, X_val, y_val)
    baseline_score = compute_bss(full_preds, y_val)[2]
    print(f"  [baseline: 전체 피처] Val Score={baseline_score:.2f} (best_iteration={full_best_iter}, {time.time()-t0:.1f}s)")

    preds_list = []
    for i, seed in enumerate(BAG_SEEDS[:n_members]):
        cols = sample_columns(all_cols, FEATURE_FRAC, seed)
        t0 = time.time()
        preds, best_iter = train_one(cols, X_train, y_train, X_val, y_val)
        score = compute_bss(preds, y_val)[2]
        print(f"  member {i+1} (bag_seed={seed}, n_cols={len(cols)}): Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s)")
        preds_list.append(preds)

    for k in range(2, len(preds_list) + 1):
        ens_pred = np.mean(preds_list[:k], axis=0)
        ens_score = compute_bss(ens_pred, y_val)[2]
        print(f"  [{k}-member 평균] Val Score={ens_score:.2f} (vs baseline(전체피처) {baseline_score:.2f}, delta={ens_score-baseline_score:+.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--n-members", type=int, default=5)
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7, args.n_members)


if __name__ == "__main__":
    main()
