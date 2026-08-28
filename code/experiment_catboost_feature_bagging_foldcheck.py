# code/experiment_catboost_feature_bagging_foldcheck.py
"""CatBoost feature-subset bagging(code/experiment_catboost_feature_bagging.py)의
dual-regime 결과 — cutoff7 5-member 평균 +1.13~+3.61(노이즈 하한선 밑) vs
season==2023 +8.35~+40.94(멤버 수에 거의 선형으로 증가) — 이 비대칭 자체가
CatBoost seed-ensemble이 보였던 패턴(cutoff7 +9.87/season==2023 +31.17,
2/3 fold·평균 +3.43으로 기각)과 같은 신호라, 같은 rolling-origin 3-fold로
재검증한다.

§35/§46/§57과 동일하게 game_type=='R'만 써서 F1 필터 함정을 피한다(R-only에서는
no-op). CatBoost 단독, 5개 피처 서브셋 멤버(FEATURE_FRAC=0.8, bag_seed=1..5)
평균 vs 전체피처 단일모델을 3개 시즌(2021/2022/2023)에서 train<val로 비교한다.

사용법:
  python -m code.experiment_catboost_feature_bagging_foldcheck
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
FEATURE_FRAC = 0.8
BAG_SEEDS = [1, 2, 3, 4, 5]


def load_r_only():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df[df["game_type"] == "R"].reset_index(drop=True)  # F1 트랩 회피
    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    return df, df_trm


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


def run_fold(df_all, df_trm, val_season):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values

    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols]

    all_cols_df = base_features + [TARGET_COL]
    train_split = df.loc[train_mask, all_cols_df].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols_df].reset_index(drop=True)
    y_val = val_split[TARGET_COL].values

    prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, prior)
    val_split = apply_te_residual_features(te_source, val_split, prior)
    all_cols = base_features + TE_RESIDUAL_COLS

    X_train, y_train = train_split[all_cols], train_split[TARGET_COL].values
    X_val = val_split[all_cols]

    t0 = time.time()
    baseline_preds, base_iter = train_one(all_cols, X_train, y_train, X_val, y_val)
    baseline_score = compute_bss(baseline_preds, y_val)[2]
    print(f"  [val={val_season}][baseline] Val Score={baseline_score:.2f} (best_iter={base_iter}, {time.time()-t0:.1f}s)")

    preds_list = []
    for seed in BAG_SEEDS:
        cols = sample_columns(all_cols, FEATURE_FRAC, seed)
        t0 = time.time()
        preds, best_iter = train_one(cols, X_train, y_train, X_val, y_val)
        score = compute_bss(preds, y_val)[2]
        print(f"  [val={val_season}] member(bag_seed={seed}, n_cols={len(cols)}): Val Score={score:.2f} (best_iter={best_iter}, {time.time()-t0:.1f}s)")
        preds_list.append(preds)

    ens_score = compute_bss(np.mean(preds_list, axis=0), y_val)[2]
    delta = ens_score - baseline_score
    print(f"  [val={val_season}] {len(BAG_SEEDS)}-member 평균={ens_score:.2f} | baseline={baseline_score:.2f} | delta={delta:+.2f} (n_train={len(train_split)}, n_val={len(val_split)})")
    return delta


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행")

    deltas = []
    for val_season in FOLD_SEASONS:
        print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===")
        delta = run_fold(df_all, df_trm, val_season)
        deltas.append(delta)

    wins = sum(d > 0 for d in deltas)
    print(f"\n{'='*70}\n{wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}\n{'='*70}")


if __name__ == "__main__":
    main()
