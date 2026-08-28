import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.mlp_model import compute_bss
from code.train import add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS
from code.trackman_pitcher_features import merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2021, 2022, 2023]


def load_r_only():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df[df["game_type"] == "R"].reset_index(drop=True)
    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    return df, df_trm


def train_and_predict(val_season, df_all, df_trm, seed=42):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values
    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)
    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols]
    all_cols = list(dict.fromkeys(base_features + [TARGET_COL, "asof_pitcher_n"]))
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

    params = dict(CATBOOST_PARAMS)
    params["random_seed"] = seed
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    t0 = time.time()
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    preds = model.predict_proba(X_val)[:, 1]
    print(f"  [val={val_season}] 학습 완료 ({time.time()-t0:.1f}s)")

    n_pitcher = val_split["asof_pitcher_n"].values
    q1 = np.quantile(n_pitcher, 0.25)
    cold_mask = n_pitcher <= q1
    cold_gap = preds[cold_mask].mean() - y_val[cold_mask].mean()
    return preds, y_val, cold_mask, cold_gap


def main():
    df_all, df_trm = load_r_only()
    fold_data = {}
    for val_season in FOLD_SEASONS:
        preds, y_val, cold_mask, cold_gap = train_and_predict(val_season, df_all, df_trm)
        fold_data[val_season] = (preds, y_val, cold_mask, cold_gap)
        print(f"  [val={val_season}] cold_gap={cold_gap:+.4f} baseline_score={compute_bss(preds, y_val)[2]:.2f}")

    print("\n=== Leave-one-out 콜드스타트 사후보정 검증 ===")
    deltas = []
    for val_season in FOLD_SEASONS:
        preds, y_val, cold_mask, _ = fold_data[val_season]
        other_gaps = [fold_data[s][3] for s in FOLD_SEASONS if s != val_season]
        correction = np.mean(other_gaps)
        baseline_score = compute_bss(preds, y_val)[2]
        corrected = preds.copy()
        corrected[cold_mask] = np.clip(corrected[cold_mask] - correction, 0, 1)
        corrected_score = compute_bss(corrected, y_val)[2]
        cold_base = compute_bss(preds[cold_mask], y_val[cold_mask])[2]
        cold_corr = compute_bss(corrected[cold_mask], y_val[cold_mask])[2]
        delta = corrected_score - baseline_score
        deltas.append(delta)
        print(f"  val={val_season}: 다른 fold서 추정한 보정량={correction:+.4f} | 전체 {baseline_score:.2f}->{corrected_score:.2f} (delta={delta:+.2f}) | 콜드subgroup {cold_base:.2f}->{cold_corr:.2f} (delta={cold_corr-cold_base:+.2f})")

    print(f"\n{sum(d>0 for d in deltas)}/{len(deltas)} fold 승리, 평균 delta={np.mean(deltas):+.2f}")


if __name__ == "__main__":
    main()
