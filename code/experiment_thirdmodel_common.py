# code/experiment_thirdmodel_common.py
""""솔로 강하면서 CatBoost/MLP와 메커니즘 자체가 다른 3rd 모델" 탐색 시리즈
(DeepFM/EBM/BART/NAM)가 공유하는 빌드 함수. `code/experiment_asof_only_thirdmodel.py`와
달리 asof-only로 피처를 제한하지 않고 CatBoost가 보는 것과 동일한 전체 피처셋
(train.py::main()과 동일한 F1필터+TE잔차+season진행분+coarse pitchmix)을 그대로 쓴다 —
이번 탐색의 목적은 상관 낮추기가 아니라 "다르게 틀리는" 강한 솔로 모델을 찾는 것이므로,
정보를 일부러 줄이지 않는다."""
import os

import numpy as np
import pandas as pd

from code.train import add_engineered_features, apply_f1_filter, apply_te_residual_features, TE_RESIDUAL_COLS, TRACKMAN_TIER_FEED
from code.mlp_model import CAT_COLS, compute_bss
from code.blend_model import predict_blend_bundle
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
REF_MODEL_PATH = "./open/reference/best_model.pkl"


def build_full(holdout, cutoff7):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    if cutoff7:
        train_mask = (train_df["season"] < 2024) | ((train_df["season"] == 2024) & (train_df["game_month"] < 7))
        val_mask = (train_df["season"] == 2024) & (train_df["game_month"] >= 7)
    else:
        train_mask = train_df["season"] < holdout
        val_mask = train_df["season"] == holdout

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    trk_holdout = 2024 if cutoff7 else holdout
    train_df, trk_tier_cols = add_all_tiers(train_df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=trk_holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=trk_holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in train_df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS and c not in trk_cat_cols]
    cat_feature_cols = [c for c in features if c not in trk_mlp_cols]  # CatBoost 입력과 동일

    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    train_split = apply_te_residual_features(train_split, train_split, te_prior)
    val_split = apply_te_residual_features(train_split, val_split, te_prior)
    cat_feature_cols = cat_feature_cols + TE_RESIDUAL_COLS

    return train_split, val_split, features, cat_feature_cols, num_cols


def load_prod_reference(val_split, features):
    import pickle
    with open(REF_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    X_val_full = val_split[features + TE_RESIDUAL_COLS]
    prod_pred = predict_blend_bundle(bundle, X_val_full)
    y_val = val_split[TARGET_COL].values
    prod_score = compute_bss(prod_pred, y_val)[2]
    return prod_pred, prod_score
