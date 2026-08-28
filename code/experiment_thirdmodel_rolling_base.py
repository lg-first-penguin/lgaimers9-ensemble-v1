# code/experiment_thirdmodel_rolling_base.py
"""21/22/23/24(+2020) season-level rolling-origin 재검증의 공용 베이스. CatBoost/MLP는
어떤 3rd-모델 후보(DeepFM/EBM/BART/NAM)를 검증하든 동일하므로, fold당 한 번만 학습해
`open/temp/experiment_rolling_base/fold_{season}.pkl`에 캐싱한다 — 후보 스크립트 4개가
각자 CatBoost+MLP를 중복 재학습하는 낭비를 피한다.

`code/experiment_meta_oof_fit.py`와 동일 관례: R-only(F1 트랩 회피), train<val_season,
CatBoost는 고정 설정 1회, MLP는 3-seed(SCREEN_SEEDS). 3rd 모델에는 `code/thirdmodel_common.py`
관례대로 MLP와 같은 피처셋(CAT_COLS + num_cols)을 준다 — TE-residual/coarse pitchmix는
CatBoost 전용이라 주지 않는다.

사용법(모듈로 import): `from code.experiment_thirdmodel_rolling_base import get_fold_base, FOLD_SEASONS`
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import (
    CAT_COLS, apply_preprocessing, compute_bss, embed_dim_for_cardinality, fit_preprocessing,
    fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.train import TE_RESIDUAL_COLS, add_engineered_features, apply_te_residual_features
from code.trackman_pitcher_features import PITCHMIX_COLS, merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2020, 2021, 2022, 2023, 2024]
SCREEN_SEEDS = [42, 123, 7]
CACHE_DIR = "./open/temp/experiment_rolling_base"


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


def compute_fold_base(df_all, df_trm, val_season, device):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values

    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in PITCHMIX_COLS]
    num_cols = [c for c in base_features if c not in CAT_COLS]

    all_cols = base_features + PITCHMIX_COLS + [TARGET_COL]
    train_split = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    y_val = val_split[TARGET_COL].values

    te_prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)

    cat_feature_cols = base_features + PITCHMIX_COLS + TE_RESIDUAL_COLS
    X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val = val_split[cat_feature_cols]
    t0 = time.time()
    cat_model, cat_best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
    cat_pred = cat_model.predict_proba(X_val)[:, 1]
    print(f"  [val={val_season}] CatBoost 완료 (best_iter={cat_best_iter}, {time.time()-t0:.1f}s)")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num)

    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val, seeds=SCREEN_SEEDS, device=device,
    )
    mlp_pred = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_val_cat, X_val_num, bin_edges=bin_edges, device=device)
    print(f"  [val={val_season}] MLP(3-seed) 완료 ({time.time()-t0:.1f}s)")

    cat_score = compute_bss(cat_pred, y_val)[2]
    mlp_score = compute_bss(mlp_pred, y_val)[2]
    print(f"  [val={val_season}] CatBoost={cat_score:.2f} | MLP={mlp_score:.2f}")

    # 3rd-모델 후보용으로 MLP와 같은 피처셋(CAT_COLS+num_cols, TE-residual/pitchmix 제외)만 남긴 raw split도 같이 저장.
    fold_data = {
        "train_split": train_split[CAT_COLS + num_cols + [TARGET_COL]].reset_index(drop=True),
        "val_split": val_split[CAT_COLS + num_cols + [TARGET_COL]].reset_index(drop=True),
        "num_cols": num_cols,
        "cat_pred": cat_pred, "mlp_pred": mlp_pred, "y_val": y_val,
        "cat_score": cat_score, "mlp_score": mlp_score,
    }
    return fold_data


def get_fold_base(val_season, df_all=None, df_trm=None, device=None, force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"fold_{val_season}.pkl")
    if not force and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    if df_all is None or df_trm is None:
        df_all, df_trm = load_r_only()
    if device is None:
        device = get_device()
    fold_data = compute_fold_base(df_all, df_trm, val_season, device)
    with open(cache_path, "wb") as f:
        pickle.dump(fold_data, f)
    return fold_data
