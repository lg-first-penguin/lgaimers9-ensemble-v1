# code/experiment_asof_only_thirdmodel.py
""""3rd 모델의 solo가 항상 약했던 이유가 트랙맨(정적 투수 요약 벡터)이라는 입력
자체의 정보 천장 때문이었다면, 대신 이미 신호가 검증된 'asof 계열' 피처만으로
RandomForest/ExcelFormer/TabNet을 학습시키면 solo가 더 강하면서도 CatBoost/MLP(둘 다
raw 상황 피처+asof를 전부 봄)와는 다르게 틀릴 수 있는가?"를 검증한다.

ASOF_ONLY_COLS는 공식 asof_* 19개 + pitcher_hand/batter_hand(선수 고정 속성) +
add_engineered_features가 만드는 파생 피처 중 "순수하게 asof_*로부터만 계산되고
행 자신의 상황(count/inning/주자/li 등)에 의존하지 않는" 15개(시즌진행분 8 +
prev-game gap류 5 + pitcher_relative_success + matchup)로 구성 — count_diff,
pitcher_count_advantage_*, count_pressure, TE-residual(그룹 키에 balls_before/
strikes_before 등 상황 컬럼이 들어감)는 "순수 asof"가 아니므로 의도적으로 제외했다.
game_type/base_state/inning 등 원시 상황 컬럼, pitcher_id/batter_id(고카디널리티
식별자, 핵심 교훈 #3 — asof 통계가 이미 정체성 정보를 담고 있어 원본 ID 실익 작음)도
전부 제외한다.

1단계: RandomForest(Regressor, MSE objective — §13.2에서 검증된 RF 최적 objective)만
빠르게 돌려 solo 점수와 프로덕션 예측과의 상관관계를 확인한다. ExcelFormer/TabNet은
결과를 보고 투자 여부를 판단한다.

사용법:
  python -m code.experiment_asof_only_thirdmodel --cutoff7 --model rf
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from code.train import add_engineered_features, apply_f1_filter, apply_te_residual_features, TE_RESIDUAL_COLS, TRACKMAN_TIER_FEED
from code.mlp_model import compute_bss
from code.blend_model import predict_blend_bundle
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix
from code.randomforest_model import REGRESSOR_PARAMS
from sklearn.ensemble import RandomForestRegressor

ASOF_CAT_COLS = ["pitcher_hand", "batter_hand"]
ASOF_NUM_COLS = [c for c in (
    "asof_pitcher_n asof_pitcher_success_rate asof_pitcher_reverse_rate asof_pitcher_middle_rate "
    "asof_pitcher_ball_rate asof_pitcher_strike_rate asof_pitcher_prev1_game_success_rate "
    "asof_pitcher_prev3_game_success_rate asof_pitcher_prev5_game_success_rate "
    "asof_pitcher_prev1_game_middle_rate asof_pitcher_prev3_game_middle_rate "
    "asof_pitcher_prev5_game_middle_rate asof_batter_n asof_batter_success_rate asof_batter_middle_rate "
    "asof_pitcher_pitchmix_n asof_pitcher_fastball_rate asof_pitcher_breaking_rate asof_pitcher_offspeed_rate "
    "pitcher_season_n pitcher_season_success_count pitcher_season_success_rate pitcher_season_rate_gap "
    "batter_season_n batter_season_success_count batter_season_success_rate batter_season_rate_gap "
    "pitcher_recent1_gap pitcher_recent3_gap pitcher_recent5_gap pitcher_relative_success pitcher_trend "
    "pitcher_consistency matchup"
).split()]

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
REF_MODEL_PATH = "./open/reference/best_model.pkl"

ASOF_OFFICIAL_COLS = [
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
ASOF_IDENTITY_COLS = ["pitcher_hand", "batter_hand"]
ASOF_DERIVED_COLS = [
    "pitcher_season_n", "pitcher_season_success_count", "pitcher_season_success_rate", "pitcher_season_rate_gap",
    "batter_season_n", "batter_season_success_count", "batter_season_success_rate", "batter_season_rate_gap",
    "pitcher_recent1_gap", "pitcher_recent3_gap", "pitcher_recent5_gap",
    "pitcher_relative_success", "pitcher_trend", "pitcher_consistency",
    "matchup",
]
ASOF_ONLY_COLS = ASOF_OFFICIAL_COLS + ASOF_IDENTITY_COLS + ASOF_DERIVED_COLS
assert set(ASOF_CAT_COLS + ASOF_NUM_COLS) == set(ASOF_ONLY_COLS)


def build(holdout, cutoff7):
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
    train_df, _ = add_all_tiers(train_df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=2024 if cutoff7 else holdout)
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=2024 if cutoff7 else holdout)

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    features = [c for c in train_df.columns if c not in ["row_id", TARGET_COL]]
    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    val_split_te = apply_te_residual_features(train_split, val_split, te_prior)

    return train_split, val_split_te, features


def run_tabnet(train_split, val_split, y_train, y_val):
    from pytorch_tabnet.tab_model import TabNetClassifier
    from code.experiment_tabnet_correlation import BrierMetric
    from code.mlp_model import fit_preprocessing, apply_preprocessing

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        train_split[ASOF_ONLY_COLS + [TARGET_COL]], ASOF_CAT_COLS, ASOF_NUM_COLS,
    )
    val_proc = apply_preprocessing(val_split[ASOF_ONLY_COLS], ASOF_CAT_COLS, ASOF_NUM_COLS, cat_encoder, num_imputer, num_scaler)

    tabnet_cols = ASOF_CAT_COLS + ASOF_NUM_COLS
    X_tr = train_proc[tabnet_cols].values.astype(np.float32)
    y_tr = y_train.astype(np.int64)
    X_val = val_proc[tabnet_cols].values.astype(np.float32)
    cat_idxs = list(range(len(ASOF_CAT_COLS)))

    t0 = time.time()
    model = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1, seed=42, verbose=0)
    model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val.astype(np.int64))], eval_metric=[BrierMetric],
        max_epochs=100, patience=15, batch_size=4096, virtual_batch_size=512,
    )
    pred = model.predict_proba(X_val)[:, 1]
    return pred, time.time() - t0


def run_excel(train_split, val_split, y_train, y_val):
    import torch
    from catboost import CatBoostClassifier, Pool
    from code.excelformer_model import compute_feature_ranking, train_excel
    from code.mlp_model import fit_preprocessing, apply_preprocessing, fit_quantile_edges, to_tensors, get_device

    # 랭킹용 가벼운 CatBoost (asof-only 피처, 카테고리 선언 없음 — hand는 이미 정수)
    t0 = time.time()
    cb_params = dict(iterations=1500, depth=6, learning_rate=0.05, loss_function="Logloss",
                      eval_metric="BrierScore", early_stopping_rounds=50, random_seed=42, verbose=False)
    cb = CatBoostClassifier(**cb_params)
    order = ASOF_CAT_COLS + ASOF_NUM_COLS
    train_pool = Pool(train_split[order], y_train)
    val_pool = Pool(val_split[order], y_val)
    cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
    ranking = compute_feature_ranking(cb, order, ASOF_CAT_COLS, ASOF_NUM_COLS)
    print(f"  [ExcelFormer용 랭킹 CatBoost] {time.time()-t0:.1f}s")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        train_split[ASOF_ONLY_COLS + [TARGET_COL]], ASOF_CAT_COLS, ASOF_NUM_COLS,
    )
    val_proc = apply_preprocessing(val_split[ASOF_ONLY_COLS], ASOF_CAT_COLS, ASOF_NUM_COLS, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr_t = to_tensors(train_proc, ASOF_CAT_COLS, ASOF_NUM_COLS, TARGET_COL)
    X_val_cat, X_val_num = to_tensors(val_proc, ASOF_CAT_COLS, ASOF_NUM_COLS)
    bin_edges = fit_quantile_edges(X_tr_num)
    device = get_device()

    t0 = time.time()
    model, best_epoch = train_excel(
        X_tr_cat, X_tr_num, y_tr_t, cat_dims=cat_dims, bin_edges=bin_edges,
        cat_cols=ASOF_CAT_COLS, num_cols=ASOF_NUM_COLS, feature_ranking=ranking,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
        device=device, verbose=False, seed=42,
    )
    from code.ft_transformer_model import batched_forward
    pred = batched_forward(model, X_val_cat, X_val_num, device=device)
    return pred, time.time() - t0


def run(holdout, cutoff7, model_name):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} | 3rd 모델: {model_name} (asof-only, {len(ASOF_ONLY_COLS)}컬럼) ===\n{'='*70}")

    train_split, val_split, full_features = build(holdout, cutoff7)
    y_train = train_split[TARGET_COL].values
    y_val = val_split[TARGET_COL].values

    with open(REF_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    X_val_full = val_split[full_features + TE_RESIDUAL_COLS]
    prod_pred = predict_blend_bundle(bundle, X_val_full)
    prod_score = compute_bss(prod_pred, y_val)[2]
    print(f"[프로덕션 reference 블렌드] Val Score={prod_score:.2f} (참고용, 동일 val 재구성)")

    X_train_asof = train_split[ASOF_ONLY_COLS]
    X_val_asof = val_split[ASOF_ONLY_COLS]

    if model_name == "rf":
        t0 = time.time()
        model = RandomForestRegressor(n_estimators=300, **REGRESSOR_PARAMS)
        model.fit(X_train_asof, y_train.astype(np.float64))
        pred = np.clip(model.predict(X_val_asof), 0, 1)
        elapsed = time.time() - t0
    elif model_name == "tabnet":
        pred, elapsed = run_tabnet(train_split, val_split, y_train, y_val)
    elif model_name == "excel":
        pred, elapsed = run_excel(train_split, val_split, y_train, y_val)
    else:
        raise NotImplementedError(model_name)

    brier, bss, score = compute_bss(pred, y_val)
    corr = np.corrcoef(pred, prod_pred)[0, 1]
    print(f"[{model_name} asof-only] Val Score={score:.2f} ({elapsed:.1f}s, n_features={len(ASOF_ONLY_COLS)})")
    print(f"[{model_name} asof-only] 프로덕션 블렌드 예측과의 상관계수: {corr:.4f}")
    return score, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--model", choices=["rf", "tabnet", "excel"], default="rf")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run(holdout, args.cutoff7, args.model)


if __name__ == "__main__":
    main()
