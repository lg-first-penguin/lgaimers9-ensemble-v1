# code/experiment_trackman_crosswalk_v2.py
"""2026-08-24 세션: 팀원이 새로 공유한 훨씬 정교한 train<->trackman 선수 ID 크로스워크
(`train-trackman_연결/player_id_global_mapping_final.csv`, 경기 전체 시퀀스 정렬 기반,
전역 confirmed_global_exact 695명 vs 우리 기존 `code/pitcher_crosswalk.py` 산출물
(pitcher_map.csv) 568명)로 tier A/B/C(`code/trackman_pitcher_features.py`)를 다시
검증한다. tier A/B/C는 이미 실전 950.81(-31.41, A+pitchmix)/869.52(-112.70, A+B+C)로
기각된 라인이지만, 그 원인은 (1) 트랙맨 컬럼 자체의 구조적 정보 부재(위치/포수요구
정보 없음), (2) 2025 트랙맨이 없어 선수별 값이 시즌 내내 고정되는 시간적 한계로
진단되었지 크로스워크 정확도 문제로 진단된 적은 없다 — 이번 실험은 "혹시 크로스워크
노이즈 자체가 신호를 깎아먹고 있었나"를 크로스워크만 바꿔서 저비용으로 확인한다.

tier A/B/C는 batter_map.csv를 쓰지 않는다(tier C의 "batter_hand"는 batter identity가
아니라 df_main 자체의 batter_hand 코드 컬럼) — 확인 완료, 이 실험도 pitcher_map만 교체.

사용법:
  python -m code.experiment_trackman_crosswalk_v2 --regime cutoff7
  python -m code.experiment_trackman_crosswalk_v2 --regime 2023
"""
import argparse
import os

import numpy as np
import pandas as pd

from code.mlp_model import (
    CAT_COLS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.catboost_model import train_catboost, predict_catboost
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix, PITCHMIX_COLS
from code.train import apply_f1_filter, add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]

TIERS = ["a", "b", "c"]


def build_new_pitcher_map():
    df = pd.read_csv("train-trackman_연결/player_id_global_mapping_final.csv", encoding="utf-8-sig")
    sub = df[(df["role"] == "pitcher") & (df["global_player_mapping_status"] == "confirmed_global_exact")].copy()
    out = pd.DataFrame({
        "pitcher_id": sub["train_player_id"].values,
        "pitcher_trackman_id": sub["current_trackman_player_id"].astype(int).values,
        "confidence": sub["global_exact_purity"].values,
        "n_pitch": sub["train_role_observations"].values,
    })
    print(f"[새 크로스워크] confirmed_global_exact 투수: {len(out)}명")
    return out


def build_split_with_tiers(cutoff7, holdout, pitcher_map, tiers=TIERS, apply_f1=True):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    if cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
        trk_holdout = 2024
    else:
        train_mask = df["season"] < holdout
        val_mask = df["season"] == holdout
        trk_holdout = holdout

    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, tiers, holdout=trk_holdout)
    trk_mlp_cols = trk_tier_cols.get("a", [])
    trk_cat_cols = trk_tier_cols.get("b", []) + trk_tier_cols.get("c", [])

    df = merge_coarse_pitchmix(df, df_trm, holdout=trk_holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in df.columns if c not in drop_cols]
    mlp_num_cols_base = [c for c in features if c not in CAT_COLS and c not in trk_cat_cols and c not in trk_mlp_cols]
    cat_feature_cols_base = [c for c in features if c not in trk_mlp_cols]

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    if apply_f1:
        train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    train_split = apply_te_residual_features(train_split, train_split, te_prior)
    val_split = apply_te_residual_features(train_split, val_split, te_prior)
    cat_feature_cols_base = cat_feature_cols_base + TE_RESIDUAL_COLS

    return train_split, val_split, mlp_num_cols_base, cat_feature_cols_base, trk_mlp_cols, trk_cat_cols


def mlp_solo_score(train_split, val_split, num_cols, device, seeds=SCREEN_SEEDS):
    tr_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    va_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(tr_proc, CAT_COLS, num_cols, TARGET_COL)
    X_va_cat, X_va_num, y_va = to_tensors(va_proc, CAT_COLS, num_cols, TARGET_COL)
    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_va_cat, X_val_num=X_va_num, y_val=y_va, seeds=seeds, device=device, verbose=False,
    )
    preds = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_va_cat, X_va_num, bin_edges=bin_edges, device=device)
    return compute_bss(preds, y_va.numpy())[2]


def catboost_solo_score(train_split, val_split, cat_cols):
    X_train = train_split[cat_cols]
    y_train = train_split[TARGET_COL].values
    X_val = val_split[cat_cols]
    y_val = val_split[TARGET_COL].values
    model, _ = train_catboost(X_train, y_train, X_val, y_val, verbose=False)
    return compute_bss(predict_catboost(model, X_val), y_val)[2]


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    old_pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    new_pitcher_map = build_new_pitcher_map()
    device = get_device()

    for map_name, pitcher_map in [("기존(568명)", old_pitcher_map), ("신규(695명)", new_pitcher_map)]:
        print(f"\n--- 크로스워크: {map_name} ---")
        train_split, val_split, mlp_num_cols, cat_feature_cols, trk_mlp_cols, trk_cat_cols = \
            build_split_with_tiers(cutoff7, holdout, pitcher_map)
        tier_a_cols = trk_mlp_cols
        tier_bc_cols = trk_cat_cols[:-len(PITCHMIX_COLS)] if len(trk_cat_cols) > len(PITCHMIX_COLS) else []

        base_mlp_cols = [c for c in mlp_num_cols if c not in tier_a_cols]
        base_cat_cols = [c for c in cat_feature_cols if c not in tier_bc_cols]

        mlp_base = mlp_solo_score(train_split, val_split, base_mlp_cols, device)
        mlp_tierA = mlp_solo_score(train_split, val_split, base_mlp_cols + tier_a_cols, device)
        cat_base = catboost_solo_score(train_split, val_split, base_cat_cols)
        cat_tierBC = catboost_solo_score(train_split, val_split, cat_feature_cols)

        print(f"[{map_name}] MLP baseline={mlp_base:.2f} | +tierA={mlp_tierA:.2f} | delta={mlp_tierA - mlp_base:+.2f}")
        print(f"[{map_name}] CatBoost baseline={cat_base:.2f} | +tierB+C+pitchmix={cat_tierBC:.2f} | delta={cat_tierBC - cat_base:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["cutoff7", "2023"], default="cutoff7")
    args = parser.parse_args()
    if args.regime == "cutoff7":
        run_regime(cutoff7=True, holdout=2024)
    else:
        run_regime(cutoff7=False, holdout=2023)
