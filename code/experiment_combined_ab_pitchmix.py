# code/experiment_combined_ab_pitchmix.py
"""2026-08-17 세션 최종 확인: `clean_trackman()`의 결측치 오필터링 버그를 고친 뒤 재검증한
tier A(투수x구종군, ->MLP)+B(+압박, ->CatBoost)와, 팀원이 제보한 coarse pitchmix
(볼카운트x손 조합, ->CatBoost)를 한 번에 합쳐서 서로 잡아먹지 않는지 확인한다.

C(+타자손)는 버그 수정 후에도 season==2023 홀드아웃에서 여전히 마이너스(-14.64)로 제외.
개별 검증 결과(2023 / cutoff7 듀얼체크 전부 통과):
  A->mlp        : +7.58  / +32.31
  B->cat        : +16.33 / +28.64
  pitchmix->cat : +12.84 / +13.05
세 피처는 조인 축이 전부 다르다(투수정체성 x 구종군 / +압박상황 / 볼카운트x손) — 겹치는
신호가 아닐 가능성이 높지만 실제로 합쳐봐야 안다(CatBoost에 B+pitchmix를 동시에 주면
서로의 분산을 깎아먹을 수도 있음).

사용법:
  python -m code.experiment_combined_ab_pitchmix --holdout 2023
  python -m code.experiment_combined_ab_pitchmix --cutoff7
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from code.blend_model import fit_meta_model
from code.catboost_model import predict_catboost, train_catboost
from code.mlp_model import (
    CAT_COLS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, make_bundle, predict_bundle,
    to_tensors, train_ensemble, QUANTILE_N_BINS,
)
from code.train import add_engineered_features
from code.trackman_pitcher_features import clean_trackman, add_all_tiers
from code.experiment_coarse_pitchmix import merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]
FULL_TIER_FEED = {"a": "mlp", "b": "cat"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024, choices=[2023, 2024])
    parser.add_argument("--apply-f1", dest="apply_f1", action="store_true", default=True)
    parser.add_argument("--no-f1", dest="apply_f1", action="store_false")
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--tiers", type=str, default="a,b",
                         help="pitcher-std 티어 중 어느 걸 포함할지 (예: 'a', 'a,b'). pitchmix는 항상 포함.")
    args = parser.parse_args()
    if args.cutoff7:
        args.holdout = 2024
    TIER_FEED = {t: FULL_TIER_FEED[t] for t in args.tiers.split(",") if t}
    label = f"COMBINED[{args.tiers}+pitchmix] {'cutoff7' if args.cutoff7 else 'holdout=' + str(args.holdout)} f1={'ON' if args.apply_f1 else 'OFF'}"

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm_full = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm_full)
    t0 = time.time()
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TIER_FEED), holdout=args.holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TIER_FEED[tier] == "cat" for c in cols]
    print(f"[{label}] pitcher-std 병합 완료 ({time.time() - t0:.1f}s) | tier별 피처 수: { {t: len(c) for t, c in trk_tier_cols.items()} }")

    df_trm_pmix = df_trm_full[["season", "balls_before", "strikes_before", "pitcher_hand",
                                "batter_hand", "pitch_type_group"]].copy()
    hand_map = {"Left": 1, "Right": 2}
    df_trm_pmix["pitcher_hand"] = df_trm_pmix["pitcher_hand"].map(hand_map)
    df_trm_pmix["batter_hand"] = df_trm_pmix["batter_hand"].map(hand_map)
    t0 = time.time()
    df = merge_coarse_pitchmix(df, df_trm_pmix, holdout=args.holdout)
    print(f"[{label}] pitchmix 병합 완료 ({time.time() - t0:.1f}s)")
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    if args.cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
    else:
        train_mask = df["season"] < args.holdout
        val_mask = df["season"] == args.holdout
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    trk_all_cols = trk_mlp_cols + trk_cat_cols
    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in trk_all_cols]
    features = base_features + trk_all_cols

    cat_features = base_features + trk_cat_cols
    mlp_num_cols = [c for c in base_features if c not in CAT_COLS] + trk_mlp_cols
    print(f"[{label}] 총 피처 수: {len(features)} | CatBoost 피처: {len(cat_features)} | MLP 수치형 피처: {len(mlp_num_cols)}")

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)

    if args.apply_f1:
        before = len(train_split)
        train_split = train_split[~((train_split["game_type"] == "F") & (train_split["season"] <= 2022))].reset_index(drop=True)
        print(f"[{label}] F1 필터 적용: {before} -> {len(train_split)}행")

    print(f"[{label}] 훈련: {len(train_split)}행 | 검증: {len(val_split)}행")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    y_val_np = val_proc[TARGET_COL].values

    device = get_device()
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
        seeds=SCREEN_SEEDS, device=device,
    )
    print(f"[{label}] MLP({len(SCREEN_SEEDS)}-seed) 학습 완료 ({time.time() - t0:.1f}s)")

    mlp_bundle = make_bundle(
        members, CAT_COLS, mlp_num_cols, cat_dims, embed_dims,
        cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges,
    )
    mlp_val_preds = predict_bundle(mlp_bundle, val_split[features], device=device)

    X_train_raw, y_train_raw = train_split[cat_features], train_split[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[cat_features], val_split[TARGET_COL].values
    t0 = time.time()
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    print(f"[{label}] CatBoost 완료 (best_iteration={catboost_best_iteration}, {time.time() - t0:.1f}s)")

    cat_val_preds = predict_catboost(catboost_model, X_val_raw)

    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    mlp_score = compute_bss(mlp_val_preds, y_val_raw)[2]
    w_cat, w_mlp, intercept, blend_score, blend_brier = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)

    print(f"\n[RESULT {label}] CatBoost={cat_score:.2f} | MLP={mlp_score:.2f} | Blend={blend_score:.2f}")


if __name__ == "__main__":
    main()
