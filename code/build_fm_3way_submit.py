# code/build_fm_3way_submit.py
"""DeepFM(튜닝판)을 기존 프로덕션 2-way(CatBoost+MLP) 번들에 3번째 모델로 추가해
`submit/model/final_retained_model.pkl`을 3-way 블렌드로 교체하는 원샷 스크립트.

dopip.py Step 3(Full Retrain)와 동일한 피처 엔지니어링(F1 필터, TE-residual, 시즌진행분,
coarse pitchmix, holdout=2025/None 컨벤션)을 그대로 재현하되, CatBoost/MLP는 이미 완습된
기존 `submit/model/final_retained_model.pkl`을 그대로 재사용(재학습 없음)하고 DeepFM만
전체 데이터로 새로 학습한다. epoch 예산은 dopip.py의 관례(각 시드의 cutoff7 스크리닝
best_epoch + FULL_RETRAIN_EPOCH_BUFFER)를 그대로 따른다 — 여기서는
`code/experiment_fm_tuned3way.py --seeds ... `가 `./open/temp/experiment_fm_tuned/fm_bundle.pkl`에
저장한 멤버별 best_epoch를 사용한다.

메타모델 가중치는 재적합하지 않고 `--w-cat --w-mlp --w-fm --intercept`로 넘겨받은 값을
그대로 쓴다(dopip.py가 reference의 meta_model을 그대로 재사용하는 것과 동일 관례 —
전체 데이터에는 held-out val이 없어 재적합이 애초에 불가능).

사용법:
  python -m code.build_fm_3way_submit --w-cat 1.257 --w-mlp 1.163 --w-fm 1.297 --intercept -1.877
"""
import argparse
import os
import pickle
import shutil
import time

import numpy as np
import pandas as pd

from code.mlp_model import CAT_COLS, ENSEMBLE_SEEDS
from code.train import (
    apply_f1_filter, add_engineered_features, apply_te_residual_features,
    TE_RESIDUAL_COLS, TRACKMAN_TIER_FEED, build_season_end_lookup,
)
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix, PITCHMIX_COLS
from code.mlp_model import fit_preprocessing, fit_quantile_edges, to_tensors, get_device
from code.fm_model import train_deepfm

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FM_CACHE_DIR = "./open/temp/experiment_fm_tuned"
REF_FINAL_MODEL = "./submit/model/final_retained_model.pkl"
FULL_RETRAIN_EPOCH_BUFFER = 5

FM_CAT_COLS = CAT_COLS + ["pitcher_id", "batter_id"]


def build_full_train_df():
    """dopip.py Step 3와 완전히 동일한 피처 엔지니어링 (F1 필터, TE-residual, 시즌진행분,
    coarse pitchmix, holdout=2025/None). full_features/cat_feature_cols/num_cols도
    dopip.py와 동일하게 재현해 기존 catboost_model/mlp_bundle과 스키마가 일치하게 한다."""
    train_df_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    train_df_raw["top_bottom"] = train_df_raw["top_bottom"].map({"T": 0, "B": 1}).astype("int64")
    train_df = train_df_raw.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    train_df, trk_tier_cols = add_all_tiers(train_df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=2025)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=None)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    league_success_mean = train_df[TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)
    train_df = apply_f1_filter(train_df)

    te_prior = train_df[TARGET_COL].mean()
    train_df = apply_te_residual_features(train_df, train_df, te_prior)

    drop_cols = ["row_id", TARGET_COL]
    full_features = [c for c in train_df.columns if c not in drop_cols and c not in TE_RESIDUAL_COLS]
    num_cols = [c for c in full_features if c not in CAT_COLS and c not in trk_cat_cols]
    return train_df, num_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-cat", type=float, required=True)
    parser.add_argument("--w-mlp", type=float, required=True)
    parser.add_argument("--w-fm", type=float, required=True)
    parser.add_argument("--intercept", type=float, required=True)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    with open(os.path.join(FM_CACHE_DIR, "fm_bundle.pkl"), "rb") as f:
        fm_screen_bundle = fm_screen = pickle.load(f)
    per_seed_epochs = [m["best_epoch"] for m in fm_screen_bundle["members"]]
    seeds = [m["seed"] for m in fm_screen_bundle["members"]]
    print(f"[build_fm_3way_submit] cutoff7 스크리닝 best_epoch: {dict(zip(seeds, per_seed_epochs))}")

    print("[build_fm_3way_submit] 전체 데이터 피처 엔지니어링 재현 중 (dopip.py Step 3와 동일)...")
    t0 = time.time()
    train_df, mlp_num_cols = build_full_train_df()
    fm_num_cols = [c for c in mlp_num_cols if c not in ("pitcher_id", "batter_id")]
    print(f"[build_fm_3way_submit] 완료: {len(train_df)}행, {time.time()-t0:.1f}s")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        train_df[FM_CAT_COLS + fm_num_cols + [TARGET_COL]], FM_CAT_COLS, fm_num_cols,
    )
    X_full_cat, X_full_num, y_full = to_tensors(train_proc, FM_CAT_COLS, fm_num_cols, TARGET_COL)
    bin_edges = fit_quantile_edges(X_full_num)
    device = get_device()
    print(f"[Device] {device}")

    full_members = []
    for seed, base_epoch in zip(seeds, per_seed_epochs):
        full_epochs = max(base_epoch, 1) + FULL_RETRAIN_EPOCH_BUFFER
        t0 = time.time()
        model, _ = train_deepfm(
            X_full_cat, X_full_num, y_full, cat_dims=cat_dims, bin_edges=bin_edges, k=args.k,
            max_epochs=full_epochs, device=device, seed=seed, verbose=False,
            embed_dropout=fm_screen_bundle["embed_dropout"],
            high_card_cat_idx=fm_screen_bundle["high_card_cat_idx"],
            high_card_weight_decay=fm_screen_bundle["high_card_weight_decay"],
        )
        full_members.append({
            "state_dict": {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()},
            "best_epoch": full_epochs, "seed": seed,
        })
        print(f"[Full Retrain][DeepFM] seed={seed} {full_epochs} epoch 학습 완료 ({time.time()-t0:.1f}s)")

    fm_bundle = {
        "members": full_members, "cat_cols": FM_CAT_COLS, "num_cols": fm_num_cols,
        "cat_dims": cat_dims, "k": args.k, "bin_edges": bin_edges,
        "cat_encoder": cat_encoder, "num_imputer": num_imputer, "num_scaler": num_scaler,
    }

    with open(REF_FINAL_MODEL, "rb") as f:
        base_bundle = pickle.load(f)

    backup_path = REF_FINAL_MODEL.replace(".pkl", "_2way_backup.pkl")
    if not os.path.exists(backup_path):
        shutil.copy2(REF_FINAL_MODEL, backup_path)
        print(f"[백업] 기존 2-way final_retained_model.pkl -> {backup_path}")

    final_bundle = dict(base_bundle)
    final_bundle["fm_bundle"] = fm_bundle
    final_bundle["meta_model"] = {"w_cat": args.w_cat, "w_mlp": args.w_mlp, "w_fm": args.w_fm, "intercept": args.intercept}

    with open(REF_FINAL_MODEL, "wb") as f:
        pickle.dump(final_bundle, f)
    print(f"[build_fm_3way_submit] 3-way 번들 저장 완료: {REF_FINAL_MODEL} (meta_model={final_bundle['meta_model']})")


if __name__ == "__main__":
    main()
