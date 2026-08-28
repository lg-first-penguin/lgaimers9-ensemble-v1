# code/experiment_pair_matchup_v2.py
"""code/experiment_pair_matchup.py의 수정판. 첫 채택판(실전 1028.75, -12.65 회귀)의
버그를 고친 뒤 재검증한다 — 버그: train_mask/val_mask로 나누기 전에 전체 df에
apply_pair_matchup_rowlevel을 한 번에 돌려서, val_split의 각 행이 "같은 시즌 앞부분에
이미 만난" 페어까지 누적 커버리지에 포함시켰다(로컬 pair_n>0 93.8%/95.3%). 그런데 실제
제출은 2024년 말에 얼린 정적 lookup을 2025 test.csv에 그대로 병합만 하므로 이런
"같은 시즌 내 누적"이 구조적으로 존재할 수 없다(실측: season==2024/2023 각각 51.16%/
54.10%만 "그 시즌 이전에 이미 만난 적 있음" — 로컬 검증과 실제 15%p 이상 차이).

이 버전은 code/train.py의 수정된 apply_pair_matchup_rowlevel(train_split 전용, +
PAIR_K_SMOOTH=20 empirical-Bayes 축소)/build_pair_matchup_lookup/apply_pair_matchup_static
(val_split엔 train_split 기준 정적 lookup만 적용, TE-residual과 동일 패턴)을 그대로
가져와 쓴다 — 로컬 검증도 실전과 동일한 "얼린 lookup" 방식으로 val을 만든다.

사용법: python -m code.experiment_pair_matchup_v2 [--holdout 2023|2024] [--cutoff7]
"""
import argparse
import os

import numpy as np
import pandas as pd

from code.mlp_model import (
    CAT_COLS, QUANTILE_D, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model
from code.train import (
    TE_RESIDUAL_COLS, TRACKMAN_TIER_FEED, add_engineered_features, apply_f1_filter,
    apply_te_residual_features, apply_pair_matchup_rowlevel, build_pair_matchup_lookup,
    apply_pair_matchup_static, PAIR_COLS,
)
from code.trackman_pitcher_features import add_all_tiers, clean_trackman, merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]

REF_CAT = 706.56
REF_MLP7 = 738.55
REF_BLEND = 753.37


def build_split_with_pair(cutoff7=True, holdout=2024, apply_f1=True):
    """thirdmodel_common.build_split과 동일하되, pair-matchup은 train_split만 causal
    source로 쓰고 val_split엔 정적 lookup을 적용한다(실전과 동일 방식)."""
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

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=trk_holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]

    df = merge_coarse_pitchmix(df, df_trm, holdout=trk_holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in df.columns if c not in drop_cols]
    mlp_num_cols = [c for c in features if c not in CAT_COLS and c not in trk_cat_cols and c not in trk_mlp_cols]
    cat_feature_cols = [c for c in features if c not in trk_mlp_cols]

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    if apply_f1:
        before = len(train_split)
        train_split = apply_f1_filter(train_split)
        print(f"[F1 필터] {before} -> {len(train_split)}행")

    te_prior = train_split[TARGET_COL].mean()
    train_split = apply_te_residual_features(train_split, train_split, te_prior)
    val_split = apply_te_residual_features(train_split, val_split, te_prior)
    cat_feature_cols = cat_feature_cols + TE_RESIDUAL_COLS

    # 페어 매치업: train_split만 causal, val_split은 그 train_split 기준 정적 lookup(실전과 동일)
    train_split = apply_pair_matchup_rowlevel(train_split)
    pair_lookup = build_pair_matchup_lookup(train_split)
    val_split = apply_pair_matchup_static(val_split, pair_lookup)

    print(f"[build_split_with_pair v2] train={len(train_split)} val={len(val_split)} "
          f"mlp_num_cols={len(mlp_num_cols)} cat_feature_cols={len(cat_feature_cols)}")
    return train_split, val_split, mlp_num_cols, cat_feature_cols


def report_coverage(train_split, val_split):
    tr_cov = (train_split["pair_n"] > 0).mean()
    val_cov = (val_split["pair_n"] > 0).mean()
    print(f"[커버리지-v2(실전과 동일한 얼린 lookup 기준)] train: pair_n>0 비율={tr_cov:.4f} "
          f"({(train_split['pair_n']>0).sum()}/{len(train_split)})")
    print(f"[커버리지-v2] val: pair_n>0 비율={val_cov:.4f} ({(val_split['pair_n']>0).sum()}/{len(val_split)}) "
          f"<- 첫 채택판은 여기가 부풀려져 있었음(cutoff7 93.8%/95.3%)")
    return tr_cov, val_cov


def run_one(name, train_split, val_split, mlp_num_cols, cat_feature_cols):
    print(f"\n{'='*20} 실험: {name} {'='*20}")
    print(f"mlp_num_cols={len(mlp_num_cols)} cat_feature_cols={len(cat_feature_cols)}")

    X_train_raw, y_train_raw = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[cat_feature_cols], val_split[TARGET_COL].values
    cat_model, cat_best_iter = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_val_preds = predict_catboost(cat_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    print(f"[CatBoost] best_iter={cat_best_iter} score={cat_score:.2f} (ref={REF_CAT:.2f}, delta={cat_score-REF_CAT:+.2f})")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    device = get_device()

    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=val_proc[TARGET_COL].values,
        seeds=SCREEN_SEEDS, device=device,
    )
    ens_pred = predict_ensemble(
        members, cat_dims, len(mlp_num_cols), embed_dims,
        X_val_cat, X_val_num, bin_edges=bin_edges, quantile_d=QUANTILE_D, device=device,
    )
    mlp_score = compute_bss(ens_pred, y_val_raw)[2]
    print(f"[MLP {len(SCREEN_SEEDS)}-seed] score={mlp_score:.2f} (ref 7-seed={REF_MLP7:.2f}, delta={mlp_score-REF_MLP7:+.2f})")

    w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_val_preds, ens_pred, y_val_raw)
    print(f"[2-way blend] score={blend_score:.2f} (ref={REF_BLEND:.2f}, delta={blend_score-REF_BLEND:+.2f}) "
          f"weights cat={w_cat:.3f} mlp={w_mlp:.3f} intercept={intercept:.3f}")
    return dict(name=name, cat_score=cat_score, mlp_score=mlp_score, blend_score=blend_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()

    regime = "cutoff7" if args.cutoff7 else f"season=={args.holdout}"
    train_split, val_split, base_mlp_cols, base_cat_cols = build_split_with_pair(
        cutoff7=args.cutoff7, holdout=args.holdout,
    )
    report_coverage(train_split, val_split)

    results = []
    results.append(run_one(
        "baseline(pair 미포함)", train_split, val_split, base_mlp_cols, base_cat_cols,
    ))
    results.append(run_one(
        "pair -> MLP", train_split, val_split, base_mlp_cols + PAIR_COLS, base_cat_cols,
    ))
    results.append(run_one(
        "pair -> CatBoost", train_split, val_split, base_mlp_cols, base_cat_cols + PAIR_COLS,
    ))
    results.append(run_one(
        "pair -> both", train_split, val_split, base_mlp_cols + PAIR_COLS, base_cat_cols + PAIR_COLS,
    ))

    print(f"\n{'='*20} 요약 v2 ({regime}) {'='*20}")
    base = results[0]
    for r in results:
        d_cat, d_mlp, d_blend = r["cat_score"] - base["cat_score"], r["mlp_score"] - base["mlp_score"], r["blend_score"] - base["blend_score"]
        print(f"{r['name']:20s}: CatBoost={r['cat_score']:.2f}({d_cat:+.2f}) "
              f"MLP(3-seed)={r['mlp_score']:.2f}({d_mlp:+.2f}) blend={r['blend_score']:.2f}({d_blend:+.2f})")


if __name__ == "__main__":
    main()
