# code/experiment_trackman_nanfill.py
"""§38.7 사후조사 후속 실험: 트랙맨 pitcher-std/pressure 재도전(§38)이 실제 리더보드에서
869.52로 대폭 하락한 것과 별개로, §38.3 스크리닝 표를 재검토한 결과 season==2023 홀드아웃
(pre-ABS)에서는 채택된 3개 조합(A->mlp, B->cat, C->cat)이 전부 마이너스였다는 사실이
드러났다(-3.48/-10.48/-10.95) — 당시 이걸 "2023은 ABS 이전 레짐이라 다른 게 당연하다"는
가설로 기각하고 cutoff=7(2024) 결과만으로 프로덕션에 반영했는데, 실제 리더보드가 이 가설을
반증했다.

이 스크립트는 별개의 가설을 하나 더 검증한다: `merge_asof_pitcher_std`가 크로스워크에 아예
없는(신인/미매칭) 투수의 trkstd 피처를 전부 0.0으로 채우는데, 이 0이 "실제로 낮은 값"과
"데이터 없음"을 구분 못 해 CatBoost/MLP 양쪽에 잘못된 신호를 준다는 가설. 0-fill을 제거해
NaN을 그대로 흘려보내면 CatBoost는 네이티브 NaN 처리(분할 방향 학습)를, MLP 쪽은
`fit_preprocessing`의 `SimpleImputer(median)`(다른 모든 수치형 피처와 동일한 처리)을
받는다. season==2023 홀드아웃에서 이 변경만으로 델타가 뒤집히는지 확인한다 — 안 뒤집히면
0-fill이 원인이 아니라 트랙맨 자체(커버된 투수 포함)가 pre-ABS 레짐에서 무신호/역신호라는
쪽에 더 무게가 실린다.

`code/trackman_pitcher_features.py`/`code/experiment_trackman_pitcherstd_pressure.py`와
달리, `build_pitcher_lookup`의 두 fillna(단일 투구 버킷 std=NaN->0, 커버된 투수의 일부
피벗 셀 결측->0)는 그대로 둔다 — 이번 실험 대상은 `merge_asof_pitcher_std` 마지막의
"크로스워크에 아예 없는 투수 전체 행"에 대한 fillna(0.0)뿐이다.

사용법:
  python -m code.experiment_trackman_nanfill --holdout 2023 --tier a --feed-to mlp
  python -m code.experiment_trackman_nanfill --holdout 2023 --tier b --feed-to cat
  python -m code.experiment_trackman_nanfill --holdout 2023 --tier c --feed-to cat
  python -m code.experiment_trackman_nanfill --cutoff7 --tier c --feed-to cat
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
from code.trackman_pitcher_features import clean_trackman, build_pitcher_lookup

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]


def merge_asof_pitcher_std_nanfill(df_main, df_trm_clean, pitcher_map, tier, holdout):
    """merge_asof_pitcher_std와 동일하나, 크로스워크에 없는 투수(pitcher_map 미매칭 또는
    train.csv에 아예 없던 신인)의 trkstd 피처를 0.0이 아니라 NaN으로 남겨둔다."""
    pieces = []
    feature_cols = None
    for season in sorted(df_main["season"].unique()):
        cutoff_season = min(season, holdout - 1)
        trm_cut = df_trm_clean[df_trm_clean["season"] <= cutoff_season]
        lookup = build_pitcher_lookup(trm_cut, pitcher_map, tier)
        if feature_cols is None:
            feature_cols = [c for c in lookup.columns if c != "pitcher_id"]
        rows = df_main[df_main["season"] == season]
        merged = pd.merge(rows, lookup, on="pitcher_id", how="left")
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True)
    # 의도적으로 fillna(0.0) 생략 — NaN을 그대로 남겨 CatBoost 네이티브 처리 /
    # MLP SimpleImputer(median) 처리에 맡긴다.
    return result, feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024, choices=[2023, 2024])
    parser.add_argument("--tier", type=str, required=True, choices=["a", "b", "c", "f"])
    parser.add_argument("--feed-to", type=str, required=True, choices=["both", "cat", "mlp"])
    parser.add_argument("--apply-f1", dest="apply_f1", action="store_true", default=True)
    parser.add_argument("--no-f1", dest="apply_f1", action="store_false")
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    if args.cutoff7:
        args.holdout = 2024
    label = f"NANFILL {'cutoff7' if args.cutoff7 else 'holdout=' + str(args.holdout)} tier={args.tier} feed={args.feed_to} f1={'ON' if args.apply_f1 else 'OFF'}"

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    t0 = time.time()
    df, trk_feature_cols = merge_asof_pitcher_std_nanfill(df, df_trm_clean, pitcher_map, args.tier, args.holdout)
    cov = df[trk_feature_cols[0]].notna().mean() if trk_feature_cols else 0.0
    print(f"[{label}] 트랙맨 병합 완료 ({time.time() - t0:.1f}s) | 파생 피처 수: {len(trk_feature_cols)} | non-NaN 커버리지: {cov:.1%}")

    if args.cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
    else:
        train_mask = df["season"] < args.holdout
        val_mask = df["season"] == args.holdout
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in trk_feature_cols]
    features = base_features + trk_feature_cols

    cat_features = base_features + (trk_feature_cols if args.feed_to in ("both", "cat") else [])
    mlp_num_cols = [c for c in base_features if c not in CAT_COLS] + \
        (trk_feature_cols if args.feed_to in ("both", "mlp") else [])
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
