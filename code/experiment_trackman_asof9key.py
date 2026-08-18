# code/experiment_trackman_asof9key.py
"""트랙맨 병합을 'season 컷오프 + 9-key 매칭'으로 재설계하는 실험.

배경: `code/experiment_trackman_ablation.py`에서 트랙맨 피처를 아예 드롭한 게
season-drop fallback 패치(752.42)보다도 나았다(795.13). 원인은 학습/서빙 분포 불일치
(train-serve skew) — 학습 시점엔 항상 season이 정확히 매칭되는 "날카로운" 분포를 보고,
서빙 시점엔 항상 season 매칭에 실패해 9-key로 재매칭된 "뭉뚱그려진" 분포를 보게 되어,
모델이 학습한 것과 실제로 받는 입력의 의미가 어긋난다.

이 실험은 애초에 학습 때도 서빙 때와 "같은 종류"의 분포를 보게 만들어 skew 자체를
없앤다: 매칭 키에서 season을 아예 빼고(9-key), 대신 시간 순서를 지키기 위해 각 행의
season 이하(<=) 트랙맨 데이터만 사용하는 asof 누적 컷오프를 건다.

- 학습 행(season=S<2024): trackman_history의 season<=S인 행만으로 9-key 룩업을 만들어
  매칭 (이후 시즌 정보를 보지 않으므로 리크 없음).
- 검증 행(season=2024): trackman_history의 season<=2024, 즉 전체(2019~2024)로 매칭.
- 실제 배포 행(season=2025, 트랙맨 커버리지 밖): trackman_history의 season<=2025도
  사실상 전체(2019~2024)로 매칭 — 검증 행과 정확히 같은 조건이다.

즉 이 설계는 검증(2024)과 실제 배포(2025)가 "asof 컷오프 = 전체 트랙맨 히스토리"로
동일하게 수렴하기 때문에, 이전 실험처럼 trackman_history에서 season==2024를 인위적으로
지워 "블라인드 상황"을 흉내 낼 필요가 없다 — 로컬 val 점수가 곧 실제 배포 조건과
동일한 매칭 분포를 이미 재현한다.

사용법:
  python -m code.experiment_trackman_asof9key
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from code.blend_model import fit_meta_model, predict_meta
from code.catboost_model import predict_catboost, train_catboost
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device,
    make_bundle, predict_bundle, to_tensors, train_ensemble,
)
from code.train import add_engineered_features

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
CACHE_PATH = "./open/temp/experiment_trackman_asof9key_bundle.pkl"


def build_lookup(df_trm, cols):
    """process_trackman_features_safe와 동일한 집계/피벗 로직. cols가 곧 groupby 매칭 키."""
    grouped = df_trm.groupby(cols + ["pitch_type_group", "auto_pitch_type"])
    g1 = grouped[["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
                  "extension", "rel_height", "rel_side", "zone_speed"]].agg(["mean", "std"])
    g2 = g1.reset_index()
    std_cols = [c for c in g2.columns if "std" in c]
    g2[std_cols] = g2[std_cols].fillna(0)
    g2.columns = ["_".join(c).strip("_") for c in g2.columns]
    g3 = g2.drop(columns="auto_pitch_type")
    g3 = g3.groupby(cols + ["pitch_type_group"]).agg(["mean"])
    pivoted = g3.unstack(level="pitch_type_group")
    pivoted.columns = [f"{c[0]}_{c[1]}_{c[2]}" for c in pivoted.columns]
    tm_final = pivoted.reset_index().fillna(0)
    if "top_bottom" in tm_final.columns:
        tm_final["top_bottom"] = tm_final["top_bottom"].map({"Top": 0, "Bottom": 1}).astype(np.int64)
    for col in ["batter_hand", "pitcher_hand"]:
        if col in tm_final.columns:
            tm_final[col] = tm_final[col].map({"Left": 1, "Right": 2}).astype(np.int64)
    return tm_final


def merge_asof_trackman(df_main, df_trm, key_cols):
    """행의 season 이하(<=)로만 컷오프한 트랙맨 데이터를, season을 뺀 key_cols로 매칭.

    df_main의 고유 season 값마다 컷오프된 트랙맨 부분집합으로 룩업을 새로 만들어 병합한다
    (season 수가 적어 — 최대 6개 — 루프 비용은 무시할 만한 수준).
    """
    df_main_copy = df_main.copy()
    df_main_copy["top_bottom"] = df_main_copy["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    pieces = []
    feature_cols = None
    for season in sorted(df_main_copy["season"].unique()):
        trm_cut = df_trm[df_trm["season"] <= season]
        lookup = build_lookup(trm_cut, key_cols)
        if feature_cols is None:
            feature_cols = [c for c in lookup.columns if c not in key_cols]
        rows = df_main_copy[df_main_copy["season"] == season]
        merged = pd.merge(rows, lookup, on=key_cols, how="left")
        pieces.append(merged)

    result = pd.concat(pieces, ignore_index=True)
    result[feature_cols] = result[feature_cols].fillna(0)
    return result, feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024, choices=[2023, 2024])
    parser.add_argument("--apply-f1", dest="apply_f1", action="store_true", default=True)
    parser.add_argument("--no-f1", dest="apply_f1", action="store_false")
    args = parser.parse_args()
    label = f"holdout={args.holdout} f1={'ON' if args.apply_f1 else 'OFF'}"

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")

    full_match_cols = [c for c in df.columns if c in df_trm.columns and c != "row_id"]
    key_cols = [c for c in full_match_cols if c != "season"]
    print(f"[{label}] asof 컷오프 + 9-key 매칭: {key_cols}")

    t0 = time.time()
    df, feature_cols = merge_asof_trackman(df, df_trm, key_cols)
    print(f"[{label}] 트랙맨 asof 병합 완료 ({time.time() - t0:.1f}s) | 파생 피처 수: {len(feature_cols)}")

    train_mask = df["season"] < args.holdout
    val_mask = df["season"] == args.holdout
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in df.columns if c not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS]
    print(f"[{label}] 총 피처 수: {len(features)} (범주형 {len(CAT_COLS)}, 수치형 {len(num_cols)})")

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)

    if args.apply_f1:
        before = len(train_split)
        train_split = train_split[~((train_split["game_type"] == "F") & (train_split["season"] <= 2022))].reset_index(drop=True)
        print(f"[{label}] F1 필터 적용: {before} -> {len(train_split)}행 ({before - len(train_split)}행 제거)")

    print(f"[{label}] 훈련 데이터: {len(train_split)}행 | 검증 데이터: {len(val_split)}행")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_val_np = val_proc[TARGET_COL].values

    device = get_device()
    print(f"[Device] {device}")

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
        seeds=ENSEMBLE_SEEDS, device=device,
    )
    print(f"[{label}] MLP 앙상블 학습 완료 ({time.time() - t0:.1f}s)")

    mlp_bundle = make_bundle(
        members, CAT_COLS, num_cols, cat_dims, embed_dims,
        cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges,
    )

    X_train_raw, y_train_raw = train_split[features], train_split[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[features], val_split[TARGET_COL].values
    t0 = time.time()
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    print(f"[{label}] CatBoost 학습 완료 (best_iteration={catboost_best_iteration}, {time.time() - t0:.1f}s)")

    mlp_val_preds = predict_bundle(mlp_bundle, X_val_raw, device=device)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)

    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    mlp_score = compute_bss(mlp_val_preds, y_val_raw)[2]
    w_cat, w_mlp, intercept, blend_score, blend_brier = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)

    print(f"\n[RESULT {label}] CatBoost={cat_score:.2f} | MLP={mlp_score:.2f} | Blend={blend_score:.2f}")
    print(f"  (w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f})")

    cache_path = f"./open/temp/experiment_trackman_asof9key_h{args.holdout}_f1{'on' if args.apply_f1 else 'off'}_bundle.pkl"
    os.makedirs("./open/temp", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "catboost_model": catboost_model, "mlp_bundle": mlp_bundle,
            "meta_model": {"w_cat": w_cat, "w_mlp": w_mlp, "intercept": intercept},
            "catboost_best_iteration": catboost_best_iteration,
            "cat_score": cat_score, "mlp_score": mlp_score, "blend_score": blend_score,
        }, f)
    print(f"\n번들 캐시 저장: {cache_path}")


if __name__ == "__main__":
    main()
