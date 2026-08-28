# code/experiment_batter_trackman_identity.py
"""tier A(투수 정체성 x 구종군별 트랙맨 물리 지표 mean/std, ->MLP)의 타자판을 시도한다 —
"타자 identity 기반 trackman 집계"는 이 프로젝트에서 아직 시도된 적 없는 조합.

주의(사전 검토에서 제기된 우려, 이 실험으로 데이터 확인): `control_success`는 투수의 제구
실행 결과이지 타자의 스윙/결과가 아니다. "이 타자가 과거에 상대한 공들의 물리 특성"은
사실 "이 타자가 상대해온 투수들의 구위 수준"을 대리하는 간접 신호일 뿐이라, 투수 identity판
(tier A)보다 신호 경로가 한 단계 더 간접적이다. tier A 자체도 로컬에서는 통과했지만 실전
리더보드에서 -31.41로 무너진 전례가 있어(A+coarse pitchmix 조합, 950.81) 같은 실패 반복
위험이 있다 — 그래서 프로덕션 반영 전 dual-regime CatBoost+MLP 블렌드 스크리닝부터 한다.

크로스워크는 code/pitcher_crosswalk.py가 이미 만들어 둔 ./open/temp/batter_map.csv를 쓴다
(506명, train batter_id 커버리지는 pitcher_map과 비슷한 수준 — 외부 데이터 아님).
tier A와 동일하게 그룹핑 축은 구종군(pitch_type_group)만 쓰고(추가 축 없음), ->MLP만 피딩.

사용법:
  python -m code.experiment_batter_trackman_identity --holdout 2023
  python -m code.experiment_batter_trackman_identity --cutoff7
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
from code.trackman_pitcher_features import METRICS, clean_trackman

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]
BATTER_PREFIX = "trkstdBatA_"


def build_batter_lookup(df_trm_clean, batter_map):
    """build_pitcher_lookup(code/trackman_pitcher_features.py)의 타자판: 추가 그룹핑 축
    없이 batter_id x pitch_type_group별 mean/std만 wide pivot."""
    merged = df_trm_clean.merge(batter_map[["batter_trackman_id", "batter_id"]],
                                 on="batter_trackman_id", how="inner")
    group_cols = ["batter_id", "pitch_type_group"]

    g = merged.groupby(group_cols)[METRICS].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    std_cols = [c for c in g.columns if c.endswith("_std")]
    g[std_cols] = g[std_cols].fillna(0.0)

    pivoted = g.set_index(group_cols).unstack(level="pitch_type_group")
    pivoted.columns = ["_".join(str(x) for x in c) for c in pivoted.columns]
    pivoted = pivoted.reset_index().fillna(0.0)
    rename = {c: BATTER_PREFIX + c for c in pivoted.columns if c != "batter_id"}
    return pivoted.rename(columns=rename)


def merge_asof_batter_std(df_main, df_trm_clean, batter_map, holdout):
    """merge_asof_pitcher_std와 동일한 as-of 로직(시즌별 cutoff=min(season, holdout-1))을
    batter_id 축으로 반복."""
    pieces = []
    feature_cols = None
    for season in sorted(df_main["season"].unique()):
        cutoff_season = min(season, holdout - 1)
        trm_cut = df_trm_clean[df_trm_clean["season"] <= cutoff_season]
        lookup = build_batter_lookup(trm_cut, batter_map)
        if feature_cols is None:
            feature_cols = [c for c in lookup.columns if c != "batter_id"]
        rows = df_main[df_main["season"] == season]
        merged = pd.merge(rows, lookup, on="batter_id", how="left")
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True)
    result[feature_cols] = result[feature_cols].fillna(0.0)
    return result, feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024, choices=[2023, 2024])
    parser.add_argument("--apply-f1", dest="apply_f1", action="store_true", default=True)
    parser.add_argument("--no-f1", dest="apply_f1", action="store_false")
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    if args.cutoff7:
        args.holdout = 2024
    label = f"BATTER-IDENTITY(->MLP) {'cutoff7' if args.cutoff7 else 'holdout=' + str(args.holdout)} f1={'ON' if args.apply_f1 else 'OFF'}"

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    batter_map = pd.read_csv("./open/temp/batter_map.csv")
    df_trm_full = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm_full)
    t0 = time.time()
    df, trk_mlp_cols = merge_asof_batter_std(df, df_trm_clean, batter_map, holdout=args.holdout)
    print(f"[{label}] batter-std 병합 완료 ({time.time() - t0:.1f}s) | 피처 수: {len(trk_mlp_cols)}")

    if args.cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
    else:
        train_mask = df["season"] < args.holdout
        val_mask = df["season"] == args.holdout
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in trk_mlp_cols]
    features = base_features + trk_mlp_cols
    cat_features = base_features  # CatBoost는 두 variant 모두 baseline 그대로 (신규 피처 미피딩)

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)

    if args.apply_f1:
        before = len(train_split)
        train_split = train_split[~((train_split["game_type"] == "F") & (train_split["season"] <= 2022))].reset_index(drop=True)
        print(f"[{label}] F1 필터 적용: {before} -> {len(train_split)}행")

    print(f"[{label}] 훈련: {len(train_split)}행 | 검증: {len(val_split)}행 | 신규 피처 수: {len(trk_mlp_cols)}")

    X_train_raw, y_train_raw = train_split[cat_features], train_split[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[cat_features], val_split[TARGET_COL].values
    t0 = time.time()
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    print(f"[{label}] CatBoost 완료 (best_iteration={catboost_best_iteration}, {time.time() - t0:.1f}s)")
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]

    device = get_device()
    results = {}
    for tag, mlp_extra in [("baseline", []), ("+batter_identity(->MLP)", trk_mlp_cols)]:
        mlp_num_cols = [c for c in base_features if c not in CAT_COLS] + mlp_extra
        mlp_features = base_features + mlp_extra

        train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
        val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)

        X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
        X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
        y_val_np = val_proc[TARGET_COL].values

        embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
        bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

        t0 = time.time()
        members = train_ensemble(
            X_tr_cat, X_tr_num, y_tr,
            cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
            seeds=SCREEN_SEEDS, device=device,
        )
        print(f"[{label}][{tag}] MLP({len(SCREEN_SEEDS)}-seed) 학습 완료 ({time.time() - t0:.1f}s)")

        mlp_bundle = make_bundle(
            members, CAT_COLS, mlp_num_cols, cat_dims, embed_dims,
            cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges,
        )
        mlp_val_preds = predict_bundle(mlp_bundle, val_split[mlp_features], device=device)

        mlp_score = compute_bss(mlp_val_preds, y_val_raw)[2]
        w_cat, w_mlp, intercept, blend_score, blend_brier = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)
        print(f"[RESULT {label}][{tag}] CatBoost={cat_score:.2f} | MLP={mlp_score:.2f} | Blend={blend_score:.2f}")
        results[tag] = (mlp_score, blend_score)

    base_mlp, base_blend = results["baseline"]
    new_mlp, new_blend = results["+batter_identity(->MLP)"]
    print(f"\n{'='*70}\n[{label}] delta: MLP {new_mlp-base_mlp:+.2f} | Blend {new_blend-base_blend:+.2f}\n{'='*70}")


if __name__ == "__main__":
    main()
