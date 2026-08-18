# code/experiment_coarse_pitchmix.py
"""팀원의 XGBoost 트랙에서 보고된 "coarse pitchmix" 피처를 우리 CatBoost+MLP 파이프라인
(cutoff=7 프로덕션 스플릿, F1 필터)으로 독립 검증한다.

배경(팀원 리포트, 2026-08-17 세션에서 전달받음): trackman_history.csv를
(balls_before, strikes_before, pitcher_hand, batter_hand) 4축으로 그룹핑해 그룹별
pitch_type_group(fastball/breaking/offspeed/other) 비율을 계산, 그 비율 4개를 새 피처로
원본 행에 상황 조합 기준으로 병합한다. 투수 정체성이나 season을 조인 키로 쓰지 않는다는
점이 핵심 — 이번 세션 내내 실패한 tier A/B/C(크로스워크 커버리지 문제)나 asof9key(season
불일치 문제)와 달리 이 두 실패 원인을 구조적으로 피해간다. 팀원 쪽 결과(2023+2024 이중검증,
리크 수정판): XGBoost +38.20, CatBoost +25.16, MLP +0.47(거의 무변화, MLP는 이미
balls/strikes/hand를 원재료로 받아 상호작용을 스스로 학습 가능한 것으로 추정). 실전
리더보드 927(구종비중만 적용, F1 필터 없이, XGBoost 트랙 기준).

리크 방지: 팀원 리포트에 "검증 대상 시즌까지 포함한 전체 데이터를 넘기던" 리크 버그가
있었다고 명시돼 있다. 이 스크립트는 처음부터 val 시즌(cutoff=7이면 season==2024) 자기
자신의 트랙맨은 pitchmix 테이블 계산에서 제외한다(trackman season < holdout만 사용) —
프로젝트의 다른 asof 피처들과 동일한 관례. 실제 제출(Full Retrain) 단계에서는 이 테이블을
trackman_history.csv 전체(2019~2024)로 다시 계산하게 된다 — 실제 평가 시즌(2025)은
애초에 trackman에 없으므로 이 구분 자체가 asof 피처들과 달리 시즌 무관한 정적 테이블이라
학습/서빙 분포 불일치 문제가 없다(§38의 근본 실패 원인 중 하나였던 season 조인 구조적
불일치가 애초에 발생할 수 없는 설계).

사용법:
  python -m code.experiment_coarse_pitchmix --cutoff7 --feed-to cat
  python -m code.experiment_coarse_pitchmix --cutoff7 --feed-to both
  python -m code.experiment_coarse_pitchmix --holdout 2023 --feed-to cat
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

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]
COARSE_COLS = ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]
PITCHMIX_COLS = ["coarse_pitchmix_fastball", "coarse_pitchmix_breaking",
                  "coarse_pitchmix_offspeed", "coarse_pitchmix_other"]


def compute_coarse_pitchmix(df_trm):
    """(balls_before, strikes_before, pitcher_hand, batter_hand) 조합별 pitch_type_group
    비율을 계산해 wide 포맷으로 반환한다. 결측 조합 대비 전체 평균(global fallback)도 함께
    반환한다."""
    counts = df_trm.groupby(COARSE_COLS + ["pitch_type_group"]).size().reset_index(name="n")
    totals = counts.groupby(COARSE_COLS)["n"].transform("sum")
    counts["ratio"] = counts["n"] / totals
    pivoted = counts.pivot_table(
        index=COARSE_COLS, columns="pitch_type_group", values="ratio", fill_value=0.0,
    ).reset_index()
    pivoted = pivoted.rename(columns={
        "fastball": "coarse_pitchmix_fastball", "breaking": "coarse_pitchmix_breaking",
        "offspeed": "coarse_pitchmix_offspeed", "other": "coarse_pitchmix_other",
    })
    for c in PITCHMIX_COLS:
        if c not in pivoted.columns:
            pivoted[c] = 0.0
    global_ratio = df_trm["pitch_type_group"].value_counts(normalize=True)
    fallback = {
        "coarse_pitchmix_fastball": global_ratio.get("fastball", 0.0),
        "coarse_pitchmix_breaking": global_ratio.get("breaking", 0.0),
        "coarse_pitchmix_offspeed": global_ratio.get("offspeed", 0.0),
        "coarse_pitchmix_other": global_ratio.get("other", 0.0),
    }
    return pivoted[COARSE_COLS + PITCHMIX_COLS], fallback


def merge_coarse_pitchmix(df_main, df_trm, holdout):
    """val 시즌(holdout) 자기 자신의 트랙맨은 테이블 계산에서 제외한다(리크 방지).
    holdout=None이면 전체 trackman(2019~2024)을 그대로 쓴다(실전 Full Retrain과 동일)."""
    trm_cut = df_trm if holdout is None else df_trm[df_trm["season"] < holdout]
    lookup, fallback = compute_coarse_pitchmix(trm_cut)
    merged = pd.merge(df_main, lookup, on=COARSE_COLS, how="left")
    for c in PITCHMIX_COLS:
        merged[c] = merged[c].fillna(fallback[c])
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024, choices=[2023, 2024])
    parser.add_argument("--feed-to", type=str, default="cat", choices=["both", "cat", "mlp"])
    parser.add_argument("--apply-f1", dest="apply_f1", action="store_true", default=True)
    parser.add_argument("--no-f1", dest="apply_f1", action="store_false")
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    if args.cutoff7:
        args.holdout = 2024
    label = f"PITCHMIX {'cutoff7' if args.cutoff7 else 'holdout=' + str(args.holdout)} feed={args.feed_to} f1={'ON' if args.apply_f1 else 'OFF'}"

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig",
                          usecols=["season"] + COARSE_COLS + ["pitch_type_group"])
    # train.csv는 pitcher_hand/batter_hand를 정수코드(1/2)로 익명화해서 제공하지만
    # trackman_history.csv는 "Right"/"Left" 문자열 그대로다. 두 코드 공간이 겹치지 않는
    # pitcher_id/trackman_id(EXPERIMENTS.md 14.1/25.1)와 달리 손은 값이 2개뿐이라
    # 크로스워크(pitcher_map.csv/batter_map.csv)로 방향을 역산할 수 있었다: train 1<->Left,
    # 2<->Right (투수 568명 중 1명, 타자 506명 중 1명만 불일치 — 스위치히터 등 노이즈로 간주).
    hand_map = {"Left": 1, "Right": 2}
    df_trm["pitcher_hand"] = df_trm["pitcher_hand"].map(hand_map)
    df_trm["batter_hand"] = df_trm["batter_hand"].map(hand_map)
    t0 = time.time()
    df = merge_coarse_pitchmix(df, df_trm, holdout=args.holdout)
    print(f"[{label}] pitchmix 병합 완료 ({time.time() - t0:.1f}s)")

    if args.cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
    else:
        train_mask = df["season"] < args.holdout
        val_mask = df["season"] == args.holdout
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in PITCHMIX_COLS]
    features = base_features + PITCHMIX_COLS

    cat_features = base_features + (PITCHMIX_COLS if args.feed_to in ("both", "cat") else [])
    mlp_num_cols = [c for c in base_features if c not in CAT_COLS] + \
        (PITCHMIX_COLS if args.feed_to in ("both", "mlp") else [])
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
