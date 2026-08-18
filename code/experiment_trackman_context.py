# code/experiment_trackman_context.py
"""트랙맨을 '배경 상황(context)' 피처로 재도입하는 실험 — 정밀 매칭이 아닌 뭉뚱그린 집계.

이전 시도(`code/experiment_trackman_asof9key.py`)는 game_month/inning/top_bottom까지
포함한 9-key로 asof 누적 매칭했고, season==2024 홀드아웃에서는 +8.91점 이득처럼
보였지만 season==2023 홀드아웃에서는 -29.11점 손해로 뒤집혀 평균 -1.35 — 즉 신호가
아니라 노이즈였다(EXPERIMENTS.md §17.2/§17.3, PROJECT_HISTORY.md §14-§16).

이번 실험은 설계를 근본적으로 다르게 한다:
  - 매칭 키를 훨씬 성글게 줄인다: [season_phase(초/중/후반), balls_before,
    strikes_before, outs_before, pitcher_hand, batter_hand] 6-key만 사용.
    game_month를 그대로 쓰지 않고 3구간(early=3~5월, mid=6~8월, late=9~11월)으로
    뭉뚱그린 이유는 이렇게 하면 test.csv(항상 2025시즌)도 trackman_history의
    2019~2024 아무 시즌에서나 같은 phase로 자연스럽게 매칭되어, 이전처럼
    "학습 때는 정밀 매칭·서빙 때는 성긴 매칭"이라는 분포 불일치(train-serve skew)
    자체가 구조적으로 생기지 않는다.
  - trackman_history.csv에는 base_state/runner_on_* 컬럼이 없다 — 주자상황은
    매칭 키에 넣을 수 없다(데이터에 없는 정보를 만들어낼 수 없음).
  - 구종(pitch_type_group)별로 나누지 않고 상황 전체를 뭉뚱그려 8개 물리 지표
    [rel_speed, spin_rate, induced_vert_break, horz_break, extension, rel_height,
    rel_side, zone_speed]의 mean/std만 낸다 (이전 9-key 스크립트는 pitch_type_group x
    auto_pitch_type로 피벗해 64개 컬럼을 만들었는데, 이게 상황별 표본을 잘게 쪼개
    노이즈를 키웠을 가능성이 있다).
  - 새 피처 stint_pitch_no: trackman_game_id 내에서 pitcher_trackman_id가 바뀔 때마다
    1로 재시작하는 "이 투수의 이번 등판 내 투구수". 그 자체를 9번째 지표로
    mean/std 집계에 포함시켜 "이 상황이 보통 몇 구째쯤 나오는가(피로도 근사치)"를
    반영한다. (train/test에는 game_id가 없어 행 자체의 피로도는 만들 수 없고,
    "이 상황의 평균적 등판 내 위치"라는 배경지식만 붙일 수 있다.)
  - 이전 스크립트처럼 season asof 컷오프(학습 행은 season<=그 행의 season, 검증/실제
    배포는 season<=2024=전체)를 유지해 미래 정보 leak을 막는다.
  - 희소 셀 대비 fallback: 6-key 그룹이 있으면 그 값을, 없으면(매칭 실패) 같은
    asof 컷오프 내 전체 평균/표준편차로 대체한다.

결과가 2024 단독 기준으로만 좋아 보이는지 반드시 2023 홀드아웃으로도 확인한다
(이 프로젝트의 반복된 교훈: season==2024 단독은 그 시즌의 변동성과 모델 개선을
구분하지 못한다).

사용법:
  python -m code.experiment_trackman_context --holdout 2024
  python -m code.experiment_trackman_context --holdout 2023
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from code.blend_model import fit_meta_model
from code.catboost_model import predict_catboost, train_catboost
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device,
    make_bundle, predict_bundle, to_tensors, train_ensemble,
)
from code.train import add_engineered_features

DATA_DIR = "./open/data"
TARGET_COL = "control_success"

METRICS = [
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed", "stint_pitch_no",
]
GROUP_KEYS = ["season_phase", "balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand"]


def season_phase_from_month(month_series):
    return pd.cut(
        month_series,
        bins=[0, 5, 8, 12],
        labels=["early", "mid", "late"],
    ).astype(str)


def add_stint_pitch_no(df_trm):
    """trackman_game_id 내에서 pitcher_trackman_id가 바뀔 때마다 1로 재시작하는
    투구수(=이번 등판 내 투구수)를 계산한다."""
    df_trm = df_trm.sort_values(["trackman_game_id", "pitch_no"]).reset_index(drop=True)
    same_pitcher = df_trm["pitcher_trackman_id"] == df_trm.groupby("trackman_game_id")["pitcher_trackman_id"].shift()
    stint_id = (~same_pitcher).cumsum()
    df_trm["stint_pitch_no"] = df_trm.groupby(stint_id).cumcount() + 1
    return df_trm


def build_context_lookup(df_trm_cut):
    grouped = df_trm_cut.groupby(GROUP_KEYS)[METRICS]
    agg = grouped.agg(["mean", "std"])
    agg.columns = [f"ctx_{m}_{stat}" for m, stat in agg.columns]
    agg["ctx_n"] = grouped.size()
    agg = agg.reset_index()
    std_cols = [c for c in agg.columns if c.endswith("_std")]
    agg[std_cols] = agg[std_cols].fillna(0)
    return agg


def merge_context_asof(df_main, df_trm):
    """df_main의 각 행에 asof 컷오프(season<=그 행의 season)로 만든 상황별 mean/std
    트랙맨 컨텍스트 피처를 붙인다. 매칭 실패 셀은 같은 컷오프의 전체 평균/표준편차로 대체."""
    df_main = df_main.copy()
    df_main["season_phase"] = season_phase_from_month(df_main["game_month"])

    ctx_cols = [f"ctx_{m}_{stat}" for m in METRICS for stat in ("mean", "std")] + ["ctx_n"]

    pieces = []
    for season in sorted(df_main["season"].unique()):
        trm_cut = df_trm[df_trm["season"] <= season]
        rows = df_main[df_main["season"] == season]
        if len(trm_cut) == 0:
            merged = rows.copy()
            for c in ctx_cols:
                merged[c] = 0.0
            pieces.append(merged)
            continue

        lookup = build_context_lookup(trm_cut)
        merged = pd.merge(rows, lookup, on=GROUP_KEYS, how="left")

        global_mean = trm_cut[METRICS].mean()
        global_std = trm_cut[METRICS].std()
        for m in METRICS:
            merged[f"ctx_{m}_mean"] = merged[f"ctx_{m}_mean"].fillna(global_mean[m])
            merged[f"ctx_{m}_std"] = merged[f"ctx_{m}_std"].fillna(global_std[m])
        merged["ctx_n"] = merged["ctx_n"].fillna(0)
        pieces.append(merged)

    result = pd.concat(pieces, ignore_index=True)
    return result, ctx_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024, choices=[2023, 2024])
    parser.add_argument("--apply-f1", dest="apply_f1", action="store_true", default=True)
    parser.add_argument("--no-f1", dest="apply_f1", action="store_false")
    parser.add_argument("--no-context", dest="apply_context", action="store_false", default=True,
                         help="컨텍스트 피처 없이(트랙맨 미사용) 현재 CatBoost 하이퍼파라미터/코드로 기준선을 재현")
    args = parser.parse_args()
    label = f"holdout={args.holdout} f1={'ON' if args.apply_f1 else 'OFF'} ctx={'ON' if args.apply_context else 'OFF'}"

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    if args.apply_context:
        df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
        df_trm["season_phase"] = season_phase_from_month(df_trm["game_month"])
        df_trm["pitcher_hand"] = df_trm["pitcher_hand"].map({"Left": 1, "Right": 2}).astype(np.int64)
        df_trm["batter_hand"] = df_trm["batter_hand"].map({"Left": 1, "Right": 2}).astype(np.int64)

        t0 = time.time()
        df_trm = add_stint_pitch_no(df_trm)
        print(f"[{label}] stint_pitch_no 계산 완료 ({time.time() - t0:.1f}s)")

        t0 = time.time()
        df, ctx_cols = merge_context_asof(df, df_trm)
        print(f"[{label}] 트랙맨 컨텍스트 asof 병합 완료 ({time.time() - t0:.1f}s) | 신규 피처 수: {len(ctx_cols)}")
    else:
        print(f"[{label}] 컨텍스트 피처 미적용 (기준선)")

    train_mask = df["season"] < args.holdout
    val_mask = df["season"] == args.holdout
    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL, "season_phase"]
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


if __name__ == "__main__":
    main()
