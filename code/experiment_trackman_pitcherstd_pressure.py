# code/experiment_trackman_pitcherstd_pressure.py
"""트랙맨 재도전: 진짜 투수 정체성(시퀀스 크로스워크) x 상황(카운트 압박) x 구종군별
물리 지표(평균+표준편차)를 피처로 추가하는 실험.

배경 (EXPERIMENTS.md §14/§17/§25 참고): 지금까지 트랙맨은 두 갈래로만 시도됐다.
  (1) 상황 지문 매칭 — 투수 정체성을 모른 채 "비슷한 상황의 아무 투구" 평균 (실패, -40.4)
  (2) 팀원 크로스워크 기반 투수별 통산 물리 특성 — 정체성은 알지만 상황 조건 없음
      (실패, -38.2 ~ -45.6; 산포/재현성만 격리해도 -38.2)
이번엔 그 둘을 합친, 아직 아무도 안 해본 조합을 테스트한다: 진짜 pitcher_id로 식별한
투수가, "카운트 압박 상황"(2스트라이크 이상 또는 3볼 이상, code/train.py::add_engineered_features의
count_pressure와 동일한 압박 정의)에서 구종군별로 물리 지표가 얼마나 갈리는지(평균+표준편차)를
피처로 준다.

크로스워크는 code/pitcher_crosswalk.py가 만든 ./open/temp/pitcher_map.csv를 쓴다
(외부 데이터 아님 — train.csv/trackman_history.csv 두 공식 CSV의 컬럼만으로 시퀀스 재구성).
data_description.md에는 없는 수동 검토(사람이 판정한 이상치) 결과를 반영해 트랙맨을
먼저 클렌징한다: inning<1, balls/strikes/outs_before 범위 밖, extension<=0,
zone_speed>rel_speed, trackman_id 제외 완전 중복 행을 제거하고, 손(pitcher_hand)이
여러 값으로 기록된 투수는 다수 기록된 손만 남긴다.

asof9key(§17.2)와 같은 이유로 season<=S 컷오프를 건다(학습 시점도 서빙 시점과 같은
"전체 히스토리" 분포를 보게 해 학습/서빙 분포 불일치를 피함).

사용법:
  python -m code.experiment_trackman_pitcherstd_pressure --holdout 2024 --tier none
  python -m code.experiment_trackman_pitcherstd_pressure --holdout 2024 --tier a
  python -m code.experiment_trackman_pitcherstd_pressure --holdout 2024 --tier b
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
METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
           "extension", "rel_height", "rel_side", "zone_speed"]


def clean_trackman(df):
    """2026-08-17 세션에서 버그 수정: extension/zone_speed/rel_speed 결측(NaN)행을
    `x>0`/`x<=y` 비교의 False 부작용으로 통째로 삭제하던 것을 막고, 결측은 통과시킨다.
    상세: code/trackman_pitcher_features.py::clean_trackman 참고."""
    before = len(df)
    mask = (
        (df["inning"] >= 1)
        & df["balls_before"].between(0, 3)
        & df["strikes_before"].between(0, 2)
        & df["outs_before"].between(0, 2)
        & (df["extension"].isna() | (df["extension"] > 0))
        & (df["zone_speed"].isna() | df["rel_speed"].isna() | (df["zone_speed"] <= df["rel_speed"]))
    )
    df = df[mask].copy()
    dup_cols = [c for c in df.columns if c != "trackman_id"]
    df = df.drop_duplicates(subset=dup_cols)

    hand_counts = df.groupby(["pitcher_trackman_id", "pitcher_hand"]).size().reset_index(name="n")
    majority_hand = hand_counts.sort_values("n", ascending=False).drop_duplicates("pitcher_trackman_id")
    df = df.merge(majority_hand[["pitcher_trackman_id", "pitcher_hand"]],
                  on=["pitcher_trackman_id", "pitcher_hand"], how="inner")

    after = len(df)
    print(f"  트랙맨 클렌징: {before} -> {after}행 ({before - after}행 제거)")
    return df


TIER_SPECS = {
    "a": ([], "trkstdA_"),
    "b": (["pressure"], "trkstdB_"),
    "c": (["batter_hand"], "trkstdC_"),
    "f": (["batter_hand", "pressure"], "trkstdF_"),
}


def build_pitcher_lookup(df_trm_clean, pitcher_map, tier):
    merged = df_trm_clean.merge(pitcher_map[["pitcher_trackman_id", "pitcher_id"]],
                                 on="pitcher_trackman_id", how="inner")
    extra_keys, prefix = TIER_SPECS[tier]
    if "pressure" in extra_keys:
        merged["pressure"] = ((merged["balls_before"] >= 3) | (merged["strikes_before"] >= 2)).astype(np.int64)
    pivot_levels = extra_keys + ["pitch_type_group"]
    group_cols = ["pitcher_id"] + pivot_levels

    g = merged.groupby(group_cols)[METRICS].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    std_cols = [c for c in g.columns if c.endswith("_std")]
    g[std_cols] = g[std_cols].fillna(0.0)

    pivoted = g.set_index(group_cols).unstack(level=pivot_levels)
    pivoted.columns = ["_".join(str(x) for x in c) for c in pivoted.columns]
    pivoted = pivoted.reset_index().fillna(0.0)
    rename = {c: prefix + c for c in pivoted.columns if c != "pitcher_id"}
    return pivoted.rename(columns=rename)


def merge_asof_pitcher_std(df_main, df_trm_clean, pitcher_map, tier, holdout):
    """holdout 시즌 자체(및 그 이후)의 트랙맨은 절대 쓰지 않는다 — 실제 배포(season=2025)는
    트랙맨 커버리지(~2024)가 목표 시즌보다 항상 한 시즌 앞서 끊겨 있어 목표 시즌 자기 자신의
    트랙맨을 절대 볼 수 없기 때문. holdout 행은 cutoff=holdout-1로 클램프해 이 간극을 재현한다
    (안 그러면 시즌 내 미래 경기 트랙맨이 검증에 새어 들어가는 within-season leak이 생김)."""
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
    result[feature_cols] = result[feature_cols].fillna(0.0)
    return result, feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024, choices=[2023, 2024])
    parser.add_argument("--tier", type=str, default="none", choices=["none", "a", "b", "c", "f"])
    parser.add_argument("--feed-to", type=str, default="both", choices=["both", "cat", "mlp"],
                         help="트랙맨 피처를 어느 모델에게 줄지: both(둘 다)/cat(CatBoost만)/mlp(MLP만)")
    parser.add_argument("--apply-f1", dest="apply_f1", action="store_true", default=True)
    parser.add_argument("--no-f1", dest="apply_f1", action="store_false")
    parser.add_argument("--cutoff7", action="store_true",
                         help="홀드아웃을 시즌 전체 대신 code/train.py의 실제 프로덕션 스플릿(cutoff=7: "
                              "학습=season<2024+2024년 1~6월, 검증=2024년 7~10월)으로 대체. --holdout은 "
                              "무시되고 항상 2024로 취급(트랙맨 asof 컷오프도 동일하게 season<=2023 클램프).")
    args = parser.parse_args()
    if args.cutoff7:
        args.holdout = 2024
    label = f"{'cutoff7' if args.cutoff7 else 'holdout=' + str(args.holdout)} tier={args.tier} feed={args.feed_to} f1={'ON' if args.apply_f1 else 'OFF'}"

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    if args.tier != "none":
        pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
        df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
        df_trm_clean = clean_trackman(df_trm)
        t0 = time.time()
        df, trk_feature_cols = merge_asof_pitcher_std(df, df_trm_clean, pitcher_map, args.tier, args.holdout)
        cov = (df[trk_feature_cols[0]] != 0).mean() if trk_feature_cols else 0.0
        print(f"[{label}] 트랙맨 병합 완료 ({time.time() - t0:.1f}s) | 파생 피처 수: {len(trk_feature_cols)} | 비-0 커버리지(대략): {cov:.1%}")
    else:
        trk_feature_cols = []

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
