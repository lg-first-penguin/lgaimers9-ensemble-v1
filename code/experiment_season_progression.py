# code/experiment_season_progression.py
"""팀원 제보 피처(투수/타자 "시즌 진행분" 8개) 검증.

기존 asof_pitcher_success_rate/asof_batter_success_rate는 커리어 전체 누적값이라
베테랑 선수일수록 "최근 시즌 컨디션" 신호가 희석된다. 이 피처는 각 행의 시즌 시작
시점 기준값(직전 시즌 마지막 행의 누적치 + 그 행 자신의 결과)을 빼서, "이번 시즌
들어서만의" 성공률과, 그것이 커리어 누적 성공률과 얼마나 벌어져 있는지(rate_gap)를
분리해낸다. 시즌 경계를 넘는 lookup이므로 반드시 스플릿 전(df 전체)에서 계산해야
val 구간(2024 7~10월) 행이 자신의 직전 시즌(2023)을 정상적으로 참조한다 — 이건 이미
완결된 과거 시즌이라 미래 정보 누출이 아니다(add_engineered_features와 동일 패턴).

CatBoost 단독(baseline CATBOOST_PARAMS, cutoff7 스플릿, pitchmix-only 트랙맨)으로
빠르게 신규 피처 유무를 비교한다. cutoff7 pitchmix-only CatBoost baseline은 이번
세션 재검증에서 이미 654.14로 확인됨 — 이 스크립트의 "피처 없음" 결과가 그 수치를
재현하는지도 함께 확인한다.

사용법:
  python -m code.experiment_season_progression --cutoff7
  python -m code.experiment_season_progression --holdout 2023
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import compute_bss
from code.train import add_engineered_features, apply_f1_filter
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
TIER_FEED = {}  # pitchmix-only (tier A 제외) — 이번 세션 결정과 동일 구성


def add_season_progression_features(df):
    df = df.copy()

    for role, id_col, n_col, rate_col in [
        ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
        ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
    ]:
        cum_n = df[n_col]
        cum_success = (df[n_col] * df[rate_col]).round()

        idx = df.groupby([id_col, "season"])[n_col].idxmax()
        season_end = df.loc[idx, [id_col, "season", n_col, rate_col, TARGET_COL]].copy()
        season_end.columns = [id_col, "season", "end_n", "end_rate", "end_success"]
        season_end["next_season"] = season_end["season"] + 1

        merged = df[[id_col, "season"]].merge(
            season_end[[id_col, "next_season", "end_n", "end_rate", "end_success"]],
            left_on=[id_col, "season"], right_on=[id_col, "next_season"], how="left",
        )
        pre_n = (merged["end_n"] + 1).fillna(0).values
        pre_success = ((merged["end_n"] * merged["end_rate"]).round().fillna(0) + merged["end_success"].fillna(0)).values

        season_n = np.maximum(cum_n.values - pre_n, 0)
        season_success = np.maximum(cum_success.values - pre_success, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            season_rate = np.where(season_n > 0, season_success / season_n, np.nan)

        df[f"{role}_season_n"] = season_n
        df[f"{role}_season_success_count"] = season_success
        df[f"{role}_season_success_rate"] = season_rate
        df[f"{role}_season_rate_gap"] = season_rate - df[rate_col].values

    return df


SEASON_PROGRESSION_COLS = [
    "pitcher_season_n", "pitcher_season_success_count", "pitcher_season_success_rate", "pitcher_season_rate_gap",
    "batter_season_n", "batter_season_success_count", "batter_season_success_rate", "batter_season_rate_gap",
]


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def build_split(holdout, cutoff7):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm_full = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm_full)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TIER_FEED), holdout=holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TIER_FEED[tier] == "cat" for c in cols]

    df = merge_coarse_pitchmix(df, df_trm_full, holdout=holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    if cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
    else:
        train_mask = df["season"] < holdout
        val_mask = df["season"] == holdout

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    return df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols


def run_regime(holdout, cutoff7):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols = build_split(holdout, cutoff7)

    for tag, use_new in [("피처 없음(baseline)", False), ("+시즌진행분 8개", True)]:
        dfx = add_season_progression_features(df) if use_new else df
        extra_cols = SEASON_PROGRESSION_COLS if use_new else []

        drop_cols = ["row_id", TARGET_COL]
        base_features = [c for c in dfx.columns if c not in drop_cols and c not in trk_mlp_cols and c not in trk_cat_cols and c not in extra_cols]
        cat_features = base_features + trk_cat_cols + extra_cols

        train_split = dfx.loc[train_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        val_split = dfx.loc[val_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        train_split = apply_f1_filter(train_split)

        X_train, y_train = train_split[cat_features], train_split[TARGET_COL].values
        X_val, y_val = val_split[cat_features], val_split[TARGET_COL].values

        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"[{tag}] Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_features)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7)


if __name__ == "__main__":
    main()
