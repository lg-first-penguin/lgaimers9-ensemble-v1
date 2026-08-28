# code/experiment_trackB_gamelevel_features.py
"""팀원(외부 조언)이 제안한 "Track B" 경기단위 시계열 피처 6개를 이 프로젝트의
dual-regime(cutoff7 + season==2023) CatBoost 단독 관례로 재현/검증한다.

제안 문서의 p_season_gap은 이미 채택된 pitcher_season_rate_gap(§45)과 동일 개념이라
재검증하지 않는다. 나머지 6개만 신규:
  p_prev2_gap / p_prev10_gap  : 투수 직전 N경기(투구수 가중) 성공률 - asof 통산 성공률
  p_std10                     : 투수 직전 10경기(경기별 평균) 성공률의 표준편차 (기복, gap 아님)
  b_prev3/5/10_gap             : 타자 직전 N경기(투구수 가중) 성공률 - asof 통산 성공률

경기 경계는 원본 데이터에 game_id가 없어 (season, game_month, game_dayofweek,
home_team, away_team) 조합이 바뀌는 지점으로 추정한다(홈/원정팀은 top_bottom으로
역산: top_bottom==T면 투수가 홈팀). 이 프로젝트 train.csv에 직접 적용해 검증한 결과
4,821경기로 분리됨(제안 문서가 인용한 수치와 정확히 일치) — 신뢰할 만한 근사로 채택.

프로덕션 베이스라인(시즌진행분+TE-residual+coarse pitchmix, code/train.py와 동일
구성)에 이 6개를 추가했을 때 CatBoost 단독 스코어 변화를 dual-regime으로 비교한다.
season==2023 레짐에서는 val 시즌 자신의 경기 이력을 학습에 포함시키지 않도록
경기단위 lookup도 시즌 경계를 존중해서 계산한다(add_trackB_features는 전체 df에서
한 번에 계산 — shift(1)이 경기 단위라 시즌을 넘어가도 이전 시즌 마지막 경기를
참조하는 게 정상 동작, 시즌진행분 lookup과 동일한 "완결된 과거는 누출 아님" 원칙).

사용법:
  python -m code.experiment_trackB_gamelevel_features --cutoff7
  python -m code.experiment_trackB_gamelevel_features --holdout 2023
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import compute_bss
from code.train import add_engineered_features, apply_f1_filter, apply_te_residual_features, TE_RESIDUAL_COLS, TRACKMAN_TIER_FEED
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"

TRACKB_COLS = ["p_prev2_gap", "p_prev10_gap", "p_std10", "b_prev3_gap", "b_prev5_gap", "b_prev10_gap"]


def add_trackB_features(df):
    """df: control_success를 포함한 전체 train.csv(split 이전). top_bottom은 0/1
    (0=T)로 이미 매핑돼 있어야 한다."""
    df = df.copy()
    home_p = np.where(df["top_bottom"] == 0, df["pitcher_team_id"], df["batter_team_id"])
    key = pd.DataFrame({
        "a": df["season"].values, "b": df["game_month"].values, "c": df["game_dayofweek"].values,
        "d": home_p,
        "e": np.where(df["top_bottom"] == 0, df["batter_team_id"], df["pitcher_team_id"]),
    })
    df["_gid"] = (key != key.shift()).any(axis=1).cumsum().values

    pg = df.groupby(["pitcher_id", "_gid"])[TARGET_COL].agg(["sum", "size"]).reset_index()
    pg = pg.sort_values(["pitcher_id", "_gid"])
    for n in [2, 10]:
        r = pg.groupby("pitcher_id").rolling(n, min_periods=n)[["sum", "size"]].sum()
        pg[f"p_prev{n}"] = (r["sum"] / r["size"]).values
    pg[["p_prev2", "p_prev10"]] = pg.groupby("pitcher_id")[["p_prev2", "p_prev10"]].shift(1)

    pmg = df.groupby(["pitcher_id", "_gid"])[TARGET_COL].mean().reset_index()
    pmg = pmg.sort_values(["pitcher_id", "_gid"])
    pmg["p_std10"] = pmg.groupby("pitcher_id")[TARGET_COL].rolling(10, min_periods=5).std().reset_index(drop=True).values
    pmg["p_std10"] = pmg.groupby("pitcher_id")["p_std10"].shift(1)

    bg = df.groupby(["batter_id", "_gid"])[TARGET_COL].agg(["sum", "size"]).reset_index()
    bg = bg.sort_values(["batter_id", "_gid"])
    for n in [3, 5, 10]:
        r = bg.groupby("batter_id").rolling(n, min_periods=n)[["sum", "size"]].sum()
        bg[f"b_prev{n}"] = (r["sum"] / r["size"]).values
    bg[["b_prev3", "b_prev5", "b_prev10"]] = bg.groupby("batter_id")[["b_prev3", "b_prev5", "b_prev10"]].shift(1)

    df = df.merge(pg[["pitcher_id", "_gid", "p_prev2", "p_prev10"]], on=["pitcher_id", "_gid"], how="left")
    df = df.merge(pmg[["pitcher_id", "_gid", "p_std10"]], on=["pitcher_id", "_gid"], how="left")
    df = df.merge(bg[["batter_id", "_gid", "b_prev3", "b_prev5", "b_prev10"]], on=["batter_id", "_gid"], how="left")

    df["p_prev2_gap"] = df["p_prev2"] - df["asof_pitcher_success_rate"]
    df["p_prev10_gap"] = df["p_prev10"] - df["asof_pitcher_success_rate"]
    df["b_prev3_gap"] = df["b_prev3"] - df["asof_batter_success_rate"]
    df["b_prev5_gap"] = df["b_prev5"] - df["asof_batter_success_rate"]
    df["b_prev10_gap"] = df["b_prev10"] - df["asof_batter_success_rate"]
    # p_std10은 gap이 아니라 절대 표준편차값 그대로 둔다

    return df.drop(columns=["_gid", "p_prev2", "p_prev10", "b_prev3", "b_prev5", "b_prev10"])


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
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm_full = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm_full)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]

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
    df = add_trackB_features(df)

    return df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols


def run_regime(holdout, cutoff7):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols = build_split(holdout, cutoff7)

    for tag, use_new in [("baseline(프로덕션 피처셋)", False), ("+TrackB 6개", True)]:
        extra_cols = TRACKB_COLS if use_new else []
        exclude_cols = [c for c in TRACKB_COLS if c not in extra_cols]

        drop_cols = ["row_id", TARGET_COL]
        base_features = [
            c for c in df.columns
            if c not in drop_cols and c not in trk_mlp_cols and c not in trk_cat_cols and c not in exclude_cols
        ]
        cat_features = base_features + trk_cat_cols

        train_split = df.loc[train_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        val_split = df.loc[val_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        train_split = apply_f1_filter(train_split)

        te_prior = train_split[TARGET_COL].mean()
        te_source = train_split
        train_split = apply_te_residual_features(te_source, train_split, te_prior)
        val_split = apply_te_residual_features(te_source, val_split, te_prior)
        cat_features_full = cat_features + TE_RESIDUAL_COLS

        X_train, y_train = train_split[cat_features_full], train_split[TARGET_COL].values
        X_val, y_val = val_split[cat_features_full], val_split[TARGET_COL].values

        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"[{tag}] Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_features_full)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7)


if __name__ == "__main__":
    main()
