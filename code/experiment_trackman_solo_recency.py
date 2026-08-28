# code/experiment_trackman_solo_recency.py
"""트랙맨-only solo 진단 3번째 버전: 커리어 전체 mean/std(64컬럼, baseline)에
"가장 최근 시즌만의" 평균과 "최근-커리어 격차"를 추가해, season-progression 피처와
같은 트릭(전체 누적 vs 이번 구간만)을 트랙맨 물리량에도 적용했을 때 오르는지 확인.

asof 클램프는 기존과 동일(cutoff_season = min(season, holdout-1))하게 유지하고,
그 클램프된 트랙맨 안에서 "career"(전체 클램프 구간)와 "recent"(그 중 cutoff_season
시즌 하나만)를 나눈다 — season-progression이 "커리어 누적 vs 직전 시즌 이후(이번
시즌)"를 나눈 것과 동일한 구조. recent 데이터가 없는 투수(그 시즌 트랙맨 미기록)는
gap=0(정보 없음 = 커리어와 동일 가정)으로 폴백.

thread_count=2 (다른 세션이 지금 code.train으로 프로덕션 재학습 중이라 더 보수적으로 캡).

사용법: python -m code.experiment_trackman_solo_recency
"""
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.train import apply_f1_filter
from code.mlp_model import compute_bss
from code.trackman_pitcher_features import clean_trackman

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
THREAD_COUNT = 2
ITERATIONS = 800
EARLY_STOPPING_ROUNDS = 40

METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
           "extension", "rel_height", "rel_side", "zone_speed"]


def load_base():
    t0 = time.time()
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    is_futures = df_trm_clean["pitcher_team"].str.startswith("MIN_")
    before = len(df_trm_clean)
    f1_mask = is_futures & (df_trm_clean["season"] <= 2022)
    df_trm_clean = df_trm_clean[~f1_mask].reset_index(drop=True)
    print(f"  트랙맨 F1-equiv 필터: {before} -> {len(df_trm_clean)}행 ({f1_mask.sum()}행 제거)")
    print(f"[load_base] 완료 (경과 {time.time()-t0:.1f}s)")
    return train_df, pitcher_map, df_trm_clean


def _agg_mean_std(merged, group_cols):
    g = merged.groupby(group_cols)[METRICS].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    std_cols = [c for c in g.columns if c.endswith("_std")]
    g[std_cols] = g[std_cols].fillna(0.0)
    return g


def build_pitcher_lookup_recency(trm_cut, pitcher_map, cutoff_season):
    merged = trm_cut.merge(pitcher_map[["pitcher_trackman_id", "pitcher_id"]],
                            on="pitcher_trackman_id", how="inner")
    group_cols = ["pitcher_id", "pitch_type_group"]

    career = _agg_mean_std(merged, group_cols)  # mean/std, 기존 baseline과 동일
    recent_src = merged[merged["season"] == cutoff_season]
    recent_mean = recent_src.groupby(group_cols)[METRICS].mean()
    recent_mean.columns = [f"{c}_recentmean" for c in recent_mean.columns]
    recent_mean = recent_mean.reset_index()

    g = career.merge(recent_mean, on=group_cols, how="left")
    for m in METRICS:
        g[f"{m}_gap"] = g[f"{m}_recentmean"] - g[f"{m}_mean"]
    stat_cols = [c for c in g.columns if c not in group_cols]
    g[stat_cols] = g[stat_cols].fillna(0.0)

    pivoted = g.set_index(group_cols).unstack(level="pitch_type_group")
    pivoted.columns = ["_".join(str(x) for x in c) for c in pivoted.columns]
    pivoted = pivoted.reset_index().fillna(0.0)
    rename = {c: "trkrec_" + c for c in pivoted.columns if c != "pitcher_id"}
    return pivoted.rename(columns=rename)


def merge_asof_recency(df_main, df_trm_clean, pitcher_map, holdout):
    pieces = []
    feature_cols = None
    for season in sorted(df_main["season"].unique()):
        cutoff_season = min(season, holdout - 1)
        trm_cut = df_trm_clean[df_trm_clean["season"] <= cutoff_season]
        lookup = build_pitcher_lookup_recency(trm_cut, pitcher_map, cutoff_season)
        if feature_cols is None:
            feature_cols = [c for c in lookup.columns if c != "pitcher_id"]
        rows = df_main[df_main["season"] == season]
        merged = pd.merge(rows, lookup, on="pitcher_id", how="left")
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True)
    result[feature_cols] = result[feature_cols].fillna(0.0)
    return result, feature_cols


def run_regime(name, train_df, pitcher_map, df_trm_clean, holdout, train_mask_fn, val_mask_fn):
    t0 = time.time()
    df_main = apply_f1_filter(train_df.copy())
    df_main, feature_cols = merge_asof_recency(df_main, df_trm_clean, pitcher_map, holdout=holdout)

    train_mask = train_mask_fn(df_main)
    val_mask = val_mask_fn(df_main)
    X_train, y_train = df_main.loc[train_mask, feature_cols], df_main.loc[train_mask, TARGET_COL]
    X_val, y_val = df_main.loc[val_mask, feature_cols], df_main.loc[val_mask, TARGET_COL]

    print(f"\n[{name}] n_features={len(feature_cols)} (mean/std/recentmean/gap x 8지표 x 4구종군) "
          f"n_train={len(X_train)} n_val={len(X_val)}")

    model = CatBoostClassifier(
        iterations=ITERATIONS, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        loss_function="Logloss", eval_metric="BrierScore", random_seed=42,
        thread_count=THREAD_COUNT, early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False,
    )
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    preds = model.predict_proba(X_val)[:, 1]
    brier, bss, score = compute_bss(preds, y_val.values)
    print(f"[{name}] best_iter={model.get_best_iteration()} | recency score={score:.2f} (BSS={bss:.5f}) | "
          f"경과 {time.time()-t0:.1f}s")

    importances = pd.Series(model.get_feature_importance(train_pool), index=feature_cols)
    print(f"[{name}] top-8 피처 중요도:\n{importances.sort_values(ascending=False).head(8)}")

    return dict(name=name, score=score, n_val=len(X_val), n_features=len(feature_cols))


def main():
    train_df, pitcher_map, df_trm_clean = load_base()
    results = []

    results.append(run_regime(
        "cutoff7", train_df, pitcher_map, df_trm_clean, holdout=2024,
        train_mask_fn=lambda df: (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7)),
        val_mask_fn=lambda df: (df["season"] == 2024) & (df["game_month"] >= 7),
    ))
    results.append(run_regime(
        "season2023", train_df, pitcher_map, df_trm_clean, holdout=2023,
        train_mask_fn=lambda df: df["season"] < 2023,
        val_mask_fn=lambda df: df["season"] == 2023,
    ))

    baseline_meanstd = {"cutoff7": 156.15, "season2023": 332.30}
    baseline_rich = {"cutoff7": 163.27, "season2023": 322.97}
    print("\n" + "=" * 80)
    print(f"{'레짐':<12}{'mean/std(64)':>14}{'+richagg(192)':>16}{'+recency(128)':>16}{'delta vs meanstd':>18}")
    for r in results:
        b1, b2 = baseline_meanstd[r["name"]], baseline_rich[r["name"]]
        print(f"{r['name']:<12}{b1:>14.2f}{b2:>16.2f}{r['score']:>16.2f}{r['score']-b1:>+18.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
