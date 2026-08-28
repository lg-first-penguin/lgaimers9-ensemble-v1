# code/experiment_trackman_solo_richagg.py
"""트랙맨-only solo 진단(code/experiment_trackman_solo.py, mean/std 64컬럼) 대비, 투수별
집계 함수를 mean/std 2개에서 mean/std/skew/q25/q50/q75 6개로 늘리면 solo 스코어가
유의미하게 오르는지 확인하는 저비용 실험. T-JEPA/ICL 같은 무거운 표현학습을 투입하기
전에 "정보 천장(투수당 정적 벡터 1개)이 진짜 문제인지, 그저 mean/std 요약이 조악해서인지"
를 먼저 가른다 — 여기서도 안 오르면 더 무거운 아키텍처 투자 근거가 약해진다.

CPU thread_count 캡(다른 세션과 코어 공유, 다만 확인 결과 현재 다른 세션은 유휴 상태라
4로 완화), 단일 시드. code/experiment_trackman_solo.py와 완전히 같은 F1-equiv 트랙맨
필터·asof 클램프·CatBoost 설정·검증 레짐을 써서 mean/std(64컬럼) 결과와 직접 비교 가능.

사용법: python -m code.experiment_trackman_solo_richagg
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
THREAD_COUNT = 4
ITERATIONS = 800
EARLY_STOPPING_ROUNDS = 40

METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
           "extension", "rel_height", "rel_side", "zone_speed"]

# 이전 실험(mean/std, 64컬럼)과 직접 비교하기 위해 mean/std는 그대로 두고 skew/quantile만 추가
def _q25(s):
    return s.quantile(0.25)


def _q50(s):
    return s.quantile(0.50)


def _q75(s):
    return s.quantile(0.75)


_q25.__name__, _q50.__name__, _q75.__name__ = "q25", "q50", "q75"
AGG_FUNCS = ["mean", "std", "skew", _q25, _q50, _q75]


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


def build_pitcher_lookup_rich(df_trm_clean, pitcher_map):
    merged = df_trm_clean.merge(pitcher_map[["pitcher_trackman_id", "pitcher_id"]],
                                 on="pitcher_trackman_id", how="inner")
    group_cols = ["pitcher_id", "pitch_type_group"]
    g = merged.groupby(group_cols)[METRICS].agg(AGG_FUNCS)
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    stat_cols = [c for c in g.columns if c not in group_cols]
    g[stat_cols] = g[stat_cols].fillna(0.0)

    pivoted = g.set_index(group_cols).unstack(level="pitch_type_group")
    pivoted.columns = ["_".join(str(x) for x in c) for c in pivoted.columns]
    pivoted = pivoted.reset_index().fillna(0.0)
    rename = {c: "trkrich_" + c for c in pivoted.columns if c != "pitcher_id"}
    return pivoted.rename(columns=rename)


def merge_asof_rich(df_main, df_trm_clean, pitcher_map, holdout):
    pieces = []
    feature_cols = None
    for season in sorted(df_main["season"].unique()):
        cutoff_season = min(season, holdout - 1)
        trm_cut = df_trm_clean[df_trm_clean["season"] <= cutoff_season]
        lookup = build_pitcher_lookup_rich(trm_cut, pitcher_map)
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
    df_main, feature_cols = merge_asof_rich(df_main, df_trm_clean, pitcher_map, holdout=holdout)

    train_mask = train_mask_fn(df_main)
    val_mask = val_mask_fn(df_main)
    X_train, y_train = df_main.loc[train_mask, feature_cols], df_main.loc[train_mask, TARGET_COL]
    X_val, y_val = df_main.loc[val_mask, feature_cols], df_main.loc[val_mask, TARGET_COL]

    print(f"\n[{name}] n_features={len(feature_cols)} (mean/std/skew/q25/q50/q75 x 8지표 x 4구종군) "
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
    print(f"[{name}] best_iter={model.get_best_iteration()} | 리치집계 score={score:.2f} (BSS={bss:.5f}) | "
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

    baseline = {"cutoff7": 156.15, "season2023": 332.30}  # mean/std 64컬럼 결과 (experiment_trackman_solo.py)
    print("\n" + "=" * 70)
    print(f"{'레짐':<12}{'n_val':>10}{'mean/std(64)':>14}{'+skew/quantile(192)':>22}{'delta':>10}")
    for r in results:
        b = baseline[r["name"]]
        print(f"{r['name']:<12}{r['n_val']:>10}{b:>14.2f}{r['score']:>22.2f}{r['score']-b:>+10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
