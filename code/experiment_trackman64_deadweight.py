# code/experiment_trackman64_deadweight.py
"""팀원(조유담) 파이프라인과의 3가지 구조적 차이 중 1번째: 트랙맨 물리량 64컬럼
(`feature_engineering.py::process_trackman_features_safe`) 유지 여부.

이 join의 `match_cols`는 df_main/df_trm 공통 컬럼(season, game_month,
game_dayofweek, inning, top_bottom, balls_before, strikes_before, outs_before,
pitcher_hand, batter_hand)의 교집합이라 season이 포함된다 — 우리 프로젝트가
과거에 이미 시도했다가 "실전에서 100% 상수로 fallback, 신뢰할 수 있는 신호
없음(2023+2024 평균 delta≈0)"으로 결론 내고 제거한 바로 그 fingerprint 방식
(CLAUDE.md "Trackman history" 절 참고)과 동일한 메커니즘이다.

이 스크립트를 로컬 dual-regime(cutoff7/season2023)으로 그대로 돌리면 **자동으로
실제 서버 상황이 재현된다**: 이 조인의 학습용 트랙맨 테이블은 항상
"train_split 자신의 최대 season/month까지"로 시간필터링되는데, val_split의
(season, game_month) 조합은 정의상 그 필터링 범위 밖에 있으므로(cutoff7 val=
season2024 7월 이후, season2023 val=season2023 자체) match_cols에 season이 낀
채로는 val 쪽이 구조적으로 전부 미매치(NaN) -> train 쪽에서 계산한 고정
평균값으로 채워짐 -> 사실상 상수 64컬럼이 된다. 이게 바로 실전 제출(2025시즌)에서
벌어지는 일과 정확히 같은 모양이라, 별도 시뮬레이션 트릭 없이 자연스럽게 같은
조건을 재현한다.

CatBoost 전용으로 테스트(우리 repo의 "실험적/트랙맨류 피처는 CatBoost 전용"
관례를 따름, MLP quantile 임베딩과의 상호작용은 별도 검증인 §2 스크립트가 담당).
노이즈를 줄이기 위해 우리 프로덕션과 동일한 5-seed CatBoost 앙상블로 비교한다.

사용법: python -m code.experiment_trackman64_deadweight --both
"""
import argparse
import time

import numpy as np
import pandas as pd

from code.catboost_model import CATBOOST_SEED_POOL, train_catboost_ensemble
from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, TARGET_COL, DATA_DIR

METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
           "rel_height", "rel_side", "zone_speed"]


def build_trackman64(train_split, val_split, df_trm_raw):
    """팀원 `process_trackman_features_safe`을 그대로 재현. train_split만 보고
    시간필터링한 트랙맨 집계 테이블을 만들어 train/val 양쪽에 조인한다. val의
    결측은 train 쪽에서 계산한 컬럼평균(=train 자신의 통계, 규칙 위반 아님)으로
    채운다 — 팀원 CLAUDE.md가 명시한 "test 유래 평균 금지" 규칙을 지키는 준수판과
    동일한 fallback."""
    max_season = train_split["season"].max()
    max_month = train_split.loc[train_split["season"] == max_season, "game_month"].max()
    df_trm = df_trm_raw.copy()
    future_mask = (df_trm["season"] > max_season) | ((df_trm["season"] == max_season) & (df_trm["game_month"] > max_month))
    df_trm = df_trm[~future_mask].reset_index(drop=True)
    print(f"    [시간필터] 트랙맨 {len(df_trm_raw)} -> {len(df_trm)}행 (train 최대 {max_season}년 {max_month}월까지)")

    match_cols = [c for c in train_split.columns if c in df_trm.columns and c != "row_id"]
    df_trm["top_bottom"] = df_trm["top_bottom"].map({"Top": 0, "Bottom": 1}).astype(np.int64)
    df_trm["pitcher_hand"] = df_trm["pitcher_hand"].map({"Left": 1, "Right": 2}).astype(np.int64)
    df_trm["batter_hand"] = df_trm["batter_hand"].map({"Left": 1, "Right": 2}).astype(np.int64)

    grouped = df_trm.groupby(match_cols + ["pitch_type_group", "auto_pitch_type"])
    g1 = grouped[METRICS].agg(["mean", "std"])
    g2 = g1.reset_index()
    std_cols = [c for c in g2.columns if "std" in c]
    g2[std_cols] = g2[std_cols].fillna(0)
    g2.columns = ["_".join(col).strip("_") for col in g2.columns]
    g3 = g2.drop(columns="auto_pitch_type")
    g3 = g3.groupby(match_cols + ["pitch_type_group"]).agg(["mean"])
    pivoted = g3.unstack(level="pitch_type_group")
    pivoted.columns = [f"{c[0]}_{c[1]}_{c[2]}" for c in pivoted.columns]
    tm_final = pivoted.reset_index().fillna(0)
    new_cols = [c for c in tm_final.columns if c not in match_cols]
    print(f"    [트랙맨64] {len(new_cols)}개 컬럼 생성, match_cols={match_cols}")

    train_merged = pd.merge(train_split, tm_final, on=match_cols, how="left")
    fill_vals = train_merged[new_cols].mean()
    n_nan_train = train_merged[new_cols].isna().any(axis=1).sum()
    train_merged[new_cols] = train_merged[new_cols].fillna(fill_vals)

    val_merged = pd.merge(val_split, tm_final, on=match_cols, how="left")
    n_nan_val = val_merged[new_cols].isna().any(axis=1).sum()
    val_merged[new_cols] = val_merged[new_cols].fillna(fill_vals)
    print(f"    [매치율] train 미매치 {n_nan_train}/{len(train_merged)}행, val 미매치 {n_nan_val}/{len(val_merged)}행")

    return train_merged, val_merged, new_cols


def run_regime(cutoff7, holdout, df_trm_raw):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (트랙맨64 상수컬럼 유지 여부) ===\n{'='*70}", flush=True)

    train_split, val_split, _mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    y_val = val_split[TARGET_COL].values

    t0 = time.time()
    tm_train, tm_val, new_cols = build_trackman64(train_split, val_split, df_trm_raw)
    print(f"    빌드 완료 ({time.time()-t0:.1f}s)", flush=True)

    def ensemble_score(cols, tr, va, tag):
        t0 = time.time()
        X_train = tr[cols]
        y_train = tr[TARGET_COL].values
        X_val = va[cols]
        results = train_catboost_ensemble(X_train, y_train, X_val, y_val, seeds=CATBOOST_SEED_POOL, verbose=False)
        preds_list = [m.predict_proba(X_val)[:, 1] for m, _ in results]
        ensemble_pred = np.mean(preds_list, axis=0)
        score = compute_bss(ensemble_pred, y_val)[2]
        print(f"  [{tag}] 5-seed 앙상블 Val Score={score:.2f} ({time.time()-t0:.1f}s)", flush=True)
        return score

    base_score = ensemble_score(cat_feature_cols, train_split, val_split, "baseline(트랙맨64 없음)")
    tm_score = ensemble_score(cat_feature_cols + new_cols, tm_train, tm_val, "+트랙맨64 상수컬럼")

    delta = tm_score - base_score
    print(f"\n  delta(+트랙맨64 - baseline) = {delta:+.2f}", flush=True)
    return base_score, tm_score, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    import os
    df_trm_raw = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")

    results = []
    if args.both:
        results.append(("cutoff7", run_regime(True, 2024, df_trm_raw)))
        results.append(("season2023", run_regime(False, 2023, df_trm_raw)))
    else:
        holdout = 2024 if args.cutoff7 else args.holdout
        label = "cutoff7" if args.cutoff7 else f"season{holdout}"
        results.append((label, run_regime(args.cutoff7, holdout, df_trm_raw)))

    print("\n" + "=" * 70)
    print(f"{'레짐':<14}{'baseline':>12}{'+트랙맨64':>12}{'delta':>10}")
    for name, (b, t, d) in results:
        print(f"{name:<14}{b:>12.2f}{t:>12.2f}{d:>+10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
