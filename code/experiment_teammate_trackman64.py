# code/experiment_teammate_trackman64.py
"""팀원(조유담) 실전 1059.72 레시피의 두 번째 구조적 차이 — season을 조인 키에 포함한
트랙맨 물리량 64개(구종군별 mean+std) — 를 우리 프로덕션 피처셋에 그대로 이식해
isolate 테스트한다. CatBoost 하이퍼파라미터는 우리 값 그대로 두고 이 피처만 추가한다
(`code/experiment_teammate_catboost_hparams.py`가 반대로 하이퍼파라미터만 isolate했고,
cutoff7에서 -13.14로 오히려 손해였다 — 그렇다면 팀원의 실전 우위는 하이퍼파라미터가
아니라 이 피처일 가능성이 남는다).

**중요한 구조적 우려**: 조인 키에 `season`이 들어간다(`balls_before, strikes_before,
pitcher_hand, batter_hand, inning, top_bottom, season, game_month`). 이건 이 프로젝트가
CLAUDE.md/PROJECT_HISTORY.md에 명시적으로 문서화한 옛날 버그 패턴과 정확히 같은 구조다 —
trackman_history.csv는 2019~2024만 있고 실제 평가 데이터(test.csv)는 항상 season=2025라
**실전에서는 이 조인이 100% 실패해 64개 컬럼 전부가 (구종군별) 상수값으로 fallback된다**
(팀원 자신도 이 사실을 알고 문서화함). `code/trackman_pitcher_features.py::COARSE_COLS`가
의도적으로 season/inning/top_bottom/game_month를 뺀 것도 바로 이 문제를 피하기 위해서였다
(§37 이후 확립된 설계 원칙). 로컬 검증(cutoff7/season2023 val은 둘 다 season<=2024라
트랙맨이 커버하는 시즌)에서는 조인이 실제로 종종 성공하므로, 여기서 플러스가 나오더라도
**실전(2025)에서는 재현되지 않을 구조적 위험이 있다** — 그 가능성을 명확히 안고 테스트한다.

사용법:
  python -m code.experiment_teammate_trackman64 --cutoff7
  python -m code.experiment_teammate_trackman64 --holdout 2023
"""
import argparse
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, TARGET_COL
from code.trackman_pitcher_features import clean_trackman, METRICS, _HAND_CODE

DATA_DIR = "./open/data"

JOIN_KEYS = ["balls_before", "strikes_before", "pitcher_hand", "batter_hand",
             "inning", "top_bottom", "season", "game_month"]
PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]
TM64_COLS = [f"tm64_{g}_{stat}_{m}" for g in PITCH_GROUPS for stat in ["mean", "std"] for m in METRICS]


def compute_trackman64_lookup(df_trm_clean):
    df = df_trm_clean.copy()
    df["pitcher_hand"] = df["pitcher_hand"].map(_HAND_CODE)
    df["batter_hand"] = df["batter_hand"].map(_HAND_CODE)
    df["top_bottom"] = df["top_bottom"].map({"Top": 0, "Bottom": 1})

    agg = df.groupby(JOIN_KEYS + ["pitch_type_group"])[METRICS].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    wide = agg.pivot_table(index=JOIN_KEYS, columns="pitch_type_group",
                            values=[f"{m}_{stat}" for m in METRICS for stat in ["mean", "std"]])
    new_cols = []
    for m_stat, g in wide.columns:
        m, stat = m_stat.rsplit("_", 1)
        new_cols.append(f"tm64_{g}_{stat}_{m}")
    wide.columns = new_cols
    wide = wide.reset_index()
    for c in TM64_COLS:
        if c not in wide.columns:
            wide[c] = np.nan

    # 팀원 설명: "트랙맨 자체 평균으로 fallback" — 구종군별 전역 평균/표준편차로 해석
    fb = df.groupby("pitch_type_group")[METRICS].agg(["mean", "std"])
    fb.columns = [f"{a}_{b}" for a, b in fb.columns]
    fallback = {}
    for g in PITCH_GROUPS:
        for stat in ["mean", "std"]:
            for m in METRICS:
                col = f"tm64_{g}_{stat}_{m}"
                fallback[col] = fb.loc[g, f"{m}_{stat}"] if g in fb.index else 0.0
    return wide[JOIN_KEYS + TM64_COLS], fallback


def merge_trackman64(df_main, df_trm_clean, holdout):
    trm_cut = df_trm_clean if holdout is None else df_trm_clean[df_trm_clean["season"] < holdout]
    lookup, fallback = compute_trackman64_lookup(trm_cut)
    merged = pd.merge(df_main, lookup, on=JOIN_KEYS, how="left")
    match_rate = merged[TM64_COLS[0]].notna().mean()
    for c in TM64_COLS:
        merged[c] = merged[c].fillna(fallback[c])
    return merged, match_rate


def train_one(X_train, y_train, X_val, y_val):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1], int(model.get_best_iteration())


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (팀원 트랙맨64 isolate 테스트, 우리 하이퍼파라미터 그대로) ===\n{'='*70}")

    train_split, val_split, _mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)

    df_trm_full = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm_full)
    trm_holdout = 2024 if cutoff7 else holdout

    t0 = time.time()
    train_tm, train_match_rate = merge_trackman64(train_split, df_trm_clean, holdout=trm_holdout)
    val_tm, val_match_rate = merge_trackman64(val_split, df_trm_clean, holdout=trm_holdout)
    print(f"  트랙맨64 병합 완료 (train 매칭률={train_match_rate:.1%}, val 매칭률={val_match_rate:.1%}, "
          f"{time.time()-t0:.1f}s) — 실전(2025)에서는 이 매칭률이 구조적으로 0%가 된다")

    X_train_base = train_split[cat_feature_cols]
    y_train = train_split[TARGET_COL].values
    X_val_base = val_split[cat_feature_cols]
    y_val = val_split[TARGET_COL].values

    t0 = time.time()
    base_preds, base_iter = train_one(X_train_base, y_train, X_val_base, y_val)
    base_score = compute_bss(base_preds, y_val)[2]
    print(f"  [우리 baseline(트랙맨64 없음)] Val Score={base_score:.2f} (best_iteration={base_iter}, {time.time()-t0:.1f}s)")

    cat_feature_cols_tm = cat_feature_cols + TM64_COLS
    X_train_tm = train_tm[cat_feature_cols_tm]
    X_val_tm = val_tm[cat_feature_cols_tm]
    t0 = time.time()
    tm_preds, tm_iter = train_one(X_train_tm, y_train, X_val_tm, y_val)
    tm_score = compute_bss(tm_preds, y_val)[2]
    print(f"  [+트랙맨64(64컬럼)] Val Score={tm_score:.2f} (best_iteration={tm_iter}, {time.time()-t0:.1f}s)")

    print(f"\n  delta = {tm_score - base_score:+.2f}")
    return base_score, tm_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(args.cutoff7, holdout)


if __name__ == "__main__":
    main()
