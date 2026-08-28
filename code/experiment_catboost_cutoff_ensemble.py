# code/experiment_catboost_cutoff_ensemble.py
"""CatBoost "cutoff-diversity" 앙상블 — 서로 다른 train/val 시간 경계(cutoff month)로
학습한 멤버들을 평균낸다.

배경: `code/tune.py` 재튜닝 이전 cutoff 스윕(EXPERIMENTS.md §35.3)에서 weight=1
기준 cutoff 4~9 전부 CatBoost 단독 플러스였다("산 모양", cutoff=7이 가장 크고
신뢰할 만해 채택). 지금까지 CatBoost 다양성 축은 시드(§54, 노이즈로 기각)와
피처 서브셋(feature bagging)만 시도됐는데, 둘 다 "같은 학습 데이터, 다른 난수"라
다양성이 얕다. cutoff 자체를 바꾸면 멤버마다 **학습에 포함되는 2024년 데이터
양이 실제로 다르므로**, 시드보다 더 근본적인 다양성일 수 있다.

**리크 방지 설계**: val_cutoff_month=N에서 평가할 때, 앙상블 멤버들의
train_cutoff_month는 전부 <= N으로 제한한다 (그래야 모든 멤버의 학습 데이터가
val 구간(month>=N)을 절대 포함하지 않는다 — cutoff=8 멤버를 cutoff=7 val
평가에 섞으면 val의 7월 데이터가 그 멤버 학습에 들어가 리크가 된다).

TE-residual(`apply_te_residual_features`)은 causal 소스가 train_split 자신이므로
멤버마다(= train_cutoff_month마다) 새로 계산한다 — season 진행분/coarse
pitchmix/F1필터 이외 나머지 피처 엔지니어링은 cutoff에 안 좌우되므로 한 번만
계산해 재사용한다(`load_base_df`).

3-fold(val_cutoff_month ∈ {6,7,8}, 이 프로젝트 cutoff 스윕의 "산 모양" 중 가장
신뢰할 만한 중심부)로 스크리닝한다 — season==2023 레짐은 이 메커니즘(2024년 내
cutoff 경계)이 구조적으로 적용되지 않으므로(season2023 val엔애초에 2024 데이터가
전혀 안 들어감) 대신 val_cutoff_month를 바꿔가며 이 프로젝트의 "dual-regime
대신 rolling-origin" 규율(핵심 교훈 #28)을 지킨다.

사용법:
  python -m code.experiment_catboost_cutoff_ensemble
  python -m code.experiment_catboost_cutoff_ensemble --val-cutoffs 6 7 8 --n-members 4
"""
import argparse
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.mlp_model import compute_bss
from code.train import (
    TRACKMAN_TIER_FEED, add_engineered_features, apply_f1_filter,
    apply_te_residual_features, TE_RESIDUAL_COLS,
)
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"


def load_base_df():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm_full = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm_full)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=2024)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]

    df = merge_coarse_pitchmix(df, df_trm_full, holdout=2024)

    prod_train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
    league_success_mean = df.loc[prod_train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)
    return df, trk_mlp_cols


def build_member_split(df, trk_mlp_cols, train_cutoff_month, val_cutoff_month):
    train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < train_cutoff_month))
    val_mask = (df["season"] == 2024) & (df["game_month"] >= val_cutoff_month)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in df.columns if c not in drop_cols and c not in trk_mlp_cols]

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    train_split = apply_te_residual_features(train_split, train_split, te_prior)
    val_split = apply_te_residual_features(train_split, val_split, te_prior)
    cat_feature_cols = features + TE_RESIDUAL_COLS

    X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val, y_val = val_split[cat_feature_cols], val_split[TARGET_COL].values
    return X_train, y_train, X_val, y_val


def train_member(X_train, y_train, X_val, y_val):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1], int(model.get_best_iteration())


def run_fold(df, trk_mlp_cols, val_cutoff_month, n_members):
    member_cutoffs = list(range(max(4, val_cutoff_month - n_members + 1), val_cutoff_month + 1))
    print(f"\n{'='*70}\n=== fold: val_cutoff_month={val_cutoff_month} (val=2024년 {val_cutoff_month}월~) | "
          f"members train_cutoff_month={member_cutoffs} ===\n{'='*70}")

    y_val_ref = None
    preds_by_cutoff = {}
    for tc in member_cutoffs:
        t0 = time.time()
        X_train, y_train, X_val, y_val = build_member_split(df, trk_mlp_cols, tc, val_cutoff_month)
        if y_val_ref is None:
            y_val_ref = y_val
        else:
            assert len(y_val) == len(y_val_ref), "val set이 멤버마다 달라짐 — cutoff 경계 버그"
        preds, best_iter = train_member(X_train, y_train, X_val, y_val)
        score = compute_bss(preds, y_val)[2]
        preds_by_cutoff[tc] = preds
        print(f"  [train_cutoff={tc}] train={len(y_train)}행 Val Score={score:.2f} "
              f"(best_iteration={best_iter}, {time.time()-t0:.1f}s)")

    baseline_score = compute_bss(preds_by_cutoff[val_cutoff_month], y_val_ref)[2]
    print(f"\n  baseline(단일, train_cutoff={val_cutoff_month}, 현 프로덕션과 동일)={baseline_score:.2f}")

    ordered_cutoffs = sorted(preds_by_cutoff.keys())
    for k in range(2, len(ordered_cutoffs) + 1):
        subset = ordered_cutoffs[-k:]
        ens_pred = np.mean([preds_by_cutoff[tc] for tc in subset], axis=0)
        ens_score = compute_bss(ens_pred, y_val_ref)[2]
        print(f"  [{k}-member 앙상블 {subset}] Val Score={ens_score:.2f} "
              f"(vs baseline {baseline_score:.2f}, delta={ens_score-baseline_score:+.2f})")
    return baseline_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-cutoffs", type=int, nargs="+", default=[6, 7, 8])
    parser.add_argument("--n-members", type=int, default=4)
    args = parser.parse_args()

    df, trk_mlp_cols = load_base_df()
    for vc in args.val_cutoffs:
        run_fold(df, trk_mlp_cols, vc, args.n_members)


if __name__ == "__main__":
    main()
