# code/experiment_fleague_subgroup.py
"""game_type=='F'(퓨처스) 서브그룹의 사후 재보정 가능성 진단.

cutoff7 val(2024 7~10월)에서 F 서브그룹 score(494.3)가 R 서브그룹(762.4)보다
크게 낮았던 것의 후속. F는 이 프로젝트에서 이미 두 번 문제를 일으킨 축이라
(F1 필터: F 성공률-game_type 관계가 2023년부터 역전; TRACKMAN_TIER_FEED 관련
아님) 안전하게 다루려면 아래 제약을 지켜야 한다:

- rolling-origin fold-check(FOLD_SEASONS=[2021,2022,2023], train<val_season)를
  그대로 못 쓴다 — val_season<=2023인 fold는 train<val_season이 season<=2022만
  포함해서 F1 필터(season<=2022 F행 제거)가 train의 F행을 통째로 지워버리는
  함정(핵심 교훈 #20/#28)에 정확히 걸린다.
- 대신 프로덕션과 동일한 "cutoff=7"식 스플릿(train에 val_year 상반기까지 포함,
  F1 필터는 season<=2022만 걸러서 val_year==2023이면 2023년 1~6월 F행은 학습에
  남는다)을 val_year in {2023, 2024}에 대해 각각 구성한다. 2024가 실제 프로덕션
  cutoff7이고, 2023은 그것과 구조적으로 동일한 한 해 전 버전 — 두 개의 독립된
  "진짜" 레짐으로 F 서브그룹 방향 일치 여부와 leave-one-out 보정 전이를 볼 수 있다.

CatBoost 단독(seed=42)만 쓴다 — 신호의 존재 여부를 보는 진단이라 블렌드/MLP는
생략(fold-check들의 기존 관례와 동일).
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.mlp_model import compute_bss
from code.train import add_engineered_features, apply_f1_filter, apply_te_residual_features, TE_RESIDUAL_COLS, TRACKMAN_TIER_FEED
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
VAL_YEARS = [2023, 2024]


def build_cutoff_split(val_year):
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_mask = (train_df["season"] < val_year) | ((train_df["season"] == val_year) & (train_df["game_month"] < 7))
    val_mask = (train_df["season"] == val_year) & (train_df["game_month"] >= 7)

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    train_df, _ = add_all_tiers(train_df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=val_year)
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=val_year)

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    features = [c for c in train_df.columns if c not in ["row_id", TARGET_COL]]
    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)
    cat_feature_cols = features + TE_RESIDUAL_COLS

    X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val, y_val = val_split[cat_feature_cols], val_split[TARGET_COL].values
    return X_train, y_train, X_val, y_val, val_split["game_type"].values


def train_catboost(X_train, y_train, X_val, y_val, seed=42):
    params = dict(CATBOOST_PARAMS)
    params["random_seed"] = seed
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1]


def report_gap(name, preds, y):
    gap = preds.mean() - y.mean()
    _, _, score = compute_bss(preds, y)
    print(f"    [{name}] n={len(y)}  ō={y.mean():.4f}  mean_pred={preds.mean():.4f}  gap={gap:+.4f}  score={score:.2f}")
    return gap, score


def main():
    fold_data = {}
    for val_year in VAL_YEARS:
        print(f"\n=== cutoff7식 스플릿: val_year={val_year} (train은 {val_year}년 1~6월까지 포함) ===")
        X_train, y_train, X_val, y_val, game_type = build_cutoff_split(val_year)
        t0 = time.time()
        preds = train_catboost(X_train, y_train, X_val, y_val)
        print(f"  학습 완료 ({time.time()-t0:.1f}s, n_train={len(y_train)}, n_val={len(y_val)})")

        f_mask = game_type == "F"
        r_mask = game_type == "R"
        print(f"  game_type 분포: F={f_mask.sum()}, R={r_mask.sum()}")
        report_gap("전체", preds, y_val)
        f_gap, f_score = report_gap("F 서브그룹", preds[f_mask], y_val[f_mask])
        report_gap("R 서브그룹", preds[r_mask], y_val[r_mask])

        fold_data[val_year] = dict(preds=preds, y_val=y_val, f_mask=f_mask, f_gap=f_gap, f_score=f_score)

    print(f"\n{'='*78}\n=== F 서브그룹 gap 방향 비교 ===\n{'='*78}")
    for val_year in VAL_YEARS:
        d = fold_data[val_year]
        print(f"  val_year={val_year}: F gap={d['f_gap']:+.4f}  F score={d['f_score']:.2f}")

    print(f"\n{'='*78}\n=== leave-one-out F 서브그룹 사후보정 전이 검증 ===\n{'='*78}")
    for val_year in VAL_YEARS:
        other_year = [y for y in VAL_YEARS if y != val_year][0]
        correction = fold_data[other_year]["f_gap"]
        d = fold_data[val_year]
        preds, y_val, f_mask = d["preds"], d["y_val"], d["f_mask"]
        baseline_score = compute_bss(preds, y_val)[2]
        corrected = preds.copy()
        corrected[f_mask] = np.clip(corrected[f_mask] - correction, 0, 1)
        corrected_score = compute_bss(corrected, y_val)[2]
        f_base_score = compute_bss(preds[f_mask], y_val[f_mask])[2]
        f_corr_score = compute_bss(corrected[f_mask], y_val[f_mask])[2]
        print(f"  val_year={val_year}: {other_year}년서 추정한 보정량={correction:+.4f} | 전체 {baseline_score:.2f}->{corrected_score:.2f} (delta={corrected_score-baseline_score:+.2f}) | F subgroup {f_base_score:.2f}->{f_corr_score:.2f} (delta={f_corr_score-f_base_score:+.2f})")


if __name__ == "__main__":
    main()
