# code/experiment_career_trajectory_foldcheck.py
"""code/experiment_career_trajectory.py의 dual-regime 결과(cutoff7 +0.59 거의 평평,
season==2023 +32.06 큰 플러스)를 rolling-origin 3-fold로 재검증한다.

이 프로젝트에서 "한쪽 레짐은 평평, 다른 쪽은 크게 플러스"라는 모양은 CatBoost
seed-ensemble(cutoff7 +1.51 거의 평평 / season==2023 +12.23, 이후 3-fold에서
2/3·평균+3.43으로 기각)과 li==0 필터(cutoff7 +8.62/season==2023 +26.49 둘 다 플러스로
더 유리했는데도 3-fold에서 1/3·평균-10.58로 기각)에서 반복적으로 노이즈로 확정된
패턴이라 반드시 거쳐야 하는 검증 단계다.

li_zero_filter_foldcheck.py와 동일한 방법론: game_type=='R'만 써서 F1 필터가 이른
cutoff(train<2021 등)에서 학습 구간의 F행을 통째로 지우는 함정을 피한다. CatBoost
단독(random_seed=42 고정, 두 variant 완전히 동일 시드)으로 2021/2022/2023을 각각
val로 삼아 train<val rolling-origin 검증한다.

career_trajectory lookup은 각 fold의 R-only 전체 df_all(스플릿 이전)에서 계산한다 —
시즌진행분과 동일하게 각 행은 자기 시즌보다 앞선 시즌만 참조하므로(season_m1/m2 <
자기 season), df_all에 미래 시즌 데이터가 섞여 있어도 특정 행의 피처 계산에는 절대
쓰이지 않는다(구조적으로 안전).

사용법:
  python -m code.experiment_career_trajectory_foldcheck
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.experiment_career_trajectory import TRAJ_COLS, apply_trajectory_features, build_trajectory_tables
from code.mlp_model import compute_bss
from code.train import add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS
from code.trackman_pitcher_features import PITCHMIX_COLS, merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2021, 2022, 2023]


def load_r_only():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df[df["game_type"] == "R"].reset_index(drop=True)  # F1 트랩 회피
    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    return df, df_trm


def train_one(X_train, y_train, X_val, y_val):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1], int(model.get_best_iteration())


def run_fold(df_all, df_trm, traj_tables, val_season):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values

    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)
    df = apply_trajectory_features(df, traj_tables)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in TRAJ_COLS]

    all_cols = base_features + TRAJ_COLS + [TARGET_COL]
    train_split_base = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    y_val = val_split[TARGET_COL].values

    prior = train_split_base[TARGET_COL].mean()
    te_source = train_split_base
    train_split_te = apply_te_residual_features(te_source, train_split_base, prior)
    val_split_te = apply_te_residual_features(te_source, val_split, prior)

    results = {}
    for tag, extra in [("baseline", []), ("+career_trajectory(4)", TRAJ_COLS)]:
        cat_feature_cols = base_features + TE_RESIDUAL_COLS + extra
        X_train, y_train = train_split_te[cat_feature_cols], train_split_te[TARGET_COL].values
        X_val = val_split_te[cat_feature_cols]

        t0 = time.time()
        preds, best_iter = train_one(X_train, y_train, X_val, y_val)
        score = compute_bss(preds, y_val)[2]
        print(f"  [val={val_season}][{tag}] Val Score={score:.2f} (best_iter={best_iter}, {time.time()-t0:.1f}s, "
              f"n_train={len(train_split_te)}, n_val={len(val_split_te)}, n_features={len(cat_feature_cols)})")
        results[tag] = score

    delta = results["+career_trajectory(4)"] - results["baseline"]
    print(f"  [val={val_season}] delta={delta:+.2f}")
    return delta


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행")
    traj_tables = build_trajectory_tables(df_all)

    deltas = []
    for val_season in FOLD_SEASONS:
        print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===")
        delta = run_fold(df_all, df_trm, traj_tables, val_season)
        deltas.append(delta)

    wins = sum(d > 0 for d in deltas)
    print(f"\n{'='*70}\n{wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}\n{'='*70}")


if __name__ == "__main__":
    main()
