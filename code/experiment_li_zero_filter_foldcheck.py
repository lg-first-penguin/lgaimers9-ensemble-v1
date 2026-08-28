# code/experiment_li_zero_filter_foldcheck.py
"""li==0 필터(code/experiment_li_zero_filter.py)의 dual-regime 결과(cutoff7 +8.62 /
season==2023 +26.49, 둘 다 양수)를, CatBoost seed-ensemble이 똑같이 듀얼레짐을
통과했다가(+9.87/+31.17) rolling-origin 3-fold에서 노이즈로 확정 기각된 전례
(code/experiment_catboost_seed_ensemble_foldcheck.py, 2/3 fold·평균 +3.43)를
근거로 재검증한다.

§35/§46/§57과 동일하게 game_type=='R'만 써서 F1 필터가 이른 cutoff(train<2021 등)에서
학습 구간의 F행을 통째로 지우는 함정을 피한다(R-only에서는 F1 필터가 어차피 no-op).
CatBoost 단독(random_seed=42 고정, 두 variant 완전히 동일 시드)으로 3개 시즌
(2021/2022/2023)을 각각 val로 삼아 train<val rolling-origin 검증한다.

li==0 필터는 학습 데이터에서만 적용한다(apply_f1_filter와 동일한 컨벤션) — 검증
데이터는 그대로 둔다.

사용법:
  python -m code.experiment_li_zero_filter_foldcheck
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
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


def apply_li_zero_filter(df):
    before = len(df)
    filtered = df[df["li"] != 0].reset_index(drop=True)
    print(f"    [li==0 필터] {before} -> {len(filtered)}행 ({before - len(filtered)}행 제거)")
    return filtered


def train_one(X_train, y_train, X_val, y_val):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1], int(model.get_best_iteration())


def run_fold(df_all, df_trm, val_season):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values

    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols]

    all_cols = base_features + [TARGET_COL]
    train_split_base = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    y_val = val_split[TARGET_COL].values

    results = {}
    for tag, train_split in [("baseline", train_split_base), ("li_zero_filtered", apply_li_zero_filter(train_split_base))]:
        prior = train_split[TARGET_COL].mean()
        te_source = train_split
        train_split_te = apply_te_residual_features(te_source, train_split, prior)
        val_split_te = apply_te_residual_features(te_source, val_split, prior)
        cat_feature_cols = base_features + TE_RESIDUAL_COLS

        X_train, y_train = train_split_te[cat_feature_cols], train_split_te[TARGET_COL].values
        X_val = val_split_te[cat_feature_cols]

        t0 = time.time()
        preds, best_iter = train_one(X_train, y_train, X_val, y_val)
        score = compute_bss(preds, y_val)[2]
        print(f"  [val={val_season}][{tag}] Val Score={score:.2f} (best_iter={best_iter}, {time.time()-t0:.1f}s, n_train={len(train_split_te)}, n_val={len(val_split_te)})")
        results[tag] = score

    delta = results["li_zero_filtered"] - results["baseline"]
    print(f"  [val={val_season}] delta={delta:+.2f}")
    return delta


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행")

    deltas = []
    for val_season in FOLD_SEASONS:
        print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===")
        delta = run_fold(df_all, df_trm, val_season)
        deltas.append(delta)

    wins = sum(d > 0 for d in deltas)
    print(f"\n{'='*70}\n{wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}\n{'='*70}")


if __name__ == "__main__":
    main()
