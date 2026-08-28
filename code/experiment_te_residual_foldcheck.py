# code/experiment_te_residual_foldcheck.py
"""Track A(target-encoding 잔차 6개, ->CatBoost 전용)의 cutoff7(+6.97)/season==2023
(+20.49) 듀얼레짐 결과에 대해, 다른 세션이 §33(CatBoost GPU 재튜닝 — 듀얼레짐
전부 우호적이고 로컬 블렌드 +17.61이었는데도 실전 제출은 -3.64로 뒤집힌 전례)을
근거로 "검증 윈도우 특유의 우연일 수 있다"고 지적한 데 대한 rolling-origin
재검증이다.

§35("2024만 유독 절벽" 재검증)와 동일하게 game_type=='R'만 써서, F1 필터가 이른
cutoff(예: train<2021)에서 학습 구간의 F행을 통째로 지워버리는 함정(핵심 교훈
#20/#28)을 피한다. CatBoost 단독(CATBOOST_PARAMS의 random_seed=42 고정, 두
variant가 완전히 같은 시드)으로 3개 시즌(2021/2022/2023)을 각각 val로 삼아
train<val 방식 rolling-origin 검증한다. 이 fold check의 목적은 "신호 자체가
여러 시점에 걸쳐 일반화되는가"이지 블렌드 동역학이 아니므로 MLP는 생략한다
(시간도 크게 단축).

사용법:
  python -m code.experiment_te_residual_foldcheck
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_target_encoding_residual import TE_RESIDUAL_COLS, add_te_residual_features
from code.mlp_model import compute_bss
from code.train import add_engineered_features
from code.trackman_pitcher_features import PITCHMIX_COLS, merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2021, 2022, 2023]


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def load_r_only():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df[df["game_type"] == "R"].reset_index(drop=True)  # F1 트랩 회피(§35와 동일)

    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    return df, df_trm


def run_fold(df_all, df_trm, val_season):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values

    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in PITCHMIX_COLS]

    all_cols = base_features + PITCHMIX_COLS + [TARGET_COL]
    train_split = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    y_val = val_split[TARGET_COL].values

    prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = add_te_residual_features(te_source, train_split, prior)
    val_split = add_te_residual_features(te_source, val_split, prior)

    results = {}
    for tag, extra in [("baseline", []), ("+TE잔차6개", TE_RESIDUAL_COLS)]:
        cat_feature_cols = base_features + PITCHMIX_COLS + extra
        X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
        X_val = val_split[cat_feature_cols]
        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"  [val={val_season}][{tag}] Val Score={score:.2f} (best_iter={best_iter}, {time.time()-t0:.1f}s, n_train={len(train_split)}, n_val={len(val_split)})")
        results[tag] = score
    return results


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행")

    deltas = []
    for val_season in FOLD_SEASONS:
        print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===")
        r = run_fold(df_all, df_trm, val_season)
        delta = r["+TE잔차6개"] - r["baseline"]
        deltas.append(delta)
        print(f"  delta: {delta:+.2f}")

    wins = sum(d > 0 for d in deltas)
    print(f"\n{'='*70}\n{wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}\n{'='*70}")


if __name__ == "__main__":
    main()
