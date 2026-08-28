# code/experiment_te_pcnt_bhand_foldcheck.py
"""다른 세션이 제보한 신규 3-way 축(pitcher x count x batter_hand, residual std
최고치 0.0350로 팀원 기록에서 지목됐던 축)의 단일 홀드아웃 결과(cutoff7 +4.01,
season2023 +16.18 — 둘 다 6축 자체의 노이즈 문턱 근처/이하)를, 이미 검증에 쓴
rolling-origin 3-fold 인프라(code/experiment_te_residual_foldcheck.py와 동일하게
2021/2022/2023을 각각 val로, game_type=='R'만 써서 F1 트랩 회피, §35 방식)로
재검증한다. code/train.py의 TE_AXES/TE_MAIN_AXES/causal_smoothed_te_encode를 그대로
재사용(수정 없음, import만) — 6축(현재 프로덕션) vs 7축(+신규)을 CatBoost 단독으로
비교한다.

사용법:
  python -m code.experiment_te_pcnt_bhand_foldcheck
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import compute_bss
from code.train import TE_AXES, TE_MAIN_AXES, add_engineered_features, causal_smoothed_te_encode
from code.trackman_pitcher_features import PITCHMIX_COLS, merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2021, 2022, 2023]

NEW_AXIS = ("te_p_cnt_bhand", ["pitcher_id", "balls_before", "strikes_before", "batter_hand"], "p_main")


def apply_te_residuals(source_df, query_df, prior, axes):
    """code/train.py::apply_te_residual_features와 동일 로직이지만 axes를 인자로
    받아 6축/7축을 같은 함수로 비교할 수 있게 한 버전."""
    query_df = query_df.copy()
    mains = {}
    covered_any = np.zeros(len(query_df), dtype=np.int64)
    for name, group_cols in TE_MAIN_AXES:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        mains[name] = enc
        covered_any = np.maximum(covered_any, covered)
    res_cols = []
    for name, group_cols, main_key in axes:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        query_df[f"{name}_res"] = enc - mains[main_key]
        covered_any = np.maximum(covered_any, covered)
        res_cols.append(f"{name}_res")
    query_df["te_covered"] = covered_any
    return query_df, res_cols + ["te_covered"]


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

    results = {}
    for tag, axes in [("6축(기존)", TE_AXES), ("7축(+p×cnt×bhand)", TE_AXES + [NEW_AXIS])]:
        tr, res_cols = apply_te_residuals(train_split, train_split, prior, axes)
        va, _ = apply_te_residuals(train_split, val_split, prior, axes)
        cat_feature_cols = base_features + PITCHMIX_COLS + res_cols

        X_train, y_train = tr[cat_feature_cols], tr[TARGET_COL].values
        X_val = va[cat_feature_cols]
        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"  [val={val_season}][{tag}] Val Score={score:.2f} (best_iter={best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_feature_cols)})")
        results[tag] = score
    return results


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행")

    deltas = []
    for val_season in FOLD_SEASONS:
        print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===")
        r = run_fold(df_all, df_trm, val_season)
        delta = r["7축(+p×cnt×bhand)"] - r["6축(기존)"]
        deltas.append(delta)
        print(f"  delta: {delta:+.2f}")

    wins = sum(d > 0 for d in deltas)
    print(f"\n{'='*70}\n{wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}\n{'='*70}")


if __name__ == "__main__":
    main()
