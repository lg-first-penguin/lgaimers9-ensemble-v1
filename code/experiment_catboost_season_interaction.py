# code/experiment_catboost_season_interaction.py
"""2026-08-24 세션: `code/experiment_mlp_ensemble_role.py`에서 MLP 3-seed 유망 ->
7-seed 반전으로 기각된 `season_gap_product`(pitcher_season_rate_gap x
batter_season_rate_gap)를 CatBoost 단독(시드 고정, 노이즈 훨씬 적음)으로도
확인해본다. pitchmix/te_single_axis는 이미 CatBoost 프로덕션 피처에 포함돼 있어
재테스트 대상이 아니다 — season_gap_product만 CatBoost에도 없는 새 컬럼이다.

사용법: python -m code.experiment_catboost_season_interaction
"""
import numpy as np

from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, TARGET_COL
from code.catboost_model import train_catboost, predict_catboost


def add_season_interaction(df):
    df = df.copy()
    p_gap = df["pitcher_season_rate_gap"].fillna(0.0)
    b_gap = df["batter_season_rate_gap"].fillna(0.0)
    df["season_gap_product"] = p_gap * b_gap
    conf = np.minimum(np.log1p(df["pitcher_season_n"]), np.log1p(df["batter_season_n"]))
    df["season_gap_product_weighted"] = p_gap * b_gap * conf
    return df


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    y_val = val_split[TARGET_COL].values

    X_train_base = train_split[cat_feature_cols]
    X_val_base = val_split[cat_feature_cols]
    model_base, it_base = train_catboost(X_train_base, train_split[TARGET_COL].values, X_val_base, y_val, verbose=False)
    base_score = compute_bss(predict_catboost(model_base, X_val_base), y_val)[2]
    print(f"[baseline] Val Score: {base_score:.2f} (best_iteration={it_base})")

    tr_season = add_season_interaction(train_split)
    va_season = add_season_interaction(val_split)
    season_cols = cat_feature_cols + ["season_gap_product", "season_gap_product_weighted"]
    X_train_s = tr_season[season_cols]
    X_val_s = va_season[season_cols]
    model_s, it_s = train_catboost(X_train_s, tr_season[TARGET_COL].values, X_val_s, y_val, verbose=False)
    s_score = compute_bss(predict_catboost(model_s, X_val_s), y_val)[2]
    print(f"[+season_interaction(2)] Val Score: {s_score:.2f} (best_iteration={it_s})")

    print(f"\nDelta: {s_score - base_score:+.2f}")
    return base_score, s_score


if __name__ == "__main__":
    run_regime(cutoff7=True, holdout=2024)
    run_regime(cutoff7=False, holdout=2023)
