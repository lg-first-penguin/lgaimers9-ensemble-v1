# code/experiment_reverse_rate_season_progression.py
"""asof_pitcher_reverse_rate(공식 피처, control_success 기준 Cohen's d=-0.160 -- 트랙맨
물리지표 최댓값(d≈0.13)보다 크고 asof_pitcher_success_rate(d=0.169)와 비슷한 크기)에
code/train.py::SEASON_PROGRESSION_SPECS와 동일한 시즌진행분 분해 메커니즘을 적용해본다.

asof_pitcher_success_rate와 달리 reverse_rate는 행 단위 원본 이벤트 플래그가 없어(요약
통계만 제공됨) build_season_end_lookup의 "시즌 마지막 행 자신의 실제 결과(end_success)를
누적치에 더하는" 트릭을 그대로 쓸 수 없다. 근사: pre_n = end_n(마지막 행 자체는 미포함),
pre_success = round(end_n*end_rate) -- 투구 1건 오차, asof_pitcher_n 중앙값이 수천
단위라 무시 가능.

사용법:
  python -m code.experiment_reverse_rate_season_progression --regime cutoff7
  python -m code.experiment_reverse_rate_season_progression --regime 2023
"""
import argparse

import numpy as np
import pandas as pd

from code.mlp_model import (
    CAT_COLS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model
from code.thirdmodel_common import build_split, TARGET_COL

SCREEN_SEEDS = [42, 123, 7]

REV_SPEC = ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate")
REV_COLS = ["pitcher_reverse_season_n", "pitcher_reverse_season_count",
            "pitcher_reverse_season_rate", "pitcher_reverse_season_rate_gap"]


def build_reverse_season_end_lookup(df):
    role, id_col, n_col, rate_col = REV_SPEC
    idx = df.groupby([id_col, "season"])[n_col].idxmax()
    season_end = df.loc[idx, [id_col, "season", n_col, rate_col]].copy()
    season_end.columns = ["id", "season", "end_n", "end_rate"]
    season_end["season"] = season_end["season"] + 1
    return season_end


def apply_reverse_season_progression(df, lookup):
    df = df.copy()
    role, id_col, n_col, rate_col = REV_SPEC
    merged = df[[id_col, "season"]].merge(lookup, left_on=[id_col, "season"], right_on=["id", "season"], how="left")
    pre_n = merged["end_n"].fillna(0).values
    pre_success = (merged["end_n"] * merged["end_rate"]).round().fillna(0).values

    cum_n = df[n_col].values
    cum_success = np.round(df[n_col].values * df[rate_col].values)
    season_n = np.maximum(cum_n - pre_n, 0)
    season_success = np.maximum(cum_success - pre_success, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        season_rate = np.where(season_n > 0, season_success / season_n, np.nan)

    df["pitcher_reverse_season_n"] = season_n
    df["pitcher_reverse_season_count"] = season_success
    df["pitcher_reverse_season_rate"] = season_rate
    df["pitcher_reverse_season_rate_gap"] = season_rate - df[rate_col].values
    return df


def mlp_solo_score(train_split, val_split, num_cols, device, seeds=SCREEN_SEEDS):
    tr_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    va_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(tr_proc, CAT_COLS, num_cols, TARGET_COL)
    X_va_cat, X_va_num, y_va = to_tensors(va_proc, CAT_COLS, num_cols, TARGET_COL)
    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_va_cat, X_val_num=X_va_num, y_val=y_va, seeds=seeds, device=device, verbose=False,
    )
    preds = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_va_cat, X_va_num, bin_edges=bin_edges, device=device)
    return preds, compute_bss(preds, y_va.numpy())[2]


def catboost_solo_score(train_split, val_split, cat_cols):
    X_train = train_split[cat_cols]
    y_train = train_split[TARGET_COL].values
    X_val = val_split[cat_cols]
    y_val = val_split[TARGET_COL].values
    model, _ = train_catboost(X_train, y_train, X_val, y_val, verbose=False)
    preds = predict_catboost(model, X_val)
    return preds, compute_bss(preds, y_val)[2]


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    y_val = val_split[TARGET_COL].values
    device = get_device()

    lookup = build_reverse_season_end_lookup(train_split)
    tr_rev = apply_reverse_season_progression(train_split, lookup)
    va_rev = apply_reverse_season_progression(val_split, lookup)

    # baseline
    cat_preds_base, cat_base = catboost_solo_score(train_split, val_split, cat_feature_cols)
    mlp_preds_base, mlp_base = mlp_solo_score(train_split, val_split, mlp_num_cols, device)
    _, _, _, blend_base, _ = fit_meta_model(cat_preds_base, mlp_preds_base, y_val)

    # +reverse season progression (both models, same convention as success_rate season progression)
    cat_preds_new, cat_new = catboost_solo_score(tr_rev, va_rev, cat_feature_cols + REV_COLS)
    mlp_preds_new, mlp_new = mlp_solo_score(tr_rev, va_rev, mlp_num_cols + REV_COLS, device)
    _, _, _, blend_new, _ = fit_meta_model(cat_preds_new, mlp_preds_new, y_val)

    print(f"[baseline] CatBoost={cat_base:.2f} | MLP={mlp_base:.2f} | 2-way={blend_base:.2f}")
    print(f"[+reverse_season(4)] CatBoost={cat_new:.2f} | MLP={mlp_new:.2f} | 2-way={blend_new:.2f}")
    print(f"Delta: CatBoost={cat_new - cat_base:+.2f} | MLP={mlp_new - mlp_base:+.2f} | 2-way={blend_new - blend_base:+.2f}")
    return {
        "cat_delta": cat_new - cat_base, "mlp_delta": mlp_new - mlp_base, "blend_delta": blend_new - blend_base,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["cutoff7", "2023", "both"], default="both")
    args = parser.parse_args()
    results = {}
    if args.regime in ("cutoff7", "both"):
        results["cutoff7"] = run_regime(cutoff7=True, holdout=2024)
    if args.regime in ("2023", "both"):
        results["2023"] = run_regime(cutoff7=False, holdout=2023)
    print("\n=== 요약 ===")
    for k, v in results.items():
        print(f"{k}: {v}")
