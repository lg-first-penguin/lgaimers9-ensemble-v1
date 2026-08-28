# code/experiment_asof_season_progression_all.py
"""시즌진행분(season progression, code/train.py::SEASON_PROGRESSION_SPECS)과 동일한
메커니즘(전 시즌 끝 누적치를 빼서 "이번 시즌만의" 값 + 커리어 대비 격차 분리)을,
asof_pitcher_reverse_rate(별도 fork에서 진행 중, 이 스크립트 범위 아님)를 제외한
나머지 7개 공식 asof_* 비율 컬럼에 적용한다:
  - asof_pitcher_middle_rate/ball_rate/strike_rate (n=asof_pitcher_n)
  - asof_batter_middle_rate (n=asof_batter_n)
  - asof_pitcher_fastball_rate/breaking_rate/offspeed_rate
    (n=asof_pitcher_pitchmix_n -- 실측 확인 결과 asof_pitcher_n과 항상 동일해 n_col로
    asof_pitcher_n을 그대로 씀)

원조 시즌진행분(asof_pitcher_success_rate)은 raw per-row 이벤트가 control_success라서
시즌 마지막 행 자신의 실제 결과(end_success)를 더해 "그 시즌 끝난 시점의 진짜 최종
누적치"를 정확히 복원했다. 여기 7개 컬럼은 원본 이벤트 플래그가 없으므로
pre_n=end_n, pre_count=round(end_n*end_rate) 근사만 쓴다(마지막 투구 1건 오차,
n이 수천 단위라 무시 가능).

사용법:
  python -m code.experiment_asof_season_progression_all --regime cutoff7
  python -m code.experiment_asof_season_progression_all --regime 2023
  python -m code.experiment_asof_season_progression_all --regime cutoff7 --combined a,b,c
"""
import argparse

import numpy as np
import pandas as pd

from code.mlp_model import (
    CAT_COLS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.catboost_model import train_catboost, predict_catboost
from code.thirdmodel_common import build_split, TARGET_COL

SCREEN_SEEDS = [42, 123, 7]

GENERIC_SPECS = [
    ("pitcher_middle", "pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate"),
    ("pitcher_ball", "pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate"),
    ("pitcher_strike", "pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate"),
    ("batter_middle", "batter_id", "asof_batter_n", "asof_batter_middle_rate"),
    ("pitcher_fastball", "pitcher_id", "asof_pitcher_n", "asof_pitcher_fastball_rate"),
    ("pitcher_breaking", "pitcher_id", "asof_pitcher_n", "asof_pitcher_breaking_rate"),
    ("pitcher_offspeed", "pitcher_id", "asof_pitcher_n", "asof_pitcher_offspeed_rate"),
]


def build_generic_season_end_lookup(df, specs):
    tables = []
    for name, id_col, n_col, rate_col in specs:
        idx = df.groupby([id_col, "season"])[n_col].idxmax()
        season_end = df.loc[idx, [id_col, "season", n_col, rate_col]].copy()
        season_end.columns = ["id", "season", "end_n", "end_rate"]
        season_end["season"] = season_end["season"] + 1
        season_end.insert(0, "name", name)
        tables.append(season_end)
    return pd.concat(tables, ignore_index=True)


def apply_generic_season_progression(df, lookup, specs):
    df = df.copy()
    new_cols = []
    for name, id_col, n_col, rate_col in specs:
        lut = lookup.loc[lookup["name"] == name, ["id", "season", "end_n", "end_rate"]]
        merged = df[[id_col, "season"]].merge(
            lut, left_on=[id_col, "season"], right_on=["id", "season"], how="left",
        )
        pre_n = merged["end_n"].fillna(0).values
        pre_count = (merged["end_n"] * merged["end_rate"]).round().fillna(0).values

        cum_n = df[n_col].values
        cum_count = np.round(df[n_col].values * df[rate_col].values)
        season_n = np.maximum(cum_n - pre_n, 0)
        season_count = np.maximum(cum_count - pre_count, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            season_rate = np.where(season_n > 0, season_count / season_n, np.nan)

        c_n, c_cnt, c_rate, c_gap = (f"{name}_season_n", f"{name}_season_count",
                                      f"{name}_season_rate", f"{name}_season_rate_gap")
        df[c_n] = season_n
        df[c_cnt] = season_count
        df[c_rate] = season_rate
        df[c_gap] = season_rate - df[rate_col].values
        new_cols += [c_n, c_cnt, c_rate, c_gap]
    return df, new_cols


def catboost_solo_score(train_split, val_split, cols):
    X_train, y_train = train_split[cols], train_split[TARGET_COL].values
    X_val, y_val = val_split[cols], val_split[TARGET_COL].values
    model, _ = train_catboost(X_train, y_train, X_val, y_val, verbose=False)
    return compute_bss(predict_catboost(model, X_val), y_val)[2]


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
    return compute_bss(preds, y_va.numpy())[2]


def run(regime, specs_subset=None, label=""):
    cutoff7 = regime == "cutoff7"
    holdout = 2024 if cutoff7 else 2023
    print(f"\n{'='*70}\n=== 레짐: {regime} | 후보: {label or 'ALL'} ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)

    lookup = build_generic_season_end_lookup(train_split, GENERIC_SPECS)
    specs = specs_subset if specs_subset is not None else GENERIC_SPECS

    tr2, new_cols = apply_generic_season_progression(train_split, lookup, specs)
    va2, _ = apply_generic_season_progression(val_split, lookup, specs)

    cat_base = catboost_solo_score(train_split, val_split, cat_feature_cols)
    cat_new = catboost_solo_score(tr2, va2, cat_feature_cols + new_cols)

    device = get_device()
    mlp_base = mlp_solo_score(train_split, val_split, mlp_num_cols, device)
    mlp_new = mlp_solo_score(tr2, va2, mlp_num_cols + new_cols, device)

    print(f"[{label or 'ALL'}] CatBoost solo: base={cat_base:.2f} new={cat_new:.2f} delta={cat_new-cat_base:+.2f}")
    print(f"[{label or 'ALL'}] MLP solo:      base={mlp_base:.2f} new={mlp_new:.2f} delta={mlp_new-mlp_base:+.2f}")
    return cat_new - cat_base, mlp_new - mlp_base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["cutoff7", "2023"], default="cutoff7")
    parser.add_argument("--only", type=str, default=None, help="콤마구분 name 목록 (GENERIC_SPECS의 name), 지정 안하면 개별 전체 순회")
    args = parser.parse_args()

    if args.only:
        names = set(args.only.split(","))
        subset = [s for s in GENERIC_SPECS if s[0] in names]
        run(args.regime, subset, label="+".join(names))
    else:
        results = {}
        for spec in GENERIC_SPECS:
            name = spec[0]
            cat_d, mlp_d = run(args.regime, [spec], label=name)
            results[name] = (cat_d, mlp_d)
        print(f"\n{'='*70}\n=== {args.regime} 개별 요약 ===")
        for name, (cat_d, mlp_d) in results.items():
            print(f"  {name:20s} CatBoost delta={cat_d:+7.2f} | MLP delta={mlp_d:+7.2f}")
