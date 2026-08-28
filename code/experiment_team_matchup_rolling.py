# code/experiment_team_matchup_rolling.py
"""팀 매치업 피처(code/train.py::apply_team_matchup_features)의 cutoff7/season==2023
결과가 CatBoost 피딩 방향에서 정반대로 뒤집혀서(cutoff7 -6.94 vs 2023 +28.46), DeepFM 등
3rd-모델 후보를 닫을 때 썼던 season-level rolling-origin 타이브레이크를 여기도 적용한다.
23/24(cutoff7으로 이미 근사)는 결과가 있으니 2021/2022 두 폴드만 추가로 본다
(`code/experiment_thirdmodel_rolling_base.py`와 동일 관례: R-only로 F1 트랩 회피,
train<val_season, CatBoost 고정설정 1회 + MLP 3-seed).

폴드마다 baseline/→MLP/→CatBoost/→both 4변형을 새로 학습한다(캐시된 fold_base는
CAT_COLS+num_cols만 저장하고 TE-residual/pitchmix 포함 CatBoost 피처는 저장 안 하므로
재사용 불가 — 피처 엔지니어링 자체는 가벼우니 이번에도 전량 재계산).

사용법: python -m code.experiment_team_matchup_rolling
"""
import time

import numpy as np

from code.blend_model import fit_meta_model
from code.catboost_model import CAT_FEATURES
from code.experiment_thirdmodel_rolling_base import load_r_only, train_catboost_custom
from code.mlp_model import (
    CAT_COLS, QUANTILE_D, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.train import (
    TE_RESIDUAL_COLS, TEAM_MATCHUP_RESIDUAL_COLS, add_engineered_features,
    apply_te_residual_features, apply_team_matchup_features,
)
from code.trackman_pitcher_features import PITCHMIX_COLS, merge_coarse_pitchmix

TARGET_COL = "control_success"
FOLD_SEASONS = [2021, 2022]
SCREEN_SEEDS = [42, 123, 7]


def build_fold(df_all, df_trm, val_season):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values

    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)

    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in PITCHMIX_COLS]
    num_cols = [c for c in base_features if c not in CAT_COLS]

    all_cols = base_features + PITCHMIX_COLS + [TARGET_COL]
    train_split = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)

    te_prior = train_split[TARGET_COL].mean()
    train_split = apply_te_residual_features(train_split, train_split, te_prior)
    val_split = apply_te_residual_features(train_split, val_split, te_prior)
    cat_feature_cols = base_features + PITCHMIX_COLS + TE_RESIDUAL_COLS

    tm_prior = train_split[TARGET_COL].mean()
    train_split = apply_team_matchup_features(train_split, train_split, tm_prior)
    val_split = apply_team_matchup_features(train_split, val_split, tm_prior)

    print(f"  [val={val_season}] train={len(train_split)} val={len(val_split)} "
          f"team_matchup_covered train/val={train_split['team_matchup_covered'].mean():.4f}/"
          f"{val_split['team_matchup_covered'].mean():.4f}")
    return train_split, val_split, num_cols, cat_feature_cols


def run_variant(train_split, val_split, mlp_num_cols, cat_feature_cols, device):
    X_train_raw, y_train_raw = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[cat_feature_cols], val_split[TARGET_COL].values
    cat_model, cat_best_iter = train_catboost_custom(X_train_raw, y_train_raw, X_val_raw, y_val_raw, CAT_FEATURES)
    cat_pred = cat_model.predict_proba(X_val_raw)[:, 1]
    cat_score = compute_bss(cat_pred, y_val_raw)[2]

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_raw, seeds=SCREEN_SEEDS, device=device,
    )
    mlp_pred = predict_ensemble(
        members, cat_dims, len(mlp_num_cols), embed_dims, X_val_cat, X_val_num,
        bin_edges=bin_edges, quantile_d=QUANTILE_D, device=device,
    )
    mlp_score = compute_bss(mlp_pred, y_val_raw)[2]

    w_cat, w_mlp, ic, blend_score, _ = fit_meta_model(cat_pred, mlp_pred, y_val_raw)
    return dict(cat_score=cat_score, mlp_score=mlp_score, blend_score=blend_score)


def main():
    df_all, df_trm = load_r_only()
    device = get_device()

    all_results = {}
    for val_season in FOLD_SEASONS:
        print(f"\n{'='*20} fold: train<{val_season} -> val=={val_season} (R-only) {'='*20}")
        train_split, val_split, num_cols, cat_feature_cols = build_fold(df_all, df_trm, val_season)

        variants = {
            "baseline": (num_cols, cat_feature_cols),
            "team_matchup->MLP": (num_cols + TEAM_MATCHUP_RESIDUAL_COLS, cat_feature_cols),
            "team_matchup->CatBoost": (num_cols, cat_feature_cols + TEAM_MATCHUP_RESIDUAL_COLS),
            "team_matchup->both": (num_cols + TEAM_MATCHUP_RESIDUAL_COLS, cat_feature_cols + TEAM_MATCHUP_RESIDUAL_COLS),
        }
        fold_res = {}
        for name, (mlp_cols, cat_cols) in variants.items():
            t0 = time.time()
            res = run_variant(train_split, val_split, mlp_cols, cat_cols, device)
            print(f"  [{name}] CatBoost={res['cat_score']:.2f} MLP(3-seed)={res['mlp_score']:.2f} "
                  f"Blend={res['blend_score']:.2f} ({time.time()-t0:.1f}s)")
            fold_res[name] = res
        all_results[val_season] = fold_res

    print(f"\n{'='*30} 요약 (2021/2022 rolling-origin) {'='*30}")
    for name in ["baseline", "team_matchup->MLP", "team_matchup->CatBoost", "team_matchup->both"]:
        line = f"{name:26s}: "
        for val_season in FOLD_SEASONS:
            base = all_results[val_season]["baseline"]["blend_score"]
            cur = all_results[val_season][name]["blend_score"]
            d = cur - base
            line += f"[{val_season}] {cur:.2f}({d:+.2f})  "
        print(line)


if __name__ == "__main__":
    main()
