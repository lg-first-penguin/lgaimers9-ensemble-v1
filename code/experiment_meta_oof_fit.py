# code/experiment_meta_oof_fit.py
"""MLP 고도화 5단계: CatBoost+MLP 스태킹 메타모델(w_cat/w_mlp/intercept)을 지금처럼
cutoff7 val 한 구간의 예측만으로 fit하는 대신, 여러 시즌에 걸친 rolling-origin
예측을 모아 fit하면 더 강건한(특정 구간 노이즈에 덜 흔들리는) 가중치가 나오는지 확인한다.

game_type=='R'만 써서(F1 필터가 이른 cutoff에서 학습 구간의 F행을 통째로 지워버리는
함정을 피함, code/experiment_reverse_season_foldcheck.py와 동일 관례) 2021~2024를
train<val 방식 rolling-origin으로 훑으며 각 fold의 CatBoost(고정 시드)+MLP(3-seed)
예측을 뽑는다. "미래 시즌을 예측할 때 메타 가중치를 어떻게 fit하는 게 나은가"라는
실제 프로덕션 용례를 그대로 재현하기 위해, 각 대상 fold i에 대해:
  - single: 바로 직전 fold(i-1) 하나의 예측만으로 메타모델 fit (지금 관례와 동일한 방식)
  - pooled: i보다 앞선 모든 fold를 합쳐서 메타모델 fit
두 방식으로 fold i의 블렌드 점수를 비교한다(2021은 이전 fold가 없어 대상에서 제외).

사용법:
  python -m code.experiment_meta_oof_fit
"""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.blend_model import fit_meta_model, predict_meta
from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import (
    CAT_COLS, apply_preprocessing, compute_bss, embed_dim_for_cardinality, fit_preprocessing,
    fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble, QUANTILE_N_BINS,
)
from code.train import TE_RESIDUAL_COLS, add_engineered_features, apply_te_residual_features
from code.trackman_pitcher_features import PITCHMIX_COLS, merge_coarse_pitchmix

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2020, 2021, 2022, 2023, 2024]
SCREEN_SEEDS = [42, 123, 7]


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
    df = df[df["game_type"] == "R"].reset_index(drop=True)  # F1 트랩 회피

    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    return df, df_trm


def run_fold(df_all, df_trm, val_season, device):
    """이 fold의 (cat_pred, mlp_pred, y_val)을 반환한다."""
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
    y_val = val_split[TARGET_COL].values

    te_prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, te_prior)
    val_split = apply_te_residual_features(te_source, val_split, te_prior)

    cat_feature_cols = base_features + PITCHMIX_COLS + TE_RESIDUAL_COLS
    X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val = val_split[cat_feature_cols]
    t0 = time.time()
    cat_model, cat_best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
    cat_pred = cat_model.predict_proba(X_val)[:, 1]
    print(f"  [val={val_season}] CatBoost 완료 (best_iter={cat_best_iter}, {time.time()-t0:.1f}s)")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val, seeds=SCREEN_SEEDS, device=device,
    )
    mlp_pred = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_val_cat, X_val_num, bin_edges=bin_edges, device=device)
    print(f"  [val={val_season}] MLP(3-seed) 완료 ({time.time()-t0:.1f}s)")

    cat_score = compute_bss(cat_pred, y_val)[2]
    mlp_score = compute_bss(mlp_pred, y_val)[2]
    print(f"  [val={val_season}] CatBoost={cat_score:.2f} | MLP={mlp_score:.2f}")

    return cat_pred, mlp_pred, y_val


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행")
    device = get_device()

    fold_preds = {}
    for val_season in FOLD_SEASONS:
        print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===")
        fold_preds[val_season] = run_fold(df_all, df_trm, val_season, device)

    print(f"\n{'='*70}\n메타모델 fit 방식 비교 (single-window vs pooled-OOF)\n{'='*70}")
    results = []
    for i, val_season in enumerate(FOLD_SEASONS):
        if i == 0:
            continue  # 2021은 이전 fold가 없어 대상에서 제외
        prior_seasons = FOLD_SEASONS[:i]
        target_cat, target_mlp, target_y = fold_preds[val_season]

        # single: 바로 직전 fold 하나로만 메타모델 fit (지금 프로덕션과 동일한 방식)
        single_season = prior_seasons[-1]
        s_cat, s_mlp, s_y = fold_preds[single_season]
        w_cat_s, w_mlp_s, ic_s, _, _ = fit_meta_model(s_cat, s_mlp, s_y)
        single_pred = predict_meta(w_cat_s, w_mlp_s, ic_s, target_cat, target_mlp)
        single_score = compute_bss(single_pred, target_y)[2]

        # pooled: target 이전의 모든 fold를 합쳐서 메타모델 fit
        pooled_cat = np.concatenate([fold_preds[s][0] for s in prior_seasons])
        pooled_mlp = np.concatenate([fold_preds[s][1] for s in prior_seasons])
        pooled_y = np.concatenate([fold_preds[s][2] for s in prior_seasons])
        w_cat_p, w_mlp_p, ic_p, _, _ = fit_meta_model(pooled_cat, pooled_mlp, pooled_y)
        pooled_pred = predict_meta(w_cat_p, w_mlp_p, ic_p, target_cat, target_mlp)
        pooled_score = compute_bss(pooled_pred, target_y)[2]

        delta = pooled_score - single_score
        results.append(delta)
        print(f"[target={val_season}] single(fit on {single_season})={single_score:.2f} "
              f"(w_cat={w_cat_s:.3f}, w_mlp={w_mlp_s:.3f}) | "
              f"pooled(fit on {prior_seasons})={pooled_score:.2f} "
              f"(w_cat={w_cat_p:.3f}, w_mlp={w_mlp_p:.3f}) | delta={delta:+.2f}")

    wins = sum(d > 0 for d in results)
    print(f"\n{'='*70}\n{wins}/{len(results)} fold 승리, 평균 delta: {np.mean(results):+.2f}\n{'='*70}")


if __name__ == "__main__":
    main()
