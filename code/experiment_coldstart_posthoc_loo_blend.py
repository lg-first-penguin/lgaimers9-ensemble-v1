# code/experiment_coldstart_posthoc_loo_blend.py
"""code/experiment_coldstart_posthoc_loo.py는 CatBoost 단독으로만 콜드스타트
leave-one-out 사후보정을 검증했다. "정말 손해가 없냐"는 질문에 제대로 답하려면
실제 프로덕션이 쓰는 CatBoost+MLP 블렌드 레벨에서도 같은 검증을 해야 한다 —
CatBoost 단독에서 본 효과가 메타모델 재가중치를 거치면 사라지거나(§54의 CatBoost
시드 앙상블이 겪은 일) 부호가 바뀔 수 있기 때문이다.

rolling-origin 3-fold(R-only, train<val_season, val_season∈{2021,2022,2023})를
그대로 쓰되, MLP는 스크리닝 규모(3-seed, MLP_SEEDS_SCREEN)로 학습해 CatBoost+MLP
블렌드를 만들고, 콜드스타트(asof_pitcher_n 하위 25%) 구간에만 다른 fold에서
추정한 보정량을 적용하는 leave-one-out을 블렌드 예측값에 대해 수행한다.
"""
import time

import numpy as np
import pandas as pd

from code.blend_model import fit_meta_model, predict_meta
from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.train import add_engineered_features, apply_te_residual_features, TE_RESIDUAL_COLS
from code.trackman_pitcher_features import PITCHMIX_COLS, merge_coarse_pitchmix
from catboost import CatBoostClassifier, Pool

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
FOLD_SEASONS = [2021, 2022, 2023]
MLP_SEEDS_SCREEN = [42, 123, 7]


def load_r_only():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df[df["game_type"] == "R"].reset_index(drop=True)
    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    return df, df_trm


def build_features(df_all, df_trm, val_season):
    train_mask = (df_all["season"] < val_season).values
    val_mask = (df_all["season"] == val_season).values
    league_success_mean = df_all.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df_all.copy(), league_success_mean)
    df = merge_coarse_pitchmix(df, df_trm, holdout=val_season)
    return df, train_mask, val_mask


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


def run_fold(df_all, df_trm, val_season):
    df, train_mask, val_mask = build_features(df_all, df_trm, val_season)
    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols]

    all_cols = list(dict.fromkeys(base_features + [TARGET_COL, "asof_pitcher_n"]))
    train_split = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    y_val = val_split[TARGET_COL].values

    prior = train_split[TARGET_COL].mean()
    te_source = train_split
    train_split = apply_te_residual_features(te_source, train_split, prior)
    val_split = apply_te_residual_features(te_source, val_split, prior)
    cat_feature_cols = base_features + TE_RESIDUAL_COLS

    X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val = val_split[cat_feature_cols]

    t0 = time.time()
    cat_pred = train_catboost(X_train, y_train, X_val, y_val)
    print(f"  [val={val_season}] CatBoost 완료 ({time.time()-t0:.1f}s) solo={compute_bss(cat_pred, y_val)[2]:.2f}")

    # MLP: coarse pitchmix(->CatBoost 전용)는 num_cols에서 제외해 프로덕션과 동일하게 맞춘다.
    num_cols = [c for c in cat_feature_cols if c not in CAT_COLS and c not in PITCHMIX_COLS]
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, TARGET_COL)
    y_val_np = val_proc[TARGET_COL].values
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    t0 = time.time()
    device = get_device()
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np, seeds=MLP_SEEDS_SCREEN, device=device,
    )
    mlp_pred = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_val_cat, X_val_num, bin_edges=bin_edges, device=device)
    print(f"  [val={val_season}] MLP 3-seed 완료 ({time.time()-t0:.1f}s) solo={compute_bss(mlp_pred, y_val_np)[2]:.2f}")

    assert np.array_equal(y_val, y_val_np), "행 순서 불일치"

    w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_pred, mlp_pred, y_val)
    blend_pred = predict_meta(w_cat, w_mlp, intercept, cat_pred, mlp_pred)
    print(f"  [val={val_season}] 블렌드 score={blend_score:.2f} (w_cat={w_cat:.3f} w_mlp={w_mlp:.3f})")

    n_pitcher = val_split["asof_pitcher_n"].values
    q1 = np.quantile(n_pitcher, 0.25)
    cold_mask = n_pitcher <= q1
    cold_gap = blend_pred[cold_mask].mean() - y_val[cold_mask].mean()
    print(f"  [val={val_season}] 콜드(하위25%) gap={cold_gap:+.4f} (n={cold_mask.sum()})")

    return {"val_season": val_season, "blend_pred": blend_pred, "y_val": y_val, "cold_mask": cold_mask, "cold_gap": cold_gap, "blend_score": blend_score}


def main():
    df_all, df_trm = load_r_only()
    print(f"[R-only 로드] {len(df_all)}행")

    fold_data = {}
    for val_season in FOLD_SEASONS:
        print(f"\n=== fold: train<{val_season} -> val=={val_season} (블렌드) ===")
        fold_data[val_season] = run_fold(df_all, df_trm, val_season)

    print(f"\n{'='*78}\n=== 블렌드 레벨 leave-one-out 콜드스타트 사후보정 검증 ===\n{'='*78}")
    deltas = []
    for val_season in FOLD_SEASONS:
        d = fold_data[val_season]
        other_gaps = [fold_data[s]["cold_gap"] for s in FOLD_SEASONS if s != val_season]
        correction = np.mean(other_gaps)
        preds, y_val, cold_mask = d["blend_pred"], d["y_val"], d["cold_mask"]
        baseline_score = compute_bss(preds, y_val)[2]
        corrected = preds.copy()
        corrected[cold_mask] = np.clip(corrected[cold_mask] - correction, 0, 1)
        corrected_score = compute_bss(corrected, y_val)[2]
        cold_base = compute_bss(preds[cold_mask], y_val[cold_mask])[2]
        cold_corr = compute_bss(corrected[cold_mask], y_val[cold_mask])[2]
        delta = corrected_score - baseline_score
        deltas.append(delta)
        print(f"  val={val_season}: 다른 fold서 추정한 보정량={correction:+.4f} | 전체 {baseline_score:.2f}->{corrected_score:.2f} (delta={delta:+.2f}) | 콜드subgroup {cold_base:.2f}->{cold_corr:.2f} (delta={cold_corr-cold_base:+.2f})")

    print(f"\n{sum(x>0 for x in deltas)}/{len(deltas)} fold 승리, 평균 delta={np.mean(deltas):+.2f}")


if __name__ == "__main__":
    main()
