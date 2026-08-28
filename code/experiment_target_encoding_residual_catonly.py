# code/experiment_target_encoding_residual_catonly.py
"""code/experiment_target_encoding_residual.py의 결과(cutoff7: CatBoost +12.05 /
MLP -24.51 / Blend -2.38, season==2023: 전부 +) 를 보고 "TE 잔차 6개를 CatBoost에만
먹이자"는 제안을 검증한다. MLP는 TE 피처를 아예 안 보므로 baseline과 완전히 동일한
입력으로 학습된다 — 그래서 MLP는 딱 한 번만 학습하고, baseline CatBoost / TE-추가
CatBoost 두 번만 새로 학습해서 각각 그 동일한 MLP 예측과 블렌드한다(재검증 시간
단축, MLP 7-seed 학습이 가장 오래 걸리는 부분이라 이 절감 효과가 큼).

사용법:
  python -m code.experiment_target_encoding_residual_catonly --cutoff7
  python -m code.experiment_target_encoding_residual_catonly --holdout 2023
"""
import argparse
import time

from catboost import CatBoostClassifier, Pool

from code.blend_model import fit_meta_model
from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_season_progression import TARGET_COL, build_split
from code.experiment_target_encoding_residual import TE_RESIDUAL_COLS, add_te_residual_features
from code.train import apply_f1_filter
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors,
    train_ensemble,
)


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def run_regime(holdout, cutoff7):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (TE 잔차 -> CatBoost 전용) ===\n{'='*70}")

    df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols = build_split(holdout, cutoff7)
    drop_cols = ["row_id", TARGET_COL]
    base_features = [c for c in df.columns if c not in drop_cols and c not in trk_mlp_cols and c not in trk_cat_cols]

    all_cols = base_features + trk_mlp_cols + trk_cat_cols + [TARGET_COL]
    train_split = df.loc[train_mask, all_cols].reset_index(drop=True)
    val_split = df.loc[val_mask, all_cols].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)
    y_val_raw = val_split[TARGET_COL].values

    prior = train_split[TARGET_COL].mean()
    t0 = time.time()
    te_source = train_split
    train_split = add_te_residual_features(te_source, train_split, prior)
    val_split = add_te_residual_features(te_source, val_split, prior)
    print(f"[TE 잔차 피처 생성] {time.time()-t0:.1f}s (prior={prior:.4f})")

    # MLP는 TE 피처를 절대 안 봄 -> baseline과 동일 -> 딱 한 번만 학습
    cat_feature_cols_base = base_features + trk_cat_cols
    mlp_num_cols = [c for c in base_features if c not in CAT_COLS] + trk_mlp_cols

    device = get_device()
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr_t = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num)

    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr_t, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_raw,
        seeds=ENSEMBLE_SEEDS, device=device, verbose=False,
    )
    mlp_preds = predict_ensemble(members, cat_dims, len(mlp_num_cols), embed_dims, X_val_cat, X_val_num, bin_edges=bin_edges, device=device)
    mlp_score = compute_bss(mlp_preds, y_val_raw)[2]
    print(f"[MLP(7-seed), TE 미포함, 두 variant 공용] Val Score={mlp_score:.2f} ({time.time()-t0:.1f}s)")

    results = {}
    for tag, cat_feature_cols in [("baseline", cat_feature_cols_base), ("+TE잔차6개(CatBoost만)", cat_feature_cols_base + TE_RESIDUAL_COLS)]:
        X_train, y_train = train_split[cat_feature_cols], train_split[TARGET_COL].values
        X_val = val_split[cat_feature_cols]

        t0 = time.time()
        cat_model, cat_best_iter = train_catboost_custom(X_train, y_train, X_val, y_val_raw, CAT_FEATURES)
        cat_preds = cat_model.predict_proba(X_val)[:, 1]
        cat_score = compute_bss(cat_preds, y_val_raw)[2]
        print(f"[{tag}][CatBoost] Val Score={cat_score:.2f} (best_iteration={cat_best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_feature_cols)})")

        w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_preds, mlp_preds, y_val_raw)
        print(f"[{tag}][Blend w/ 공용 MLP] w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f} -> Val Score={blend_score:.2f}")
        results[tag] = (cat_score, blend_score)

    base = results["baseline"]
    print(f"\n--- {label} 요약 (MLP={mlp_score:.2f} 공용, vs baseline) ---")
    for tag in ["baseline", "+TE잔차6개(CatBoost만)"]:
        c, b = results[tag]
        print(f"  {tag}: CatBoost {c:.2f}({c-base[0]:+.2f})  Blend {b:.2f}({b-base[1]:+.2f})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7)


if __name__ == "__main__":
    main()
