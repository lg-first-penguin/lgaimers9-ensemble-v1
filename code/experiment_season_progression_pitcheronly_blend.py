# code/experiment_season_progression_pitcheronly_blend.py
"""code/experiment_season_progression_ablation.py는 CatBoost 단독으로만 확인했다
(cutoff7: full8 689.61 vs pitcher-only 690.42, 거의 동률/노이즈 수준 / season==2023:
full8 721.68 vs pitcher-only 681.79, full8이 +39.89 뚜렷한 우세 — 두 레짐이 갈렸고
season==2023 쪽이 노이즈 문턱을 넘는 크기라 "타자 시즌진행분 유지(full8)"로 잠정
결론냈다). MLP는 CatBoost와 상호작용을 학습하는 방식이 근본적으로 달라서
(트리 분기 vs 임베딩+층), 같은 피처 제거가 반대로 나올 수 있다는 게 이 프로젝트의
반복된 패턴(핵심 교훈 #23/#24 — 서브모델별로 반드시 따로 확인) — 이 스크립트는
CatBoost 단독뿐 아니라 MLP(7-seed) 단독, 그리고 프로덕션 블렌드까지 3variant x
3지표로 재확인한다.

사용법:
  python -m code.experiment_season_progression_pitcheronly_blend --cutoff7
  python -m code.experiment_season_progression_pitcheronly_blend --holdout 2023
"""
import argparse
import time

from catboost import CatBoostClassifier, Pool

from code.blend_model import fit_meta_model
from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_season_progression import SEASON_PROGRESSION_COLS, TARGET_COL, build_split
from code.train import apply_f1_filter
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors,
    train_ensemble,
)

PITCHER_COLS = [c for c in SEASON_PROGRESSION_COLS if c.startswith("pitcher_")]
BATTER_COLS = [c for c in SEASON_PROGRESSION_COLS if c.startswith("batter_")]

VARIANTS = [
    ("none", SEASON_PROGRESSION_COLS),
    ("pitcher-only", BATTER_COLS),
    ("full8", []),
]


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
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols = build_split(holdout, cutoff7)
    drop_cols = ["row_id", TARGET_COL]

    results = {}
    for tag, exclude_cols in VARIANTS:
        base_features = [
            c for c in df.columns
            if c not in drop_cols and c not in trk_mlp_cols and c not in trk_cat_cols and c not in exclude_cols
        ]
        cat_feature_cols = base_features + trk_cat_cols
        mlp_num_cols = [c for c in base_features if c not in CAT_COLS] + trk_mlp_cols

        train_split = df.loc[train_mask, cat_feature_cols + [TARGET_COL]].reset_index(drop=True)
        val_split = df.loc[val_mask, cat_feature_cols + [TARGET_COL]].reset_index(drop=True)
        train_split = apply_f1_filter(train_split)
        y_val_raw = val_split[TARGET_COL].values

        t0 = time.time()
        cat_model, cat_best_iter = train_catboost_custom(
            train_split[cat_feature_cols], train_split[TARGET_COL].values,
            val_split[cat_feature_cols], y_val_raw, CAT_FEATURES,
        )
        cat_preds = cat_model.predict_proba(val_split[cat_feature_cols])[:, 1]
        cat_score = compute_bss(cat_preds, y_val_raw)[2]
        print(f"[{tag}][CatBoost] Val Score={cat_score:.2f} (best_iteration={cat_best_iter}, {time.time()-t0:.1f}s)")

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
        print(f"[{tag}][MLP(7-seed)] Val Score={mlp_score:.2f} ({time.time()-t0:.1f}s)")

        w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_preds, mlp_preds, y_val_raw)
        print(f"[{tag}][Blend] w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f} -> Val Score={blend_score:.2f}")

        results[tag] = (cat_score, mlp_score, blend_score)

    base = results["none"]
    print(f"\n--- {label} 요약 (vs none) ---")
    print(f"  {'variant':<14}{'CatBoost':>12}{'MLP(7seed)':>14}{'Blend':>12}")
    for tag, _ in VARIANTS:
        c, m, b = results[tag]
        print(f"  {tag:<14}{c:>8.2f}({c-base[0]:+.2f}) {m:>8.2f}({m-base[1]:+.2f}) {b:>8.2f}({b-base[2]:+.2f})")
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
