# code/experiment_reverse_rate_catboost_only.py
"""asof_pitcher_reverse_rate 시즌진행분(code/experiment_reverse_rate_season_progression.py)이
MLP/블렌드에서 cutoff7<->season2023 부호반전으로 기각됐으나, CatBoost 단독은 양쪽 다
플러스였다(cutoff7 +1.80/season2023 +26.87). TE-residual/coarse pitchmix가 CatBoost
전용으로 채택된 것과 같은 패턴인지 확인하기 위해, 이번엔 4컬럼을 CatBoost에만 먹이고
MLP은 건드리지 않는다(MLP는 ENSEMBLE_SEEDS 7개 전체로 한 번만 학습해 baseline/신규
버전 양쪽의 블렌드 계산에 재사용 -- 어차피 피처가 안 바뀌므로 재학습 불필요).

사용법:
  python -m code.experiment_reverse_rate_catboost_only --regime cutoff7
  python -m code.experiment_reverse_rate_catboost_only --regime 2023
"""
import argparse

from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model
from code.thirdmodel_common import build_split, TARGET_COL
from code.experiment_reverse_rate_season_progression import (
    REV_COLS, apply_reverse_season_progression, build_reverse_season_end_lookup,
)


def mlp_solo_score(train_split, val_split, num_cols, device, seeds=ENSEMBLE_SEEDS):
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
    print(f"\n{'='*70}\n=== 레짐: {label} (CatBoost 전용, MLP 7-seed 고정) ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    y_val = val_split[TARGET_COL].values
    device = get_device()

    lookup = build_reverse_season_end_lookup(train_split)
    tr_rev = apply_reverse_season_progression(train_split, lookup)
    va_rev = apply_reverse_season_progression(val_split, lookup)

    # MLP: 피처 변경 없음 -> 7-seed로 한 번만 학습해 재사용
    mlp_preds, mlp_score = mlp_solo_score(train_split, val_split, mlp_num_cols, device, seeds=ENSEMBLE_SEEDS)
    print(f"[MLP 7-seed 고정] solo={mlp_score:.2f}")

    # CatBoost: baseline vs +reverse_season(4, CatBoost 전용)
    cat_preds_base, cat_base = catboost_solo_score(train_split, val_split, cat_feature_cols)
    cat_preds_new, cat_new = catboost_solo_score(tr_rev, va_rev, cat_feature_cols + REV_COLS)

    _, _, _, blend_base, _ = fit_meta_model(cat_preds_base, mlp_preds, y_val)
    _, _, _, blend_new, _ = fit_meta_model(cat_preds_new, mlp_preds, y_val)

    print(f"[baseline] CatBoost={cat_base:.2f} | 2-way={blend_base:.2f}")
    print(f"[+reverse_season(4)->CatBoost전용] CatBoost={cat_new:.2f} | 2-way={blend_new:.2f}")
    print(f"Delta: CatBoost={cat_new - cat_base:+.2f} | 2-way={blend_new - blend_base:+.2f}")
    return {"cat_delta": cat_new - cat_base, "blend_delta": blend_new - blend_base}


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
