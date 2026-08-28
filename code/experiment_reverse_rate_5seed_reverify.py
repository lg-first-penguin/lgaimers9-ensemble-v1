# code/experiment_reverse_rate_5seed_reverify.py
"""팀원(조유담) 파이프라인과의 3가지 구조적 차이 중 3번째, 그리고 교차 파이프라인
불일치 재검증: `asof_pitcher_reverse_rate` 시즌진행분 분해.

우리 repo는 이 피처를 이미 기각했다(`code/experiment_reverse_rate_season_progression.py`,
단일시드 CatBoost + 3-seed MLP, cutoff7 -13.01 / season2023 +35.36 레짐반전).
그런데 팀원 쪽은 정확히 같은 피처(동일 근사 방식)를 half_2024/expand_2023 양쪽 fold
모두에서 플러스로 확인하고 채택, 실전에서도 1085.24 -> ~1090으로 재확인됐다.

§78(CatBoost 하이퍼파라미터 재검증)에서 단일시드 CatBoost의 레짐반전이 5-seed
앙상블로 노이즈를 걷어내니 사라진 전례가 있으므로, 이 피처도 같은 방식으로
재검증한다: CatBoost는 5-seed 앙상블(우리 프로덕션과 동일 `CATBOOST_SEED_POOL`),
MLP는 3-seed 스크리닝 유지(비용 때문에, quantile PLE 자체는 §2에서 별도 검증).

사용법: python -m code.experiment_reverse_rate_5seed_reverify --both
"""
import argparse
import time

import numpy as np

from code.catboost_model import CATBOOST_SEED_POOL, train_catboost_ensemble
from code.blend_model import fit_meta_model
from code.mlp_model import (
    CAT_COLS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.thirdmodel_common import build_split, TARGET_COL
from code.experiment_reverse_rate_season_progression import (
    build_reverse_season_end_lookup, apply_reverse_season_progression, REV_COLS,
)

SCREEN_SEEDS = [42, 123, 7]


def catboost_ensemble_score(train_split, val_split, cols, y_val, tag):
    t0 = time.time()
    X_train = train_split[cols]
    y_train = train_split[TARGET_COL].values
    X_val = val_split[cols]
    results = train_catboost_ensemble(X_train, y_train, X_val, y_val, seeds=CATBOOST_SEED_POOL, verbose=False)
    preds_list = [m.predict_proba(X_val)[:, 1] for m, _ in results]
    ensemble_pred = np.mean(preds_list, axis=0)
    score = compute_bss(ensemble_pred, y_val)[2]
    print(f"  [CatBoost {tag}] 5-seed 앙상블={score:.2f} ({time.time()-t0:.1f}s)", flush=True)
    return ensemble_pred, score


def mlp_ensemble_score(train_split, val_split, num_cols, y_val, device, tag):
    t0 = time.time()
    tr_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    va_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(tr_proc, CAT_COLS, num_cols, TARGET_COL)
    X_va_cat, X_va_num, _y_va = to_tensors(va_proc, CAT_COLS, num_cols, TARGET_COL)
    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_va_cat, X_val_num=X_va_num, y_val=y_val, seeds=SCREEN_SEEDS, device=device, verbose=False,
    )
    preds = predict_ensemble(members, cat_dims, len(num_cols), embed_dims, X_va_cat, X_va_num, bin_edges=bin_edges, device=device)
    score = compute_bss(preds, y_val)[2]
    print(f"  [MLP {tag}] 3-seed 앙상블={score:.2f} ({time.time()-t0:.1f}s)", flush=True)
    return preds, score


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (reverse_rate 시즌분해, 5-seed CatBoost 재검증) ===\n{'='*70}", flush=True)

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    y_val = val_split[TARGET_COL].values
    device = get_device()

    lookup = build_reverse_season_end_lookup(train_split)
    tr_rev = apply_reverse_season_progression(train_split, lookup)
    va_rev = apply_reverse_season_progression(val_split, lookup)

    cat_preds_base, cat_base = catboost_ensemble_score(train_split, val_split, cat_feature_cols, y_val, "baseline")
    mlp_preds_base, mlp_base = mlp_ensemble_score(train_split, val_split, mlp_num_cols, y_val, device, "baseline")
    _, _, _, blend_base, _ = fit_meta_model(cat_preds_base, mlp_preds_base, y_val)

    cat_preds_new, cat_new = catboost_ensemble_score(tr_rev, va_rev, cat_feature_cols + REV_COLS, y_val, "+reverse_season")
    mlp_preds_new, mlp_new = mlp_ensemble_score(tr_rev, va_rev, mlp_num_cols + REV_COLS, y_val, device, "+reverse_season")
    _, _, _, blend_new, _ = fit_meta_model(cat_preds_new, mlp_preds_new, y_val)

    print(f"\n[baseline] CatBoost(5seed)={cat_base:.2f} | MLP(3seed)={mlp_base:.2f} | Blend={blend_base:.2f}")
    print(f"[+reverse_season] CatBoost(5seed)={cat_new:.2f} | MLP(3seed)={mlp_new:.2f} | Blend={blend_new:.2f}")
    print(f"Delta: CatBoost={cat_new-cat_base:+.2f} | MLP={mlp_new-mlp_base:+.2f} | Blend={blend_new-blend_base:+.2f}")
    return {"cat_delta": cat_new - cat_base, "mlp_delta": mlp_new - mlp_base, "blend_delta": blend_new - blend_base}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    results = {}
    if args.both:
        results["cutoff7"] = run_regime(True, 2024)
        results["season2023"] = run_regime(False, 2023)
    else:
        holdout = 2024 if args.cutoff7 else args.holdout
        label = "cutoff7" if args.cutoff7 else f"season{holdout}"
        results[label] = run_regime(args.cutoff7, holdout)

    print("\n=== 요약 ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
