# code/experiment_hparam_reverify.py
"""§41(950.81, 982.22 대비 -31.41 하락) 원인 후보로 지목된 "cutoff=7·tier A+coarse
pitchmix 도입 이후 CatBoost/MLP 하이퍼파라미터가 한 번도 재탐색되지 않았다"를
검증하기 위한 재탐색(`code/tune.py`, `code/tune_mlp.py`) 결과를 7-seed 프로덕션
앙상블 + 양쪽 레짐(cutoff7/season==2023)으로 재확인한다.

`code/tune_mlp.py`는 비용 때문에 3-seed(SCREEN_SEEDS)로 스크리닝하므로, §40에서 확인한
3-seed 노이즈(~19점) 때문에 best trial이 진짜 개선인지 알 수 없다 — 이 스크립트가
그 재검증 단계다. CatBoost는 결정적(seed 고정)이라 재검증 필요성이 상대적으로 작지만,
40 trial Optuna가 cutoff7 단일 홀드아웃(11만행)에 과최적화됐을 위험은 여전히 있어
동일하게 두 레짐으로 재확인한다.

사용법:
  python -m code.experiment_hparam_reverify --cutoff7
  python -m code.experiment_hparam_reverify --holdout 2023
"""
import argparse
import json
import time

import numpy as np
from catboost import CatBoostClassifier, Pool

import code.experiment_residual_correction_9_10 as split_mod
import code.mlp_model as mlp_model
from code.blend_model import fit_meta_model
from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_residual_correction_9_10 import build_split
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors,
    train_ensemble,
)

TARGET_COL = "control_success"
CATBOOST_TUNE_JSON = "./open/temp/best_hparams.json"
MLP_TUNE_JSON = "./open/temp/best_mlp_hparams.json"
CATBOOST_TUNABLE_KEYS = [
    "depth", "learning_rate", "l2_leaf_reg", "random_strength",
    "bagging_temperature", "border_count", "min_data_in_leaf",
]


def train_catboost_custom(X_train, y_train, X_val, y_val, override=None):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    if override:
        for k in CATBOOST_TUNABLE_KEYS:
            params[k] = override[k]
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def run_regime(holdout, cutoff7, catboost_override, mlp_override, pitchmix_only=False, skip_catboost_tuned=False):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    if pitchmix_only:
        label += " (pitchmix-only, tier A 제외)"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    orig_tier_feed = split_mod.TIER_FEED
    if pitchmix_only:
        split_mod.TIER_FEED = {}
    try:
        train_split, val_split, features, cat_features, mlp_num_cols = build_split(holdout, cutoff7)
    finally:
        split_mod.TIER_FEED = orig_tier_feed
    device = get_device()

    X_train_raw = train_split[cat_features]
    y_train_raw = train_split[TARGET_COL].values
    X_val_raw = val_split[cat_features]
    y_val_raw = val_split[TARGET_COL].values

    cat_tags = [("baseline", None)] if skip_catboost_tuned else [("baseline", None), ("tuned", catboost_override)]
    results = {}
    for tag, override in cat_tags:
        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train_raw, y_train_raw, X_val_raw, y_val_raw, override)
        preds = model.predict_proba(X_val_raw)[:, 1]
        score = compute_bss(preds, y_val_raw)[2]
        results[f"cat_{tag}"] = (preds, score)
        print(f"[CatBoost-{tag}] Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s)")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr_t = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num)

    orig_dropout = mlp_model.DROPOUT
    for tag, override in [("baseline", None), ("tuned", mlp_override)]:
        t0 = time.time()
        kwargs = {}
        if override:
            mlp_model.DROPOUT = override["dropout"]
            kwargs = dict(lr=override["lr"], weight_decay=override["weight_decay"], batch_size=override["batch_size"])
        else:
            mlp_model.DROPOUT = orig_dropout
        members = train_ensemble(
            X_tr_cat, X_tr_num, y_tr_t, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_raw,
            seeds=ENSEMBLE_SEEDS, device=device, verbose=False, **kwargs,
        )
        mlp_model.DROPOUT = orig_dropout
        preds = predict_ensemble(members, cat_dims, len(mlp_num_cols), embed_dims, X_val_cat, X_val_num, bin_edges=bin_edges, device=device)
        score = compute_bss(preds, y_val_raw)[2]
        results[f"mlp_{tag}"] = (preds, score)
        print(f"[MLP(7-seed)-{tag}] Val Score={score:.2f} ({time.time()-t0:.1f}s)")

    print(f"\n[블렌드 비교 — {label}]")
    for cat_tag, _ in cat_tags:
        for mlp_tag in ["baseline", "tuned"]:
            cat_preds, cat_score = results[f"cat_{cat_tag}"]
            mlp_preds, mlp_score = results[f"mlp_{mlp_tag}"]
            w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_preds, mlp_preds, y_val_raw)
            print(f"  CatBoost={cat_tag}({cat_score:.2f}) + MLP={mlp_tag}({mlp_score:.2f}) -> Blend={blend_score:.2f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--pitchmix-only", action="store_true",
                         help="tier A를 빼고 coarse pitchmix만 남긴 후보 피처셋으로 검증(EXPERIMENTS.md §43)")
    parser.add_argument("--mlp-json", type=str, default=MLP_TUNE_JSON)
    parser.add_argument("--skip-catboost-tuned", action="store_true",
                         help="CatBoost 재탐색은 이미 기각 확정(§42)이라 재실행하지 않고 baseline만 씀")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout

    catboost_override = None
    if not args.skip_catboost_tuned:
        with open(CATBOOST_TUNE_JSON) as f:
            catboost_override = json.load(f)["best_params"]
        print(f"[CatBoost tuned params] {catboost_override}")
    with open(args.mlp_json) as f:
        mlp_override = json.load(f)["best_params"]
    print(f"[MLP tuned params] {mlp_override}")

    run_regime(holdout, args.cutoff7, catboost_override, mlp_override,
                pitchmix_only=args.pitchmix_only, skip_catboost_tuned=args.skip_catboost_tuned)


if __name__ == "__main__":
    main()
