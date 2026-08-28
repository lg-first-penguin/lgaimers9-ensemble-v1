# code/experiment_catboost_te_retune.py
"""2026-08-25: TE-residual(2026-08-20 도입)을 처음으로 포함한 `code/tune.py`
Optuna 재탐색(`open/temp/best_hparams.json`) 결과를 프로덕션 값과 비교 검증한다.

이전 CatBoost 재탐색 시도 세 번(§43/§45, §72 boosting grid, §74 팀원 하이퍼파라미터)
모두 "dual-regime 단일 시드에서는 통과처럼 보이지만 cutoff7(프로덕션 승격 기준)이
마이너스/노이즈, season==2023만 크게 플러스"인 동일한 과적합 패턴으로 최종 기각됐다
(핵심 교훈 #28: dual-regime 단일 시드 통과만으로는 채택 근거가 안 됨). 이번엔
TE-residual이 포함된 첫 재탐색이라는 점이 다르지만, 같은 방식으로 검증한다:
1) dual-regime(cutoff7 단일시드 + season==2023 단일시드) 통과 여부를 먼저 보고
2) 통과하면 §72와 동일하게 3-seed(42/123/7) 재검증까지 거친 뒤에만 프로덕션 반영을
   고려한다.

**Optuna 탐색 자체는 CatBoost 솔로 점수만 최적화한다(의도적)** — 핵심 교훈 #23
("블렌드 델타 플러스만으로는 유용성을 알 수 없다, 메타모델 재가중치가 서브모델
훼손을 가릴 수 있다")에 따라, 탐색 목적함수를 블렌드로 바꾸지 않는다. 대신 이
재검증 스크립트에서 solo(CatBoost) 델타와, 실제 MLP 7-seed 앙상블 + 스태킹
메타모델(`code/blend_model.py::fit_meta_model`, 매 조합마다 새로 fit)을 포함한
blend 델타를 **둘 다** 리포트해서, 솔로 개선이 블렌드에서도 재현되는지 /
메타모델 재가중치에만 얹혀가는 가짜 개선은 아닌지 구분한다(`code/
experiment_hparam_reverify.py`와 동일한 관례, 다만 그 스크립트가 쓰던
`experiment_residual_correction_9_10.build_split`은 TE-residual을 호출하지
않는 구식 스냅샷이라 여기서는 `code/thirdmodel_common.py::build_split`로
교체했다).

프로덕션과 동일한 피처셋/스플릿을 그대로 재사용한다.

사용법:
  python -m code.experiment_catboost_te_retune --cutoff7
  python -m code.experiment_catboost_te_retune --holdout 2023
  python -m code.experiment_catboost_te_retune --cutoff7 --seeds 42 123 7
  python -m code.experiment_catboost_te_retune --cutoff7 --solo-only   # MLP/블렌드 생략, 빠른 확인용
"""
import argparse
import json
import time

from catboost import CatBoostClassifier, Pool

from code.blend_model import fit_meta_model
from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.thirdmodel_common import build_split, TARGET_COL

TUNE_JSON = "./open/temp/best_hparams.json"
TUNABLE_KEYS = [
    "depth", "learning_rate", "l2_leaf_reg", "random_strength",
    "bagging_temperature", "border_count", "min_data_in_leaf",
]


def train_catboost_one(params_override, X_train, y_train, X_val, y_val, seed=None):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    if params_override:
        for k in TUNABLE_KEYS:
            params[k] = params_override[k]
    if seed is not None:
        params["random_seed"] = seed
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    preds = model.predict_proba(X_val)[:, 1]
    return preds, int(model.get_best_iteration())


def train_mlp_once(train_split, val_split, mlp_num_cols, y_val, device):
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num)

    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val, seeds=ENSEMBLE_SEEDS, device=device, verbose=False,
    )
    preds = predict_ensemble(members, cat_dims, len(mlp_num_cols), embed_dims, X_val_cat, X_val_num, bin_edges=bin_edges, device=device)
    return preds


def run_regime(cutoff7, holdout, tuned_params, seeds, solo_only):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (baseline vs TE-residual 포함 재탐색) ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    X_train = train_split[cat_feature_cols]
    y_train = train_split[TARGET_COL].values
    X_val = val_split[cat_feature_cols]
    y_val = val_split[TARGET_COL].values

    mlp_preds = None
    if not solo_only:
        t0 = time.time()
        mlp_preds = train_mlp_once(train_split, val_split, mlp_num_cols, y_val, get_device())
        mlp_score = compute_bss(mlp_preds, y_val)[2]
        print(f"  [MLP({len(ENSEMBLE_SEEDS)}-seed), 두 CatBoost 후보 공통] Val Score={mlp_score:.2f} ({time.time()-t0:.1f}s)")

    solo_deltas, blend_deltas = [], []
    for seed in seeds:
        t0 = time.time()
        base_preds, base_iter = train_catboost_one(None, X_train, y_train, X_val, y_val, seed=seed)
        base_score = compute_bss(base_preds, y_val)[2]
        tuned_preds, tuned_iter = train_catboost_one(tuned_params, X_train, y_train, X_val, y_val, seed=seed)
        tuned_score = compute_bss(tuned_preds, y_val)[2]
        solo_delta = tuned_score - base_score
        solo_deltas.append(solo_delta)
        elapsed = time.time() - t0
        line = (f"  [seed={seed}] CatBoost solo: baseline={base_score:.2f}(iter={base_iter}) "
                f"tuned={tuned_score:.2f}(iter={tuned_iter}) delta={solo_delta:+.2f}")

        if mlp_preds is not None:
            _, _, _, base_blend, _ = fit_meta_model(base_preds, mlp_preds, y_val)
            _, _, _, tuned_blend, _ = fit_meta_model(tuned_preds, mlp_preds, y_val)
            blend_delta = tuned_blend - base_blend
            blend_deltas.append(blend_delta)
            line += f" | blend(+MLP+메타모델): baseline={base_blend:.2f} tuned={tuned_blend:.2f} delta={blend_delta:+.2f}"
        line += f" ({elapsed:.1f}s)"
        print(line)

    avg_solo = sum(solo_deltas) / len(solo_deltas)
    wins_solo = sum(1 for d in solo_deltas if d > 0)
    print(f"  --- {label}: solo {wins_solo}/{len(solo_deltas)} seed 승, 평균 solo delta={avg_solo:+.2f}", end="")
    if blend_deltas:
        avg_blend = sum(blend_deltas) / len(blend_deltas)
        wins_blend = sum(1 for d in blend_deltas if d > 0)
        print(f" | blend {wins_blend}/{len(blend_deltas)} seed 승, 평균 blend delta={avg_blend:+.2f} ---")
    else:
        print(" ---")
    return solo_deltas, blend_deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--tune-json", type=str, default=TUNE_JSON)
    parser.add_argument("--solo-only", action="store_true", help="MLP/블렌드 생략, CatBoost 솔로만 빠르게 확인")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout

    with open(args.tune_json) as f:
        tuned_params = json.load(f)["best_params"]
    print(f"[튜닝된 파라미터] {tuned_params}")

    run_regime(args.cutoff7, holdout, tuned_params, args.seeds, args.solo_only)


if __name__ == "__main__":
    main()
