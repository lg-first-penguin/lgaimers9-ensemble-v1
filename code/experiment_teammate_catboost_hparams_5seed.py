# code/experiment_teammate_catboost_hparams_5seed.py
"""`experiment_teammate_catboost_hparams.py`의 재검증판. 그 스크립트는 단일 시드
(random_seed=42)로만 비교해서 cutoff7 -13.14 / season==2023 +34.15(레짐 반전)로
기각됐는데, CatBoost는 thread_count만 바꿔도 단일시드에서 7~19pt가 흔들리는
노이즈가 이미 문서화돼 있어(teammate_catboost_mlp_track_983.md side finding),
그 반전이 "진짜 손해"인지 "단일시드 노이즈"인지 분리가 안 된 채로 남아있었다.

우리 프로덕션이 이미 5-seed CatBoost 배깅(`CATBOOST_SEED_POOL`)을 쓰고 있으므로,
같은 5개 시드로 baseline(CATBOOST_PARAMS)과 팀원 하이퍼파라미터(TEAMMATE_PARAMS)를
각각 앙상블 학습해서 평균낸 예측으로 dual-regime 비교한다 — 새로 Optuna를 돌리는
게 아니라 기존 isolate 테스트의 노이즈만 줄인 재확인.

사용법:
  python -m code.experiment_teammate_catboost_hparams_5seed --cutoff7
  python -m code.experiment_teammate_catboost_hparams_5seed --holdout 2023
  python -m code.experiment_teammate_catboost_hparams_5seed --both   # 두 레짐 순차 실행
"""
import argparse
import time

import numpy as np
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, CATBOOST_SEED_POOL
from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, TARGET_COL
from code.experiment_teammate_catboost_hparams import TEAMMATE_PARAMS, TEAMMATE_ITERATIONS

THREAD_COUNT = 4  # 두 config 모두 동일하게 고정 (thread_count 자체가 노이즈원이라 통제)


def train_one_seed(overrides, iterations, seed, X_train, y_train, X_val, y_val):
    params = dict(CATBOOST_PARAMS)
    params.update(overrides)
    params["iterations"] = iterations
    params["early_stopping_rounds"] = 50
    params["random_seed"] = seed
    params["thread_count"] = THREAD_COUNT
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1], int(model.get_best_iteration())


def train_ensemble(name, overrides, iterations, seeds, X_train, y_train, X_val, y_val, y_val_arr):
    preds_list = []
    t0 = time.time()
    for seed in seeds:
        preds, best_iter = train_one_seed(overrides, iterations, seed, X_train, y_train, X_val, y_val)
        solo_score = compute_bss(preds, y_val_arr)[2]
        preds_list.append(preds)
        print(f"    seed={seed:>6} best_iter={best_iter:>4} solo={solo_score:.2f}", flush=True)
    ensemble_pred = np.mean(preds_list, axis=0)
    ensemble_score = compute_bss(ensemble_pred, y_val_arr)[2]
    print(f"  [{name}] 5-seed 앙상블 Val Score={ensemble_score:.2f} ({time.time()-t0:.1f}s)")
    return ensemble_score


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (팀원 CatBoost 하이퍼파라미터, 5-seed 앙상블 재검증) ===\n{'='*70}")

    train_split, val_split, _mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    X_train = train_split[cat_feature_cols]
    y_train = train_split[TARGET_COL].values
    X_val = val_split[cat_feature_cols]
    y_val = val_split[TARGET_COL].values

    print("  [baseline: 우리 CATBOOST_PARAMS]")
    base_score = train_ensemble("baseline", {}, 1500, CATBOOST_SEED_POOL, X_train, y_train, X_val, y_val, y_val)

    print("  [팀원 하이퍼파라미터]")
    tm_score = train_ensemble("팀원", TEAMMATE_PARAMS, TEAMMATE_ITERATIONS, CATBOOST_SEED_POOL, X_train, y_train, X_val, y_val, y_val)

    delta = tm_score - base_score
    print(f"\n  delta(5-seed 앙상블, 팀원-baseline) = {delta:+.2f}")
    return base_score, tm_score, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    results = []
    if args.both:
        results.append(("cutoff7", run_regime(True, 2024)))
        results.append(("season2023", run_regime(False, 2023)))
    else:
        holdout = 2024 if args.cutoff7 else args.holdout
        label = "cutoff7" if args.cutoff7 else f"season{holdout}"
        results.append((label, run_regime(args.cutoff7, holdout)))

    print("\n" + "=" * 70)
    print(f"{'레짐':<14}{'baseline':>12}{'팀원':>12}{'delta':>10}")
    for name, (b, t, d) in results:
        print(f"{name:<14}{b:>12.2f}{t:>12.2f}{d:>+10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
