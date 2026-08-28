# code/experiment_catboost_boosting_grid.py
"""CatBoost `boosting_type`(Ordered/Plain) x `grow_policy`(SymmetricTree/
Depthwise/Lossguide) 그리드 스크리닝.

`code/catboost_model.py::CATBOOST_PARAMS`는 depth/learning_rate/l2_leaf_reg/
random_strength/bagging_temperature/border_count/min_data_in_leaf는
`code/tune.py`(Optuna 40 trials)로 반복 재탐색됐지만, boosting_type과
grow_policy는 한 번도 명시적으로 튜닝되거나 언급된 적이 없다(기본값 "Auto" ->
학습 행 수가 5만 행을 넘으면 CatBoost가 자동으로 Plain을 선택하므로, 지금
프로덕션은 사실상 이미 Plain+SymmetricTree다). F1 필터로 학습 데이터가 줄어든
지금, Ordered boosting(소표본에서 과적합에 강함)이나 Depthwise/Lossguide 트리
성장 방식이 프로덕션 조합을 이길 수 있는지 확인한다.

유효 조합(Ordered는 grow_policy=SymmetricTree만 지원):
  1. Plain + SymmetricTree   <- 현재 프로덕션과 동일(재현 baseline)
  2. Plain + Depthwise
  3. Plain + Lossguide
  4. Ordered + SymmetricTree

프로덕션과 동일한 피처셋/스플릿(`code/thirdmodel_common.py::build_split` — F1
필터, cutoff7 스플릿, season-progression, TE-residual, coarse pitchmix 전부
포함)을 그대로 재사용한다.

사용법:
  python -m code.experiment_catboost_boosting_grid --cutoff7
  python -m code.experiment_catboost_boosting_grid --holdout 2023
"""
import argparse
import time

from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS, MAX_ITERATIONS, EARLY_STOPPING_ROUNDS
from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, TARGET_COL

COMBOS = [
    ("Plain", "SymmetricTree"),
    ("Plain", "Depthwise"),
    ("Plain", "Lossguide"),
    ("Ordered", "SymmetricTree"),
]


def train_one(boosting_type, grow_policy, X_train, y_train, X_val, y_val, seed=None):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = MAX_ITERATIONS
    params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    params["boosting_type"] = boosting_type
    params["grow_policy"] = grow_policy
    if seed is not None:
        params["random_seed"] = seed
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    preds = model.predict_proba(X_val)[:, 1]
    return preds, int(model.get_best_iteration())


def run_regime(cutoff7, holdout, combos=None, seeds=(None,)):
    combos = combos if combos is not None else COMBOS
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (CatBoost boosting_type x grow_policy 그리드) ===\n{'='*70}")

    train_split, val_split, _mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    X_train = train_split[cat_feature_cols]
    y_train = train_split[TARGET_COL].values
    X_val = val_split[cat_feature_cols]
    y_val = val_split[TARGET_COL].values

    results = {}
    for boosting_type, grow_policy in combos:
        for seed in seeds:
            t0 = time.time()
            try:
                preds, best_iter = train_one(boosting_type, grow_policy, X_train, y_train, X_val, y_val, seed=seed)
            except Exception as e:
                print(f"  [{boosting_type}+{grow_policy} seed={seed}] FAILED: {e}")
                continue
            score = compute_bss(preds, y_val)[2]
            elapsed = time.time() - t0
            results[(boosting_type, grow_policy, seed)] = score
            print(f"  [{boosting_type}+{grow_policy} seed={seed}] Val Score={score:.2f} (best_iteration={best_iter}, {elapsed:.1f}s)")

    for seed in seeds:
        baseline = results.get(("Plain", "SymmetricTree", seed))
        if baseline is None:
            continue
        print(f"\n  --- seed={seed}: baseline(Plain+SymmetricTree)={baseline:.2f} 대비 delta ---")
        for combo, score in results.items():
            if combo == ("Plain", "SymmetricTree", seed) or combo[2] != seed:
                continue
            print(f"  {combo[0]}+{combo[1]}: delta={score - baseline:+.2f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--only-ordered", action="store_true",
                         help="Plain+SymmetricTree(baseline) vs Ordered+SymmetricTree만 실행")
    parser.add_argument("--seeds", type=int, nargs="+", default=[None],
                         help="random_seed 후보들 (기본: CATBOOST_PARAMS 고정값 42)")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    combos = [("Plain", "SymmetricTree"), ("Ordered", "SymmetricTree")] if args.only_ordered else COMBOS
    run_regime(args.cutoff7, holdout, combos=combos, seeds=args.seeds)


if __name__ == "__main__":
    main()
