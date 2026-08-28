# code/experiment_teammate_catboost_hparams.py
"""팀원(조유담) 실전 1059.72 레시피에서 보고된 CatBoost 하이퍼파라미터를, 우리
현재 프로덕션 피처셋(season진행분+TE-residual+coarse pitchmix, tier A 없음)에
그대로 이식해 dual-regime으로 isolate 테스트한다.

팀원 레시피와 우리 레시피를 대조한 결과, 파생피처12/시즌진행분8/TrackA(TE-residual)6/
구종비중4는 전부 우리 repo와 완전히 동일했다(컬럼명까지 일치) — 구조적 차이는
(1) 이 CatBoost 하이퍼파라미터, (2) season을 조인키로 쓰는 트랙맨 물리량 64개
(2025 실전에선 상수로 fallback, 별도 실험 `code/experiment_teammate_trackman64.py`
에서 다룸) 딱 둘뿐이다. 이 스크립트는 (1)만 isolate — 트랙맨 64개는 안 건드리고
하이퍼파라미터 효과만 순수하게 본다.

사용법:
  python -m code.experiment_teammate_catboost_hparams --cutoff7
  python -m code.experiment_teammate_catboost_hparams --holdout 2023
"""
import argparse
import time

from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import compute_bss
from code.thirdmodel_common import build_split, TARGET_COL

TEAMMATE_PARAMS = dict(
    depth=7,
    learning_rate=0.02034,
    l2_leaf_reg=14.806,
    random_strength=7.992,
    bagging_temperature=0.00872,
    border_count=167,
    min_data_in_leaf=35,
    bootstrap_type="Bayesian",
    loss_function="Logloss",
    eval_metric="BrierScore",
    random_seed=42,
    verbose=False,
)
TEAMMATE_ITERATIONS = 1462


def train_one(overrides, iterations, X_train, y_train, X_val, y_val):
    params = dict(CATBOOST_PARAMS)
    params.update(overrides)
    params["iterations"] = iterations
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model.predict_proba(X_val)[:, 1], int(model.get_best_iteration())


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (팀원 CatBoost 하이퍼파라미터 isolate 테스트) ===\n{'='*70}")

    train_split, val_split, _mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout, apply_f1=True)
    X_train = train_split[cat_feature_cols]
    y_train = train_split[TARGET_COL].values
    X_val = val_split[cat_feature_cols]
    y_val = val_split[TARGET_COL].values

    t0 = time.time()
    base_preds, base_iter = train_one({}, 1500, X_train, y_train, X_val, y_val)
    base_score = compute_bss(base_preds, y_val)[2]
    print(f"  [우리 baseline] Val Score={base_score:.2f} (best_iteration={base_iter}, {time.time()-t0:.1f}s)")

    t0 = time.time()
    tm_preds, tm_iter = train_one(TEAMMATE_PARAMS, TEAMMATE_ITERATIONS, X_train, y_train, X_val, y_val)
    tm_score = compute_bss(tm_preds, y_val)[2]
    print(f"  [팀원 하이퍼파라미터] Val Score={tm_score:.2f} (best_iteration={tm_iter}, {time.time()-t0:.1f}s)")

    print(f"\n  delta = {tm_score - base_score:+.2f}")
    return base_score, tm_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(args.cutoff7, holdout)


if __name__ == "__main__":
    main()
