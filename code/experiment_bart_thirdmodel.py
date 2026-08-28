# code/experiment_bart_thirdmodel.py
"""3rd-모델 탐색 3번 후보: BART(Bayesian Additive Regression Trees, stochtree 패키지의
StochTreeBARTBinaryClassifier). CatBoost의 그리디 gradient boosting과 달리 베이지안
백피팅(MCMC)으로 다수의 얕은 트리를 사후분포에서 표집한다 — 트리 기반이지만 학습
동역학(그리디 최적화 vs 사후표집)이 근본적으로 다른 후보.

전체 피처(cat_feature_cols와 동일, pitcher_id/batter_id 포함, CatBoost가 이미 이 둘을
raw 수치로 받는 것과 동일 취급)를 CAT_COLS(7개)만 ordinal-encode하고 나머지는 그대로
수치로 넣는다 — 트리는 스케일링이 필요 없다.

사용법: python -m code.experiment_bart_thirdmodel --cutoff7 [--num-gfr N --num-mcmc N]
"""
import argparse
import time

import numpy as np

from code.experiment_thirdmodel_common import build_full, load_prod_reference, TARGET_COL
from code.mlp_model import CAT_COLS, compute_bss, fit_preprocessing, apply_preprocessing


def run(holdout, cutoff7, num_gfr=5, num_mcmc=20):
    from stochtree import StochTreeBARTBinaryClassifier

    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} | 3rd 모델: BART (num_gfr={num_gfr}, num_mcmc={num_mcmc}) ===\n{'='*70}")

    train_split, val_split, features, cat_feature_cols, num_cols_full = build_full(holdout, cutoff7)
    y_train = train_split[TARGET_COL].values.astype(np.int64)
    y_val = val_split[TARGET_COL].values

    prod_pred, prod_score = load_prod_reference(val_split, features)
    print(f"[프로덕션 reference 블렌드] Val Score={prod_score:.2f} (참고용, 동일 val 재구성)")

    bart_num_cols = [c for c in cat_feature_cols if c not in CAT_COLS]
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        train_split[CAT_COLS + bart_num_cols + [TARGET_COL]], CAT_COLS, bart_num_cols,
    )
    val_proc = apply_preprocessing(val_split[CAT_COLS + bart_num_cols], CAT_COLS, bart_num_cols, cat_encoder, num_imputer, num_scaler)

    order = CAT_COLS + bart_num_cols
    X_train = train_proc[order].values.astype(np.float64)
    X_val = val_proc[order].values.astype(np.float64)
    print(f"[BART] n_features={len(order)} (cat={len(CAT_COLS)}, num={len(bart_num_cols)}), n_train={len(X_train)}, n_val={len(X_val)}")

    t0 = time.time()
    model = StochTreeBARTBinaryClassifier(num_gfr=num_gfr, num_burnin=0, num_mcmc=num_mcmc)
    model.fit(X_train, y_train)
    fit_elapsed = time.time() - t0

    t0 = time.time()
    pred = model.predict_proba(X_val)[:, 1]
    pred_elapsed = time.time() - t0

    brier, bss, score = compute_bss(pred, y_val)
    corr = np.corrcoef(pred, prod_pred)[0, 1]
    print(f"[BART] Val Score={score:.2f} (fit {fit_elapsed:.1f}s, predict {pred_elapsed:.1f}s)")
    print(f"[BART] 프로덕션 블렌드 예측과의 상관계수: {corr:.4f}")
    return score, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--num-gfr", type=int, default=5)
    parser.add_argument("--num-mcmc", type=int, default=20)
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run(holdout, args.cutoff7, num_gfr=args.num_gfr, num_mcmc=args.num_mcmc)


if __name__ == "__main__":
    main()
