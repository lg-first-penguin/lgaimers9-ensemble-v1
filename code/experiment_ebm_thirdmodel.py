# code/experiment_ebm_thirdmodel.py
"""3rd-모델 탐색 2번 후보: EBM(Explainable Boosting Machine, interpret-core). CatBoost의
그리디 트리 부스팅(ordered boosting)이나 MLP의 SGD 딥넷과 달리, EBM은 피처 하나씩
라운드로빈으로 얕은(leaf<=2) 부스팅을 도는 "cyclic gradient boosting of shape functions"
+ bagging(outer_bags) 앙상블이라는, 이 프로젝트에 아직 없는 완전히 다른 학습 절차다.
고카디널리티 카테고리(pitcher_id/batter_id)도 cat_smooth/min_cat_samples라는, CatBoost의
ordered target statistics와는 다른 자체 평활화 메커니즘으로 처리한다 — 같은 정보를 다른
방식으로 규제했을 때 에러 패턴이 갈라지는지 확인하는 것이 핵심 가설.

사용법: python -m code.experiment_ebm_thirdmodel --cutoff7 [--fast]
"""
import argparse
import time

import numpy as np

from code.experiment_thirdmodel_common import build_full, load_prod_reference, TARGET_COL
from code.mlp_model import CAT_COLS, compute_bss

EBM_CAT_COLS = CAT_COLS + ["pitcher_id", "batter_id"]


def run(holdout, cutoff7, fast=False):
    from interpret.glassbox import ExplainableBoostingClassifier

    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    mode = "fast(outer_bags=4,interactions=0)" if fast else "full(outer_bags=14,interactions=3x)"
    print(f"\n{'='*70}\n=== 레짐: {label} | 3rd 모델: EBM {mode} ===\n{'='*70}")

    train_split, val_split, features, cat_feature_cols, num_cols_full = build_full(holdout, cutoff7)
    y_train = train_split[TARGET_COL].values
    y_val = val_split[TARGET_COL].values

    prod_pred, prod_score = load_prod_reference(val_split, features)
    print(f"[프로덕션 reference 블렌드] Val Score={prod_score:.2f} (참고용, 동일 val 재구성)")

    ebm_num_cols = [c for c in num_cols_full if c not in ("pitcher_id", "batter_id")]
    ebm_cols = EBM_CAT_COLS + ebm_num_cols

    X_train = train_split[ebm_cols].copy()
    X_val = val_split[ebm_cols].copy()
    for c in EBM_CAT_COLS:
        X_train[c] = X_train[c].astype(str).astype("category")
        X_val[c] = X_val[c].astype(str).astype("category")

    kwargs = dict(random_state=42, n_jobs=-1)
    if fast:
        kwargs.update(outer_bags=4, interactions=0)
    model = ExplainableBoostingClassifier(**kwargs)

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    pred = model.predict_proba(X_val)[:, 1]
    brier, bss, score = compute_bss(pred, y_val)
    corr = np.corrcoef(pred, prod_pred)[0, 1]
    print(f"[EBM] Val Score={score:.2f} ({elapsed:.1f}s, n_cat={len(EBM_CAT_COLS)}, n_num={len(ebm_num_cols)})")
    print(f"[EBM] 프로덕션 블렌드 예측과의 상관계수: {corr:.4f}")
    return score, corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run(holdout, args.cutoff7, fast=args.fast)


if __name__ == "__main__":
    main()
