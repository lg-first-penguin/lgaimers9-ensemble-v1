# code/experiment_3way_stack.py
"""사용자 요청: 팀원 스타일 CatBoost(A-H 피처, F1필터 OFF, 5-seed bagging —
`code/experiment_teammate_catboost_replica.py`에서 이미 확인됨: 두 레짐 다
season2023은 사실상 붕괴(best_iter 0~4, score ~10-22)했지만 사용자가 그대로
강행 요청) + 우리 프로덕션 CatBoost(F1 ON, 이번에 5-seed로 확장) + MLP(N-seed)
를 3-way 스태킹해서 기존 2-way(우리 CatBoost 1seed + MLP 7seed) 대비 이득이
있는지 확인한다.

입력 소스:
  - 팀원 replica CatBoost 예측: `/tmp/teammate_replica_{label}_f1off_5seed_preds.npy`
    + `/tmp/teammate_replica_{label}_yval.npy` (experiment_teammate_catboost_replica.py
    --save-preds로 이미 생성됨). 행 순서는 이 스크립트의 val_split과 동일한 로직
    (원본 train.csv를 그대로 읽어 동일 season/game_month 마스크만 적용, 셔플 없음)
    으로 만들어졌으므로 정렬 없이 그대로 정렬 일치한다 — 길이 비교로 sanity check.
  - 우리 CatBoost/MLP: 이 스크립트 자체에서 production 스플릿(`thirdmodel_common
    .build_split`)으로 학습.

메타모델은 `code/blend_model.py::fit_meta_model`(2피처 로지스틱회귀)을 N피처로
일반화한 버전을 자체 구현한다(프로덕션 코드는 건드리지 않음, 이 실험 스크립트
전용).

사용법:
  python -m code.experiment_3way_stack --cutoff7 --mlp-seeds 7
  python -m code.experiment_3way_stack --holdout 2023 --mlp-seeds 7
"""
import argparse
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from code.catboost_model import train_catboost
from code.experiment_catboost_seed_ensemble import SEED_POOL as OUR_CAT_SEED_POOL, train_one as train_catboost_seed
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS,
    embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    fit_quantile_edges, to_tensors, train_ensemble, predict_ensemble, compute_bss,
)
from code.thirdmodel_common import build_split, TARGET_COL


def fit_meta_nway(pred_list, y_val):
    X = np.column_stack(pred_list)
    model = LogisticRegression()
    model.fit(X, y_val)
    z = X @ model.coef_[0] + model.intercept_[0]
    blend = 1.0 / (1.0 + np.exp(-z))
    brier, bss, score = compute_bss(blend, y_val)
    return model.coef_[0], float(model.intercept_[0]), score


def run(cutoff7, holdout, mlp_seeds, n_cat_seeds):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 3-way 스태킹: {label} (mlp_seeds={mlp_seeds}, our_cat_seeds={n_cat_seeds}) ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout)
    y_val = val_split[TARGET_COL].values
    X_train_cat, y_train_cat = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val_cat = val_split[cat_feature_cols]

    # --- 우리 CatBoost: 1seed(현재 프로덕션과 동일) + n_cat_seeds 배깅 ---
    our_cat_preds_list = []
    for seed in OUR_CAT_SEED_POOL[:n_cat_seeds]:
        t0 = time.time()
        preds, best_iter = train_catboost_seed(seed, X_train_cat, y_train_cat, X_val_cat, y_val)
        score = compute_bss(preds, y_val)[2]
        print(f"  [우리 CatBoost] seed={seed}: Val Score={score:.2f} ({time.time()-t0:.1f}s)")
        our_cat_preds_list.append(preds)
    our_cat_1seed = our_cat_preds_list[0]
    our_cat_nseed = np.mean(our_cat_preds_list, axis=0)
    print(f"  [우리 CatBoost] 1seed={compute_bss(our_cat_1seed, y_val)[2]:.2f} | "
          f"{n_cat_seeds}seed={compute_bss(our_cat_nseed, y_val)[2]:.2f}")

    # --- 팀원 replica CatBoost (F1=OFF, 5seed, 이미 저장됨) ---
    tm_pred_path = f"/tmp/teammate_replica_{label}_f1off_5seed_preds.npy"
    tm_yval_path = f"/tmp/teammate_replica_{label}_yval.npy"
    tm_cat_preds = np.load(tm_pred_path)
    tm_yval = np.load(tm_yval_path)
    assert len(tm_cat_preds) == len(y_val), f"길이 불일치: teammate={len(tm_cat_preds)} vs 여기={len(y_val)}"
    assert np.array_equal(tm_yval, y_val), "행 정렬 불일치: y_val이 다름 (teammate replica와 이 스크립트의 val_split 순서가 어긋남)"
    print(f"  [팀원 replica CatBoost] score={compute_bss(tm_cat_preds, y_val)[2]:.2f} (사전 저장분 로드, 정렬 확인 완료)")

    # --- MLP N-seed ---
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat_t, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    seeds = (ENSEMBLE_SEEDS + [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13])[:mlp_seeds]
    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat_t, X_val_num=X_val_num, y_val=y_val,
        seeds=seeds, verbose=True,
    )
    print(f"[MLP {mlp_seeds}-seed 학습 완료] {time.time()-t0:.1f}s")
    mlp_preds = predict_ensemble(members, cat_dims, len(mlp_num_cols), embed_dims, X_val_cat_t, X_val_num, bin_edges=bin_edges)
    print(f"  [MLP {mlp_seeds}-seed] score={compute_bss(mlp_preds, y_val)[2]:.2f}")

    # --- 스태킹 비교 ---
    variants = {
        "2-way(현재 프로덕션 재현: our_cat1+mlp)": [our_cat_1seed, mlp_preds],
        f"2-way(our_cat{n_cat_seeds}seed+mlp)": [our_cat_nseed, mlp_preds],
        f"3-way(teammate_cat+our_cat1+mlp)": [tm_cat_preds, our_cat_1seed, mlp_preds],
        f"3-way(teammate_cat+our_cat{n_cat_seeds}seed+mlp)": [tm_cat_preds, our_cat_nseed, mlp_preds],
    }
    print(f"\n--- {label} 스태킹 결과 (mlp_seeds={mlp_seeds}) ---")
    base_score = None
    for tag, preds in variants.items():
        coefs, intercept, score = fit_meta_nway(preds, y_val)
        if base_score is None:
            base_score = score
        print(f"  [{tag}] Blend Score={score:.2f} (delta vs 2-way base={score-base_score:+.2f}) coefs={coefs} intercept={intercept:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--mlp-seeds", type=int, default=7)
    parser.add_argument("--n-cat-seeds", type=int, default=5)
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run(args.cutoff7, holdout, args.mlp_seeds, args.n_cat_seeds)


if __name__ == "__main__":
    main()
