# code/experiment_bart_tuned3way.py
"""BART 튜닝판 + 3-way 스태킹 확인. 1차 스크리닝에서 `general_params`의 `num_threads`
기본값이 1(싱글스레드)이라는 걸 확인 — 8코어 중 1개만 썼다. `num_threads=-1`로 멀티스레드
전환하고, 그만큼 확보한 여유로 문헌 기본값(Chipman et al. BART 논문/stochtree 기본값
num_trees=200, num_gfr=10, num_mcmc=100 — 1차 스크리닝의 num_gfr=5/num_mcmc=20은 사후분포
표집이 너무 적었다)까지 정식으로 돌린다.

사용법: python -m code.experiment_bart_tuned3way [--num-gfr 10] [--num-mcmc 100] [--num-threads -1]
"""
import argparse
import os
import time

import numpy as np

from code.experiment_thirdmodel_base import load_cache
from code.experiment_3way_stack import fit_meta_model_n
from code.mlp_model import CAT_COLS, compute_bss, fit_preprocessing, apply_preprocessing

TARGET_COL = "control_success"
CACHE_DIR = "./open/temp/experiment_bart_tuned"


def run(num_gfr=10, num_mcmc=100, num_threads=-1):
    from stochtree import StochTreeBARTBinaryClassifier

    train_split, val_split, cat_val_preds, mlp_val_preds, y_val, meta = load_cache()
    cat_feature_cols = meta["cat_feature_cols"]
    bart_num_cols = [c for c in cat_feature_cols if c not in CAT_COLS]

    print(f"[BART-tuned] num_gfr={num_gfr} num_mcmc={num_mcmc} num_threads={num_threads}")
    print(f"[baseline] CatBoost={meta['catboost_score']:.2f} | MLP={meta['mlp_score']:.2f} | 2-way={meta['baseline_2way_score']:.2f}")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(
        train_split[CAT_COLS + bart_num_cols + [TARGET_COL]], CAT_COLS, bart_num_cols,
    )
    val_proc = apply_preprocessing(val_split[CAT_COLS + bart_num_cols], CAT_COLS, bart_num_cols, cat_encoder, num_imputer, num_scaler)
    order = CAT_COLS + bart_num_cols
    X_train = train_proc[order].values.astype(np.float64)
    X_val = val_proc[order].values.astype(np.float64)
    y_train = train_split[TARGET_COL].values.astype(np.int64)

    t0 = time.time()
    model = StochTreeBARTBinaryClassifier(
        num_gfr=num_gfr, num_burnin=0, num_mcmc=num_mcmc,
        general_params={"num_threads": num_threads, "random_seed": 42},
    )
    model.fit(X_train, y_train)
    fit_elapsed = time.time() - t0

    pred = model.predict_proba(X_val)[:, 1]
    bart_score = compute_bss(pred, y_val)[2]
    corr_cat = np.corrcoef(pred, cat_val_preds)[0, 1]
    corr_mlp = np.corrcoef(pred, mlp_val_preds)[0, 1]
    print(f"[BART-tuned] solo={bart_score:.2f} (fit {fit_elapsed:.1f}s) | corr(cat)={corr_cat:.4f} corr(mlp)={corr_mlp:.4f}")

    weights, intercept, score_3way, _ = fit_meta_model_n([cat_val_preds, mlp_val_preds, pred], y_val)
    delta = score_3way - meta["baseline_2way_score"]
    print("\n" + "=" * 90)
    print(f"2-way={meta['baseline_2way_score']:.2f} | 3-way(+BART)={score_3way:.2f} (delta {delta:+.2f}) "
          f"| weights cat={weights[0]:.3f} mlp={weights[1]:.3f} bart={weights[2]:.3f} intercept={intercept:.3f}")
    print("=" * 90)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(os.path.join(CACHE_DIR, "bart_val_preds.npz"), val_preds=pred)
    print(f"[cache] saved val_preds -> {CACHE_DIR}/bart_val_preds.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gfr", type=int, default=10)
    parser.add_argument("--num-mcmc", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=-1)
    args = parser.parse_args()
    run(num_gfr=args.num_gfr, num_mcmc=args.num_mcmc, num_threads=args.num_threads)


if __name__ == "__main__":
    main()
