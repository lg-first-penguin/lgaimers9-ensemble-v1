# code/experiment_thirdmodel_rolling_bart.py
"""BART 3-way delta를 2020/2021/2022/2023/2024 season-level rolling-origin으로
재검증한다(`code/experiment_thirdmodel_rolling_base.py`의 CatBoost/MLP fold 캐시 재사용).
BART는 MCMC 사후표집 자체가 이미 여러 트리를 평균하는 구조라 seed 앙상블 대신
num_mcmc(사후표집 수)를 §67.3 튜닝판(100)보다 올려 DeepFM 7-seed에 준하는 강건성을 준다
(num_gfr=10 유지, num_mcmc 100->200).

사용법: python -m code.experiment_thirdmodel_rolling_bart
"""
import time

import numpy as np

from code.blend_model import fit_meta_model, predict_meta
from code.experiment_3way_stack import fit_meta_model_n
from code.experiment_thirdmodel_rolling_base import CAT_COLS, FOLD_SEASONS, get_fold_base, load_r_only
from code.mlp_model import apply_preprocessing, compute_bss, fit_preprocessing

TARGET_COL = "control_success"
NUM_GFR = 10
NUM_MCMC = 200
NUM_THREADS = -1


def run_bart_fold(fold_data):
    from stochtree import StochTreeBARTBinaryClassifier

    train_split, val_split, num_cols = fold_data["train_split"], fold_data["val_split"], fold_data["num_cols"]
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    order = CAT_COLS + num_cols
    X_train = train_proc[order].values.astype(np.float64)
    X_val = val_proc[order].values.astype(np.float64)
    y_train = train_split[TARGET_COL].values.astype(np.int64)

    model = StochTreeBARTBinaryClassifier(
        num_gfr=NUM_GFR, num_burnin=0, num_mcmc=NUM_MCMC,
        general_params={"num_threads": NUM_THREADS, "random_seed": 42},
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


def main():
    df_all, df_trm = load_r_only()

    fold_results = {}
    for val_season in FOLD_SEASONS:
        print(f"\n=== fold: train<{val_season} -> val=={val_season} (R-only) ===")
        fold_data = get_fold_base(val_season, df_all, df_trm)
        t0 = time.time()
        bart_pred = run_bart_fold(fold_data)
        print(f"  [val={val_season}] BART(num_mcmc={NUM_MCMC}) 완료 ({time.time()-t0:.1f}s)")
        fold_results[val_season] = (fold_data["cat_pred"], fold_data["mlp_pred"], bart_pred, fold_data["y_val"])

    print(f"\n{'='*90}\nfold별 2-way vs 3-way(+BART)\n{'='*90}")
    deltas = []
    for val_season in FOLD_SEASONS:
        cat_p, mlp_p, bart_p, y = fold_results[val_season]
        w_cat, w_mlp, ic, _, _ = fit_meta_model(cat_p, mlp_p, y)
        score_2way = compute_bss(predict_meta(w_cat, w_mlp, ic, cat_p, mlp_p), y)[2]

        weights, intercept, score_3way, _ = fit_meta_model_n([cat_p, mlp_p, bart_p], y)
        delta = score_3way - score_2way
        deltas.append(delta)
        bart_score = compute_bss(bart_p, y)[2]
        corr_cat, corr_mlp = np.corrcoef(bart_p, cat_p)[0, 1], np.corrcoef(bart_p, mlp_p)[0, 1]
        print(f"[val={val_season}] BART solo={bart_score:.2f} corr(cat)={corr_cat:.3f} corr(mlp)={corr_mlp:.3f} | "
              f"2-way={score_2way:.2f} | 3-way={score_3way:.2f} (delta {delta:+.2f})")

    wins = sum(d > 0 for d in deltas)
    print(f"\n{'='*90}\n{wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}, 표준편차: {np.std(deltas):.2f}\n{'='*90}")


if __name__ == "__main__":
    main()
