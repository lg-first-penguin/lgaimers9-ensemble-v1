# code/experiment_thirdmodel_rolling_ebm.py
"""EBM 3-way delta를 2020/2021/2022/2023/2024 season-level rolling-origin으로
재검증한다(`code/experiment_thirdmodel_rolling_base.py`의 CatBoost/MLP fold 캐시 재사용).
EBM은 seed 앙상블 대신 outer_bags 내부 배깅으로 강건성을 확보하는 모델이라 fold당 1회만
학습하되, outer_bags를 §67.3 튜닝판(8)에서 문헌 기본값(14)으로 올려 DeepFM 7-seed에
준하는 강건성을 준다(1차 스크리닝 때의 outer_bags=14 크래시는 n_jobs=-1/max_bins=1024
조합의 메모리 문제였고, n_jobs=4/max_bins=256로 이미 완화된 상태이므로 재시도).

사용법: python -m code.experiment_thirdmodel_rolling_ebm
"""
import time

import numpy as np

from code.blend_model import fit_meta_model, predict_meta
from code.experiment_3way_stack import fit_meta_model_n
from code.experiment_thirdmodel_rolling_base import CAT_COLS, FOLD_SEASONS, get_fold_base, load_r_only
from code.mlp_model import compute_bss

TARGET_COL = "control_success"
EBM_CAT_COLS = CAT_COLS + ["pitcher_id", "batter_id"]
N_JOBS = 4
MAX_BINS = 256
OUTER_BAGS = 14
INTERACTIONS = 20


def run_ebm_fold(fold_data):
    from interpret.glassbox import ExplainableBoostingClassifier

    train_split, val_split, num_cols = fold_data["train_split"], fold_data["val_split"], fold_data["num_cols"]
    ebm_num_cols = [c for c in num_cols if c not in ("pitcher_id", "batter_id")]
    ebm_cols = EBM_CAT_COLS + ebm_num_cols

    X_train = train_split[ebm_cols].copy()
    X_val = val_split[ebm_cols].copy()
    for c in EBM_CAT_COLS:
        X_train[c] = X_train[c].astype(str).astype("category")
        X_val[c] = X_val[c].astype(str).astype("category")
    y_train = train_split[TARGET_COL].values

    model = ExplainableBoostingClassifier(
        random_state=42, n_jobs=N_JOBS, max_bins=MAX_BINS, outer_bags=OUTER_BAGS, interactions=INTERACTIONS,
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
        ebm_pred = run_ebm_fold(fold_data)
        print(f"  [val={val_season}] EBM(outer_bags={OUTER_BAGS}) 완료 ({time.time()-t0:.1f}s)")
        fold_results[val_season] = (fold_data["cat_pred"], fold_data["mlp_pred"], ebm_pred, fold_data["y_val"])

    print(f"\n{'='*90}\nfold별 2-way vs 3-way(+EBM)\n{'='*90}")
    deltas = []
    for val_season in FOLD_SEASONS:
        cat_p, mlp_p, ebm_p, y = fold_results[val_season]
        w_cat, w_mlp, ic, _, _ = fit_meta_model(cat_p, mlp_p, y)
        score_2way = compute_bss(predict_meta(w_cat, w_mlp, ic, cat_p, mlp_p), y)[2]

        weights, intercept, score_3way, _ = fit_meta_model_n([cat_p, mlp_p, ebm_p], y)
        delta = score_3way - score_2way
        deltas.append(delta)
        ebm_score = compute_bss(ebm_p, y)[2]
        corr_cat, corr_mlp = np.corrcoef(ebm_p, cat_p)[0, 1], np.corrcoef(ebm_p, mlp_p)[0, 1]
        print(f"[val={val_season}] EBM solo={ebm_score:.2f} corr(cat)={corr_cat:.3f} corr(mlp)={corr_mlp:.3f} | "
              f"2-way={score_2way:.2f} | 3-way={score_3way:.2f} (delta {delta:+.2f})")

    wins = sum(d > 0 for d in deltas)
    print(f"\n{'='*90}\n{wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}, 표준편차: {np.std(deltas):.2f}\n{'='*90}")


if __name__ == "__main__":
    main()
