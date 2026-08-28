# code/experiment_ebm_tuned3way.py
"""EBM 튜닝판 + 3-way 스태킹 확인. 1차 스크리닝(fast=outer_bags=4,interactions=0)은
solo 209.50로 약했고, 기본 설정(outer_bags=14, interactions='3x', n_jobs=-1, max_bins=1024)은
joblib 워커가 메모리 부족(OOM 추정, /dev/shm 9.8G에서 KeyError)으로 크래시했다.
문헌/실무 권장 대책:
  1. n_jobs를 코어 절반 정도로 낮춰 워커당 메모리 여유 확보 (interpret 문서 권장:
     n_jobs=-1은 대용량 데이터에서 메모리 경합을 일으킬 수 있음).
  2. max_bins을 1024->256으로 낮춰 히스토그램 메모리 절감 (EBM 논문/구현 모두 256이
     실무 기본값에 가까움 — 1024는 과함).
  3. interactions는 문자열 '3x' 대신 명시적 정수(예: 20)로 고정해 자동 탐지 비용 절감.
  4. outer_bags은 8로 완화(기본 14의 절반) — bagging 이득은 있지만 4개보다 안정적.
결측치는 imputation 없이 그대로 둔다 — EBM의 missing='separate'가 결측을 별도 bin으로
다루는 네이티브 방식이라(대회 asof 결측이 실제로 정보량이 있을 수 있음, cold-start),
CatBoost/MLP의 median-impute와는 다르게 이 부분도 굳이 맞추지 않는다.

사용법: python -m code.experiment_ebm_tuned3way [--n-jobs 4] [--max-bins 256] [--outer-bags 8] [--interactions 20]
"""
import argparse
import os
import time

import numpy as np

from code.experiment_thirdmodel_base import load_cache
from code.experiment_3way_stack import fit_meta_model_n
from code.mlp_model import CAT_COLS, compute_bss

TARGET_COL = "control_success"
EBM_CAT_COLS = CAT_COLS + ["pitcher_id", "batter_id"]
CACHE_DIR = "./open/temp/experiment_ebm_tuned"


def run(n_jobs=4, max_bins=256, outer_bags=8, interactions=20):
    from interpret.glassbox import ExplainableBoostingClassifier

    train_split, val_split, cat_val_preds, mlp_val_preds, y_val, meta = load_cache()
    ebm_num_cols = [c for c in meta["mlp_num_cols"] if c not in ("pitcher_id", "batter_id")]
    ebm_cols = EBM_CAT_COLS + ebm_num_cols

    print(f"[EBM-tuned] n_jobs={n_jobs} max_bins={max_bins} outer_bags={outer_bags} interactions={interactions}")
    print(f"[baseline] CatBoost={meta['catboost_score']:.2f} | MLP={meta['mlp_score']:.2f} | 2-way={meta['baseline_2way_score']:.2f}")

    X_train = train_split[ebm_cols].copy()
    X_val = val_split[ebm_cols].copy()
    for c in EBM_CAT_COLS:
        X_train[c] = X_train[c].astype(str).astype("category")
        X_val[c] = X_val[c].astype(str).astype("category")
    y_train = train_split[TARGET_COL].values

    model = ExplainableBoostingClassifier(
        random_state=42, n_jobs=n_jobs, max_bins=max_bins, outer_bags=outer_bags, interactions=interactions,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    pred = model.predict_proba(X_val)[:, 1]
    ebm_score = compute_bss(pred, y_val)[2]
    corr_cat = np.corrcoef(pred, cat_val_preds)[0, 1]
    corr_mlp = np.corrcoef(pred, mlp_val_preds)[0, 1]
    print(f"[EBM-tuned] solo={ebm_score:.2f} ({elapsed:.1f}s) | corr(cat)={corr_cat:.4f} corr(mlp)={corr_mlp:.4f}")

    weights, intercept, score_3way, _ = fit_meta_model_n([cat_val_preds, mlp_val_preds, pred], y_val)
    delta = score_3way - meta["baseline_2way_score"]
    print("\n" + "=" * 90)
    print(f"2-way={meta['baseline_2way_score']:.2f} | 3-way(+EBM)={score_3way:.2f} (delta {delta:+.2f}) "
          f"| weights cat={weights[0]:.3f} mlp={weights[1]:.3f} ebm={weights[2]:.3f} intercept={intercept:.3f}")
    print("=" * 90)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(os.path.join(CACHE_DIR, "ebm_val_preds.npz"), val_preds=pred)
    print(f"[cache] saved val_preds -> {CACHE_DIR}/ebm_val_preds.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-bins", type=int, default=256)
    parser.add_argument("--outer-bags", type=int, default=8)
    parser.add_argument("--interactions", type=int, default=20)
    args = parser.parse_args()
    run(n_jobs=args.n_jobs, max_bins=args.max_bins, outer_bags=args.outer_bags, interactions=args.interactions)


if __name__ == "__main__":
    main()
