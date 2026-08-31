# code/build_m4_meta_bundle.py
"""[메타 M4] 현행 1117.03 번들의 meta_model 만 'Brier-목적 LogOP' 로 교체.

동기 (code/experiment_yudam_meta_methods.py):
  현행 메타 = LogisticRegression on [p_cat, p_mlp] (log-loss 최소화).
  대회 지표는 Brier(=MSE) 인데 메타가 log-loss 를 최소화하는 불일치.
  M4 = sigmoid(w1*logit(p_cat) + w2*logit(p_mlp) + b), (w1,w2,b) 를 Brier 직접 최소화.
  rolling-origin 4-fold: M2/M3/M4 전부 4/4 승, M4 mean Δ +6.46 (cutoff7 3-seed +3.82 /
  7-seed +2.43). M5(심플렉스 제약)는 붕괴 기각.

베이스 모델(CatBoost 5-seed full-retrain + MLP 7-seed full-retrain)·정적 lookup 3종은
현행 번들에서 그대로 재사용 — **바뀌는 건 meta_model dict 하나뿐.** meta 가중치는
cutoff7 stage-1(cutoff7 학습 베이스 + held-out val, 트랙맨64 제거)로 재fit 한다
(full-retrain 베이스로 fit 하면 val 리크 — §89 candidate A 버그와 동일).

산출 meta_model = {"w_cat","w_mlp","intercept","space":"logit_brier"} — script.py /
blend_model.py 가 space 키로 자동 분기 (없으면 raw = 기존 동작, 하위호환).

실행:
  python -m code.build_m4_meta_bundle              # STAGING 에만 빌드
  python -m code.build_m4_meta_bundle --promote    # 백업 후 submit/ 반영
"""
import argparse
import gc
import hashlib
import os
import pickle
import shutil
import time

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, LinearRegression

from code.experiment_yudam_common import build_split, _yudam_catboost_params, TARGET
from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    to_tensors, train_ensemble, make_bundle, predict_bundle, get_device, compute_bss,
)
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS

EPS = 1e-6
PROD_DIR = "./submit/model"
PROD_PKL = os.path.join(PROD_DIR, "final_retained_model.pkl")
SP = "/tmp/claude-1000/-home-user-contest-mlp-lgaimers9/80d8950c-489c-47b5-93de-fb7a77e4f443/scratchpad"
STAGING_DIR = os.path.join(SP, "submit_m4_meta")
STAGING_PKL = os.path.join(STAGING_DIR, "final_retained_model.pkl")
LOOKUP_CSVS = ("season_end_lookup.csv", "rate_end_lookup.csv", "te_source.csv")
EXTRA_CSVS = ("trackman_match_table.csv", "trackman_match_table_gap.csv")


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _bss(pred, y):
    return compute_bss(np.clip(pred, 0.0, 1.0), y)[2]


def _brier(pred, y):
    return np.mean((np.clip(pred, 0.0, 1.0) - y) ** 2)


def stage1_meta(device):
    print("\n" + "=" * 72 + "\n[stage-1] cutoff7 (트랙맨64-free) 베이스 학습 -> 메타 fit\n" + "=" * 72, flush=True)
    tr, va, num_cols, cat_cols, all_cols = build_split("cutoff7", verbose=True)
    yv = va[TARGET].values.astype(np.float64)

    tp, ce, ni, ns, cat_dims = fit_preprocessing(tr, CAT_COLS, num_cols)
    vp = apply_preprocessing(va, CAT_COLS, num_cols, ce, ni, ns)
    Xtc, Xtn, ytr = to_tensors(tp, CAT_COLS, num_cols, TARGET)
    Xvc, Xvn, _ = to_tensors(vp, CAT_COLS, num_cols, TARGET)
    del tp, vp
    gc.collect()
    eds = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        Xtc, Xtn, ytr, cat_dims=cat_dims, num_numeric_feats=len(num_cols),
        embed_dims=eds, bin_edges=None, X_val_cat=Xvc, X_val_num=Xvn, y_val=yv,
        seeds=list(YUDAM_ENSEMBLE_SEEDS), device=device, verbose=True,
    )
    mb = make_bundle(members, CAT_COLS, num_cols, cat_dims, eds, ce, ni, ns, bin_edges=None)
    pm = predict_bundle(mb, va[all_cols], device=device).astype(np.float64)

    cb_res = train_catboost_ensemble(
        tr[cat_cols], tr[TARGET].values, va[cat_cols], yv,
        seeds=list(YUDAM_CATBOOST_SEEDS), verbose=True, params=_yudam_catboost_params(),
    )
    pc = predict_catboost_ensemble([m for m, _ in cb_res], va[cat_cols]).astype(np.float64)

    cat_solo, mlp_solo = _bss(pc, yv), _bss(pm, yv)
    print(f"\n[stage-1] cat_solo={cat_solo:.2f} | mlp_solo={mlp_solo:.2f} | n_val={len(yv)}", flush=True)

    # --- 참고: M1/M2/M3 ---
    m1 = LogisticRegression().fit(np.column_stack([pc, pm]), yv)
    p1 = _sig(m1.coef_[0][0] * pc + m1.coef_[0][1] * pm + m1.intercept_[0])
    lc, lm = _logit(pc), _logit(pm)
    m2 = LogisticRegression().fit(np.column_stack([lc, lm]), yv)
    p2 = _sig(m2.coef_[0][0] * lc + m2.coef_[0][1] * lm + m2.intercept_[0])
    m3 = LinearRegression().fit(np.column_stack([pc, pm]), yv)
    p3 = m3.coef_[0] * pc + m3.coef_[1] * pm + m3.intercept_

    # --- M4: Brier 직접최소화, sigmoid(w1*logit pc + w2*logit pm + b) ---
    x0 = np.array([m2.coef_[0][0], m2.coef_[0][1], m2.intercept_[0]])
    res = minimize(lambda x: _brier(_sig(x[0] * lc + x[1] * lm + x[2]), yv), x0,
                   method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-11, maxiter=8000))
    w_cat, w_mlp, intercept = (float(v) for v in res.x)
    p4 = _sig(w_cat * lc + w_mlp * lm + intercept)

    print(f"[stage-1] M1 logistic_raw   blend={_bss(p1, yv):.2f}  (현행 방식)", flush=True)
    print(f"[stage-1] M2 logistic_logit blend={_bss(p2, yv):.2f}", flush=True)
    print(f"[stage-1] M3 linreg_raw     blend={_bss(p3, yv):.2f}", flush=True)
    print(f"[stage-1] M4 logop_brier    blend={_bss(p4, yv):.2f}  <- 채택  "
          f"(w_cat={w_cat:.5f} w_mlp={w_mlp:.5f} b={intercept:.5f}, converged={res.success})", flush=True)

    del tr, va, members, mb, Xtc, Xtn, Xvc, Xvn, cb_res
    gc.collect()
    return dict(w_cat=w_cat, w_mlp=w_mlp, intercept=intercept, space="logit_brier")


def build(meta):
    os.makedirs(STAGING_DIR, exist_ok=True)
    with open(PROD_PKL, "rb") as f:
        bundle = pickle.load(f)
    old_meta = dict(bundle["meta_model"])
    bundle["meta_model"] = meta   # 베이스 모델·cat_feature_cols·cb_iters 전부 그대로
    with open(STAGING_PKL, "wb") as f:
        pickle.dump(bundle, f)
    for c in LOOKUP_CSVS + EXTRA_CSVS:
        src = os.path.join(PROD_DIR, c)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(STAGING_DIR, c))
    print(f"\n[build] {STAGING_PKL}", flush=True)
    print(f"[build] 기존 meta {old_meta}", flush=True)
    print(f"[build] 신규 meta {meta}", flush=True)
    print(f"[build] 베이스 모델/lookup 무변경 — meta_model dict 만 교체", flush=True)


def promote():
    ts = time.strftime("%Y%m%d_%H%M%S")
    bdir = f"./open/former_model/submit_pre_m4meta_{ts}"
    os.makedirs(bdir, exist_ok=True)
    for c in ("final_retained_model.pkl",) + LOOKUP_CSVS + EXTRA_CSVS:
        src = os.path.join(PROD_DIR, c)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(bdir, c))
    shutil.copy2("./submit/script.py", os.path.join(bdir, "script.py"))
    print(f"[promote] 현행 submit/ 백업 -> {bdir}  (pkl md5 {_md5(PROD_PKL)})", flush=True)
    shutil.copy2(STAGING_PKL, PROD_PKL)
    print(f"[promote] STAGING pkl -> {PROD_PKL}  (md5 {_md5(PROD_PKL)})", flush=True)
    print(f"[promote] lookup CSV 3종은 무변경(내용 동일). script.py 는 별도 패치 필요.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()
    device = get_device()
    print(f"device={device}", flush=True)
    meta = stage1_meta(device)
    gc.collect()
    build(meta)
    if args.promote:
        promote()
    else:
        print("\n(STAGING 만 빌드됨. --promote 로 백업+반영. script.py 패치도 잊지 말 것.)", flush=True)


if __name__ == "__main__":
    main()
