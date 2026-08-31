# code/experiment_yudam_meta_methods.py
"""[메타모델 방법론 5종 비교] rolling-origin 4-fold, 같은 베이스 예측으로 채점.

현행 프로덕션 메타 = LogisticRegression on [p_cat, p_mlp] (log-loss 최소화) +
sigmoid(w_cat*p + w_mlp*p + b). 대회 지표는 Brier(=MSE) 인데 메타는 log-loss 를
최소화한다는 불일치가 핵심 동기 (이 프로젝트 교훈 #10: RF 가 목적함수를 MSE 로 바꿔
+28%). 서베이(Ranjan-Gneiting, Yao2018, decision-focused pooling 2024) 도 "평가
스코어링룰에 가중치를 최적화" 가 원칙이라 함.

5종 (전부 val 전체 fit + val 전체 채점 = 이 repo 관례, EXPERIMENTS 숫자와 직접 비교):
  M1 logistic_raw   : LogisticRegression([p_cat, p_mlp])            <- 현행 프로덕션
  M2 logistic_logit : LogisticRegression([logit p_cat, logit p_mlp])  (LogOP, log-loss fit)
  M3 linreg_raw     : LinearRegression([p_cat, p_mlp]) -> y, clip     (Brier 직접최소화, unconstrained + b)
  M4 logop_brier    : minimize Brier of sigmoid(w1*logit p_cat + w2*logit p_mlp + b)  (지표정렬 LogOP)
  M5 simplex_brier  : w in [0,1], minimize Brier of  w*p_cat + (1-w)*p_mlp  (optimal linear pool, no b)

레짐: cutoff7 / 2023 / 2022 / 2021 (rolling-origin). 베이스 = raw-concat MLP 3-seed +
CatBoost 3-seed v2 HP, 트랙맨64 제거(1117 baseline). fold 당 베이스 1회 학습 후 5종 채점.

사용법:  python -m code.experiment_yudam_meta_methods
"""
import gc
import json
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

import os as _os
# 기본 = rolling-origin 4-fold, 3-seed 스크리닝.
# META_METHODS_REGIMES / META_METHODS_SEEDS 환경변수로 재확인 런 구성 가능
#   예) META_METHODS_REGIMES=cutoff7 META_METHODS_SEEDS=7  (production seed 재확인)
REGIMES = _os.environ.get("META_METHODS_REGIMES", "cutoff7,2023,2022,2021").split(",")
_NS = int(_os.environ.get("META_METHODS_SEEDS", "3"))
_SEED_POOL = (42, 123, 7, 2024, 99, 555, 31337)
MLP_SEEDS = _SEED_POOL[:_NS]
CB_SEEDS = _SEED_POOL[:min(_NS, 5)]
EPS = 1e-6
OUT = ("/tmp/claude-1000/-home-user-contest-mlp-lgaimers9/"
       "80d8950c-489c-47b5-93de-fb7a77e4f443/scratchpad/meta_methods.json")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _bss(pred, y):
    return compute_bss(np.clip(pred, 0.0, 1.0), y)[2]


def _brier(pred, y):
    return np.mean((np.clip(pred, 0.0, 1.0) - y) ** 2)


def m1_logistic_raw(pc, pm, y):
    clf = LogisticRegression().fit(np.column_stack([pc, pm]), y)
    w = clf.coef_[0]; b = float(clf.intercept_[0])
    pred = _sig(w[0] * pc + w[1] * pm + b)
    return pred, dict(w_cat=float(w[0]), w_mlp=float(w[1]), intercept=b, space="raw")


def m2_logistic_logit(pc, pm, y):
    lc, lm = _logit(pc), _logit(pm)
    clf = LogisticRegression().fit(np.column_stack([lc, lm]), y)
    w = clf.coef_[0]; b = float(clf.intercept_[0])
    pred = _sig(w[0] * lc + w[1] * lm + b)
    return pred, dict(w_cat=float(w[0]), w_mlp=float(w[1]), intercept=b, space="logit")


def m3_linreg_raw(pc, pm, y):
    reg = LinearRegression().fit(np.column_stack([pc, pm]), y)
    w = reg.coef_; b = float(reg.intercept_)
    pred = w[0] * pc + w[1] * pm + b
    return pred, dict(w_cat=float(w[0]), w_mlp=float(w[1]), intercept=b, space="raw_linear")


def m4_logop_brier(pc, pm, y):
    lc, lm = _logit(pc), _logit(pm)
    # 초기값 = m2 결과
    _, w0 = m2_logistic_logit(pc, pm, y)
    x0 = np.array([w0["w_cat"], w0["w_mlp"], w0["intercept"]])
    obj = lambda x: _brier(_sig(x[0] * lc + x[1] * lm + x[2]), y)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-10, maxiter=5000))
    x = res.x
    pred = _sig(x[0] * lc + x[1] * lm + x[2])
    return pred, dict(w_cat=float(x[0]), w_mlp=float(x[1]), intercept=float(x[2]),
                      space="logit_brier", converged=bool(res.success))


def m5_simplex_brier(pc, pm, y):
    # w*p_cat + (1-w)*p_mlp 의 Brier 는 w 에 대해 볼록 이차식 -> closed form
    d = pc - pm
    num = np.sum((y - pm) * d)
    den = np.sum(d * d)
    w = 0.5 if den == 0 else float(np.clip(num / den, 0.0, 1.0))
    pred = w * pc + (1.0 - w) * pm
    return pred, dict(w_cat=w, w_mlp=1.0 - w, intercept=0.0, space="simplex")


METHODS = {
    "M1_logistic_raw":   m1_logistic_raw,
    "M2_logistic_logit": m2_logistic_logit,
    "M3_linreg_raw":     m3_linreg_raw,
    "M4_logop_brier":    m4_logop_brier,
    "M5_simplex_brier":  m5_simplex_brier,
}


def _base_preds(regime, device):
    tr, va, num_cols, cat_cols, all_cols = build_split(regime, verbose=True)
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
        seeds=list(MLP_SEEDS), device=device, verbose=False,
    )
    bundle = make_bundle(members, CAT_COLS, num_cols, cat_dims, eds, ce, ni, ns, bin_edges=None)
    pm = predict_bundle(bundle, va[all_cols], device=device).astype(np.float64)

    cb_res = train_catboost_ensemble(
        tr[cat_cols], tr[TARGET].values, va[cat_cols], yv,
        seeds=list(CB_SEEDS), verbose=False, params=_yudam_catboost_params(),
    )
    pc = predict_catboost_ensemble([m for m, _ in cb_res], va[cat_cols]).astype(np.float64)

    del tr, va, members, bundle, Xtc, Xtn, Xvc, Xvn, cb_res
    gc.collect()
    return pc, pm, yv


def main():
    t0 = time.time()
    device = get_device()
    print(f"device={device} | MLP {MLP_SEEDS} / CB {CB_SEEDS}", flush=True)
    results = {}

    for regime in REGIMES:
        print(f"\n{'=' * 66}\nregime={regime}\n{'=' * 66}", flush=True)
        pc, pm, yv = _base_preds(regime, device)
        cat_solo, mlp_solo = _bss(pc, yv), _bss(pm, yv)
        print(f"[{regime}] cat_solo={cat_solo:.2f} | mlp_solo={mlp_solo:.2f} | n_val={len(yv)}", flush=True)
        reg = dict(cat_solo=cat_solo, mlp_solo=mlp_solo, n_val=int(len(yv)), methods={})
        for name, fn in METHODS.items():
            pred, info = fn(pc, pm, yv)
            score = _bss(pred, yv)
            reg["methods"][name] = dict(score=score, **info)
            print(f"[{regime}] {name:20s} blend={score:9.2f}  "
                  f"(w={info['w_cat']:.4f}/{info['w_mlp']:.4f}/{info.get('intercept', 0):.4f} {info['space']})",
                  flush=True)
        results[regime] = reg
        _dump(results)
        del pc, pm, yv
        gc.collect()

    _dump(results)
    _summary(results)
    print(f"\n총 소요 {time.time() - t0:.1f}s -> {OUT}", flush=True)


def _dump(results):
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _summary(results):
    print(f"\n{'=' * 74}\n요약: blend BSS (절대값) 및 M1 대비 Δ\n{'=' * 74}", flush=True)
    hdr = f"{'method':20s} " + " ".join(f"{r:>10s}" for r in REGIMES) + "   |  Δvs M1 (mean, wins)"
    print(hdr, flush=True)
    m1 = {r: results[r]["methods"]["M1_logistic_raw"]["score"] for r in REGIMES if r in results}
    for name in METHODS:
        scores = {r: results[r]["methods"][name]["score"] for r in REGIMES if r in results}
        deltas = [scores[r] - m1[r] for r in scores]
        wins = sum(1 for d in deltas if d > 0)
        row = f"{name:20s} " + " ".join(f"{scores[r]:10.2f}" for r in scores)
        print(f"{row}   |  {np.mean(deltas):+7.2f}  {wins}/{len(deltas)}", flush=True)
    print("\n(주: 2022/2021 은 val r 이 0.5 근방이라 BSS 분모 아티팩트로 절대값이 부풀려짐 #17 — Δ 부호/일관성만 보기)",
          flush=True)


if __name__ == "__main__":
    main()
