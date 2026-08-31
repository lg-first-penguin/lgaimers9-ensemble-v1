# code/experiment_yudam_meta_refit_logit.py
"""[메타모델 다듬기] 트랙맨64-free 피처셋에서 stage-1 메타 재fit + logit-space 메타 비교.

배경:
 - 현행 프로덕션(실전 1117.03) 번들의 메타가중치(w_cat=2.0117 / w_mlp=1.9848 /
   intercept=-2.0284)는 트랙맨64가 *살아있을 때* fit한 걸 build_trackman64_removed_bundle.py
   --reuse-ref 로 그대로 재사용한 것. CLAUDE.md 도 "트랙맨64-free 피처셋으로 stage-1
   재fit 하면 조금 더 오를 여지" 명시. => [메타 #1]
 - mlp_prune_rolling 실험에서 baseline blend 를 raw 확률 대신 logit(log-odds)로 메타에
   넣으면 4-fold 전부 양수(cutoff7 +1.38 / 2023 +8.21 / 2022 +22.17 / 2021 +1.44,
   regime-flip 없음). 이게 production seed(MLP 7-seed)에서도 유지되는지 확인. => [메타 #2]

이 스크립트: cutoff7 stage-1 을 production seed(MLP 7-seed raw-concat + CatBoost 5-seed
v2 HP, 트랙맨64 제거)로 한 번 학습하고, 같은 예측으로 3가지 메타를 채점:
  A. raw 확률 입력 + val 전체 fit  (= 메타 #1: 트랙맨64-free 재fit)
  B. logit 입력 + val 전체 fit     (= 메타 #2)
  C. 현행 프로덕션 메타가중치를 이 예측에 그대로 적용 (비교 기준선)

사용법:  python -m code.experiment_yudam_meta_refit_logit
"""
import gc
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from code.experiment_yudam_common import build_split, _yudam_catboost_params, TARGET
from code.mlp_model import (
    CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing,
    to_tensors, train_ensemble, make_bundle, predict_bundle, get_device, compute_bss,
)
from code.catboost_model import train_catboost_ensemble, predict_catboost_ensemble
from code.train import YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS

PROD_META = dict(w_cat=2.0117, w_mlp=1.9848, intercept=-2.0284)  # 실전 1117.03 번들


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _score(pred, y):
    return compute_bss(pred, y)[2]


def _fit_and_score(cat_v, mlp_v, y, space):
    X = np.column_stack([_logit(cat_v), _logit(mlp_v)]) if space == "logit" \
        else np.column_stack([cat_v, mlp_v])
    clf = LogisticRegression()
    clf.fit(X, y)
    w0, w1 = (float(c) for c in clf.coef_[0])
    b = float(clf.intercept_[0])
    z = w0 * X[:, 0] + w1 * X[:, 1] + b
    return _score(1.0 / (1.0 + np.exp(-z)), y), (w0, w1, b)


def main():
    t0 = time.time()
    device = get_device()
    print(f"device={device} | MLP {len(YUDAM_ENSEMBLE_SEEDS)}-seed / CatBoost {len(YUDAM_CATBOOST_SEEDS)}-seed",
          flush=True)

    tr, va, num_cols, cat_cols, all_cols = build_split("cutoff7", verbose=True)
    yv = va[TARGET].values

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
    bundle = make_bundle(members, CAT_COLS, num_cols, cat_dims, eds, ce, ni, ns, bin_edges=None)
    mlp_v = predict_bundle(bundle, va[all_cols], device=device)

    cb_res = train_catboost_ensemble(
        tr[cat_cols], tr[TARGET].values, va[cat_cols], yv,
        seeds=list(YUDAM_CATBOOST_SEEDS), verbose=True, params=_yudam_catboost_params(),
    )
    cat_v = predict_catboost_ensemble([m for m, _ in cb_res], va[cat_cols])

    cat_solo, mlp_solo = _score(cat_v, yv), _score(mlp_v, yv)

    A_score, A_w = _fit_and_score(cat_v, mlp_v, yv, "raw")
    B_score, B_w = _fit_and_score(cat_v, mlp_v, yv, "logit")
    m = PROD_META
    C_pred = 1.0 / (1.0 + np.exp(-(m["w_cat"] * cat_v + m["w_mlp"] * mlp_v + m["intercept"])))
    C_score = _score(C_pred, yv)

    print("\n" + "=" * 66, flush=True)
    print(f"cutoff7 (트랙맨64-free) | CatBoost solo={cat_solo:.2f} | raw-MLP solo={mlp_solo:.2f}", flush=True)
    print("-" * 66, flush=True)
    print(f"C. 현행 프로덕션 메타 그대로 적용        blend={C_score:.2f}   (w_cat={m['w_cat']:.4f} w_mlp={m['w_mlp']:.4f} b={m['intercept']:.4f})", flush=True)
    print(f"A. raw 확률 입력 + val 전체 재fit  [메타#1]  blend={A_score:.2f}   Δvs C {A_score - C_score:+.2f}   (w={A_w[0]:.4f}/{A_w[1]:.4f}/{A_w[2]:.4f})", flush=True)
    print(f"B. logit 입력 + val 전체 fit       [메타#2]  blend={B_score:.2f}   Δvs A {B_score - A_score:+.2f}   (w={B_w[0]:.4f}/{B_w[1]:.4f}/{B_w[2]:.4f})", flush=True)
    print("=" * 66, flush=True)
    print(f"\n총 소요 {time.time() - t0:.1f}s", flush=True)

    out = dict(
        cat_solo=cat_solo, mlp_solo=mlp_solo, n_val=int(len(yv)),
        C_prod_meta=dict(score=C_score, **PROD_META),
        A_raw_refit=dict(score=A_score, w_cat=A_w[0], w_mlp=A_w[1], intercept=A_w[2]),
        B_logit_refit=dict(score=B_score, w_cat=B_w[0], w_mlp=B_w[1], intercept=B_w[2]),
    )
    with open("/tmp/claude-1000/-home-user-contest-mlp-lgaimers9/"
              "80d8950c-489c-47b5-93de-fb7a77e4f443/scratchpad/meta_refit_logit.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
