# code/experiment_context_blend_constrained.py
"""[Context 블렌드 실험 2/2] constrained context-dependent blend weight, LOFO 검증.

배경: yudam 이 자유로운 CatBoost 메타(context 피처 다수 + 카운트)로 "행마다 다른 블렌드
규칙"을 시도했으나 LOFO 재검증에서 방향 불안정(데이터 적은 fold 크게 이김 / 실전과
가까운 half_2024·expand_2023 짐) → 기각 (yudam EXPERIMENTS.md §6-18).

여기서는 **constrained 버전**: context 축을 1개만, functional form 도 자유 트리가 아니라
**로지스틱 회귀 + 1축 상호작용항**으로 제한한다.

  M_fixed  : sigmoid(a·L_cat + b·L_mlp + c)                          (3 params, 현행형)
  M_ctx    : sigmoid((a0+a1·z̃)·L_cat + (b0+b1·z̃)·L_mlp + (c0+c1·z̃)) (6 params)
             = LogisticRegression on [L_cat, L_mlp, z̃·L_cat, z̃·L_mlp, z̃]
  M_ctx_reg: 위와 동일하나 C=0.1 (상호작용항까지 강하게 규제 = 더 constrained)

L_* = logit(clip(p)), z̃ = pooled-train 표준화 context 축. 축은 1개씩 스윕.

LOFO: 홀드아웃 fold f 를 제외한 3 fold((p_cat,p_mlp,z,y))를 합쳐 메타 fit → f 에 frozen
적용 → BSS. 4 fold 각각 홀드아웃. yudam 을 죽인 바로 그 프로토콜.

판정: mean Δ > 0 **이고** 실전과 가까운 fold(cutoff7=half_2024, 2023)가 안 져야 함.
2021/2022 만 이기고 cutoff7/2023 지면 → yudam 과 동일, 기각.

선행: `python -m code.experiment_context_blend_cache` (fold pkl 4개 생성)
사용법: python -m code.experiment_context_blend_constrained
        python -m code.experiment_context_blend_constrained --axes li,count_pressure
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

FOLDS = ["cutoff7", "2023", "2022", "2021"]
REALISTIC = {"cutoff7", "2023"}  # 실전(2025)과 구조적으로 가까운 fold
CACHE_DIR = "scratchpad"
DEFAULT_AXES = ["li", "count_pressure", "count_diff", "strikes_before",
                "same_hand", "num_runners_on", "home_win_expectancy", "outs_before"]
EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _bss(p, y):
    y = np.asarray(y, dtype=np.float64)
    r = y.mean()
    brier = np.mean((np.asarray(p, dtype=np.float64) - y) ** 2)
    base = r * (1 - r)
    return max(0.0, 100000.0 * (1.0 - brier / base))


def load_folds():
    out = {}
    for f in FOLDS:
        path = os.path.join(CACHE_DIR, f"ctxblend_fold_{f}.pkl")
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        d["L_cat"] = _logit(d["p_cat"])
        d["L_mlp"] = _logit(d["p_mlp"])
        out[f] = d
    return out


def fit_fixed(tr):
    X = np.column_stack([tr["L_cat"], tr["L_mlp"]])
    clf = LogisticRegression(C=1e6, max_iter=1000).fit(X, tr["y"])
    return clf


def apply_fixed(clf, d):
    X = np.column_stack([d["L_cat"], d["L_mlp"]])
    return clf.predict_proba(X)[:, 1]


def _ctx_design(L_cat, L_mlp, zt):
    return np.column_stack([L_cat, L_mlp, zt * L_cat, zt * L_mlp, zt])


def fit_ctx(tr, axis, C):
    z = tr["ctx"][axis].to_numpy(dtype=np.float64)
    mu = np.nanmedian(z)
    z = np.where(np.isfinite(z), z, mu)
    m, s = z.mean(), z.std()
    s = s if s > 1e-9 else 1.0
    zt = (z - m) / s
    X = _ctx_design(tr["L_cat"], tr["L_mlp"], zt)
    clf = LogisticRegression(C=C, max_iter=2000).fit(X, tr["y"])
    return clf, (m, s, mu)


def apply_ctx(clf, norm, axis, d):
    m, s, mu = norm
    z = d["ctx"][axis].to_numpy(dtype=np.float64)
    z = np.where(np.isfinite(z), z, mu)
    zt = (z - m) / s
    X = _ctx_design(d["L_cat"], d["L_mlp"], zt)
    return clf.predict_proba(X)[:, 1]


def pool(folds, names):
    keys = ["L_cat", "L_mlp", "y"]
    d = {k: np.concatenate([folds[n][k] for n in names]) for k in keys}
    import pandas as pd
    d["ctx"] = pd.concat([folds[n]["ctx"] for n in names], ignore_index=True)
    return d


def run_axis(folds, axis):
    rows = []
    coefs = []
    for held in FOLDS:
        tr_names = [f for f in FOLDS if f != held]
        tr = pool(folds, tr_names)
        te = folds[held]

        fx = fit_fixed(tr)
        s_fixed = _bss(apply_fixed(fx, te), te["y"])

        cx, norm = fit_ctx(tr, axis, C=1.0)
        s_ctx = _bss(apply_ctx(cx, norm, axis, te), te["y"])

        cxr, normr = fit_ctx(tr, axis, C=0.1)
        s_ctx_r = _bss(apply_ctx(cxr, normr, axis, te), te["y"])

        # 상호작용 계수: [L_cat, L_mlp, z*L_cat, z*L_mlp, z] -> a1,b1,c1 = idx 2,3,4
        a1, b1, c1 = cx.coef_[0][2], cx.coef_[0][3], cx.coef_[0][4]
        coefs.append((a1, b1, c1))
        rows.append(dict(held=held, s_fixed=s_fixed, d_ctx=s_ctx - s_fixed,
                         d_ctx_reg=s_ctx_r - s_fixed))
    return rows, coefs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", type=str, default=",".join(DEFAULT_AXES))
    args = ap.parse_args()
    folds = load_folds()
    print("fold sizes / solo:")
    for f in FOLDS:
        d = folds[f]
        print(f"  {f:>8}: n={len(d['y'])}  cat_solo={d.get('cat_solo', float('nan')):.1f} "
              f"mlp_solo={d.get('mlp_solo', float('nan')):.1f}  ctx cols={list(d['ctx'].columns)}", flush=True)

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    axes = [a for a in axes if a in folds[FOLDS[0]]["ctx"].columns]
    print(f"\naxes to sweep: {axes}", flush=True)

    summary = []
    for axis in axes:
        rows, coefs = run_axis(folds, axis)
        print(f"\n{'='*72}\n=== axis = {axis} ===\n{'='*72}", flush=True)
        print(f"{'held-out':>9} | {'M_fixed':>9} | {'Δ M_ctx':>9} | {'Δ M_ctx_reg':>12}", flush=True)
        for r in rows:
            tag = " (실전近)" if r["held"] in REALISTIC else ""
            print(f"{r['held']:>9} | {r['s_fixed']:9.2f} | {r['d_ctx']:+9.2f} | {r['d_ctx_reg']:+12.2f}{tag}", flush=True)
        d_ctx = np.array([r["d_ctx"] for r in rows])
        d_reg = np.array([r["d_ctx_reg"] for r in rows])
        real_ctx = np.array([r["d_ctx"] for r in rows if r["held"] in REALISTIC])
        real_reg = np.array([r["d_ctx_reg"] for r in rows if r["held"] in REALISTIC])
        a1m = np.mean([c[0] for c in coefs]); b1m = np.mean([c[1] for c in coefs]); c1m = np.mean([c[2] for c in coefs])
        a1s = np.std([c[0] for c in coefs]); b1s = np.std([c[1] for c in coefs])
        print(f"  mean Δ  M_ctx={d_ctx.mean():+.2f} (real folds {real_ctx.mean():+.2f}) | "
              f"M_ctx_reg={d_reg.mean():+.2f} (real {real_reg.mean():+.2f})", flush=True)
        print(f"  상호작용 계수 (LOFO 4fit 평균±std): a1(z·L_cat)={a1m:+.3f}±{a1s:.3f}  "
              f"b1(z·L_mlp)={b1m:+.3f}±{b1s:.3f}  c1(z)={c1m:+.3f}", flush=True)
        verdict = ("후보" if (d_ctx.mean() > 0 and real_ctx.min() > -2) else
                   "기각(실전近 fold 손해 or mean<0)")
        summary.append((axis, d_ctx.mean(), real_ctx.mean(), d_reg.mean(), real_reg.mean(), verdict))

    print(f"\n{'='*84}\n=== 종합 (constrained context blend, LOFO 4fold) ===\n{'='*84}", flush=True)
    print(f"{'axis':>20} | {'meanΔ ctx':>10} | {'realΔ ctx':>10} | {'meanΔ reg':>10} | {'realΔ reg':>10} | verdict", flush=True)
    for axis, m, rm, mr, rmr, v in summary:
        print(f"{axis:>20} | {m:+10.2f} | {rm:+10.2f} | {mr:+10.2f} | {rmr:+10.2f} | {v}", flush=True)
    print("\n판정: meanΔ>0 이고 cutoff7·2023(실전近) 둘 다 안 져야 후보. "
          "2021/2022만 이기면 yudam과 동일 = 기각.", flush=True)


if __name__ == "__main__":
    main()
