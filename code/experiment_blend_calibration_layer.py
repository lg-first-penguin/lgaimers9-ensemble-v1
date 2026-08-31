# code/experiment_blend_calibration_layer.py
"""[Calibration layer 실험] 최종 블렌드 예측 위에 monotonic 보정층, LOFO 검증.

배경: M4(메타 fit 목적함수 log-loss→Brier)는 선형 메타의 *재fit* 이었지 블렌드 출력에
붙이는 monotonic 보정층이 아니었다. BSS 분해상 REL(신뢰도) 은 거의 포화(blend REL
7.68pt, RES 지배 — memory: bss_rel_res_decomposition_diagnostic) 라 여지는 작지만
그 구체적 형태(isotonic / Platt / beta)는 미시도.

**핵심 동기(사용자)**: 팀원 실험에서 블렌드에 *상수* 보정치를 더했더니 실전 +9 였던 적이
있음. CLAUDE.md 의 기존 결론은 "팀원은 intercept 없는 고정가중 평균(0.61·cat+0.39·mlp)을
써서 global bias 를 상수가 잡아준 것 / 이 repo 는 stacking 메타에 intercept 가 있어
`mean_pred-mean_actual≈+0.00002` 로 이미 무의미". 근데 지금은 yudam 레시피(2-input
로지스틱 메타, 70/30 fit)로 바뀌었으니 재확인. 그래서 monotonic 보정기뿐 아니라
**순수 상수 보정(shift)** 도 같이 측정한다:
  - fold별 global bias `mean(p_blend)-mean(y)` 및 부호 일관성 (상수가 실전에서 통하려면
    4 fold 부호가 같아야 함)
  - `shift`      : logit(p') = logit(p) + b*   (1-param, 팀원 상수보정의 logit공간 버전)
  - `shift_prob` : p' = clip(p + c*, 0, 1)      (확률공간 상수)
  - Platt 의 기울기 w 가 ~1 이면 Platt = shift 와 사실상 동일(=상수효과)

`experiment_context_blend_cache.py` 가 만든 fold pkl(p_cat/p_mlp/y) 재사용. 각 fold 의
블렌드 예측을 만든 뒤(2가지 소스), 보정기를 3 fold 로 fit → 홀드아웃 fold 에 frozen
적용 → BSS. 4 fold 각각 홀드아웃 (LOFO, yudam 컨텍스트 메타를 죽인 프로토콜과 동일).

블렌드 소스:
  prod_w  : 1117 번들 프로덕션 메타 가중치 고정 (w_cat=2.0117 w_mlp=1.9848 b=-2.0284)
  lofo_fit: 홀드아웃 제외 3 fold 로 [L_cat,L_mlp]→y 로지스틱 재fit (context 실험의 M_fixed 와 동일)

보정기:
  isotonic : IsotonicRegression(out_of_bounds='clip')  — 비모수 monotonic
  platt    : sigmoid(w·logit(p_blend) + b)  — 1.5-param, 과적합 덜함
  beta     : sigmoid(a·ln p + b·ln(1-p) + c)  — Kull 2017, 3-param

판정: mean Δ > 0 이고 cutoff7·2023(실전近) 안 져야 후보. 아니면 기각.

선행: python -m code.experiment_context_blend_cache
사용법: python -m code.experiment_blend_calibration_layer
"""
import os
import pickle

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

FOLDS = ["cutoff7", "2023", "2022", "2021"]
REALISTIC = {"cutoff7", "2023"}
CACHE_DIR = "scratchpad"
EPS = 1e-6
PROD_META = dict(w_cat=2.0117, w_mlp=1.9848, b=-2.0284)


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _bss(p, y):
    y = np.asarray(y, np.float64)
    r = y.mean()
    brier = np.mean((np.clip(p, 0, 1) - y) ** 2)
    return max(0.0, 100000.0 * (1.0 - brier / (r * (1 - r))))


def load_folds():
    out = {}
    for f in FOLDS:
        with open(os.path.join(CACHE_DIR, f"ctxblend_fold_{f}.pkl"), "rb") as fh:
            d = pickle.load(fh)
        d["L_cat"], d["L_mlp"] = _logit(d["p_cat"]), _logit(d["p_mlp"])
        out[f] = d
    return out


def blend_prod(d):
    return _sig(PROD_META["w_cat"] * d["p_cat"] + PROD_META["w_mlp"] * d["p_mlp"] + PROD_META["b"])


def make_lofo_blender(folds, tr_names):
    L = np.concatenate([folds[n]["L_cat"] for n in tr_names])
    M = np.concatenate([folds[n]["L_mlp"] for n in tr_names])
    y = np.concatenate([folds[n]["y"] for n in tr_names])
    clf = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack([L, M]), y)
    return lambda d: clf.predict_proba(np.column_stack([d["L_cat"], d["L_mlp"]]))[:, 1]


# ---- calibrators: fit(p, y) -> transform(p) ----
def cal_isotonic(p, y):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    return lambda q: ir.predict(q)


def cal_platt(p, y):
    z = _logit(p).reshape(-1, 1)
    clf = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    t = lambda q: clf.predict_proba(_logit(q).reshape(-1, 1))[:, 1]
    t.slope = float(clf.coef_[0][0])
    t.intercept = float(clf.intercept_[0])
    return t


def cal_shift(p, y):
    """logit(p') = logit(p) + b*, b* = argmin log-loss (기울기 1 고정, 순수 상수 shift)."""
    from scipy.optimize import minimize_scalar
    z = _logit(p)
    y = np.asarray(y, np.float64)

    def nll(b):
        q = np.clip(_sig(z + b), EPS, 1 - EPS)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    b = minimize_scalar(nll, bounds=(-3, 3), method="bounded").x
    t = lambda q: _sig(_logit(q) + b)
    t.b = float(b)
    return t


def cal_shift_prob(p, y):
    """p' = clip(p + c*, 0, 1), c* = argmin Brier (확률공간 상수)."""
    from scipy.optimize import minimize_scalar
    p = np.asarray(p, np.float64)
    y = np.asarray(y, np.float64)
    c = minimize_scalar(lambda c: np.mean((np.clip(p + c, 0, 1) - y) ** 2),
                        bounds=(-0.2, 0.2), method="bounded").x
    t = lambda q: np.clip(q + c, 0, 1)
    t.c = float(c)
    return t


def cal_beta(p, y):
    pc = np.clip(p, EPS, 1 - EPS)
    X = np.column_stack([np.log(pc), np.log(1 - pc)])
    clf = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)

    def _t(q):
        qc = np.clip(q, EPS, 1 - EPS)
        return clf.predict_proba(np.column_stack([np.log(qc), np.log(1 - qc)]))[:, 1]
    return _t


CALS = {"isotonic": cal_isotonic, "platt": cal_platt, "beta": cal_beta,
        "shift": cal_shift, "shift_prob": cal_shift_prob}


def run_source(folds, source):
    rows = []
    for held in FOLDS:
        tr_names = [f for f in FOLDS if f != held]
        te = folds[held]

        if source == "prod_w":
            blend = blend_prod
        else:
            blend = make_lofo_blender(folds, tr_names)

        p_tr = np.concatenate([blend(folds[n]) for n in tr_names])
        y_tr = np.concatenate([folds[n]["y"] for n in tr_names])
        p_te = blend(te)
        base = _bss(p_te, te["y"])
        bias_te = float(p_te.mean() - te["y"].mean())
        bias_tr = float(p_tr.mean() - y_tr.mean())

        row = dict(held=held, base=base, bias_te=bias_te, bias_tr=bias_tr)
        for cname, cfit in CALS.items():
            t = cfit(p_tr, y_tr)
            row[f"d_{cname}"] = _bss(t(p_te), te["y"]) - base
            if cname == "shift":
                row["shift_b"] = t.b
            elif cname == "shift_prob":
                row["shift_c"] = t.c
            elif cname == "platt":
                row["platt_slope"] = t.slope
        rows.append(row)
    return rows


def main():
    folds = load_folds()
    print("fold sizes:", {f: len(folds[f]["y"]) for f in FOLDS}, flush=True)

    summary = []
    for source in ("prod_w", "lofo_fit"):
        rows = run_source(folds, source)
        print(f"\n{'='*90}\n=== blend source = {source} ===\n{'='*90}", flush=True)
        print("  global bias (mean p_blend - mean y):", flush=True)
        for r in rows:
            tag = " (실전近)" if r["held"] in REALISTIC else ""
            print(f"    {r['held']:>9}: held-out bias={r['bias_te']:+.5f}  train(3fold) bias={r['bias_tr']:+.5f}  "
                  f"| shift b*={r.get('shift_b', 0):+.4f}  shift_prob c*={r.get('shift_c', 0):+.5f}  "
                  f"platt slope={r.get('platt_slope', 1):.3f}{tag}", flush=True)
        biases = [r["bias_te"] for r in rows]
        print(f"    -> held-out bias 부호: {['%+.4f' % b for b in biases]}  "
              f"({'일관' if (all(b > 0 for b in biases) or all(b < 0 for b in biases)) else '부호갈림 → 상수보정 실전 불가'})", flush=True)
        hdr = f"\n{'held-out':>9} | {'base BSS':>9} | " + " | ".join(f"Δ {c:>9}" for c in CALS)
        print(hdr, flush=True)
        for r in rows:
            tag = " (실전近)" if r["held"] in REALISTIC else ""
            print(f"{r['held']:>9} | {r['base']:9.2f} | " +
                  " | ".join(f"{r['d_'+c]:+11.2f}" for c in CALS) + tag, flush=True)
        for c in CALS:
            alld = np.array([r["d_" + c] for r in rows])
            reald = np.array([r["d_" + c] for r in rows if r["held"] in REALISTIC])
            v = "후보" if (alld.mean() > 0 and reald.min() > -2) else "기각"
            print(f"  {c:>9}: meanΔ={alld.mean():+.2f}  realΔ(mean)={reald.mean():+.2f}  "
                  f"min={alld.min():+.2f}  -> {v}", flush=True)
            summary.append((source, c, alld.mean(), reald.mean(), alld.min(), v))

    print(f"\n{'='*80}\n=== 종합 (calibration layer, LOFO 4fold) ===\n{'='*80}", flush=True)
    print(f"{'source':>9} | {'calibrator':>10} | {'meanΔ':>8} | {'realΔ':>8} | {'minΔ':>8} | verdict", flush=True)
    for s, c, m, rm, mn, v in summary:
        print(f"{s:>9} | {c:>10} | {m:+8.2f} | {rm:+8.2f} | {mn:+8.2f} | {v}", flush=True)
    print("\n판정: meanΔ>0 이고 cutoff7·2023 안 져야 후보. BSS 분해상 REL 포화라 소폭 예상.", flush=True)


if __name__ == "__main__":
    main()
