# code/experiment_calibration.py
"""CatBoost+MLP 스태킹 블렌드 확률에 사후 보정(calibration)을 한 겹 더 씌우면
이득이 있는지 검증하는 실험.

지금까지 시도한 "모델 개수를 늘리기"(§12/§22~24)와 "메타모델을 비선형으로
바꾸기"(§32, `code/experiment_meta_nonlinear.py`)는 둘 다 기각됐다. 이 스크립트는
전혀 다른 각도 — 두 모델을 어떻게 합칠지가 아니라, **이미 합쳐진 최종 확률이
얼마나 잘 보정(calibrated)돼 있는지**를 본다. BSS(Brier Skill Score)는 확률
보정 품질에 직접 민감한 지표라서, 모델 결합 방식과 독립적으로 시도해볼 가치가
있다. CatBoost/MLP 재학습이 전혀 필요 없다 — `code/experiment_meta_nonlinear.py`가
만들어둔 캐시(`open/temp/experiment_meta_nonlinear/`)를 그대로 재사용한다.

각 rolling-origin fold에서: (1) 기존과 동일하게 meta_train 구간에서 선형
스태킹(`fit_meta_model`)을 학습해 블렌드 확률을 얻고, (2) 그 블렌드 확률 위에
보정 함수를 meta_train 구간에서 추가로 학습해(입력 1개: 블렌드 확률 자체)
meta_val에 적용한다. 보정이 없는 (1)번만 쓴 경우와 비교한다.

사용법:
  python -m code.experiment_calibration --step screen
  python -m code.experiment_calibration --step screen --candidate isotonic
"""
import argparse
import os

import numpy as np

from code.experiment_meta_nonlinear import load_cache, CACHE_DIR as BASE_CACHE_DIR, FOLD_BOUNDARIES
from code.blend_model import fit_meta_model, predict_meta
from code.mlp_model import compute_bss


def fit_isotonic(blend_tr, y_tr):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(blend_tr, y_tr)
    return lambda blend_va: iso.predict(blend_va)


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt(blend_tr, y_tr):
    """Platt scaling: 블렌드 확률을 logit으로 되돌린 뒤 1차원 로지스틱 회귀로 재보정."""
    from sklearn.linear_model import LogisticRegression
    X_tr = _logit(blend_tr).reshape(-1, 1)
    model = LogisticRegression()
    model.fit(X_tr, y_tr)

    def predict_fn(blend_va):
        X_va = _logit(blend_va).reshape(-1, 1)
        return model.predict_proba(X_va)[:, 1]
    return predict_fn


CANDIDATES = {
    "isotonic": fit_isotonic,
    "platt": fit_platt,
}


def foldcheck_one(name, fit_fn, cat_val_preds, mlp_val_preds, y_val, game_month):
    deltas = []
    rows = []
    for lo, hi in FOLD_BOUNDARIES:
        val_mask = (game_month == lo) | (game_month == hi)
        train_mask = game_month < lo

        cat_tr, mlp_tr, y_tr = cat_val_preds[train_mask], mlp_val_preds[train_mask], y_val[train_mask]
        cat_va, mlp_va, y_va = cat_val_preds[val_mask], mlp_val_preds[val_mask], y_val[val_mask]

        w_cat, w_mlp, intercept, _, _ = fit_meta_model(cat_tr, mlp_tr, y_tr)
        blend_tr = predict_meta(w_cat, w_mlp, intercept, cat_tr, mlp_tr)
        blend_va = predict_meta(w_cat, w_mlp, intercept, cat_va, mlp_va)
        baseline_score = compute_bss(blend_va, y_va)[2]

        calibrate_fn = fit_fn(blend_tr, y_tr)
        calibrated_va = calibrate_fn(blend_va)
        candidate_score = compute_bss(calibrated_va, y_va)[2]

        delta = candidate_score - baseline_score
        deltas.append(delta)
        rows.append((f"{lo}~{hi}월", train_mask.sum(), val_mask.sum(), baseline_score, candidate_score, delta))

    wins = sum(1 for d in deltas if d > 0)
    print(f"\n[{name}] rolling-origin 3-fold foldcheck (vs 보정 없는 선형 스태킹, 각 fold에서 둘 다 재학습)")
    print(f"{'val 구간':<10} {'n_train':>8} {'n_val':>8} {'보정없음':>10} {name:>14} {'delta':>8}")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:>8} {r[2]:>8} {r[3]:>10.2f} {r[4]:>14.2f} {r[5]:>+8.2f}")
    print(f"[{name}] {wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}")
    return deltas


def step_screen(candidate=None):
    if not os.path.exists(os.path.join(BASE_CACHE_DIR, "meta.pkl")):
        raise RuntimeError("먼저 `python -m code.experiment_meta_nonlinear --step base`를 실행해 캐시를 만들어야 합니다.")

    cat_val_preds, mlp_val_preds, y_val, game_month, meta = load_cache()
    print(f"[screen] 캐시 로드 완료 (code/experiment_meta_nonlinear의 base 캐시 재사용) — "
          f"기존 2-way 선형 스태킹(전체 season==2024) 기준 점수: {meta['baseline_2way_score']:.2f}")

    names = [candidate] if candidate else list(CANDIDATES.keys())
    summary = {}
    for name in names:
        deltas = foldcheck_one(name, CANDIDATES[name], cat_val_preds, mlp_val_preds, y_val, game_month)
        summary[name] = deltas

    if len(summary) > 1:
        print("\n=== 후보 비교 요약 ===")
        print(f"{'후보':<10} {'fold 승수':>10} {'평균 delta':>12}")
        for name, deltas in summary.items():
            wins = sum(1 for d in deltas if d > 0)
            print(f"{name:<10} {wins:>8}/3 {np.mean(deltas):>+12.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["screen"])
    parser.add_argument("--candidate", default=None, choices=list(CANDIDATES.keys()),
                         help="후보 하나만 실행 (생략 시 전체 실행)")
    args = parser.parse_args()
    step_screen(candidate=args.candidate)


if __name__ == "__main__":
    main()
