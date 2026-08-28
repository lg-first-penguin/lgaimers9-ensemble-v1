# code/experiment_meta_tuning_3way.py
"""[cat_pred, mlp_pred, fm_pred] 3-way 메타모델 후보를 `code/experiment_meta_tuning.py`와
동일한 원칙(TimeSeriesSplit, 시간 인과 방향 유지)으로 비교한다. fm_pred는
`code/experiment_fm_tuned3way.py --seeds ... `가 `./open/temp/experiment_fm_tuned/fm_val_preds.npz`에
저장한 7-seed 평균 예측을 그대로 쓴다(재학습 없음).

사용법: python -m code.experiment_meta_tuning_3way
"""
import os

import numpy as np

from code.experiment_thirdmodel_base import load_cache
from code.experiment_3way_stack import fit_meta_model_n
from code.mlp_model import compute_bss

FM_CACHE_DIR = "./open/temp/experiment_fm_tuned"


def build_meta_features_3way(cat_p, mlp_p, fm_p):
    return np.column_stack([cat_p, mlp_p, fm_p, cat_p * mlp_p, cat_p * fm_p, mlp_p * fm_p])


def fit_plain_logreg(X, y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    return model


def fit_logreg_cv(X, y):
    from sklearn.linear_model import LogisticRegressionCV
    model = LogisticRegressionCV(cv=5, Cs=10, max_iter=2000)
    model.fit(X, y)
    return model


CANDIDATES_3WAY = ["2way_baseline", "3way_plain", "3way_cv", "3way_meta_features_plain", "3way_meta_features_cv"]


def eval_candidate(name, cat_tr, mlp_tr, fm_tr, y_tr, cat_va, mlp_va, fm_va, y_va):
    if name == "2way_baseline":
        from code.blend_model import fit_meta_model, predict_meta
        w_cat, w_mlp, intercept, _, _ = fit_meta_model(cat_tr, mlp_tr, y_tr)
        pred = predict_meta(w_cat, w_mlp, intercept, cat_va, mlp_va)
    elif name == "3way_plain":
        X_tr = np.column_stack([cat_tr, mlp_tr, fm_tr])
        X_va = np.column_stack([cat_va, mlp_va, fm_va])
        model = fit_plain_logreg(X_tr, y_tr)
        pred = model.predict_proba(X_va)[:, 1]
    elif name == "3way_cv":
        X_tr = np.column_stack([cat_tr, mlp_tr, fm_tr])
        X_va = np.column_stack([cat_va, mlp_va, fm_va])
        model = fit_logreg_cv(X_tr, y_tr)
        pred = model.predict_proba(X_va)[:, 1]
    elif name == "3way_meta_features_plain":
        X_tr, X_va = build_meta_features_3way(cat_tr, mlp_tr, fm_tr), build_meta_features_3way(cat_va, mlp_va, fm_va)
        model = fit_plain_logreg(X_tr, y_tr)
        pred = model.predict_proba(X_va)[:, 1]
    elif name == "3way_meta_features_cv":
        X_tr, X_va = build_meta_features_3way(cat_tr, mlp_tr, fm_tr), build_meta_features_3way(cat_va, mlp_va, fm_va)
        model = fit_logreg_cv(X_tr, y_tr)
        pred = model.predict_proba(X_va)[:, 1]
    else:
        raise ValueError(name)
    return compute_bss(pred, y_va)[2]


def main():
    from sklearn.model_selection import TimeSeriesSplit

    train_split, val_split, cat_val_preds, mlp_val_preds, y_val, meta = load_cache()
    npz = np.load(os.path.join(FM_CACHE_DIR, "fm_val_preds.npz"))
    fm_val_preds = npz["fm_val_preds"]

    game_month = val_split["game_month"].values
    assert np.all(np.diff(game_month) >= 0)
    print(f"[3way meta-tuning] val n={len(y_val)}")
    print(f"[baseline] CatBoost={meta['catboost_score']:.2f} | MLP={meta['mlp_score']:.2f} | 2-way(전체윈도우)={meta['baseline_2way_score']:.2f}")
    fm_score = compute_bss(fm_val_preds, y_val)[2]
    print(f"[DeepFM(7-seed, tuned)] solo={fm_score:.2f}")

    print("\n[전체윈도우 in-sample 참고 — 과신 금지]")
    full_scores = {}
    for name in CANDIDATES_3WAY:
        s = eval_candidate(name, cat_val_preds, mlp_val_preds, fm_val_preds, y_val, cat_val_preds, mlp_val_preds, fm_val_preds, y_val)
        full_scores[name] = s
        print(f"  {name:<26} {s:.2f}")

    N_SPLITS = 8
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_results = {name: [] for name in CANDIDATES_3WAY}
    print(f"\n[TimeSeriesSplit n_splits={N_SPLITS}]")
    for tr_idx, va_idx in tscv.split(np.zeros(len(y_val))):
        for name in CANDIDATES_3WAY:
            s = eval_candidate(
                name,
                cat_val_preds[tr_idx], mlp_val_preds[tr_idx], fm_val_preds[tr_idx], y_val[tr_idx],
                cat_val_preds[va_idx], mlp_val_preds[va_idx], fm_val_preds[va_idx], y_val[va_idx],
            )
            fold_results[name].append(s)

    print("\n" + "=" * 100)
    print(f"{'candidate':<26} {'전체윈도우':>12} {'CV평균':>10} {'CV표준편차':>10} {'paired Δ평균':>13} {'paired Δ표준편차':>16}")
    baseline_folds = np.array(fold_results["2way_baseline"])
    for name in CANDIDATES_3WAY:
        folds = np.array(fold_results[name])
        paired_delta = folds - baseline_folds
        marker = "  <- 현재 프로덕션(2-way)" if name == "2way_baseline" else ""
        print(f"{name:<26} {full_scores[name]:>12.2f} {folds.mean():>10.2f} {folds.std():>10.2f} {paired_delta.mean():>13.2f} {paired_delta.std():>16.2f}{marker}")
    print("=" * 100)


if __name__ == "__main__":
    main()
