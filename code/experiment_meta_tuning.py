# code/experiment_meta_tuning.py
"""스태킹 메타모델(`code/blend_model.py::fit_meta_model`) 자체를 튜닝한다. 현재 프로덕션은
plain `LogisticRegression()`(C=1.0 고정, 정규화 강도 튜닝 없음) — [cat_pred, mlp_pred] 2피처
선형 결합뿐이다. 문헌/실무에서 검증된 스태킹 개선 레버 4가지를 시도한다:

  1. LogisticRegressionCV: 정규화 강도(C)를 5-fold CV로 자동 선택.
  2. 메타-피처 엔지니어링(Sill et al. 2009 "Feature-Weighted Linear Stacking"): 원본
     [cat,mlp] 2개 대신 상호작용(cat*mlp), 불일치도(|cat-mlp|), 평균((cat+mlp)/2)을
     추가한 5피처 선형 결합 — 여전히 선형이지만 두 모델이 "동의/불일치"하는 상황을
     구분할 수 있다.
  3. 비선형 메타러너(얕은 GradientBoosting) — 입력이 2~5개뿐이고 검증 표본이 ~11만행이라
     과적합 위험이 크므로 트리 수/깊이를 강하게 제한한다.
  4. Isotonic 보정: 기존 선형 블렌드 예측값에 단조 보정만 추가로 얹는다(모델 재선택 없이
     캘리브레이션만 개선하는 가장 보수적인 레버).

**폴드 설계, 두 차례 수정 경위**: 1차 시도는 cutoff7 val(2024 7~10월, 4개월치 11만행)을
7->8/7~8->9/7~9->10 식 월별 확장 윈도우로 쪼갰는데, 이 프로젝트의 실제 rolling-origin
관례(연 단위, `apply_te_residual_features`의 2021/2022/2023 방식)도 프로덕션 컨벤션도
아닌 즉석 스킴이었고, 폴드별 학습 표본이 33k/75k/108k로 들쭉날쭉해 노이즈가 컸다.
2차 시도로 시간순 대신 랜덤 K-fold(RepeatedStratifiedKFold)를 썼는데, 이건 **더 심각한
문제**를 만든다 — 메타모델을 미래 행(예: 9~10월)으로 학습해 과거 행(7월)을 맞히는 폴드가
섞여, 실제 배포(과거로만 미래를 예측)와 인과 방향이 반대인 검증이 되어버린다. 이 대회
규칙(투구 이전 정보만 사용) 정신과도 어긋나는 방법론적 오류라 폐기.

최종적으로 `sklearn.model_selection.TimeSeriesSplit`을 쓴다 — val_split의 행 순서가 이미
시간순(`game_month`가 non-decreasing으로 확인됨)이므로, 매 폴드는 항상 "과거 전체로 학습
-> 그 다음 시점 구간으로 검증"만 하고 미래->과거 방향은 절대 만들지 않는다. n_splits=8로
1차 시도의 3폴드보다 세분화해 노이즈를 평균으로 줄이되, 시간 인과 방향은 엄격히 지킨다.

사용법: python -m code.experiment_meta_tuning
"""
import numpy as np

from code.experiment_thirdmodel_base import load_cache
from code.mlp_model import compute_bss
from code.blend_model import fit_meta_model, predict_meta


def build_meta_features(cat_p, mlp_p):
    return np.column_stack([cat_p, mlp_p, cat_p * mlp_p, np.abs(cat_p - mlp_p), (cat_p + mlp_p) / 2])


def fit_logreg_cv(X, y):
    from sklearn.linear_model import LogisticRegressionCV
    model = LogisticRegressionCV(cv=5, Cs=10, max_iter=2000)
    model.fit(X, y)
    return model


def fit_gbm(X, y, seed=42):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.8,
        min_samples_leaf=200, random_state=seed,
    )
    model.fit(X, y)
    return model


def fit_isotonic(blend_pred, y):
    from sklearn.isotonic import IsotonicRegression
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(blend_pred, y)
    return model


CANDIDATES = ["baseline_logreg", "logreg_cv", "meta_features_logreg", "meta_features_logreg_cv", "gbm_2feat", "gbm_5feat", "isotonic"]


def eval_candidate(name, cat_tr, mlp_tr, y_tr, cat_va, mlp_va, y_va):
    if name == "baseline_logreg":
        w_cat, w_mlp, intercept, _, _ = fit_meta_model(cat_tr, mlp_tr, y_tr)
        pred = predict_meta(w_cat, w_mlp, intercept, cat_va, mlp_va)
    elif name == "logreg_cv":
        X_tr = np.column_stack([cat_tr, mlp_tr])
        X_va = np.column_stack([cat_va, mlp_va])
        model = fit_logreg_cv(X_tr, y_tr)
        pred = model.predict_proba(X_va)[:, 1]
    elif name == "meta_features_logreg":
        from sklearn.linear_model import LogisticRegression
        X_tr, X_va = build_meta_features(cat_tr, mlp_tr), build_meta_features(cat_va, mlp_va)
        model = LogisticRegression(max_iter=2000)
        model.fit(X_tr, y_tr)
        pred = model.predict_proba(X_va)[:, 1]
    elif name == "meta_features_logreg_cv":
        X_tr, X_va = build_meta_features(cat_tr, mlp_tr), build_meta_features(cat_va, mlp_va)
        model = fit_logreg_cv(X_tr, y_tr)
        pred = model.predict_proba(X_va)[:, 1]
    elif name == "gbm_2feat":
        X_tr = np.column_stack([cat_tr, mlp_tr])
        X_va = np.column_stack([cat_va, mlp_va])
        model = fit_gbm(X_tr, y_tr)
        pred = model.predict_proba(X_va)[:, 1]
    elif name == "gbm_5feat":
        X_tr, X_va = build_meta_features(cat_tr, mlp_tr), build_meta_features(cat_va, mlp_va)
        model = fit_gbm(X_tr, y_tr)
        pred = model.predict_proba(X_va)[:, 1]
    elif name == "isotonic":
        w_cat, w_mlp, intercept, _, _ = fit_meta_model(cat_tr, mlp_tr, y_tr)
        blend_tr = predict_meta(w_cat, w_mlp, intercept, cat_tr, mlp_tr)
        blend_va = predict_meta(w_cat, w_mlp, intercept, cat_va, mlp_va)
        iso = fit_isotonic(blend_tr, y_tr)
        pred = iso.predict(blend_va)
    else:
        raise ValueError(name)
    return compute_bss(pred, y_va)[2]


N_SPLITS = 8


def main():
    from sklearn.model_selection import TimeSeriesSplit

    train_split, val_split, cat_val_preds, mlp_val_preds, y_val, meta = load_cache()
    game_month = val_split["game_month"].values
    assert np.all(np.diff(game_month) >= 0), "val_split이 시간순이 아닙니다 — TimeSeriesSplit 전제가 깨집니다"
    print(f"[meta-tuning] val 전체 n={len(y_val)} (cutoff7 val=2024 7~10월, 행 순서=시간순 확인됨)")
    print(f"[baseline] CatBoost={meta['catboost_score']:.2f} | MLP={meta['mlp_score']:.2f} | 전체윈도우 2-way(현 프로덕션 방식)={meta['baseline_2way_score']:.2f}")

    # 전체 윈도우 in-sample(참고용, 과신 금지 — 아래 TimeSeriesSplit이 진짜 기준)
    print("\n[전체윈도우 in-sample 참고 — 과신 금지]")
    full_scores = {}
    for name in CANDIDATES:
        s = eval_candidate(name, cat_val_preds, mlp_val_preds, y_val, cat_val_preds, mlp_val_preds, y_val)
        full_scores[name] = s
        print(f"  {name:<26} {s:.2f}")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_results = {name: [] for name in CANDIDATES}
    fold_sizes = []
    print(f"\n[TimeSeriesSplit n_splits={N_SPLITS}, 항상 '과거 전체 학습 -> 다음 구간 검증', val 전체 {len(y_val)}행 대상]")
    for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(np.zeros(len(y_val)))):
        fold_sizes.append((len(tr_idx), len(va_idx)))
        print(f"  fold{fold_i+1}: train n={len(tr_idx)} (month {game_month[tr_idx[0]]}~{game_month[tr_idx[-1]]}) "
              f"-> val n={len(va_idx)} (month {game_month[va_idx[0]]}~{game_month[va_idx[-1]]})")
        for name in CANDIDATES:
            s = eval_candidate(
                name,
                cat_val_preds[tr_idx], mlp_val_preds[tr_idx], y_val[tr_idx],
                cat_val_preds[va_idx], mlp_val_preds[va_idx], y_val[va_idx],
            )
            fold_results[name].append(s)

    print("\n" + "=" * 100)
    print(f"{'candidate':<26} {'전체윈도우':>12} {'CV평균':>10} {'CV표준편차':>10} {'paired Δ평균':>13} {'paired Δ표준편차':>16}")
    baseline_folds = np.array(fold_results["baseline_logreg"])
    for name in CANDIDATES:
        folds = np.array(fold_results[name])
        avg, std = folds.mean(), folds.std()
        paired_delta = folds - baseline_folds
        marker = "  <- 현재 프로덕션" if name == "baseline_logreg" else ""
        print(f"{name:<26} {full_scores[name]:>12.2f} {avg:>10.2f} {std:>10.2f} {paired_delta.mean():>13.2f} {paired_delta.std():>16.2f}{marker}")
    print("=" * 100)
    print("(paired Δ표준편차가 paired Δ평균보다 훨씬 크면 신호가 아니라 노이즈로 판단)")


if __name__ == "__main__":
    main()
