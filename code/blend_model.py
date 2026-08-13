# code/blend_model.py
"""CatBoost + Tabular MLP 앙상블 블렌딩 유틸리티.

실측 검증 결과(`open/data` 기준, `season==2024` 홀드아웃): CatBoost 단독 818.54,
Tabular MLP 7-seed 앙상블 단독 789.58, 단순 가중 평균 블렌드(alpha≈0.55) 854.09 —
두 모델 종류를 섞는 것만으로 각 단독 모델보다 유의미하게 높은 점수를 얻었습니다.
이후 alpha 가중평균을 `[cat_pred, mlp_pred]` 2피처 로지스틱 회귀 메타모델(스태킹)로
교체 — rolling-origin 3-fold 검증에서 alpha 블렌드 대비 전 fold 우세(평균 +14.22점)를
확인하고 정식 반영. 자세한 내용은 TABULAR_MLP_REPORT.md, EXPERIMENTS.md §9 참고.

번들 스키마: {"catboost_model": <CatBoostClassifier>, "mlp_bundle": {...}, "meta_model": {"w_cat", "w_mlp", "intercept"}}
- "catboost_model"은 `catboost` 라이브러리가 제공하는 클래스라 `code/` 패키지 없이도
  (catboost만 설치되어 있으면) unpickle 가능합니다.
- "mlp_bundle"은 기존 `code/mlp_model.py::make_bundle` 결과(순수 dict)를 그대로 내포합니다.
- "meta_model"은 학습된 `sklearn.linear_model.LogisticRegression` 인스턴스가 아니라
  `w_cat`/`w_mlp`/`intercept` 순수 float만 담은 dict입니다 — 추론 시 sklearn 의존 없이
  `predict_meta`의 수식(로지스틱 함수)만으로 재현 가능하게 하기 위함(mlp_bundle이 클래스
  인스턴스 대신 dict로 저장되는 것과 동일한 이유).
"""
import numpy as np

from code.mlp_model import predict_bundle, compute_bss
from code.catboost_model import predict_catboost


def predict_meta(w_cat, w_mlp, intercept, cat_preds, mlp_preds):
    z = w_cat * cat_preds + w_mlp * mlp_preds + intercept
    return 1.0 / (1.0 + np.exp(-z))


def fit_meta_model(cat_preds, mlp_preds, y_val):
    """[cat_pred, mlp_pred]를 입력으로 하는 로지스틱 회귀 메타모델을 학습합니다.
    (w_cat, w_mlp, intercept, blend_score, blend_brier)를 반환합니다."""
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([cat_preds, mlp_preds])
    model = LogisticRegression()
    model.fit(X, y_val)
    w_cat, w_mlp = (float(c) for c in model.coef_[0])
    intercept = float(model.intercept_[0])

    blend = predict_meta(w_cat, w_mlp, intercept, cat_preds, mlp_preds)
    brier, bss, score = compute_bss(blend, y_val)
    return w_cat, w_mlp, intercept, score, brier


def make_blend_bundle(catboost_model, mlp_bundle, meta_model):
    return {
        "catboost_model": catboost_model,
        "mlp_bundle": mlp_bundle,
        "meta_model": dict(meta_model),
    }


def predict_blend_bundle(bundle, df, device=None):
    cat_preds = predict_catboost(bundle["catboost_model"], df)
    mlp_preds = predict_bundle(bundle["mlp_bundle"], df, device=device)
    meta = bundle["meta_model"]
    return predict_meta(meta["w_cat"], meta["w_mlp"], meta["intercept"], cat_preds, mlp_preds)
