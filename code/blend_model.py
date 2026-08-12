# code/blend_model.py
"""CatBoost + Tabular MLP 앙상블 블렌딩 유틸리티.

실측 검증 결과(`open/data` 기준, `season==2024` 홀드아웃): CatBoost 단독 818.54,
Tabular MLP 7-seed 앙상블 단독 789.58, 단순 가중 평균 블렌드(alpha≈0.55) 854.09 —
두 모델 종류를 섞는 것만으로 각 단독 모델보다 유의미하게 높은 점수를 얻었습니다.
자세한 내용은 TABULAR_MLP_REPORT.md 참고.

번들 스키마: {"catboost_model": <CatBoostClassifier>, "mlp_bundle": {...}, "alpha": float}
- "catboost_model"은 `catboost` 라이브러리가 제공하는 클래스라 `code/` 패키지 없이도
  (catboost만 설치되어 있으면) unpickle 가능합니다.
- "mlp_bundle"은 기존 `code/mlp_model.py::make_bundle` 결과(순수 dict)를 그대로 내포합니다.
- alpha는 최종 예측 = alpha * CatBoost 예측 + (1 - alpha) * MLP 앙상블 예측 의 가중치입니다.
"""
import numpy as np

from code.mlp_model import predict_bundle, compute_bss
from code.catboost_model import predict_catboost

ALPHA_SEARCH_STEP = 0.01


def sweep_alpha(cat_preds, mlp_preds, y_val, step=ALPHA_SEARCH_STEP):
    """alpha * cat_preds + (1-alpha) * mlp_preds 형태로 최적 blend alpha를 그리드 서치합니다."""
    best_alpha, best_score, best_brier = 0.5, -1.0, None
    for alpha in np.arange(0.0, 1.0 + step / 2, step):
        alpha = float(alpha)
        blend = alpha * cat_preds + (1 - alpha) * mlp_preds
        brier, bss, score = compute_bss(blend, y_val)
        if score > best_score:
            best_alpha, best_score, best_brier = alpha, score, brier
    return best_alpha, best_score, best_brier


def make_blend_bundle(catboost_model, mlp_bundle, alpha):
    return {
        "catboost_model": catboost_model,
        "mlp_bundle": mlp_bundle,
        "alpha": float(alpha),
    }


def predict_blend_bundle(bundle, df, device=None):
    cat_preds = predict_catboost(bundle["catboost_model"], df)
    mlp_preds = predict_bundle(bundle["mlp_bundle"], df, device=device)
    alpha = bundle["alpha"]
    return alpha * cat_preds + (1 - alpha) * mlp_preds
