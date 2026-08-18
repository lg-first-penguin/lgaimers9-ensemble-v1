# code/xgboost_model.py
"""XGBoost 학습/추론 공용 유틸리티 (3-way 스태킹 실험용, `code/experiment_3way_stack.py`).

`code/catboost_model.py`/`code/lightgbm_model.py`와 동일한 관례. 카테고리 처리는
XGBoost의 `enable_categorical=True` + pandas `category` dtype으로 처리하며,
대상 컬럼도 동일하게 `game_type`/`base_state` 두 개뿐입니다.
"""
import xgboost as xgb
from xgboost import XGBClassifier

CAT_FEATURES = ["game_type", "base_state"]

XGBOOST_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    random_state=42,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=None,
    verbosity=0,
)

EARLY_STOPPING_ROUNDS = 50


def _with_categories(X):
    X = X.copy()
    X[CAT_FEATURES] = X[CAT_FEATURES].astype("category")
    return X


def train_xgboost(X_train, y_train, X_val=None, y_val=None, n_estimators=None, verbose=False):
    """검증 세트가 주어지면 logloss 기준 early stopping을 수행하고
    (model, best_iteration)을 반환합니다. 없으면(전체 데이터 재학습) 주어진
    n_estimators만큼 고정 학습합니다.
    """
    has_val = X_val is not None and y_val is not None and len(y_val) > 0
    params = dict(XGBOOST_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if has_val:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    else:
        params.pop("early_stopping_rounds", None)

    X_train = _with_categories(X_train)
    model = XGBClassifier(**params)
    if has_val:
        X_val = _with_categories(X_val)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100 if verbose else False)
        best_iteration = int(model.best_iteration)
    else:
        model.fit(X_train, y_train, verbose=100 if verbose else False)
        best_iteration = params["n_estimators"]

    return model, best_iteration


def predict_xgboost(model, X):
    X = _with_categories(X)
    return model.predict_proba(X)[:, 1]


# 팀원이 별도 트랙(XGBoost 단독 트랙, 이 저장소의 트랙맨 조인+asof 파생피처를 그대로 이식)에서
# Optuna 2라운드(총 55 trials, 넓은 탐색 30 + 좁힌 탐색 25)로 튜닝한 값. 로컬 season==2024
# 홀드아웃 752.70 (BASE_PARAMS 값 위주 정보만 전달받아 이 저장소 파이프라인 위에서 재현 —
# 트랙맨 조인/파생피처 코드 자체는 이 저장소의 `code/train.py`에서 그대로 이식받았다고 함).
# 조기종료 기준도 logloss가 아니라 대회 채점 지표(Brier)에 직접 맞춘 커스텀 eval_metric 사용.
TUNED_XGBOOST_PARAMS = dict(
    max_depth=6,
    learning_rate=0.014399911854309236,
    subsample=0.6367420040370018,
    colsample_bytree=0.9920379472108309,
    reg_lambda=0.06047908057801884,
    reg_alpha=4.368166043300848,
    min_child_weight=12,
    gamma=0.008969955448879116,
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)

TUNED_MAX_ITERATIONS = 3000
TUNED_EARLY_STOPPING_ROUNDS = 50


def brier_eval(y_true, y_pred):
    return float(((y_pred - y_true) ** 2).mean())


def train_xgboost_tuned(X_train, y_train, X_val=None, y_val=None, n_estimators=None, verbose=False):
    """TUNED_XGBOOST_PARAMS + 커스텀 Brier eval_metric으로 학습합니다 (팀원 트랙 재현)."""
    has_val = X_val is not None and y_val is not None and len(y_val) > 0
    params = dict(TUNED_XGBOOST_PARAMS)
    params["n_estimators"] = n_estimators if n_estimators is not None else TUNED_MAX_ITERATIONS
    if has_val:
        params["eval_metric"] = brier_eval
        params["early_stopping_rounds"] = TUNED_EARLY_STOPPING_ROUNDS

    X_train = _with_categories(X_train)
    model = XGBClassifier(**params)
    if has_val:
        X_val = _with_categories(X_val)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100 if verbose else False)
        best_iteration = int(model.best_iteration)
    else:
        model.fit(X_train, y_train, verbose=100 if verbose else False)
        best_iteration = params["n_estimators"]

    return model, best_iteration
