# code/catboost_model.py
"""CatBoost 학습/추론 공용 유틸리티 (Tabular MLP와의 블렌딩용).

하이퍼파라미터는 EXPERIMENTS.md 2장의 Optuna 튜닝 결과(구 reference
`open/former_model/former_best_model_v3.pkl`, 실측 BSS 환산 818.54)를
`model.get_params()`로 그대로 추출한 값입니다 — 재현성을 위해 하드코딩.

CatBoostClassifier는 프로젝트 전용 클래스가 아니라 `catboost` 라이브러리가 제공하는
클래스이므로, `submit/requirements.txt`에 `catboost`만 명시되어 있으면 `code/`
패키지 없이도 대회 서버에서 정상적으로 unpickle됩니다 (Tabular MLP 번들처럼 dict로
감쌀 필요가 없습니다).
"""
from catboost import CatBoostClassifier, Pool

CAT_FEATURES = ["game_type", "base_state"]

CATBOOST_PARAMS = dict(
    learning_rate=0.05040411253232039,
    depth=7,
    l2_leaf_reg=3.14659036827521,
    loss_function="Logloss",
    border_count=106,
    random_seed=42,
    random_strength=3.715568024268865,
    eval_metric="BrierScore",
    bagging_temperature=0.4609270457436248,
    bootstrap_type="Bayesian",
    verbose=False,
)

MAX_ITERATIONS = 1500
EARLY_STOPPING_ROUNDS = 50
DEFAULT_FULL_RETRAIN_ITERATIONS = 500


def train_catboost(X_train, y_train, X_val=None, y_val=None, iterations=MAX_ITERATIONS, verbose=False):
    """검증 세트가 주어지면 BrierScore 기준 early stopping을 수행하고
    (model, best_iteration)을 반환합니다. 없으면(전체 데이터 재학습) 주어진
    iterations만큼 고정 학습합니다.
    """
    has_val = X_val is not None and y_val is not None and len(y_val) > 0
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = iterations
    params["verbose"] = 100 if verbose else False
    if has_val:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    model = CatBoostClassifier(**params)
    if has_val:
        train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
        val_pool = Pool(data=X_val, label=y_val, cat_features=CAT_FEATURES)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        best_iteration = int(model.get_best_iteration())
    else:
        train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
        model.fit(train_pool)
        best_iteration = iterations

    return model, best_iteration


def predict_catboost(model, X):
    return model.predict_proba(X)[:, 1]
