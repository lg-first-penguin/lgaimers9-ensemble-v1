# code/catboost_model.py
"""CatBoost 학습/추론 공용 유틸리티 (Tabular MLP와의 블렌딩용).

하이퍼파라미터는 `code/tune.py`(Optuna, 40 trials)를 트랙맨 제거 + F1 필터 적용된
현재 프로덕션 피처 구성(season<2024 학습 / season==2024 검증)으로 재실행한 결과입니다
— EXPERIMENTS.md §20. CatBoost 단독 기준 이전 하드코딩 값(689.35, 구 데이터 기준
튜닝) 대비 +32.31(721.66)로, F1 필터로 학습 데이터가 바뀐 뒤 재튜닝한 것이 유효했습니다.

CatBoostClassifier는 프로젝트 전용 클래스가 아니라 `catboost` 라이브러리가 제공하는
클래스이므로, `submit/requirements.txt`에 `catboost`만 명시되어 있으면 `code/`
패키지 없이도 대회 서버에서 정상적으로 unpickle됩니다 (Tabular MLP 번들처럼 dict로
감쌀 필요가 없습니다).
"""
from catboost import CatBoostClassifier, Pool

CAT_FEATURES = ["game_type", "base_state"]

CATBOOST_PARAMS = dict(
    learning_rate=0.027468342653742282,
    depth=6,
    l2_leaf_reg=2.36118174482295,
    loss_function="Logloss",
    border_count=32,
    random_seed=42,
    random_strength=9.996029894496754,
    eval_metric="BrierScore",
    bagging_temperature=0.46568175434742154,
    bootstrap_type="Bayesian",
    min_data_in_leaf=89,
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
