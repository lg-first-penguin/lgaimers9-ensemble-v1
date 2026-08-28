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

# 2026-08-24: 5-seed CatBoost 배깅(사용자 판단, 20-seed MLP와 함께 실전 검증).
# cutoff7 단일홀드아웃에서 소폭 양수(+9.87 solo, 블렌드 단에서는 +1.29~+7.98)였고
# 예전 rolling-origin(2/3 fold, +3.43)으로는 노이즈 취급됐던 전례가 있어 확정
# 이득이라 보기는 어렵지만, MLP 20seed와 묶어 실전 리더보드로 확인해보기로 함.
CATBOOST_SEED_POOL = [42, 123, 7, 2024, 99]


def train_catboost(X_train, y_train, X_val=None, y_val=None, iterations=MAX_ITERATIONS, verbose=False, random_seed=None, params=None, cat_features=None):
    """검증 세트가 주어지면 BrierScore 기준 early stopping을 수행하고
    (model, best_iteration)을 반환합니다. 없으면(전체 데이터 재학습) 주어진
    iterations만큼 고정 학습합니다. random_seed를 넘기면 CATBOOST_PARAMS의
    기본 seed(42)를 덮어씁니다(5-seed 배깅용). `params`를 넘기면 모듈 기본
    CATBOOST_PARAMS 대신 그 dict를 베이스로 씁니다(2026-08-27, 유담님 파이프라인
    이식 -- 서로 다른 하이퍼파라미터 세트를 같은 학습 루프로 재사용하기 위함).
    `cat_features`를 넘기면 모듈 기본 CAT_FEATURES 대신 그 목록을 씁니다.
    """
    has_val = X_val is not None and y_val is not None and len(y_val) > 0
    cat_features = cat_features if cat_features is not None else CAT_FEATURES
    params = dict(params) if params is not None else dict(CATBOOST_PARAMS)
    params.setdefault("random_seed", CATBOOST_PARAMS["random_seed"])
    params["iterations"] = iterations
    params["verbose"] = 100 if verbose else False
    if random_seed is not None:
        params["random_seed"] = random_seed
    if has_val:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    model = CatBoostClassifier(**params)
    if has_val:
        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
        val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        best_iteration = int(model.get_best_iteration())
    else:
        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
        model.fit(train_pool)
        best_iteration = iterations

    return model, best_iteration


def train_catboost_ensemble(X_train, y_train, X_val=None, y_val=None, seeds=CATBOOST_SEED_POOL,
                             iterations=MAX_ITERATIONS, per_seed_iterations=None, verbose=False,
                             params=None, cat_features=None):
    """`seeds` 각각에 대해 `train_catboost`를 호출한다. `per_seed_iterations`(seed와
    같은 길이의 리스트)가 주어지면 그 값을(전체 재학습용, early stopping 없이 고정
    iteration) 시드별로 각각 쓰고, 아니면 전부 `iterations`(검증셋 있으면 early
    stopping) 공통값을 쓴다. [(model, best_iteration), ...] 리스트를 반환한다.
    `params`/`cat_features`는 `train_catboost`로 그대로 전달된다(2026-08-27, 유담님
    하이퍼파라미터 이식용)."""
    results = []
    for i, seed in enumerate(seeds):
        seed_iterations = per_seed_iterations[i] if per_seed_iterations is not None else iterations
        model, best_iteration = train_catboost(
            X_train, y_train, X_val, y_val,
            iterations=seed_iterations, verbose=verbose, random_seed=seed,
            params=params, cat_features=cat_features,
        )
        results.append((model, best_iteration))
    return results


def predict_catboost(model, X):
    return model.predict_proba(X)[:, 1]


def predict_catboost_ensemble(models, X):
    import numpy as np
    return np.mean([predict_catboost(m, X) for m in models], axis=0)
