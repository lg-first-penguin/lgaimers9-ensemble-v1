# code/randomforest_model.py
"""RandomForest 학습/추론 공용 유틸리티 (3-way 스태킹 실험용, `code/experiment_3way_stack.py`).

sklearn RandomForestClassifier는 네이티브 범주형 지원이 없으므로, CatBoost/LightGBM/
XGBoost와 동일하게 취급하는 두 컬럼(`game_type`, `base_state`)을 고정된 값 집합
기준 정수로 직접 인코딩합니다 (데이터 설명서상 `base_state`는 8개, `game_type`은
2개 값으로 고정돼 있어 fit 가능한 OrdinalEncoder 없이도 안전합니다).

`early_stopping_rounds` 개념이 없는 모델이라, `code/catboost_model.py`/
`code/lightgbm_model.py`와 같은 함수 시그니처((model, best_iteration) 반환)를
맞추기 위해 `warm_start`로 트리를 STEP개씩 늘려가며 검증 Brier가 더 이상
개선되지 않으면(PATIENCE회 연속) 멈추는 방식으로 흉내냅니다.
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

GAME_TYPE_MAP = {"F": 0, "R": 1}
BASE_STATE_MAP = {"___": 0, "1__": 1, "_2_": 2, "__3": 3, "12_": 4, "1_3": 5, "_23": 6, "123": 7}

RANDOMFOREST_PARAMS = dict(
    max_depth=14,
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=42,
)

STEP = 50
MAX_TREES = 800
PATIENCE = 3
DEFAULT_FULL_RETRAIN_TREES = 400


def _encode_categoricals(X):
    X = X.copy()
    X["game_type"] = X["game_type"].map(GAME_TYPE_MAP)
    X["base_state"] = X["base_state"].map(BASE_STATE_MAP)
    return X


def train_randomforest(X_train, y_train, X_val=None, y_val=None, n_estimators=None, verbose=False):
    from code.mlp_model import compute_bss

    has_val = X_val is not None and y_val is not None and len(y_val) > 0
    X_train_enc = _encode_categoricals(X_train)

    if not has_val:
        params = dict(RANDOMFOREST_PARAMS)
        params["n_estimators"] = n_estimators or DEFAULT_FULL_RETRAIN_TREES
        model = RandomForestClassifier(**params)
        model.fit(X_train_enc, y_train)
        return model, params["n_estimators"]

    X_val_enc = _encode_categoricals(X_val)
    params = dict(RANDOMFOREST_PARAMS)
    params["warm_start"] = True
    model = RandomForestClassifier(n_estimators=STEP, **params)

    best_brier = float("inf")
    best_n = STEP
    no_improve = 0

    n_trees = STEP
    while n_trees <= MAX_TREES:
        model.n_estimators = n_trees
        model.fit(X_train_enc, y_train)
        val_preds = model.predict_proba(X_val_enc)[:, 1]
        brier, bss, score = compute_bss(val_preds, y_val)
        if verbose:
            print(f"[RandomForest] n_estimators={n_trees} | Val Brier: {brier:.6f} | Val Score: {score:.2f}")
        if brier < best_brier - 1e-9:
            best_brier = brier
            best_n = n_trees
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
        n_trees += STEP

    final_params = dict(RANDOMFOREST_PARAMS)
    final_params["n_estimators"] = best_n
    final_model = RandomForestClassifier(**final_params)
    final_model.fit(X_train_enc, y_train)
    return final_model, best_n


def predict_randomforest(model, X):
    X_enc = _encode_categoricals(X)
    return model.predict_proba(X_enc)[:, 1]


# RandomForestClassifier(위)는 노드 분할 기준이 Gini(클래스 순도)라 평가지표인 Brier
# score(확률 보정)와 어긋나 있음 — 단독 성능이 유독 낮았던 원인 중 하나였음. Brier
# score(`(p-y)²`의 평균)는 이진 타겟에 대한 MSE와 수학적으로 동일하므로,
# RandomForestRegressor를 0/1 타겟에 그대로 학습시키면 각 노드 분할이 기본 criterion인
# squared_error(MSE)를 직접 최소화해 Gini보다 Brier에 훨씬 가까운 목적함수가 된다.
# 실측 검증(EXPERIMENTS.md §13.2): 단독 Val Score 521.88(Classifier, leaf=50) ->
# 667.94(Regressor, leaf=500, +28%). 다만 이렇게 개선해도 3-way 스태킹 채택 기준은
# 넘지 못해(rolling-origin 평균 -2.75~-2.93) 프로덕션에는 미반영, 참고용으로만 남김.
REGRESSOR_PARAMS = dict(
    max_depth=14,
    min_samples_leaf=500,
    n_jobs=-1,
    random_state=42,
)


def train_randomforest_regressor(X_train, y_train, n_estimators=150):
    import numpy as np

    X_train_enc = _encode_categoricals(X_train)
    model = RandomForestRegressor(n_estimators=n_estimators, **REGRESSOR_PARAMS)
    model.fit(X_train_enc, y_train.astype(np.float64))
    return model


def predict_randomforest_regressor(model, X):
    import numpy as np

    X_enc = _encode_categoricals(X)
    return np.clip(model.predict(X_enc), 0, 1)
