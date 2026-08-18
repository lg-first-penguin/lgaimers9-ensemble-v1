# code/lightgbm_model.py
"""LightGBM 학습/추론 공용 유틸리티 (3-way 스태킹 실험용, `code/experiment_3way_stack.py`).

`code/catboost_model.py`와 동일한 관례: 검증 세트가 주어지면 BrierScore 기준
early stopping을 수행하고 (model, best_iteration)을 반환합니다. 카테고리 처리도
CatBoost와 동일하게 `game_type`/`base_state` 두 컬럼만 범주형으로 취급합니다
(`top_bottom`은 `process_trackman_features_safe`가 이미 0/1 정수로 매핑, 나머지
`pitcher_hand`/`batter_hand`/`*_team_id`는 원본부터 정수 ID라 특별 취급 불필요 —
CLAUDE.md "CatBoost의 categorical features는 game_type, base_state뿐" 참고).

하이퍼파라미터는 튜닝 없이 합리적인 기본값입니다 — 이 실험의 목적은 "LightGBM이
CatBoost/MLP와 다르게 틀려서 스태킹에 이득을 주는가"를 먼저 확인하는 것이라,
성능이 확인되면 그때 Optuna 등으로 별도 튜닝할 가치가 있는지 판단합니다.
"""
from lightgbm import LGBMClassifier

CAT_FEATURES = ["game_type", "base_state"]

LIGHTGBM_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=7,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    random_state=42,
    objective="binary",
    verbosity=-1,
)

EARLY_STOPPING_ROUNDS = 50


def train_lightgbm(X_train, y_train, X_val=None, y_val=None, n_estimators=None, verbose=False):
    """검증 세트가 주어지면 binary_logloss 기준 early stopping을 수행하고
    (model, best_iteration)을 반환합니다. 없으면(전체 데이터 재학습) 주어진
    n_estimators만큼 고정 학습합니다.
    """
    from lightgbm import early_stopping, log_evaluation

    has_val = X_val is not None and y_val is not None and len(y_val) > 0
    params = dict(LIGHTGBM_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators

    X_train = X_train.copy()
    X_train[CAT_FEATURES] = X_train[CAT_FEATURES].astype("category")

    model = LGBMClassifier(**params)
    if has_val:
        X_val = X_val.copy()
        X_val[CAT_FEATURES] = X_val[CAT_FEATURES].astype("category")
        callbacks = [early_stopping(EARLY_STOPPING_ROUNDS, verbose=verbose)]
        if verbose:
            callbacks.append(log_evaluation(100))
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
            categorical_feature=CAT_FEATURES, callbacks=callbacks,
        )
        best_iteration = int(model.best_iteration_)
    else:
        model.fit(X_train, y_train, categorical_feature=CAT_FEATURES)
        best_iteration = params["n_estimators"]

    return model, best_iteration


def predict_lightgbm(model, X):
    X = X.copy()
    X[CAT_FEATURES] = X[CAT_FEATURES].astype("category")
    return model.predict_proba(X)[:, 1]
