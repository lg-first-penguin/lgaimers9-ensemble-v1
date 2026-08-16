# code/experiment_meta_nonlinear.py
"""2-way(CatBoost+MLP) 스태킹의 메타모델을 비선형으로 바꾸면 이득이 있는지 검증하는 실험.

지금까지 시도된 "N번째 모델을 늘리는" 방향(EXPERIMENTS.md §12/§12.1/§12.2, §22~24)은
모두 rolling-origin fold check까지 거쳐 기각됐다 — 3번째 모델을 추가하면 기존 모델과의
상관관계 때문에 메타모델이 불안정해지는 패턴이 반복됐다. 이 스크립트는 "모델 개수를
늘리는" 대신 "기존 2개 모델(CatBoost, MLP)의 예측을 결합하는 방식 자체"를 로지스틱
회귀(선형)에서 비선형으로 바꾸는, 아직 제대로 검증된 적 없는 방향을 테스트한다.

§9에서 `cat_pred*mlp_pred` 교호작용항을 딱 한 번 시도한 적이 있지만 (a) 단일 9~10월
슬라이스만 봤고 (b) 상황 피처 9개와 같이 넣어 섞였고 (c) 트랙맨 드롭/F1 필터/quantile
embedding 이전 피처셋이었다 — 지금 조건에서 순수하게 [cat_pred, mlp_pred] 2개 입력만
유지한 채 재검증한 적은 없다. §23에서 확인된 "메타모델을 정규화하면 다중공선성에 덜
흔들린다"는 교훈을 이어받아, 여기서도 정규화된 비선형 후보들만 시험한다.

CatBoost/MLP 자체는 재학습하지 않고(`open/reference/best_model.pkl`을 그대로 로드해
추론만 수행) 메타모델만 후보별로 바꿔가며 §9.1/§12.1과 동일한 rolling-origin 3-fold
(월 버킷 확장 윈도우: train=~4월 -> val 5~6월, train=~6월 -> val 7~8월, train=~8월 ->
val 9~10월)로 비교한다. 채택 기준도 §9.1과 동일: 3-fold 전부 승리 + 뚜렷한 평균 delta.

사용법:
  python -m code.experiment_meta_nonlinear --step base       # 캐시 생성 (cat/mlp val 예측 재현)
  python -m code.experiment_meta_nonlinear --step screen      # 전체 후보 rolling-origin foldcheck
  python -m code.experiment_meta_nonlinear --step screen --candidate poly2   # 후보 하나만
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from code.train import add_engineered_features
from code.mlp_model import predict_bundle, compute_bss, get_device
from code.catboost_model import predict_catboost
from code.blend_model import fit_meta_model, predict_meta

CACHE_DIR = "./open/temp/experiment_meta_nonlinear"
REFERENCE_BUNDLE = "./open/reference/best_model.pkl"
DATA_DIR = "./open/data"
TARGET_COL = "control_success"

FOLD_BOUNDARIES = [(5, 6), (7, 8), (9, 10)]  # EXPERIMENTS.md §9.1과 동일한 월 버킷(확장 윈도우)


def step_base():
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()

    print("[base] train.py와 동일한 로직으로 season==2024 검증 split 재구성 중 (트랙맨 없음, F1 필터는 train에만 적용되므로 val엔 무관)...")
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_mask = train_df['season'] < 2024
    val_mask = train_df['season'] == 2024
    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ['row_id', TARGET_COL]
    features = [c for c in train_df.columns if c not in drop_cols]
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    print(f"[base] 검증 데이터 (2024): {len(val_split)}행 (경과 {time.time()-t0:.1f}s)")

    print(f"[base] 기존 레퍼런스 블렌드 번들 로드: {REFERENCE_BUNDLE} (재학습 없이 추론만)")
    with open(REFERENCE_BUNDLE, "rb") as f:
        bundle = pickle.load(f)
    for key in ("catboost_model", "mlp_bundle", "meta_model"):
        if key not in bundle:
            raise RuntimeError(f"{REFERENCE_BUNDLE}가 현재 블렌드 번들 스키마가 아닙니다 ('{key}' 없음) — "
                                f"dopip.py를 한 번 돌려 최신 스키마로 갱신한 뒤 다시 시도하세요.")

    device = get_device()
    X_val_raw, y_val_raw = val_split[features], val_split[TARGET_COL].values

    cat_val_preds = predict_catboost(bundle["catboost_model"], X_val_raw)
    mlp_val_preds = predict_bundle(bundle["mlp_bundle"], X_val_raw, device=device)

    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    mlp_score = compute_bss(mlp_val_preds, y_val_raw)[2]

    meta = bundle["meta_model"]
    blend_preds = predict_meta(meta["w_cat"], meta["w_mlp"], meta["intercept"], cat_val_preds, mlp_val_preds)
    blend_score = compute_bss(blend_preds, y_val_raw)[2]

    print(f"[base] CatBoost={cat_score:.2f} | MLP={mlp_score:.2f} | 2-way 선형 스태킹(기존 meta_model 그대로)={blend_score:.2f}")

    np.savez(
        os.path.join(CACHE_DIR, "base_preds.npz"),
        cat_val_preds=cat_val_preds, mlp_val_preds=mlp_val_preds, y_val=y_val_raw,
        game_month=val_split["game_month"].values,
    )
    with open(os.path.join(CACHE_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"baseline_2way_score": blend_score, "catboost_score": cat_score, "mlp_score": mlp_score}, f)
    print(f"[base] 캐시 저장 완료: {CACHE_DIR} (총 경과 {time.time()-t0:.1f}s)")


def load_cache():
    npz = np.load(os.path.join(CACHE_DIR, "base_preds.npz"))
    with open(os.path.join(CACHE_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return npz["cat_val_preds"], npz["mlp_val_preds"], npz["y_val"], npz["game_month"], meta


# ---------------------------------------------------------------------------
# 메타모델 후보들 — 전부 입력은 [cat_pred, mlp_pred] 2개로 고정 (§9: 상황 피처 추가는 이미 기각됨).
# 각 fit_* 함수는 (predict_fn) 형태로 반환 -> predict_fn(cat_va, mlp_va) -> blended prob array.
# ---------------------------------------------------------------------------

def fit_linear(cat_tr, mlp_tr, y_tr):
    """기존 프로덕션과 동일한 순수 선형 로지스틱 회귀 (베이스라인)."""
    w_cat, w_mlp, intercept, _, _ = fit_meta_model(cat_tr, mlp_tr, y_tr)
    return lambda cat_va, mlp_va: predict_meta(w_cat, w_mlp, intercept, cat_va, mlp_va)


def _poly2_design(cat_p, mlp_p):
    return np.column_stack([cat_p, mlp_p, cat_p ** 2, mlp_p ** 2, cat_p * mlp_p])


def fit_poly2_reg(cat_tr, mlp_tr, y_tr):
    """2차 다항 전개([cat, mlp, cat^2, mlp^2, cat*mlp]) + 정규화 강도를 CV로 고르는
    LogisticRegressionCV. §23의 "정규화가 다중공선성을 완화한다" 교훈을 그대로 적용."""
    from sklearn.linear_model import LogisticRegressionCV
    X_tr = _poly2_design(cat_tr, mlp_tr)
    model = LogisticRegressionCV(cv=5, Cs=10, max_iter=1000)
    model.fit(X_tr, y_tr)

    def predict_fn(cat_va, mlp_va):
        X_va = _poly2_design(cat_va, mlp_va)
        z = model.decision_function(X_va)
        return 1.0 / (1.0 + np.exp(-z))
    return predict_fn


def fit_mlp_meta(cat_tr, mlp_tr, y_tr):
    """[cat, mlp] 2입력을 받는 아주 작은 신경망 메타러너 (은닉 4유닛, 강한 L2, early_stopping).
    2피처짜리 얕은 네트워크라 로지스틱 회귀보다 일반적인 비선형 결합을 표현할 수 있다."""
    from sklearn.neural_network import MLPClassifier
    X_tr = np.column_stack([cat_tr, mlp_tr])
    model = MLPClassifier(
        hidden_layer_sizes=(4,), alpha=1.0, max_iter=2000,
        early_stopping=True, n_iter_no_change=20, random_state=42,
    )
    model.fit(X_tr, y_tr)

    def predict_fn(cat_va, mlp_va):
        X_va = np.column_stack([cat_va, mlp_va])
        return model.predict_proba(X_va)[:, 1]
    return predict_fn


def fit_gbm_meta(cat_tr, mlp_tr, y_tr):
    """[cat, mlp] 2입력짜리 아주 얕은 gradient boosting 메타러너 (max_depth=2, 트리 30개,
    learning_rate 낮게) — 트리 기반이라 로지스틱/다항 확장과는 다른 종류의 비선형을 잡을 수 있음."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X_tr = np.column_stack([cat_tr, mlp_tr])
    model = HistGradientBoostingClassifier(
        max_depth=2, max_iter=30, learning_rate=0.05,
        l2_regularization=1.0, random_state=42,
    )
    model.fit(X_tr, y_tr)

    def predict_fn(cat_va, mlp_va):
        X_va = np.column_stack([cat_va, mlp_va])
        return model.predict_proba(X_va)[:, 1]
    return predict_fn


CANDIDATES = {
    "poly2": fit_poly2_reg,
    "mlp": fit_mlp_meta,
    "gbm": fit_gbm_meta,
}


def foldcheck_one(name, fit_fn, cat_val_preds, mlp_val_preds, y_val, game_month):
    deltas = []
    rows = []
    for lo, hi in FOLD_BOUNDARIES:
        val_mask = (game_month == lo) | (game_month == hi)
        train_mask = game_month < lo

        cat_tr, mlp_tr, y_tr = cat_val_preds[train_mask], mlp_val_preds[train_mask], y_val[train_mask]
        cat_va, mlp_va, y_va = cat_val_preds[val_mask], mlp_val_preds[val_mask], y_val[val_mask]

        linear_predict = fit_linear(cat_tr, mlp_tr, y_tr)
        linear_score = compute_bss(linear_predict(cat_va, mlp_va), y_va)[2]

        candidate_predict = fit_fn(cat_tr, mlp_tr, y_tr)
        candidate_score = compute_bss(candidate_predict(cat_va, mlp_va), y_va)[2]

        delta = candidate_score - linear_score
        deltas.append(delta)
        rows.append((f"{lo}~{hi}월", train_mask.sum(), val_mask.sum(), linear_score, candidate_score, delta))

    wins = sum(1 for d in deltas if d > 0)
    print(f"\n[{name}] rolling-origin 3-fold foldcheck (vs 선형 베이스라인, 각 fold에서 둘 다 재학습)")
    print(f"{'val 구간':<10} {'n_train':>8} {'n_val':>8} {'선형':>10} {name:>14} {'delta':>8}")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:>8} {r[2]:>8} {r[3]:>10.2f} {r[4]:>14.2f} {r[5]:>+8.2f}")
    print(f"[{name}] {wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}")
    return deltas


def step_screen(candidate=None):
    if not os.path.exists(os.path.join(CACHE_DIR, "meta.pkl")):
        raise RuntimeError("먼저 `python -m code.experiment_meta_nonlinear --step base`를 실행해 캐시를 만들어야 합니다.")

    cat_val_preds, mlp_val_preds, y_val, game_month, meta = load_cache()
    print(f"[screen] 캐시 로드 완료 — 기존 2-way 선형 스태킹(전체 season==2024) 기준 점수: {meta['baseline_2way_score']:.2f}")

    names = [candidate] if candidate else list(CANDIDATES.keys())
    summary = {}
    for name in names:
        deltas = foldcheck_one(name, CANDIDATES[name], cat_val_preds, mlp_val_preds, y_val, game_month)
        summary[name] = deltas

    if len(summary) > 1:
        print("\n=== 후보 비교 요약 ===")
        print(f"{'후보':<10} {'fold 승수':>10} {'평균 delta':>12}")
        for name, deltas in summary.items():
            wins = sum(1 for d in deltas if d > 0)
            print(f"{name:<10} {wins:>8}/3 {np.mean(deltas):>+12.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["base", "screen"])
    parser.add_argument("--candidate", default=None, choices=list(CANDIDATES.keys()),
                         help="--step screen 전용: 후보 하나만 실행 (생략 시 전체 실행)")
    args = parser.parse_args()

    if args.step == "base":
        step_base()
    else:
        step_screen(candidate=args.candidate)


if __name__ == "__main__":
    main()
