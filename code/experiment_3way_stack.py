# code/experiment_3way_stack.py
"""3-way 스태킹 실험: 기존 CatBoost + MLP 앙상블 2-way 스태킹에 세 번째 모델 계열
(LightGBM/XGBoost/RandomForest)을 추가하면 이득이 있는지 검증합니다.

PROJECT_HISTORY.md "다음으로 시도해볼 만한 방향"의 "세 번째 모델 계열 추가"
항목에 해당하는 실험이며, `code/train_x30.py` 등과 동일하게 `dopip.py` 메인
파이프라인에는 아직 반영되지 않은 실험 단계 스크립트입니다. 유의미한 이득이
확인되면 그때 `code/blend_model.py`/`code/train.py`/`dopip.py`/`submit/script.py`에
정식으로 반영합니다.

`open/reference/best_model.pkl`이 이미 학습이 끝난 CatBoost+MLP 블렌드 번들이므로,
base 단계는 이를 재학습하지 않고 그대로 불러와 검증셋 추론만 수행합니다 — 무거운
학습(CatBoost + 7-seed MLP)을 반복하지 않기 위한 지름길입니다.

EXPERIMENTS.md §16에서 트랙맨 매칭의 'season' 등호 문제(실제 배포는 항상 trackman_history.csv가
못 보는 미래 시즌이라 매칭이 구조적으로 항상 실패 -> submit/script.py에 season fallback 추가)가
밝혀진 뒤, 이 스크립트가 지금까지 써온 val 피처(트랙맨이 val 시즌을 실제로 볼 수 있는 "as-is"
조건)는 실제 배포 조건과 다르다는 게 드러났다. `--val-mode`로 val 피처 생성 방식을 선택할 수
있다 (`code/experiment_trackman_season_fallback.py::merge_trackman` 재사용):
  - fallback (기본값): 실제 배포와 동일 — 트랙맨에서 val 시즌을 제거하고 season fallback 매칭
  - broken            : fallback 없이 val 시즌만 제거 (수정 전 상태, 참고/대조용)
  - as-is             : 기존 방식 그대로 (트랙맨이 val 시즌까지 실제로 보임, 실제 배포와 다른 조건)
train_split은 세 모드 모두 동일하다 — 학습 행(season<2024)은 season 등호로 항상 자기 시즌만
매칭하므로 val 시즌 존재 여부와 무관하다.

사용법:
  python -m code.experiment_3way_stack --step base                         # 캐시 생성 (val-mode=fallback 기본)
  python -m code.experiment_3way_stack --step base --val-mode as-is        # 예전 방식 재현(대조용)
  python -m code.experiment_3way_stack --step lightgbm        # 캐시 재사용, LightGBM 검증
  python -m code.experiment_3way_stack --step xgboost
  python -m code.experiment_3way_stack --step randomforest
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from code.train import add_engineered_features
from code.mlp_model import CAT_COLS, predict_bundle, compute_bss, get_device
from code.catboost_model import predict_catboost

CACHE_DIR = "./open/temp/experiment_stack3"
REFERENCE_BUNDLE = "./open/reference/best_model.pkl"
DATA_DIR = "./open/data"
TARGET_COL = "control_success"


def fit_meta_model_n(preds_list, y_val):
    """N개의 예측 배열([cat_pred, mlp_pred, third_pred, ...])을 입력으로 하는
    로지스틱 회귀 메타모델. (weights, intercept, score, brier)를 반환합니다.
    `code/blend_model.py::fit_meta_model`의 2피처 버전을 N피처로 일반화한 실험용 함수.

    plain LogisticRegression()(C=1.0 고정)은 3번째 모델이 기존 두 모델과 상관 0.9대로
    이미 높을 때 다중공선성 때문에 계수 부호까지 표본에 따라 흔들리는 게 관찰됐다
    (EXPERIMENTS.md §24.5 — ExcelFormer 가중치가 h2023 1-seed에서는 음수, 7-seed에서는
    양수). LogisticRegressionCV로 정규화 강도(C)를 내부 5-fold CV가 직접 고르게 해
    이 불안정성을 줄인다."""
    from sklearn.linear_model import LogisticRegressionCV

    X = np.column_stack(preds_list)
    model = LogisticRegressionCV(cv=5, Cs=10, max_iter=1000)
    model.fit(X, y_val)
    weights = [float(c) for c in model.coef_[0]]
    intercept = float(model.intercept_[0])

    z = sum(w * p for w, p in zip(weights, preds_list)) + intercept
    blend = 1.0 / (1.0 + np.exp(-z))
    brier, bss, score = compute_bss(blend, y_val)
    return weights, intercept, score, brier


def build_features(val_mode="fallback"):
    """train_split은 val_mode와 무관하게 항상 기존 방식(트랙맨 season 등호, 전체 트랙맨)
    그대로 만든다 — 학습 행(season<2024)은 season 등호로 이미 자기 시즌만 매칭되므로
    val 시즌(2024) 데이터가 트랙맨에 있든 없든 결과가 같다.

    val_split은 val_mode에 따라 다르게 만든다 (EXPERIMENTS.md §16 참고):
      - "fallback": 실제 배포와 동일 조건 — 트랙맨에서 val 시즌(2024)을 제거해 실제 2025
                    배포처럼 "트랙맨이 eval 시즌을 못 보는" 상황을 재현한 뒤, season을
                    뺀 9-key로 재매칭(submit/script.py::merge_trackman_features와 동일).
      - "broken"  : fallback 없이 val 시즌만 제거 — 수정 전 상태 재현(참고/대조용).
      - "as-is"   : 기존 방식 그대로(트랙맨이 val 시즌을 실제로 봄) — 이 스크립트가
                    §16 이전까지 써온 조건이며, 실제 배포와는 다르다.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")

    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)
    combined = tr_final.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_mask = combined["season"] < 2024
    league_success_mean = combined.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(combined.loc[train_mask].reset_index(drop=True), league_success_mean)

    if val_mode == "as-is":
        val_df = combined.loc[~train_mask].reset_index(drop=True)
    else:
        from code.experiment_trackman_season_fallback import merge_trackman
        val_raw = df.loc[df["season"] == 2024].reset_index(drop=True)
        df_trm_blind = df_trm[df_trm["season"] != 2024].reset_index(drop=True)
        val_df, _ = merge_trackman(val_raw, df_trm_blind, match_cols, drop_season_fallback=(val_mode == "fallback"))
    val_df = add_engineered_features(val_df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in train_df.columns if c not in drop_cols]
    missing_in_val = [c for c in features if c not in val_df.columns]
    if missing_in_val:
        raise RuntimeError(f"val_split에 train_split과 다른 컬럼 구성이 나왔습니다 (누락: {missing_in_val}) — "
                            f"트랙맨 데이터의 pitch_type_group 구성이 서브셋 간 달라졌을 수 있습니다.")

    train_split = train_df[features + [TARGET_COL]].reset_index(drop=True)
    val_split = val_df[features + [TARGET_COL]].reset_index(drop=True)
    return train_split, val_split, features


def step_base(val_mode="fallback"):
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()

    print(f"[base] 트랙맨 피처 결합(val_mode={val_mode}) + train/val(season==2024) 분할 재구성 중...")
    train_split, val_split, features = build_features(val_mode=val_mode)
    print(f"[base] 훈련 {len(train_split)}행 | 검증 {len(val_split)}행 (경과 {time.time()-t0:.1f}s)")

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
    from code.blend_model import predict_meta
    blend_preds = predict_meta(meta["w_cat"], meta["w_mlp"], meta["intercept"], cat_val_preds, mlp_val_preds)
    blend_score = compute_bss(blend_preds, y_val_raw)[2]

    print(f"[base] CatBoost={cat_score:.2f} | MLP={mlp_score:.2f} | 2-way 스태킹(기존 meta_model 그대로)={blend_score:.2f}")

    train_split.to_pickle(os.path.join(CACHE_DIR, "train_split.pkl"))
    val_split.to_pickle(os.path.join(CACHE_DIR, "val_split.pkl"))
    np.savez(
        os.path.join(CACHE_DIR, "base_preds.npz"),
        cat_val_preds=cat_val_preds, mlp_val_preds=mlp_val_preds, y_val=y_val_raw,
    )
    with open(os.path.join(CACHE_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"features": features, "baseline_2way_score": blend_score,
                     "catboost_score": cat_score, "mlp_score": mlp_score}, f)
    print(f"[base] 캐시 저장 완료: {CACHE_DIR} (총 경과 {time.time()-t0:.1f}s)")


def load_cache():
    train_split = pd.read_pickle(os.path.join(CACHE_DIR, "train_split.pkl"))
    val_split = pd.read_pickle(os.path.join(CACHE_DIR, "val_split.pkl"))
    npz = np.load(os.path.join(CACHE_DIR, "base_preds.npz"))
    with open(os.path.join(CACHE_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return train_split, val_split, npz["cat_val_preds"], npz["mlp_val_preds"], npz["y_val"], meta


def step_third_model(name):
    if not os.path.exists(os.path.join(CACHE_DIR, "meta.pkl")):
        raise RuntimeError("먼저 `python -m code.experiment_3way_stack --step base`를 실행해 캐시를 만들어야 합니다.")

    train_split, val_split, cat_val_preds, mlp_val_preds, y_val, meta = load_cache()
    features = meta["features"]
    X_train_raw, y_train_raw = train_split[features], train_split[TARGET_COL].values
    X_val_raw = val_split[features]

    t0 = time.time()
    if name == "lightgbm":
        from code.lightgbm_model import train_lightgbm, predict_lightgbm
        model, best_iter = train_lightgbm(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=True)
        third_preds = predict_lightgbm(model, X_val_raw)
    elif name == "xgboost":
        from code.xgboost_model import train_xgboost, predict_xgboost
        model, best_iter = train_xgboost(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=True)
        third_preds = predict_xgboost(model, X_val_raw)
    elif name == "xgboost_tuned":
        from code.xgboost_model import train_xgboost_tuned, predict_xgboost
        model, best_iter = train_xgboost_tuned(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=True)
        third_preds = predict_xgboost(model, X_val_raw)
    elif name == "randomforest":
        from code.randomforest_model import train_randomforest, predict_randomforest
        model, best_iter = train_randomforest(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=True)
        third_preds = predict_randomforest(model, X_val_raw)
    else:
        raise ValueError(f"알 수 없는 모델: {name}")

    third_score = compute_bss(third_preds, y_val)[2]
    print(f"[{name}] 단독 학습 완료 (best_iteration={best_iter}, 경과 {time.time()-t0:.1f}s) | 단독 Val Score: {third_score:.2f}")

    corr_cat = float(np.corrcoef(third_preds, cat_val_preds)[0, 1])
    corr_mlp = float(np.corrcoef(third_preds, mlp_val_preds)[0, 1])
    print(f"[{name}] 예측 상관관계 — vs CatBoost: {corr_cat:.4f} | vs MLP: {corr_mlp:.4f}")

    weights, intercept, score_3way, brier_3way = fit_meta_model_n([cat_val_preds, mlp_val_preds, third_preds], y_val)
    delta = score_3way - meta["baseline_2way_score"]
    print(f"[{name}] 3-way 스태킹(CatBoost+MLP+{name}) Val Score: {score_3way:.2f} "
          f"(가중치 w_cat={weights[0]:.3f} w_mlp={weights[1]:.3f} w_{name}={weights[2]:.3f} intercept={intercept:.3f})")
    print(f"[{name}] 기존 2-way 스태킹({meta['baseline_2way_score']:.2f}) 대비 delta: {delta:+.2f}")


FOLD_BOUNDARIES = [(5, 6), (7, 8), (9, 10)]  # EXPERIMENTS.md §9.1과 동일한 월 버킷(확장 윈도우)


def step_foldcheck(name):
    """EXPERIMENTS.md §9.1과 동일한 rolling-origin 3-fold(월 버킷 확장 윈도우)로
    2-way(CatBoost+MLP) vs 3-way(CatBoost+MLP+<name>) 스태킹을 비교합니다.
    CatBoost/MLP/<name> 자체는 2019~2023으로 이미 학습된 상태(재학습 없음) —
    각 fold에서는 메타모델(로지스틱 회귀)만 해당 fold의 meta_train 구간에서 재학습합니다."""
    if not os.path.exists(os.path.join(CACHE_DIR, "meta.pkl")):
        raise RuntimeError("먼저 `python -m code.experiment_3way_stack --step base`를 실행해 캐시를 만들어야 합니다.")

    train_split, val_split, cat_val_preds, mlp_val_preds, y_val, meta = load_cache()
    features = meta["features"]
    X_train_raw, y_train_raw = train_split[features], train_split[TARGET_COL].values
    X_val_raw = val_split[features]

    print(f"[{name}-foldcheck] season==2024 전체에 대해 {name} 1회 학습 (fold마다 재학습 X)...")
    if name == "lightgbm":
        from code.lightgbm_model import train_lightgbm, predict_lightgbm
        model, _ = train_lightgbm(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=False)
        third_preds = predict_lightgbm(model, X_val_raw)
    elif name == "xgboost":
        from code.xgboost_model import train_xgboost, predict_xgboost
        model, _ = train_xgboost(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=False)
        third_preds = predict_xgboost(model, X_val_raw)
    elif name == "xgboost_tuned":
        from code.xgboost_model import train_xgboost_tuned, predict_xgboost
        model, _ = train_xgboost_tuned(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=False)
        third_preds = predict_xgboost(model, X_val_raw)
    elif name == "randomforest":
        from code.randomforest_model import train_randomforest, predict_randomforest
        model, _ = train_randomforest(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=False)
        third_preds = predict_randomforest(model, X_val_raw)
    else:
        raise ValueError(f"알 수 없는 모델: {name}")

    game_month = val_split["game_month"].values
    deltas = []
    print(f"\n{'val 구간':<10} {'n_train':>8} {'n_val':>8} {'2-way':>10} {name+' 3-way':>14} {'delta':>8}")
    for lo, hi in FOLD_BOUNDARIES:
        val_mask = (game_month == lo) | (game_month == hi)
        train_mask = game_month < lo

        cat_tr, mlp_tr, third_tr, y_tr = cat_val_preds[train_mask], mlp_val_preds[train_mask], third_preds[train_mask], y_val[train_mask]
        cat_va, mlp_va, third_va, y_va = cat_val_preds[val_mask], mlp_val_preds[val_mask], third_preds[val_mask], y_val[val_mask]

        from code.blend_model import fit_meta_model, predict_meta
        w_cat, w_mlp, intercept, _, _ = fit_meta_model(cat_tr, mlp_tr, y_tr)
        two_way_preds = predict_meta(w_cat, w_mlp, intercept, cat_va, mlp_va)
        two_way_score = compute_bss(two_way_preds, y_va)[2]

        weights, intercept3, _, _ = fit_meta_model_n([cat_tr, mlp_tr, third_tr], y_tr)
        z = sum(w * p for w, p in zip(weights, [cat_va, mlp_va, third_va])) + intercept3
        three_way_preds = 1.0 / (1.0 + np.exp(-z))
        three_way_score = compute_bss(three_way_preds, y_va)[2]

        delta = three_way_score - two_way_score
        deltas.append(delta)
        print(f"{lo}~{hi}월    {train_mask.sum():>8} {val_mask.sum():>8} {two_way_score:>10.2f} {three_way_score:>14.2f} {delta:>+8.2f}")

    wins = sum(1 for d in deltas if d > 0)
    print(f"\n[{name}-foldcheck] {wins}/{len(deltas)} fold 승리, 평균 delta: {np.mean(deltas):+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["base", "lightgbm", "xgboost", "xgboost_tuned", "randomforest"])
    parser.add_argument("--foldcheck", action="store_true", help="rolling-origin 3-fold 재검증도 함께 수행")
    parser.add_argument("--val-mode", default="fallback", choices=["fallback", "broken", "as-is"],
                         help="--step base 전용: val 피처를 만들 때 트랙맨 매칭 조건 (EXPERIMENTS.md §16 참고)")
    args = parser.parse_args()

    if args.step == "base":
        step_base(val_mode=args.val_mode)
    elif args.foldcheck:
        step_foldcheck(args.step)
    else:
        step_third_model(args.step)


if __name__ == "__main__":
    main()
