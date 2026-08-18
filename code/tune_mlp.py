# code/tune_mlp.py
"""MLP 하이퍼파라미터 Optuna 재탐색.

`code/mlp_model.py`의 `LR`/`WEIGHT_DECAY`/`DROPOUT`/`BATCH_SIZE`는 §3.1에서 스윕된 값인데,
그 스윕은 수치형 피처를 표준화만 해서 그대로 concat하던 구조(quantile embedding 도입
이전, §13/§15 이전)에서 나온 결론("lr/wd 조정 무의미")이었다. 이후 수치형 처리 방식을
PLE(quantile embedding, n_bins=24)로 완전히 바꿨는데 하이퍼파라미터는 한 번도 재검증되지
않았다 — `code/tune.py`가 CatBoost를 F1 필터 도입 후 재탐색해 +32.31을 얻었던 것(§21)과
같은 상황. 아키텍처(HIDDEN1/HIDDEN2/QUANTILE_D/n_bins)는 건드리지 않고 옵티마이저·정규화
계열 하이퍼파라미터(lr, weight_decay, dropout, batch_size)만 재탐색한다.

트라이얼마다 MLP를 처음부터 학습해야 해서 CatBoost보다 훨씬 비싸다 — 3-seed 미니
앙상블(`SCREEN_SEEDS`)로 스크리닝하고, `TUNE_TRIALS` 기본값도 20으로 낮췄다. 유의미한
개선이 보이면 최적 하이퍼파라미터로 7-seed 재검증 후 `code/mlp_model.py`에 반영한다.

`DROPOUT`은 `code/mlp_model.py::TabularMLP.__init__`이 모듈 전역을 직접 참조하므로(생성자
인자가 아님), 트라이얼마다 `code.mlp_model.DROPOUT`을 임시로 덮어써서 반영한다 — 순차 실행
(단일 프로세스, 동시성 없음)이라 안전하다.

사용법:
  python -m code.tune_mlp
  TUNE_TRIALS=10 python -m code.tune_mlp
"""
import json
import os

import numpy as np
import optuna

import code.experiment_residual_correction_9_10 as split_mod
import code.mlp_model as mlp_model
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors,
    train_ensemble,
)
from code.experiment_residual_correction_9_10 import build_split
from code.experiment_mlp_subfeature import SCREEN_SEEDS

N_TRIALS = int(os.environ.get("TUNE_TRIALS", 20))
OUT_PATH = "./open/temp/best_mlp_hparams_pitchmix_only.json"


def build_data():
    """pitchmix-only 후보 피처/분할 구성(cutoff=7, F1 필터, tier A 제외, coarse
    pitchmix->CatBoost만, `code/experiment_pitchmix_only_reverify.py`와 동일 —
    `TIER_FEED`를 `{}`로 임시 monkeypatch)으로 MLP 튜닝용 텐서를 만든다. 트라이얼 간
    재사용 — 전처리기 fit은 1회만 수행.

    tier A를 빼기로 한 결정(EXPERIMENTS.md §43) 이후 재탐색 — 이전 재탐색(§41/§42)은
    tier A가 포함된 피처셋 기준이었고 레짐 간 부호가 반전해 기각됐다. MLP는 tier A
    유무에 따라 `num_cols`(mlp_num_cols)가 실제로 달라지므로(CatBoost와 달리) 새
    피처셋으로 다시 탐색해야 한다."""
    orig_tier_feed = split_mod.TIER_FEED
    split_mod.TIER_FEED = {}
    try:
        train_split, val_split, features, cat_features, num_cols = build_split(2024, cutoff7=True)
    finally:
        split_mod.TIER_FEED = orig_tier_feed

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, "control_success")
    X_val_cat, X_val_num, y_val = to_tensors(val_proc, CAT_COLS, num_cols, "control_success")

    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    return X_tr_cat, X_tr_num, y_tr, X_val_cat, X_val_num, y_val, cat_dims, embed_dims, bin_edges, len(num_cols)


def make_objective(data, device):
    X_tr_cat, X_tr_num, y_tr, X_val_cat, X_val_num, y_val, cat_dims, embed_dims, bin_edges, n_num = data
    y_val_np = y_val.numpy()

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192])

        mlp_model.DROPOUT = dropout
        members = train_ensemble(
            X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
            X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
            seeds=SCREEN_SEEDS, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            device=device, verbose=False,
        )
        preds = predict_ensemble(
            members, cat_dims, n_num, embed_dims, X_val_cat, X_val_num,
            bin_edges=bin_edges, device=device,
        )
        score = compute_bss(preds, y_val_np)[2]
        return score
    return objective


def main():
    device = get_device()
    print(f"[Device] {device}")
    print("[Tune] 데이터 로드 및 전처리 (1회만 수행, 트라이얼 간 재사용)...")
    data = build_data()
    print(f"[Tune] train: {data[0].shape[0]}행 | val: {data[3].shape[0]}행 | seeds={SCREEN_SEEDS}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

    def log_callback(study, trial):
        print(f"[Trial {trial.number:03d}] Score={trial.value:.2f} params={trial.params}")

    study.optimize(make_objective(data, device), n_trials=N_TRIALS, callbacks=[log_callback])

    print("\n" + "=" * 60)
    print(f"[Tune 완료] Best Score(3-seed): {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")
    print("=" * 60)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "best_score_3seed": study.best_value,
            "best_params": study.best_params,
        }, f, indent=2, ensure_ascii=False)
    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
