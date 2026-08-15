# code/experiment_oof_stacking.py
"""진짜 다중시즌 OOF(out-of-fold) 스태킹 실험.

지금 프로덕션 메타모델(`code/blend_model.py::fit_meta_model`)은 season==2024 단일
홀드아웃(~25만 행)의 예측값에만 피팅된다. 이 세션 초반에 논의된 "메타모델을 여러 fold의
OOF 예측으로 훨씬 크고 다양한 데이터에 피팅하면 더 안정적이지 않을까"라는 가설 — 당시엔
"3번째 모델을 편입하기로 결정된 다음에 투자할 규모"라며 보류했지만, 3번째 모델 방향이
전부 닫힌(§22~24) 뒤 이번엔 지금의 2-way(CatBoost+MLP) 메타모델 자체를 더 크고 다양한
데이터로 검증한다.

각 fold는 `code/experiment_attention.py::build_split(val_season, apply_f1=True)`를 그대로
재사용한다 — train=season<val_season, val=season==val_season이 정확히 원하는
expanding-window 구조와 일치한다:
  Fold A: train 2019-2020 -> predict 2021
  Fold B: train 2019-2021 -> predict 2022
  Fold C: train 2019-2022 -> predict 2023
(2020 예측 fold는 뺐다 — train이 2019 한 시즌뿐이라 프로덕션 규모(5개 시즌)보다 훨씬
작은 데이터로 학습한 모델이라 OOF 예측 품질의 대표성이 떨어질 수 있다.)
season==2024는 fold 구성에 전혀 쓰지 않고, 최종 비교의 유일한 out-of-sample 검증셋으로
남겨둔다 — 기존 프로덕션 컨벤션과 동일.

각 fold에서 CatBoost + MLP(7-seed 앙상블, ENSEMBLE_SEEDS)를 그 fold의 train으로 새로
학습해 val에 대한 OOF 예측을 얻고, 세 fold의 OOF 예측을 모두 이어붙여 훨씬 크고(약
3개 시즌치, ~74만 행) 다양한 메타 학습셋을 만든다. 이걸로 피팅한 메타모델
(w_cat/w_mlp/intercept)을, season<2024로 학습한 "진짜 프로덕션급" CatBoost+MLP의
season==2024 예측에 적용해서, 기존 방식(season==2024 자체에 메타모델을 피팅)과
블렌드 점수를 비교한다.

MLP는 매 fold마다 7-seed 풀 앙상블로 새로 학습한다(3-seed 스크리닝을 거치지 않음 —
이 실험 자체가 "더 엄밀하게 검증하면 뭐가 달라지는가"를 묻는 것이라 축소판으로 먼저
보는 게 실익이 적다고 판단). fold 4개(2021/2022/2023/2024) x CatBoost(~80s)+MLP
7-seed(~15~20분) 규모라 전체 실행에 1시간 이상 걸린다.

사용법:
  python -m code.experiment_oof_stacking
"""
import numpy as np

from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors,
    train_ensemble,
)
from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model, predict_meta
from code.experiment_attention import build_split

FOLD_VAL_SEASONS = [2021, 2022, 2023]
FINAL_HOLDOUT = 2024


def train_fold(val_season, device, seeds=ENSEMBLE_SEEDS):
    """`val_season`을 검증 시즌으로, 그 이전 모든 시즌을 학습으로 삼아 CatBoost+MLP를
    새로 학습하고 (cat_preds, mlp_preds, y_val)을 반환한다. OOF fold와 최종 2024 평가
    양쪽에 동일하게 쓰인다."""
    train_split, val_split, features, num_cols = build_split(val_season, apply_f1=True)
    y_val = val_split["control_success"].values

    X_train_raw = train_split[features]
    y_train_raw = train_split["control_success"].values
    X_val_raw = val_split[features]
    catboost_model, _ = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=False)
    cat_preds = predict_catboost(catboost_model, X_val_raw)

    tr_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    va_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(tr_proc, CAT_COLS, num_cols, "control_success")
    X_va_cat, X_va_num, y_va = to_tensors(va_proc, CAT_COLS, num_cols, "control_success")
    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_va_cat, X_val_num=X_va_num, y_val=y_va,
        seeds=seeds, device=device, verbose=False,
    )
    mlp_preds = predict_ensemble(
        members, cat_dims, len(num_cols), embed_dims, X_va_cat, X_va_num,
        bin_edges=bin_edges, device=device,
    )

    cat_score = compute_bss(cat_preds, y_val)[2]
    mlp_score = compute_bss(mlp_preds, y_val)[2]
    print(f"[Fold val={val_season}] n_train={len(train_split)} n_val={len(val_split)} | CatBoost={cat_score:.2f} MLP={mlp_score:.2f}")
    return cat_preds, mlp_preds, y_val


def run():
    device = get_device()
    print(f"[Device] {device}")

    oof_cat, oof_mlp, oof_y = [], [], []
    for val_season in FOLD_VAL_SEASONS:
        cat_preds, mlp_preds, y_val = train_fold(val_season, device)
        oof_cat.append(cat_preds)
        oof_mlp.append(mlp_preds)
        oof_y.append(y_val)

    oof_cat = np.concatenate(oof_cat)
    oof_mlp = np.concatenate(oof_mlp)
    oof_y = np.concatenate(oof_y)
    print(f"\n[OOF 통합] n={len(oof_y)} (folds: {FOLD_VAL_SEASONS})")

    w_cat_oof, w_mlp_oof, intercept_oof, oof_fit_score, _ = fit_meta_model(oof_cat, oof_mlp, oof_y)
    print(f"[OOF 메타모델] w_cat={w_cat_oof:.3f} w_mlp={w_mlp_oof:.3f} intercept={intercept_oof:.3f} (OOF 자체 blend score={oof_fit_score:.2f}, 참고용 — 표본 자체가 다른 fold라 프로덕션 지표와 직접 비교 불가)")

    # 최종 비교: season<2024로 학습한 "프로덕션급" CatBoost+MLP의 season==2024 예측에
    # (a) 기존 방식(season==2024 자체에 메타모델 피팅) vs (b) OOF 메타모델을 각각 적용
    print(f"\n[최종] season<{FINAL_HOLDOUT} 학습 -> season=={FINAL_HOLDOUT} 예측")
    cat_2024, mlp_2024, y_2024 = train_fold(FINAL_HOLDOUT, device)

    w_cat_std, w_mlp_std, intercept_std, standard_score, _ = fit_meta_model(cat_2024, mlp_2024, y_2024)
    oof_applied_preds = predict_meta(w_cat_oof, w_mlp_oof, intercept_oof, cat_2024, mlp_2024)
    oof_applied_score = compute_bss(oof_applied_preds, y_2024)[2]

    print(f"\n[기존 방식: season==2024에 메타모델 피팅] Blend: {standard_score:.2f} (w_cat={w_cat_std:.3f} w_mlp={w_mlp_std:.3f} intercept={intercept_std:.3f})")
    print(f"[OOF 메타모델을 2024에 적용(재피팅 없음)]  Blend: {oof_applied_score:.2f} (w_cat={w_cat_oof:.3f} w_mlp={w_mlp_oof:.3f} intercept={intercept_oof:.3f})")
    print(f"\nDelta (OOF 메타모델 - 기존 방식): {oof_applied_score - standard_score:+.2f}")


if __name__ == "__main__":
    run()
