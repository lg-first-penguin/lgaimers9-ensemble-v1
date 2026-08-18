# code/experiment_tabnet_correlation.py
""""3번째 모델"로 TabNet(Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning")을
시도해보기 전에, 이미 이 프로젝트가 겪은 선례부터 참고한다: FT-Transformer/ExcelFormer(둘 다
self-attention 기반)를 3번째 모델로 붙이는 실험이 §22-23에서 있었다. 1-seed 단일 레짐
스크리닝에서는 유망해 보였지만(3-way 스태킹 이득 평균 +14.48), 7-seed 앙상블 + 정규화된
메타모델(`LogisticRegressionCV`) + rolling-origin fold check로 제대로 재검증하니 오히려
손해(평균 -57.14)로 뒤집혔다. 원인은 그 3번째 모델의 예측이 CatBoost/MLP와 이미 상관
0.88~0.93으로 높아서, 시즌 하나짜리 검증셋에 3-피처 메타모델을 매번 새로 피팅하면
다중공선성으로 계수 부호까지 흔들렸기 때문(EXPERIMENTS.md §24.5, 핵심 교훈 #14/#15).

TabNet은 dense self-attention이 아니라 sparsemax 기반 sequential feature masking으로
트리 모델의 순차 분기 방식을 신경망으로 흉내내는 구조라, FT-Transformer/ExcelFormer나
TabM(둘 다 "신경망 계열이라 MLP와 상관관계가 높음", 핵심 교훈 #7)보다 CatBoost/MLP와 덜
상관될 가능성이 있다 — 하지만 그 반대일 수도 있다. §22의 교훈("7-seed+정규화+fold-check로
재검증하기 전엔 어느 방향인지 알 수 없다")을 존중해, 여기서는 전체 검증 파이프라인에
투자하기 전에 **1-seed로 싸게 상관관계부터 확인하는 진단**만 수행한다.

판단 기준(제안한 대화 맥락): 상관관계가 0.85 이상이면 ExcelFormer 때와 같은 결말일
가능성이 높아 여기서 멈춘다. 뚜렷이 낮으면(예: 0.7 이하) §22-23과 동일한 절차
(7-seed 앙상블 + `code/experiment_3way_stack.py::fit_meta_model_n` 정규화 메타모델 +
rolling-origin fold check)로 본격 검증한다.

TabNet 피처셋은 §22/§23의 FT-Transformer/ExcelFormer와 동일한 관례를 따라 MLP와 같은
피처셋(CAT_COLS + mlp_num_cols, 즉 tier A 트랙맨 포함)을 사용한다. CatBoost 전용
피처(coarse pitchmix 등)는 주지 않는다 — CatBoost/MLP 둘 다와 비교할 3번째 축이므로
어느 한쪽에 유리하게 피처를 몰아주지 않기 위함.

TabNet은 `pytorch-tabnet` 패키지가 필요하다(`uv pip install --python .venv/bin/python
pytorch-tabnet`로 로컬에 설치, 대회 서버 사전설치 목록엔 없음 — 채택 시
`submit/requirements.txt`에 추가 필요).

1-seed 진단 결과(cutoff7): TabNet Val=548.53 | corr(TabNet,CatBoost)=0.8198 |
corr(TabNet,MLP)=0.7898 — ExcelFormer(0.88~0.93)보다는 낮지만 "확실히 낮음"(<=0.7) 기준에는
못 미치는 애매한 지대. 1-seed 3-way 스크린 Δ+7.49(726.31→733.80)는 §22와 똑같이 신뢰
불가능한 숫자. 사용자 판단으로 본검증(7-seed 앙상블 + 양쪽 레짐) 투자 진행.

`--ensemble`: TabNet과 MLP 둘 다 SCREEN([42])이 아니라 프로덕션 ENSEMBLE_SEEDS(7개)로
학습해 §22/23과 동일한 수준으로 재검증한다. CatBoost는 원래도 결정적(deterministic
loss)이라 1회만 학습. TabNet 1-seed가 ~800s라 7-seed면 레짐당 ~1.5시간 내외 예상 —
cutoff7(프로덕션 레짐)을 먼저 돌리고, 결과가 애매하면 season==2023도 추가한다.

사용법:
  python -m code.experiment_tabnet_correlation --cutoff7   # 1-seed 진단(빠름, 완료됨)
  python -m code.experiment_tabnet_correlation --holdout 2023
  python -m code.experiment_tabnet_correlation --cutoff7 --ensemble   # 본검증(7-seed)
  python -m code.experiment_tabnet_correlation --holdout 2023 --ensemble
"""
import argparse
import time

import numpy as np
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier

from code.blend_model import fit_meta_model
from code.catboost_model import predict_catboost, train_catboost
from code.experiment_3way_stack import fit_meta_model_n
from code.experiment_residual_correction_9_10 import build_split
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device,
    make_bundle, predict_bundle, to_tensors, train_ensemble,
)

TABNET_SEED = 42


class BrierMetric(Metric):
    """TabNet의 `eval_metric`에 꽂을 수 있는 커스텀 Brier score(낮을수록 좋음)."""

    def __init__(self):
        self._name = "brier"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return float(((y_score[:, 1] - y_true) ** 2).mean())


def train_tabnet(X_tr, y_tr, X_val, y_val, cat_idxs, cat_dims, seed=TABNET_SEED, verbose=1):
    model = TabNetClassifier(
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
        seed=seed, verbose=verbose,
    )
    model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=[BrierMetric],
        max_epochs=100, patience=15, batch_size=4096, virtual_batch_size=512,
    )
    return model


def train_tabnet_ensemble(X_tr, y_tr, X_val, y_val, cat_idxs, cat_dims, seeds):
    """서로 다른 시드로 TabNet을 여러 개 학습해 (모델 리스트, 시드별 predict_proba 평균)을 반환.
    production MLP의 train_ensemble/predict_ensemble과 동일한 앙상블 관례."""
    preds_list = []
    for i, seed in enumerate(seeds):
        t0 = time.time()
        model = train_tabnet(X_tr, y_tr, X_val, y_val, cat_idxs, cat_dims, seed=seed, verbose=0)
        preds = model.predict_proba(X_val)[:, 1]
        preds_list.append(preds)
        print(f"  [TabNet ensemble {i+1}/{len(seeds)}] seed={seed} 완료 ({time.time()-t0:.1f}s)")
    return np.mean(preds_list, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--ensemble", action="store_true",
                         help="TabNet/MLP를 1-seed 대신 프로덕션 ENSEMBLE_SEEDS(7개)로 학습(본검증)")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    label = "cutoff7" if args.cutoff7 else f"holdout={holdout}"
    mode = "ensemble(7-seed)" if args.ensemble else "1-seed 진단"
    print(f"=== TabNet 3rd-model 상관관계 진단 ({label}, {mode}) ===")

    train_split, val_split, features, cat_features, mlp_num_cols = build_split(holdout, args.cutoff7)
    device = get_device()
    print(f"[Device] {device}")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    y_val_raw = val_split["control_success"].values

    # --- CatBoost(고정) ---
    X_train_raw, y_train_raw = train_split[cat_features], train_split["control_success"].values
    X_val_raw = val_split[cat_features]
    t0 = time.time()
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    print(f"[CatBoost] Val Score={cat_score:.2f} (best_iteration={catboost_best_iteration}, {time.time()-t0:.1f}s)")

    mlp_seeds = ENSEMBLE_SEEDS if args.ensemble else [TABNET_SEED]
    tabnet_seeds = ENSEMBLE_SEEDS if args.ensemble else [TABNET_SEED]
    mlp_label = f"MLP({len(mlp_seeds)}-seed)"
    tabnet_label = f"TabNet({len(tabnet_seeds)}-seed)"

    # --- MLP ---
    X_tr_cat, X_tr_num, y_tr_t = to_tensors(train_proc, CAT_COLS, mlp_num_cols, "control_success")
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, "control_success")
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)
    t0 = time.time()
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr_t, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_raw,
        seeds=mlp_seeds, device=device, verbose=False,
    )
    mlp_bundle = make_bundle(members, CAT_COLS, mlp_num_cols, cat_dims, embed_dims,
                              cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges)
    mlp_val_preds = predict_bundle(mlp_bundle, val_split, device=device)
    mlp_score = compute_bss(mlp_val_preds, y_val_raw)[2]
    print(f"[{mlp_label}] Val Score={mlp_score:.2f} ({time.time()-t0:.1f}s)")

    # --- TabNet ---
    tabnet_cols = CAT_COLS + mlp_num_cols
    X_tr = train_proc[tabnet_cols].values.astype(np.float32)
    y_tr = train_proc["control_success"].values.astype(np.int64)
    X_val = val_proc[tabnet_cols].values.astype(np.float32)
    y_val = val_proc["control_success"].values.astype(np.int64)
    cat_idxs = list(range(len(CAT_COLS)))

    t0 = time.time()
    if args.ensemble:
        print(f"\n[TabNet ensemble 시작] {len(tabnet_seeds)}개 시드")
        tabnet_val_preds = train_tabnet_ensemble(X_tr, y_tr, X_val, y_val, cat_idxs, cat_dims, tabnet_seeds)
    else:
        tabnet_model = train_tabnet(X_tr, y_tr, X_val, y_val, cat_idxs, cat_dims, seed=tabnet_seeds[0])
        tabnet_val_preds = tabnet_model.predict_proba(X_val)[:, 1]
    tabnet_score = compute_bss(tabnet_val_preds, y_val_raw)[2]
    print(f"[{tabnet_label}] Val Score={tabnet_score:.2f} ({time.time()-t0:.1f}s)")

    # --- 상관관계 진단 (핵심 질문) ---
    corr_cat = np.corrcoef(tabnet_val_preds, cat_val_preds)[0, 1]
    corr_mlp = np.corrcoef(tabnet_val_preds, mlp_val_preds)[0, 1]
    print(f"\n[상관관계] TabNet vs CatBoost: {corr_cat:.4f} | TabNet vs MLP: {corr_mlp:.4f}")
    print("(참고: ExcelFormer는 0.88~0.93에서 결국 손해로 판명 — 이 값이 뚜렷이 낮아야(예: <=0.7) 본검증 투자 가치 있음)")

    # --- 스태킹 이득 (정규화된 메타모델, §23과 동일 관례) ---
    w_cat, w_mlp, intercept2, blend2_score, _ = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)
    weights3, intercept3, blend3_score, _ = fit_meta_model_n([cat_val_preds, mlp_val_preds, tabnet_val_preds], y_val_raw)
    note = "" if args.ensemble else " (1-seed 단일 레짐 — §22처럼 노이즈일 수 있음, 채택 판단 금지)"
    print(f"\n[스태킹 이득{note}]")
    print(f"2-way Blend={blend2_score:.2f} | 3-way(+TabNet) Blend={blend3_score:.2f} (Δ{blend3_score-blend2_score:+.2f}, weights={[f'{w:.3f}' for w in weights3]})")


if __name__ == "__main__":
    main()
