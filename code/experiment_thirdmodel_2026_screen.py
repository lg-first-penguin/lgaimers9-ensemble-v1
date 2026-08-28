# code/experiment_thirdmodel_2026_screen.py
"""2026-08-21 세션: §50에서 종결됐던 "3rd-모델 스태킹" 라인을 사용자 판단으로
재개하는 1-seed 저비용 진단 스크리닝. §22/§42의 확립된 절차(전체 7-seed 검증
투자 전에 1-seed로 solo 점수 + 기존 두 모델과의 상관계수부터 확인)를 그대로 따른다.

재개 근거(`code/thirdmodel_common.py` 문서 참고): 기존 3rd-모델 실험들
(FT-Transformer/ExcelFormer §22-24, TabNet §42/§50)은 전부 TE-residual 피처(실전
+13.86) 도입 이전 또는 트랙맨 tier A가 아직 켜져 있던 스냅샷 피처셋을 썼다 — 지금
프로덕션 피처셋(real 1041.40)으로는 한 번도 재검증되지 않았다. 여기에 두 가지
새 메커니즘도 추가로 스크리닝한다: 주기함수(sin/cos) 임베딩 FT-Transformer
(`code/ft_transformer_periodic_model.py`)와 GrowNet식 boosting attention
(`code/grownet_attention_model.py`, bagging이 아니라 순차 residual-fit).

판단 기준(§42와 동일): 상관계수(CatBoost/MLP 대비)가 뚜렷이 낮으면(<=0.7 근방)
7-seed 본검증 투자 가치 있음, 0.85 이상이면 ExcelFormer/TabNet과 같은 결말일
가능성이 높아 여기서 멈춘다.

사용법:
  python -m code.experiment_thirdmodel_2026_screen --smoke                # 버그 확인용, 수 분
  python -m code.experiment_thirdmodel_2026_screen                        # 전체 1-seed 진단
  python -m code.experiment_thirdmodel_2026_screen --candidates tabnet ft_periodic
"""
import argparse
import time

import numpy as np

from code.blend_model import fit_meta_model
from code.catboost_model import predict_catboost, train_catboost
from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device,
    to_tensors, train_ensemble, predict_ensemble,
)
from code.thirdmodel_common import build_split

SEED = 42
ALL_CANDIDATES = ["tabnet", "ft_periodic", "grownet"]


def smoke_subsample(train_split, val_split, n_train=20000, n_val=5000, seed=SEED):
    rng = np.random.RandomState(seed)
    tr_idx = rng.choice(len(train_split), size=min(n_train, len(train_split)), replace=False)
    va_idx = rng.choice(len(val_split), size=min(n_val, len(val_split)), replace=False)
    return train_split.iloc[tr_idx].reset_index(drop=True), val_split.iloc[va_idx].reset_index(drop=True)


def run_tabnet(train_proc, val_proc, mlp_num_cols, cat_dims, y_val, smoke):
    from pytorch_tabnet.metrics import Metric
    from pytorch_tabnet.tab_model import TabNetClassifier

    class BrierMetric(Metric):
        def __init__(self):
            self._name = "brier"
            self._maximize = False

        def __call__(self, y_true, y_score):
            return float(((y_score[:, 1] - y_true) ** 2).mean())

    tabnet_cols = CAT_COLS + mlp_num_cols
    X_tr = train_proc[tabnet_cols].values.astype(np.float32)
    y_tr = train_proc["control_success"].values.astype(np.int64)
    X_val = val_proc[tabnet_cols].values.astype(np.float32)
    y_val_i = val_proc["control_success"].values.astype(np.int64)
    cat_idxs = list(range(len(CAT_COLS)))

    model = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1, seed=SEED, verbose=1)
    model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val_i)], eval_metric=[BrierMetric],
        max_epochs=5 if smoke else 100, patience=2 if smoke else 15,
        batch_size=1024 if smoke else 4096, virtual_batch_size=256 if smoke else 512,
    )
    return model.predict_proba(X_val)[:, 1]


def run_ft_periodic(X_tr_cat, X_tr_num, y_tr, cat_dims, num_numeric,
                     X_val_cat, X_val_num, y_val, smoke):
    from code.ft_transformer_periodic_model import batched_forward, train_ft_periodic
    model, best_epoch = train_ft_periodic(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, num_numeric=num_numeric,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val,
        max_epochs=2 if smoke else 60, patience=1 if smoke else 7,
        device=get_device(), verbose=True, seed=SEED,
    )
    print(f"  [ft_periodic] best_epoch={best_epoch}")
    return batched_forward(model, X_val_cat, X_val_num, device=get_device())


def run_grownet(X_tr_cat, X_tr_num, y_tr_np, cat_dims, bin_edges,
                 X_val_cat, X_val_num, y_val_np, smoke):
    from code.grownet_attention_model import predict_grownet, train_grownet
    stages, etas = train_grownet(
        X_tr_cat, X_tr_num, y_tr_np, cat_dims, bin_edges,
        X_val_cat, X_val_num, y_val_np,
        n_stages=2 if smoke else 5,
        stage_max_epochs=2 if smoke else 15,
        stage_patience=1 if smoke else 3,
        corrective_epochs=1 if smoke else 3,
        device=get_device(), verbose=True, seed=SEED,
    )
    return predict_grownet(stages, etas, cat_dims, bin_edges, X_val_cat, X_val_num, device=get_device())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", default=ALL_CANDIDATES, choices=ALL_CANDIDATES)
    parser.add_argument("--smoke", action="store_true", help="버그 확인용 초소형 서브샘플+epoch 캡")
    args = parser.parse_args()

    print(f"=== 3rd-모델 2026 재개 스크리닝 (candidates={args.candidates}, smoke={args.smoke}) ===")
    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=True)
    if args.smoke:
        train_split, val_split = smoke_subsample(train_split, val_split)
        print(f"[smoke] 서브샘플: train={len(train_split)} val={len(val_split)}")

    device = get_device()
    print(f"[Device] {device}")

    # --- CatBoost(대조군, 결정적이라 1회) ---
    X_train_raw, y_train_raw = train_split[cat_feature_cols], train_split["control_success"].values
    X_val_raw, y_val_raw = val_split[cat_feature_cols], val_split["control_success"].values
    t0 = time.time()
    catboost_model, cat_best_iter = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    print(f"[CatBoost] Val Score={cat_score:.2f} (best_iter={cat_best_iter}, {time.time()-t0:.1f}s)")

    # --- MLP(1-seed, §42와 동일 관례) ---
    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr_t = to_tensors(train_proc, CAT_COLS, mlp_num_cols, "control_success")
    X_val_cat, X_val_num, _ = to_tensors(val_proc, CAT_COLS, mlp_num_cols, "control_success")
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)

    t0 = time.time()
    mlp_members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr_t, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_raw,
        seeds=[SEED], device=device, verbose=False,
    )
    mlp_val_preds = predict_ensemble(mlp_members, cat_dims, len(mlp_num_cols), embed_dims,
                                      X_val_cat, X_val_num, bin_edges=bin_edges, device=device)
    mlp_score = compute_bss(mlp_val_preds, y_val_raw)[2]
    print(f"[MLP(1-seed)] Val Score={mlp_score:.2f} ({time.time()-t0:.1f}s)")

    results = {}
    for name in args.candidates:
        print(f"\n--- [{name}] 학습 시작 ---")
        t0 = time.time()
        if name == "tabnet":
            preds = run_tabnet(train_proc, val_proc, mlp_num_cols, cat_dims, y_val_raw, args.smoke)
        elif name == "ft_periodic":
            preds = run_ft_periodic(X_tr_cat, X_tr_num, y_tr_t, cat_dims, len(mlp_num_cols),
                                     X_val_cat, X_val_num, y_val_raw, args.smoke)
        elif name == "grownet":
            preds = run_grownet(X_tr_cat, X_tr_num, y_tr_t.numpy(), cat_dims, bin_edges,
                                 X_val_cat, X_val_num, y_val_raw, args.smoke)
        else:
            raise ValueError(name)
        elapsed = time.time() - t0
        score = compute_bss(preds, y_val_raw)[2]
        corr_cat = np.corrcoef(preds, cat_val_preds)[0, 1]
        corr_mlp = np.corrcoef(preds, mlp_val_preds)[0, 1]

        w_cat, w_mlp, intercept, blend2_score, _ = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)
        from code.experiment_3way_stack import fit_meta_model_n
        weights3, intercept3, blend3_score, _ = fit_meta_model_n([cat_val_preds, mlp_val_preds, preds], y_val_raw)

        results[name] = dict(score=score, corr_cat=corr_cat, corr_mlp=corr_mlp,
                              blend2=blend2_score, blend3=blend3_score, elapsed=elapsed)
        print(f"[RESULT {name}] solo={score:.2f} | corr(cat)={corr_cat:.4f} corr(mlp)={corr_mlp:.4f} "
              f"| 2-way={blend2_score:.2f} 3-way={blend3_score:.2f} (Δ{blend3_score-blend2_score:+.2f}) "
              f"| {elapsed:.1f}s")

    print("\n" + "=" * 100)
    print(f"{'candidate':<14}{'solo':>10}{'corr_cat':>10}{'corr_mlp':>10}{'2-way':>10}{'3-way':>10}{'delta':>10}{'sec':>10}")
    print(f"{'CatBoost':<14}{cat_score:>10.2f}{'—':>10}{'—':>10}{'—':>10}{'—':>10}{'—':>10}{'—':>10}")
    print(f"{'MLP(1seed)':<14}{mlp_score:>10.2f}{'—':>10}{'—':>10}{'—':>10}{'—':>10}{'—':>10}{'—':>10}")
    for name, r in results.items():
        print(f"{name:<14}{r['score']:>10.2f}{r['corr_cat']:>10.4f}{r['corr_mlp']:>10.4f}"
              f"{r['blend2']:>10.2f}{r['blend3']:>10.2f}{r['blend3']-r['blend2']:>+10.2f}{r['elapsed']:>10.1f}")
    print("=" * 100)
    print("(참고: corr가 0.85 이상이면 ExcelFormer/TabNet §22-24/§42와 같은 결말일 가능성 높음. "
          "0.7 이하로 뚜렷이 낮아야 7-seed 본검증 투자 가치. 1-seed 3-way delta는 노이즈일 수 있음 — 채택 판단 금지.)")


if __name__ == "__main__":
    main()
