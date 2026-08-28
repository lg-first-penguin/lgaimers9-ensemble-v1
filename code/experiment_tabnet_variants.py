# code/experiment_tabnet_variants.py
"""2026-08-21 세션: `code/experiment_thirdmodel_2026_screen.py`의 TabNet 1-seed 결과
(solo=484.34, corr(cat)=0.8143, corr(mlp)=0.7900 — MLP와 동일한 StandardScaler 전처리)
에 대해 사용자가 제기한 3가지 지적을 검증하는 후속 스크리닝.

1. **"MLP와 동일한 스케일링부터가 상관을 높이려고 작정한 거 아니냐"**: 타당한 지적.
   Rotation Forest(Rodriguez et al. 2006) 문헌 근거 — 그 앙상블 기법이 결정트리를
   골라 쓰는 이유가 "트리는 축(axis) 회전에 민감하기 때문"이고, PCA로 피처 공간을
   회전시키면 같은 정보량이라도 분기 경계가 달라져 앙상블 멤버 간 상관이 실제로
   낮아진다는 게 핵심 메커니즘. TabNet의 sparsemax attention도 트리처럼 축 정렬
   방식으로 피처를 하나씩 골라 쓰는 구조라 같은 원리가 적용될 가능성이 있다. 단,
   단순 스케일러 교체(RobustScaler 등)는 열 단위 선형변환이라 축을 안 바꾸므로
   Rotation Forest만큼의 효과는 이론상 기대하기 어렵다 — 그래서 v1(스케일러 교체,
   저렴한 대조군)과 v2(진짜 PCA 회전, 이론적 근거가 더 강한 버전) 둘 다 넣는다.
2. **"그래도 solo 점수를 높여봐라"**: v3 — capacity(n_d/n_a/n_steps) 증가로 solo
   점수를 CatBoost/MLP에 가깝게 밀어붙이고, 그때 상관계수가 실제로 같이 오르는지
   (오늘 3-후보 스크리닝에서 관찰된 "solo 높을수록 corr도 높다" 패턴) 직접 검증.
3. **"부스팅/주기적 임베딩 가능해?"**: 주기적 임베딩은 v4로 구현(고정 주파수
   sin/cos를 CatBoost 중요도 상위 피처에만 전처리 단계에서 추가 컬럼으로 얹음 —
   pytorch-tabnet은 수치형 전용 학습형 임베딩 레이어가 없어 이 방식이 최선).
   부스팅은 `TabNetClassifier.fit()`이 sklearn 블랙박스라 이번 스크립트에는 포함하지
   않음 — `pytorch_tabnet.tab_network.TabNet`(raw nn.Module)을 직접 써서 커스텀
   스테이지별 학습 루프를 새로 짜야 하는 별도 작업(`code/grownet_attention_model.py`급
   난이도). 여기 결과가 투자 가치를 보이면 이어서 만든다.

CatBoost/MLP(1-seed) 기준선은 한 번만 학습해 재사용(전 스크립트처럼 변형마다
다시 학습하지 않음 — 프로세스당 ~550s 절약).

사용법: python -m code.experiment_tabnet_variants
"""
import time

import numpy as np
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler

from code.blend_model import fit_meta_model
from code.catboost_model import predict_catboost, train_catboost
from code.experiment_3way_stack import fit_meta_model_n
from code.mlp_model import (
    CAT_COLS, QUANTILE_N_BINS, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device,
    to_tensors, train_ensemble, predict_ensemble,
)
from code.thirdmodel_common import build_split

SEED = 42
PERIODIC_TOPK = 6
PERIODIC_FREQS = [1.0, 2.0, 4.0, 8.0]


class BrierMetric(Metric):
    def __init__(self):
        self._name = "brier"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return float(((y_score[:, 1] - y_true) ** 2).mean())


def fit_numeric_scaler(train_split, num_cols, scaler_cls):
    """`code/mlp_model.py::fit_preprocessing`과 동일한 median-impute 로직이되,
    스케일러만 교체 가능하게 분리."""
    imputer = SimpleImputer(strategy="median")
    scaler = scaler_cls()
    imputed = imputer.fit_transform(train_split[num_cols])
    scaler.fit(imputed)
    return imputer, scaler


def apply_numeric_scaler(df, num_cols, imputer, scaler):
    imputed = imputer.transform(df[num_cols])
    return scaler.transform(imputed)


def add_periodic_features(X_num, freqs=PERIODIC_FREQS):
    """이미 표준화(평균0, 표준편차1)된 X_num(n, k)에 고정 주파수 sin/cos를 이어붙인다.
    학습 가능한 주파수가 아니라 고정값이라 `code/periodic_mlp_model.py::PeriodicEmbedding`
    (학습형)과는 다르지만, pytorch-tabnet이 학습형 수치 임베딩을 지원하지 않아 이게
    실질적 대안이다."""
    extra = []
    for f in freqs:
        extra.append(np.sin(2 * np.pi * f * X_num))
        extra.append(np.cos(2 * np.pi * f * X_num))
    return np.concatenate([X_num] + extra, axis=1)


def run_tabnet_variant(name, X_tr, y_tr, X_val, y_val_i, cat_idxs, cat_dims,
                        y_val_raw, cat_val_preds, mlp_val_preds,
                        n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
                        max_epochs=100, patience=15):
    t0 = time.time()
    model = TabNetClassifier(
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
        n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma, lambda_sparse=lambda_sparse,
        seed=SEED, verbose=1,
    )
    model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val_i)], eval_metric=[BrierMetric],
        max_epochs=max_epochs, patience=patience, batch_size=4096, virtual_batch_size=512,
    )
    preds = model.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t0
    score = compute_bss(preds, y_val_raw)[2]
    corr_cat = np.corrcoef(preds, cat_val_preds)[0, 1]
    corr_mlp = np.corrcoef(preds, mlp_val_preds)[0, 1]
    w_cat, w_mlp, intercept, blend2, _ = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)
    weights3, intercept3, blend3, _ = fit_meta_model_n([cat_val_preds, mlp_val_preds, preds], y_val_raw)
    print(f"[RESULT {name}] solo={score:.2f} | corr(cat)={corr_cat:.4f} corr(mlp)={corr_mlp:.4f} "
          f"| 2-way={blend2:.2f} 3-way={blend3:.2f} (Δ{blend3-blend2:+.2f}) | {elapsed:.1f}s")
    return dict(score=score, corr_cat=corr_cat, corr_mlp=corr_mlp, blend2=blend2, blend3=blend3, elapsed=elapsed)


def main():
    print("=== TabNet 변형 스크리닝 (스케일러/PCA회전/용량/주기피처) ===")
    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=True)
    device = get_device()
    print(f"[Device] {device}")

    # --- CatBoost/MLP 기준선 (한 번만 학습, 재사용) ---
    X_train_raw, y_train_raw = train_split[cat_feature_cols], train_split["control_success"].values
    X_val_raw, y_val_raw = val_split[cat_feature_cols], val_split["control_success"].values
    t0 = time.time()
    catboost_model, cat_best_iter = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    print(f"[CatBoost] Val Score={cat_score:.2f} (best_iter={cat_best_iter}, {time.time()-t0:.1f}s)")

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

    # --- 범주형 인코딩(모든 변형 공통, cat_encoder 재사용) ---
    # `code/mlp_model.py::apply_preprocessing`과 동일하게 문자열 캐스팅 후 +1 오프셋
    # (unknown_value=-1 -> 0, 기존 카테고리는 1부터) — cat_dims도 이 오프셋 기준으로 계산됨.
    cat_arr_tr = (cat_encoder.transform(train_split[CAT_COLS].astype(str)) + 1).astype(np.float32)
    cat_arr_val = (cat_encoder.transform(val_split[CAT_COLS].astype(str)) + 1).astype(np.float32)
    y_tr_i = train_split["control_success"].values.astype(np.int64)
    y_val_i = val_split["control_success"].values.astype(np.int64)
    cat_idxs = list(range(len(CAT_COLS)))

    results = {}

    # --- v1: RobustScaler (median/IQR) ---
    print("\n--- [v1_robust] RobustScaler ---")
    imputer_r, scaler_r = fit_numeric_scaler(train_split, mlp_num_cols, RobustScaler)
    num_tr_r = apply_numeric_scaler(train_split, mlp_num_cols, imputer_r, scaler_r)
    num_val_r = apply_numeric_scaler(val_split, mlp_num_cols, imputer_r, scaler_r)
    X_tr_v1 = np.concatenate([cat_arr_tr, num_tr_r], axis=1).astype(np.float32)
    X_val_v1 = np.concatenate([cat_arr_val, num_val_r], axis=1).astype(np.float32)
    results["v1_robust"] = run_tabnet_variant("v1_robust", X_tr_v1, y_tr_i, X_val_v1, y_val_i,
                                               cat_idxs, cat_dims, y_val_raw, cat_val_preds, mlp_val_preds)

    # --- v2: PCA 회전 (StandardScaler 적용된 수치형 전체를 회전, 성분 전부 유지) ---
    print("\n--- [v2_pca_rotation] StandardScaler + PCA 전체 회전 ---")
    imputer_s, scaler_s = fit_numeric_scaler(train_split, mlp_num_cols, StandardScaler)
    num_tr_s = apply_numeric_scaler(train_split, mlp_num_cols, imputer_s, scaler_s)
    num_val_s = apply_numeric_scaler(val_split, mlp_num_cols, imputer_s, scaler_s)
    pca = PCA(n_components=len(mlp_num_cols), random_state=SEED)
    num_tr_pca = pca.fit_transform(num_tr_s)
    num_val_pca = pca.transform(num_val_s)
    X_tr_v2 = np.concatenate([cat_arr_tr, num_tr_pca], axis=1).astype(np.float32)
    X_val_v2 = np.concatenate([cat_arr_val, num_val_pca], axis=1).astype(np.float32)
    results["v2_pca_rotation"] = run_tabnet_variant("v2_pca_rotation", X_tr_v2, y_tr_i, X_val_v2, y_val_i,
                                                      cat_idxs, cat_dims, y_val_raw, cat_val_preds, mlp_val_preds)

    # --- v3: 용량 증가 (baseline StandardScaler, n_d/n_a/n_steps 키움) ---
    print("\n--- [v3_bigcap] StandardScaler + n_d=24,n_a=24,n_steps=5 ---")
    X_tr_v3 = np.concatenate([cat_arr_tr, num_tr_s], axis=1).astype(np.float32)
    X_val_v3 = np.concatenate([cat_arr_val, num_val_s], axis=1).astype(np.float32)
    results["v3_bigcap"] = run_tabnet_variant("v3_bigcap", X_tr_v3, y_tr_i, X_val_v3, y_val_i,
                                               cat_idxs, cat_dims, y_val_raw, cat_val_preds, mlp_val_preds,
                                               n_d=24, n_a=24, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
                                               max_epochs=150, patience=20)

    # --- v4: 고정주파수 주기 피처 추가 (CatBoost 중요도 상위 PERIODIC_TOPK개 수치형에만) ---
    print("\n--- [v4_periodic] StandardScaler + 고정주파수 sin/cos(top {} 피처) ---".format(PERIODIC_TOPK))
    importances = catboost_model.get_feature_importance()
    imp_by_col = dict(zip(cat_feature_cols, importances))
    numeric_ranked = sorted(mlp_num_cols, key=lambda c: imp_by_col.get(c, 0.0), reverse=True)
    top_cols = numeric_ranked[:PERIODIC_TOPK]
    print(f"  주기 피처 대상: {top_cols}")
    top_idx = [mlp_num_cols.index(c) for c in top_cols]
    periodic_tr = add_periodic_features(num_tr_s[:, top_idx])
    periodic_val = add_periodic_features(num_val_s[:, top_idx])
    X_tr_v4 = np.concatenate([cat_arr_tr, num_tr_s, periodic_tr], axis=1).astype(np.float32)
    X_val_v4 = np.concatenate([cat_arr_val, num_val_s, periodic_val], axis=1).astype(np.float32)
    results["v4_periodic"] = run_tabnet_variant("v4_periodic", X_tr_v4, y_tr_i, X_val_v4, y_val_i,
                                                 cat_idxs, cat_dims, y_val_raw, cat_val_preds, mlp_val_preds)

    print("\n" + "=" * 100)
    print(f"{'variant':<18}{'solo':>10}{'corr_cat':>10}{'corr_mlp':>10}{'2-way':>10}{'3-way':>10}{'delta':>10}{'sec':>10}")
    print(f"{'baseline(StdScaler)':<18}{'484.34':>10}{'0.8143':>10}{'0.7900':>10}{'—':>10}{'—':>10}{'—':>10}{'—':>10}")
    for name, r in results.items():
        print(f"{name:<18}{r['score']:>10.2f}{r['corr_cat']:>10.4f}{r['corr_mlp']:>10.4f}"
              f"{r['blend2']:>10.2f}{r['blend3']:>10.2f}{r['blend3']-r['blend2']:>+10.2f}{r['elapsed']:>10.1f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
