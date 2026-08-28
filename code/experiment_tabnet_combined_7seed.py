# code/experiment_tabnet_combined_7seed.py
"""2026-08-21 세션: `code/experiment_tabnet_variants.py`의 4개 변형(v1 RobustScaler,
v2 PCA 회전, v3 용량 증가, v4 고정주파수 주기 피처)을 사용자 지시로 **하나의 조합**에
전부 적용해, 프로덕션과 동일한 엄밀도(ENSEMBLE_SEEDS 7개, `--ensemble`급 본검증)로
재검증한다.

**조합 방식**:
  1. RobustScaler(v1)로 수치형 44개 컬럼 스케일링 (median impute 후).
  2. 그 RobustScaler 결과 전체에 PCA 회전(v2, 전 성분 유지 — Rotation Forest 방식)을
     적용해 최종 "회전된 수치형" 블록을 만든다 (v1과 v2를 체이닝 — StandardScaler
     대신 RobustScaler 위에서 회전).
  3. 회전 전 RobustScaler 값 기준으로 CatBoost 중요도 상위 6개 원본 피처에 고정주파수
     sin/cos(v4)를 추가 컬럼으로 얹는다 — 회전 후 PCA 성분은 개별 원본 피처와 대응이
     없어 주기 변환의 의미가 없으므로 회전 *전* 값에 적용.
  4. 최종 수치형 입력 = [회전된 44개 성분] + [주기 피처 48개] = 92컬럼.
  5. TabNet 용량은 v3(n_d=24, n_a=24, n_steps=5, gamma=1.5, lambda_sparse=1e-4).

1-seed 스크리닝(v1~v4 개별)에서는 넷 다 상관을 오히려 높이는 방향(v1/v2)이거나
solo를 낮추는 방향(v3/v4)이었지만, 넷을 합치면 개별 효과가 상쇄/증폭될 수 있어
사용자 판단으로 직접 확인 — 1-seed 노이즈를 배제하기 위해 처음부터 ENSEMBLE_SEEDS(7)로
검증한다(§22/§42의 "본검증" 절차와 동일 엄밀도). CatBoost는 결정적이라 1회,
MLP도 프로덕션과 동일하게 7-seed 학습.

사용법: python -m code.experiment_tabnet_combined_7seed
"""
import time

import numpy as np
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from code.blend_model import fit_meta_model
from code.catboost_model import predict_catboost, train_catboost
from code.experiment_3way_stack import fit_meta_model_n
from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, apply_preprocessing, compute_bss,
    embed_dim_for_cardinality, fit_preprocessing, fit_quantile_edges, get_device,
    to_tensors, train_ensemble, predict_ensemble,
)
from code.thirdmodel_common import build_split

PERIODIC_TOPK = 6
PERIODIC_FREQS = [1.0, 2.0, 4.0, 8.0]
TABNET_KWARGS = dict(n_d=24, n_a=24, n_steps=5, gamma=1.5, lambda_sparse=1e-4)


class BrierMetric(Metric):
    def __init__(self):
        self._name = "brier"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return float(((y_score[:, 1] - y_true) ** 2).mean())


def add_periodic_features(X_num, freqs=PERIODIC_FREQS):
    extra = []
    for f in freqs:
        extra.append(np.sin(2 * np.pi * f * X_num))
        extra.append(np.cos(2 * np.pi * f * X_num))
    return np.concatenate([X_num] + extra, axis=1)


def main():
    print("=== TabNet v1+v2+v3+v4 조합, 7-seed 본검증 (cutoff7) ===")
    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=True)
    device = get_device()
    print(f"[Device] {device}")

    # --- CatBoost(결정적, 1회) ---
    X_train_raw, y_train_raw = train_split[cat_feature_cols], train_split["control_success"].values
    X_val_raw, y_val_raw = val_split[cat_feature_cols], val_split["control_success"].values
    t0 = time.time()
    catboost_model, cat_best_iter = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    print(f"[CatBoost] Val Score={cat_score:.2f} (best_iter={cat_best_iter}, {time.time()-t0:.1f}s)")

    # --- MLP(7-seed, 프로덕션과 동일) ---
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
        seeds=ENSEMBLE_SEEDS, device=device, verbose=False,
    )
    mlp_val_preds = predict_ensemble(mlp_members, cat_dims, len(mlp_num_cols), embed_dims,
                                      X_val_cat, X_val_num, bin_edges=bin_edges, device=device)
    mlp_score = compute_bss(mlp_val_preds, y_val_raw)[2]
    print(f"[MLP(7-seed)] Val Score={mlp_score:.2f} ({time.time()-t0:.1f}s)")

    # --- TabNet 조합 전처리 (한 번만, 모든 시드가 공유) ---
    cat_arr_tr = (cat_encoder.transform(train_split[CAT_COLS].astype(str)) + 1).astype(np.float32)
    cat_arr_val = (cat_encoder.transform(val_split[CAT_COLS].astype(str)) + 1).astype(np.float32)
    y_tr_i = train_split["control_success"].values.astype(np.int64)
    y_val_i = val_split["control_success"].values.astype(np.int64)
    cat_idxs = list(range(len(CAT_COLS)))

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    num_tr_robust = scaler.fit_transform(imputer.fit_transform(train_split[mlp_num_cols]))
    num_val_robust = scaler.transform(imputer.transform(val_split[mlp_num_cols]))

    pca = PCA(n_components=len(mlp_num_cols), random_state=ENSEMBLE_SEEDS[0])
    num_tr_rot = pca.fit_transform(num_tr_robust)
    num_val_rot = pca.transform(num_val_robust)

    importances = catboost_model.get_feature_importance()
    imp_by_col = dict(zip(cat_feature_cols, importances))
    numeric_ranked = sorted(mlp_num_cols, key=lambda c: imp_by_col.get(c, 0.0), reverse=True)
    top_cols = numeric_ranked[:PERIODIC_TOPK]
    print(f"[전처리] 주기 피처 대상: {top_cols}")
    top_idx = [mlp_num_cols.index(c) for c in top_cols]
    periodic_tr = add_periodic_features(num_tr_robust[:, top_idx])
    periodic_val = add_periodic_features(num_val_robust[:, top_idx])

    X_tr = np.concatenate([cat_arr_tr, num_tr_rot, periodic_tr], axis=1).astype(np.float32)
    X_val = np.concatenate([cat_arr_val, num_val_rot, periodic_val], axis=1).astype(np.float32)
    print(f"[전처리] TabNet 입력 shape: train={X_tr.shape} val={X_val.shape}")

    # --- TabNet 7-seed 앙상블 ---
    preds_list = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        t0 = time.time()
        model = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
                                  seed=seed, verbose=0, **TABNET_KWARGS)
        model.fit(
            X_tr, y_tr_i, eval_set=[(X_val, y_val_i)], eval_metric=[BrierMetric],
            max_epochs=150, patience=20, batch_size=4096, virtual_batch_size=512,
        )
        preds = model.predict_proba(X_val)[:, 1]
        preds_list.append(preds)
        seed_score = compute_bss(preds, y_val_raw)[2]
        print(f"  [TabNet 7-seed {i+1}/{len(ENSEMBLE_SEEDS)}] seed={seed} solo={seed_score:.2f} ({time.time()-t0:.1f}s)")

    tabnet_val_preds = np.mean(preds_list, axis=0)
    tabnet_score = compute_bss(tabnet_val_preds, y_val_raw)[2]
    corr_cat = np.corrcoef(tabnet_val_preds, cat_val_preds)[0, 1]
    corr_mlp = np.corrcoef(tabnet_val_preds, mlp_val_preds)[0, 1]

    w_cat, w_mlp, intercept, blend2, _ = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)
    weights3, intercept3, blend3, _ = fit_meta_model_n([cat_val_preds, mlp_val_preds, tabnet_val_preds], y_val_raw)

    print("\n" + "=" * 90)
    print(f"CatBoost={cat_score:.2f} | MLP(7-seed)={mlp_score:.2f} | TabNet-combined(7-seed)={tabnet_score:.2f}")
    print(f"corr(cat)={corr_cat:.4f} corr(mlp)={corr_mlp:.4f}")
    print(f"2-way(Cat+MLP)={blend2:.2f} | 3-way(+TabNet-combined)={blend3:.2f} (Δ{blend3-blend2:+.2f})")
    print("=" * 90)


if __name__ == "__main__":
    main()
