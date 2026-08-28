# code/experiment_thirdmodel_base.py
"""DeepFM/EBM/BART/NAM 4개 후보를 각각 튜닝 후 3-way 스태킹까지 확인하는 시리즈의 공용
베이스 캐시. CatBoost/MLP를 후보마다 재학습하면 낭비이므로, `code/thirdmodel_common.py`
(project 관례: 3rd 모델에는 CAT_COLS+mlp_num_cols를 주고, 프로덕션과 동일 cutoff7
스플릿을 씀)로 스플릿을 만들고 기존 `open/reference/best_model.pkl`(재학습 없이 추론만
— `code/experiment_3way_stack.py::step_base`와 동일한 지름길)로 cat/mlp val 예측을
한 번만 뽑아 캐시한다. 각 후보 스크립트는 이 캐시를 로드해 자기 모델만 학습하고
`code/experiment_3way_stack.py::fit_meta_model_n`으로 3-way delta를 계산한다.

사용법: python -m code.experiment_thirdmodel_base
"""
import os
import pickle

import numpy as np

from code.thirdmodel_common import build_split, TARGET_COL
from code.mlp_model import predict_bundle, compute_bss, get_device
from code.catboost_model import predict_catboost
from code.blend_model import fit_meta_model

CACHE_DIR = "./open/temp/experiment_thirdmodel4"
REFERENCE_BUNDLE = "./open/reference/best_model.pkl"


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=True)

    with open(REFERENCE_BUNDLE, "rb") as f:
        bundle = pickle.load(f)
    for key in ("catboost_model", "mlp_bundle", "meta_model"):
        if key not in bundle:
            raise RuntimeError(f"{REFERENCE_BUNDLE}가 현재 블렌드 번들 스키마가 아닙니다 ('{key}' 없음)")

    device = get_device()
    y_val = val_split[TARGET_COL].values

    cat_val_preds = predict_catboost(bundle["catboost_model"], val_split[cat_feature_cols])
    mlp_val_preds = predict_bundle(bundle["mlp_bundle"], val_split, device=device)

    cat_score = compute_bss(cat_val_preds, y_val)[2]
    mlp_score = compute_bss(mlp_val_preds, y_val)[2]
    w_cat, w_mlp, intercept, blend2, _ = fit_meta_model(cat_val_preds, mlp_val_preds, y_val)

    print(f"[base] CatBoost={cat_score:.2f} | MLP(레퍼런스 번들, 7-seed)={mlp_score:.2f} | 2-way(재적합)={blend2:.2f}")

    train_split.to_pickle(os.path.join(CACHE_DIR, "train_split.pkl"))
    val_split.to_pickle(os.path.join(CACHE_DIR, "val_split.pkl"))
    np.savez(os.path.join(CACHE_DIR, "base_preds.npz"), cat_val_preds=cat_val_preds, mlp_val_preds=mlp_val_preds, y_val=y_val)
    with open(os.path.join(CACHE_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({
            "mlp_num_cols": mlp_num_cols, "cat_feature_cols": cat_feature_cols,
            "baseline_2way_score": blend2, "catboost_score": cat_score, "mlp_score": mlp_score,
        }, f)
    print(f"[base] 캐시 저장 완료: {CACHE_DIR}")


def load_cache():
    import pandas as pd
    train_split = pd.read_pickle(os.path.join(CACHE_DIR, "train_split.pkl"))
    val_split = pd.read_pickle(os.path.join(CACHE_DIR, "val_split.pkl"))
    npz = np.load(os.path.join(CACHE_DIR, "base_preds.npz"))
    with open(os.path.join(CACHE_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return train_split, val_split, npz["cat_val_preds"], npz["mlp_val_preds"], npz["y_val"], meta


if __name__ == "__main__":
    main()
