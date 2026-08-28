# code/experiment_std_outlier_treatment.py
"""code/experiment_iqr_outlier_treatment.py의 후속 — 이상치 경계를 1.5xIQR 대신
평균 ± k*표준편차(k=2, k=3)로 계산해 동일하게 clip/flag/both 3가지를 테스트한다.
피처 대상(수치형 전체, CAT_COLS/*_id 제외)과 clip/flag/both 정의, train만으로 경계를
계산해 val에 동일 적용하는 방식은 IQR 버전과 완전히 동일 — 경계 산출 방법만 다르다.

사용법: python -m code.experiment_std_outlier_treatment
"""
import numpy as np

from code.experiment_walk4_filter import build_split_with_rowid, run_one
from code.experiment_iqr_outlier_treatment import apply_clip, apply_flags, get_numeric_features


def compute_std_bounds(train_split, features, k):
    bounds = {}
    skipped = []
    for c in features:
        mean, std = train_split[c].mean(), train_split[c].std()
        if std == 0 or np.isnan(std):
            skipped.append(c)
            continue
        bounds[c] = (mean - k * std, mean + k * std)
    if skipped:
        print(f"[STD k={k}] std=0(또는 NaN)이라 대상에서 제외된 피처 {len(skipped)}개: {skipped}")
    return bounds


def run_k(k, train_split, val_split, mlp_num_cols, cat_feature_cols):
    print(f"\n{'#'*15} k={k} (평균±{k}*표준편차) {'#'*15}")
    bounds = compute_std_bounds(train_split, get_numeric_features(mlp_num_cols, cat_feature_cols), k)
    print(f"[STD k={k}] 경계 계산된 피처 {len(bounds)}개")

    results = {}

    ts_clip = apply_clip(train_split, bounds)
    vs_clip = apply_clip(val_split, bounds)
    results["clip"] = run_one(f"STD(k={k}) 클리핑만", ts_clip, vs_clip, mlp_num_cols, cat_feature_cols, set())

    ts_flag, flag_cols = apply_flags(train_split, bounds)
    vs_flag, _ = apply_flags(val_split, bounds)
    mlp_num_cols_flag = mlp_num_cols + flag_cols
    cat_feature_cols_flag = cat_feature_cols + flag_cols
    results["flag"] = run_one(f"STD(k={k}) 플래그만", ts_flag, vs_flag, mlp_num_cols_flag, cat_feature_cols_flag, set())

    ts_both, flag_cols_b = apply_flags(train_split, bounds)
    vs_both, _ = apply_flags(val_split, bounds)
    ts_both = apply_clip(ts_both, bounds)
    vs_both = apply_clip(vs_both, bounds)
    mlp_num_cols_both = mlp_num_cols + flag_cols_b
    cat_feature_cols_both = cat_feature_cols + flag_cols_b
    results["both"] = run_one(f"STD(k={k}) 플래그+클리핑", ts_both, vs_both, mlp_num_cols_both, cat_feature_cols_both, set())

    return results


def main():
    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split_with_rowid(cutoff7=True)

    all_results = {}
    for k in (2, 3):
        all_results[k] = run_k(k, train_split, val_split, mlp_num_cols, cat_feature_cols)

    print(f"\n{'='*20} 요약 (baseline: CatBoost=706.56 MLP=738.55 blend=753.37) {'='*20}")
    for k, results in all_results.items():
        for key, r in results.items():
            print(f"k={k} {key:8s}: CatBoost={r['cat_score']:.2f} (delta {r['cat_score']-706.56:+.2f}) | "
                  f"MLP(3-seed)={r['mlp_score']:.2f} (delta {r['mlp_score']-738.55:+.2f}) | "
                  f"blend={r['blend_score']:.2f} (delta {r['blend_score']-753.37:+.2f})")


if __name__ == "__main__":
    main()
