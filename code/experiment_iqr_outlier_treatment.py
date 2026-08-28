# code/experiment_iqr_outlier_treatment.py
"""팀원(외부 AI) 제안: train.csv의 모든 수치형 피처에 대해 1.5xIQR 박스플롯 기준
"수염 밖" 이상치를 행 삭제 대신 (a) 클리핑/윈저라이징 (b) 이상치 플래그 파생피처
(c) 둘 다로 처리하면 리더보드 점수가 오른다는 제안을 검증한다. 사용자 확인 결과 이
피처 전부(모든 수치형 컬럼, *_id류 제외)에 대해 일괄 적용을 원함.

이상치 경계(Q1-1.5IQR, Q3+1.5IQR)는 TRAIN 스플릿(cutoff7, F1 필터 적용 후)만으로
계산하고 VAL에는 그대로(같은 경계로) 적용한다 — 추론 시점에 test.csv만 보고 새
경계를 계산할 수 없으므로(대회 규칙상 행별 독립 추론 원칙과도 부합), 학습 시점에
고정한 경계를 재사용하는 방식이어야 실전에서도 동일하게 재현 가능하다.

*_id로 끝나는 컬럼(pitcher_id, batter_id)은 이상치 개념 자체가 안 맞아 제외
(사용자 명시적 지시). CAT_COLS(범주형 원본 문자열 컬럼)도 당연히 대상이 아니다.

3가지 변형을 각각 CatBoost 풀 재학습 + MLP 3-seed 스크리닝으로 확인한다:
  - clip: 이상치 값을 경계값으로 클리핑(윈저라이징)
  - flag: 원본 값은 그대로 두고 컬럼별 outlier_flag_<feature> 이진 파생피처 추가
  - both: flag를 먼저 원본값 기준으로 계산한 뒤 클리핑도 적용

사용법: python -m code.experiment_iqr_outlier_treatment
"""
import numpy as np

from code.experiment_walk4_filter import build_split_with_rowid, run_one
from code.mlp_model import CAT_COLS

ID_SUFFIX = "_id"


def get_numeric_features(mlp_num_cols, cat_feature_cols):
    numeric = [c for c in cat_feature_cols if c not in CAT_COLS and not c.endswith(ID_SUFFIX)]
    return numeric


def compute_iqr_bounds(train_split, features):
    bounds = {}
    skipped = []
    for c in features:
        q1, q3 = train_split[c].quantile(0.25), train_split[c].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            skipped.append(c)
            continue
        bounds[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    if skipped:
        print(f"[IQR] IQR=0이라 대상에서 제외된 피처 {len(skipped)}개: {skipped}")
    return bounds


def apply_flags(df, bounds):
    df = df.copy()
    flag_cols = []
    for c, (lo, hi) in bounds.items():
        fc = f"outlier_flag_{c}"
        df[fc] = ((df[c] < lo) | (df[c] > hi)).astype(np.int64)
        flag_cols.append(fc)
    return df, flag_cols


def apply_clip(df, bounds):
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        df[c] = df[c].clip(lo, hi)
    return df


def main():
    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split_with_rowid(cutoff7=True)
    numeric_features = get_numeric_features(mlp_num_cols, cat_feature_cols)
    print(f"[IQR] 대상 수치형 피처 {len(numeric_features)}개 (전체 {len(cat_feature_cols)}개 중, "
          f"CAT_COLS/{ID_SUFFIX}류 제외)")

    bounds = compute_iqr_bounds(train_split, numeric_features)
    print(f"[IQR] 경계 계산된 피처 {len(bounds)}개")
    frac_outlier = {c: float(((train_split[c] < lo) | (train_split[c] > hi)).mean())
                     for c, (lo, hi) in bounds.items()}
    total_flagged_any = (
        sum(((train_split[c] < lo) | (train_split[c] > hi)) for c, (lo, hi) in bounds.items()) > 0
    ).mean()
    print(f"[IQR] 최소 1개 피처에서라도 이상치인 행 비율: {total_flagged_any:.2%}")

    results = {}

    # baseline은 code/experiment_walk4_filter.py의 REF_CAT/REF_MLP7/REF_BLEND
    # (동일 cutoff7/F1필터/피처셋) 재사용 — 아래 run_one 출력에서 자동 비교됨.

    # --- variant 1: clip only ---
    ts_clip = apply_clip(train_split, bounds)
    vs_clip = apply_clip(val_split, bounds)
    results["clip"] = run_one("IQR 클리핑만", ts_clip, vs_clip, mlp_num_cols, cat_feature_cols, set())

    # --- variant 2: flag only ---
    ts_flag, flag_cols = apply_flags(train_split, bounds)
    vs_flag, _ = apply_flags(val_split, bounds)
    mlp_num_cols_flag = mlp_num_cols + flag_cols
    cat_feature_cols_flag = cat_feature_cols + flag_cols
    print(f"[IQR] flag 변형: 피처 {len(flag_cols)}개 추가 -> mlp_num_cols {len(mlp_num_cols_flag)}, "
          f"cat_feature_cols {len(cat_feature_cols_flag)}")
    results["flag"] = run_one("IQR 플래그만", ts_flag, vs_flag, mlp_num_cols_flag, cat_feature_cols_flag, set())

    # --- variant 3: flag(원본 기준) + clip ---
    ts_both, flag_cols_b = apply_flags(train_split, bounds)
    vs_both, _ = apply_flags(val_split, bounds)
    ts_both = apply_clip(ts_both, bounds)
    vs_both = apply_clip(vs_both, bounds)
    mlp_num_cols_both = mlp_num_cols + flag_cols_b
    cat_feature_cols_both = cat_feature_cols + flag_cols_b
    results["both"] = run_one("IQR 플래그+클리핑", ts_both, vs_both, mlp_num_cols_both, cat_feature_cols_both, set())

    print(f"\n{'='*20} 요약 (baseline: CatBoost=706.56 MLP=738.55 blend=753.37) {'='*20}")
    for key, r in results.items():
        print(f"{key:8s}: CatBoost={r['cat_score']:.2f} (delta {r['cat_score']-706.56:+.2f}) | "
              f"MLP(3-seed)={r['mlp_score']:.2f} (delta {r['mlp_score']-738.55:+.2f}) | "
              f"blend={r['blend_score']:.2f} (delta {r['blend_score']-753.37:+.2f})")


if __name__ == "__main__":
    main()
