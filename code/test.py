# code/test.py
import sys
import os

# [조립 핵심 지점] 실행 디렉토리 위치 독립 무결성 보정식 주입
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pickle
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

# 이제 상위 루트 디렉토리가 시스템 패스에 잡혀있으므로 완벽하게 import 성공합니다.
from code.train import add_engineered_features
from code.mlp_model import compute_bss
from code.blend_model import predict_blend_bundle

def calculate_bss(bundle, X_val, y_val):
    """안전장치가 적용된 BSS 평가 루틴 (CatBoost+MLP 블렌드 번들 기준)"""
    preds = predict_blend_bundle(bundle, X_val)
    brier, bss, score = compute_bss(preds, y_val)
    return bss, score

def main():
    DATA_DIR = "./open/data"
    TEMP_MODEL_PATH = "./open/temp/latest_model.pkl"
    REF_MODEL_PATH = "./open/reference/best_model.pkl"

    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)

    train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # train.py와 동일하게 train-split(season<2024) 기준 리그 평균으로 파생 피처 산출
    train_mask = train_df['season'] < 2024
    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    features = [col for col in train_df.columns if col not in [ID_COL, TARGET_COL]]

    # 2024년 데이터는 학습에서 완전히 제외하고 검증셋으로만 사용
    val_split = train_df[train_df['season'] == 2024].reset_index(drop=True)

    X_val, y_val = val_split[features], val_split[TARGET_COL].values

    # 1. 신규 최신 임시 버퍼 모델 검증
    with open(TEMP_MODEL_PATH, 'rb') as f:
        latest_bundle = pickle.load(f)
    latest_bss, latest_score = calculate_bss(latest_bundle, X_val, y_val)

    print("\n" + "="*60)
    print(f"{'[CatBoost + MLP 블렌드 검증 모델 성능 리포트]':^50}")
    print(f" Brier Skill Score (BSS): {latest_bss:.5f}")
    print(f" 대회 환산 예측 점수   : {latest_score:.2f}")
    print(f" Blend 메타모델: {latest_bundle.get('meta_model')}")
    print("="*60)

    # 2. 기존 최고 Reference 모델과 성능 대조
    ref_bss = -float('inf')
    if os.path.exists(REF_MODEL_PATH):
        with open(REF_MODEL_PATH, 'rb') as f:
            ref_bundle = pickle.load(f)
        if isinstance(ref_bundle, dict) and "catboost_model" in ref_bundle and "mlp_bundle" in ref_bundle and "meta_model" in ref_bundle:
            try:
                ref_bss, ref_score = calculate_bss(ref_bundle, X_val, y_val)
                print(f" ➔ 기존 최고 Reference 모델 BSS: {ref_bss:.5f} (점수: {ref_score:.2f})")
            except Exception as e:
                # 번들 포맷(dict 키)은 현재 스태킹 구조와 같아도, 그 안의 CatBoost/MLP가
                # 기대하는 피처 스키마(컬럼 구성)가 이번 실행과 다를 수 있습니다 — 예:
                # 트랙맨 피처 제거/F1 필터 도입처럼 학습 피처 집합 자체가 바뀐 경우.
                # 이런 스키마 불일치는 예측 단계에서 예외로 드러나므로, 위의 legacy 포맷
                # 분기와 동일하게 "비교 불가 -> 신규 모델 채택"으로 처리합니다.
                ref_bss = -float('inf')
                print(f" ⚠ 기존 Reference 모델의 피처 스키마가 이번 실행과 호환되지 않습니다 ({e}). 비교를 건너뛰고 신규 모델을 채택합니다.")
        else:
            # CatBoost 단독/MLP 단독 시절의 레거시 번들, 또는 "alpha" 가중평균 시절의
            # 구 블렌드 번들("meta_model" 키가 없는 버전)은 현재 스태킹 번들 포맷과
            # 호환되지 않으므로 비교 없이 무시합니다. 이번 실행의 신규 모델이 그대로
            # NEW_BEST 로 승격되며, dopip.py가 기존 파일을 open/former_model/ 로 자동 백업합니다.
            print(" ⚠ 기존 Reference 모델이 현재 스태킹 번들 포맷이 아닙니다 (구버전 MLP/CatBoost 단독 또는 alpha 블렌드 레거시로 추정). 비교를 건너뛰고 신규 모델을 채택합니다.")

    with open("./open/temp/compare_result.txt", "w") as f:
        if latest_bss > ref_bss:
            f.write("NEW_BEST")
        else:
            f.write("KEEP_REF")
    print("✅ 검증 세트 스코어 비교 대조록 갱신 성공.")

if __name__ == "__main__":
    main()
