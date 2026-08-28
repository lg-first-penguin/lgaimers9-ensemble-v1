# code/test.py
import sys
import os

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

# 2026-08-27: code/train.py를 유담님 파이프라인 기반으로 전면 교체하면서 이 파일도
# 동일한 피처 빌드 경로로 다시 작성했다 (build_split 로직이 train.py와 반드시
# 일치해야 val_split을 정확히 재현할 수 있다 -- 이 파일은 latest_model.pkl을
# train.py가 학습한 것과 동일한 val_split으로 재평가해 reference와 비교한다).
from code.train import (
    process_trackman_features_safe, add_engineered_features, apply_f1_filter,
    apply_te_residual_features, TE_RESIDUAL_COLS, apply_same_hand, SAME_HAND_COLS,
    is_trackman64,
)
from code.mlp_model import compute_bss
from code.blend_model import predict_blend_bundle
from code.trackman_pitcher_features import merge_coarse_pitchmix


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
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")

    tr_final, _, _ = process_trackman_features_safe(df, df_trm, is_train_split=True)
    train_df = tr_final.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # train.py와 동일한 cutoff=7 분할.
    train_mask = (train_df['season'] < 2024) | ((train_df['season'] == 2024) & (train_df['game_month'] < 7))
    val_mask = (train_df['season'] == 2024) & (train_df['game_month'] >= 7)

    league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)
    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=2024)
    train_df = apply_same_hand(train_df)

    # 트랙맨64 제거 (code/train.py 와 동일 — 실전 1117.03 레시피).
    features = [col for col in train_df.columns
               if col not in [ID_COL, TARGET_COL] and not is_trackman64(col)]

    train_split = train_df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)

    te_prior = train_split[TARGET_COL].mean()
    val_split = apply_te_residual_features(train_split, val_split, te_prior)

    X_val, y_val = val_split[features + TE_RESIDUAL_COLS], val_split[TARGET_COL].values

    with open(TEMP_MODEL_PATH, 'rb') as f:
        latest_bundle = pickle.load(f)
    latest_bss, latest_score = calculate_bss(latest_bundle, X_val, y_val)

    print("\n" + "="*60)
    print(f"{'[CatBoost + MLP 블렌드 검증 모델 성능 리포트]':^50}")
    print(f" Brier Skill Score (BSS): {latest_bss:.5f}")
    print(f" 대회 환산 예측 점수   : {latest_score:.2f}")
    print(f" Blend 메타모델: {latest_bundle.get('meta_model')}")
    print("="*60)

    ref_bss = -float('inf')
    if os.path.exists(REF_MODEL_PATH):
        with open(REF_MODEL_PATH, 'rb') as f:
            ref_bundle = pickle.load(f)
        if isinstance(ref_bundle, dict) and "catboost_model" in ref_bundle and "mlp_bundle" in ref_bundle and "meta_model" in ref_bundle:
            try:
                ref_bss, ref_score = calculate_bss(ref_bundle, X_val, y_val)
                print(f" ➔ 기존 최고 Reference 모델 BSS: {ref_bss:.5f} (점수: {ref_score:.2f})")
            except Exception as e:
                ref_bss = -float('inf')
                print(f" ⚠ 기존 Reference 모델의 피처 스키마가 이번 실행과 호환되지 않습니다 ({e}). 비교를 건너뛰고 신규 모델을 채택합니다.")
        else:
            print(" ⚠ 기존 Reference 모델이 현재 스태킹 번들 포맷이 아닙니다. 비교를 건너뛰고 신규 모델을 채택합니다.")

    with open("./open/temp/compare_result.txt", "w") as f:
        if latest_bss > ref_bss:
            f.write("NEW_BEST")
        else:
            f.write("KEEP_REF")
    print("✅ 검증 세트 스코어 비교 대조록 갱신 성공.")


if __name__ == "__main__":
    main()
