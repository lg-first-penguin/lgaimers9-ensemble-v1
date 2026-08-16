# code/train.py
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

from code.mlp_model import CAT_COLS, ENSEMBLE_SEEDS, QUANTILE_N_BINS, embed_dim_for_cardinality, fit_preprocessing, apply_preprocessing, fit_quantile_edges, to_tensors, train_ensemble, make_bundle, predict_bundle, get_device
from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model, make_blend_bundle

def apply_f1_filter(df):
    """2022년 이하 시즌의 game_type=='F'(퓨처스/2군) 행을 학습에서만 제거.

    2023년부터 F의 제구 성공률이 R(1군)보다 낮아지는 방향으로 관계가 역전됐는데
    (2022 이전: F가 R보다 최대 +20.5%p 높음 -> 2023~2024: 오히려 -3.0%p 낮음),
    game_type이 CatBoost 피처 중요도 1위라 이 역전이 학습을 크게 오염시킨다
    (season==2023 단일 홀드아웃 검증 시 스코어가 0에 가깝게 붕괴). 2023년 이후 F는
    새 관계가 유효하므로 남긴다 — 전량 제거는 검증에서 더 낮은 점수를 보였다.
    가중치로 희석하는 방식(2023 이후 F에 2배 가중)도 시도됐으나 조기 종료가 첫 트리에서
    멈추는 등 실패해, 오염 구간은 제거가 유일한 해법으로 확인됐다 (EXPERIMENTS.md 참고).
    검증/추론 데이터에는 적용하지 않는다 — 학습 데이터에만 적용한다."""
    before = len(df)
    filtered = df[~((df['game_type'] == 'F') & (df['season'] <= 2022))].reset_index(drop=True)
    print(f"[F1 필터] game_type=='F' & season<=2022 제거: {before} -> {len(filtered)}행 ({before - len(filtered)}행 제거)")
    return filtered

def add_engineered_features(df, league_success_mean):
    """asof_* 및 카운트 정보를 조합한 파생 피처를 추가합니다."""
    df = df.copy()

    df['pitcher_recent1_gap'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_success_rate']
    df['pitcher_recent3_gap'] = df['asof_pitcher_prev3_game_success_rate'] - df['asof_pitcher_success_rate']
    df['pitcher_recent5_gap'] = df['asof_pitcher_prev5_game_success_rate'] - df['asof_pitcher_success_rate']

    df['pitcher_relative_success'] = df['asof_pitcher_success_rate'] - league_success_mean

    df['count_diff'] = df['strikes_before'] - df['balls_before']
    df['is_full_count'] = ((df['balls_before'] == 3) & (df['strikes_before'] == 2)).astype(np.int64)

    df['pitcher_count_advantage_raw'] = df['asof_pitcher_success_rate'] * df['count_diff']
    df['pitcher_count_advantage_rel'] = df['pitcher_relative_success'] * df['count_diff']

    df['pitcher_trend'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_prev5_game_success_rate']
    df['pitcher_consistency'] = df[[
        'asof_pitcher_prev1_game_success_rate',
        'asof_pitcher_prev3_game_success_rate',
        'asof_pitcher_prev5_game_success_rate',
    ]].std(axis=1)

    df['matchup'] = df['asof_pitcher_success_rate'] - df['asof_batter_success_rate']

    pressure_signal = df['li'] * ((df['strikes_before'] >= 2) | (df['balls_before'] >= 3)).astype(np.int64)
    df['count_pressure'] = df['pitcher_relative_success'] * pressure_signal

    return df

def main():
    DATA_DIR = "./open/data"
    target_col = 'control_success'
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)

    train_df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # ABS(자동 볼 판정 시스템) 레짐 시프트 가설 검증(EXPERIMENTS.md §35) 결과 채택:
    # 2024시즌부터 KBO가 ABS를 전면 도입해 2019~2023(구 판정체계) 데이터만으로는
    # 2024/2025(신 판정체계)를 예측하기 어렵다는 가설을, 학습에 2024 초반(3~6월)을
    # 일부 포함시켜 검증했다. cutoff=7(7월부터 검증)이 여러 cutoff(4~10) 중 블렌드
    # 기준 가장 크고 신뢰할 만한 이득(+58.03)을 보여 채택. 가중치(sample_weight)는
    # 표본이 큰 cutoff에서 오히려 baseline보다 나빠 불채택(weight=1 유지).
    train_mask = (train_df['season'] < 2024) | ((train_df['season'] == 2024) & (train_df['game_month'] < 7))
    val_mask = (train_df['season'] == 2024) & (train_df['game_month'] >= 7)

    league_success_mean = train_df.loc[train_mask, target_col].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ['row_id', target_col]
    features = [col for col in train_df.columns if col not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS]

    train_split = train_df.loc[train_mask, features + [target_col]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [target_col]].reset_index(drop=True)
    train_split = apply_f1_filter(train_split)
    print(f"훈련 데이터 (2019~2023 + 2024 3~6월, F1 필터 적용): {len(train_split)} 행 | 검증 데이터 (2024 7~10월): {len(val_split)} 행")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, num_cols, target_col)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, num_cols, target_col)
    y_val_np = val_proc[target_col].values

    device = get_device()
    print(f"[Device] {device}")

    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    bin_edges = fit_quantile_edges(X_tr_num, n_bins=QUANTILE_N_BINS)
    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=y_val_np,
        seeds=ENSEMBLE_SEEDS, device=device,
    )

    mlp_bundle = make_bundle(
        members, CAT_COLS, num_cols, cat_dims, embed_dims,
        cat_encoder, num_imputer, num_scaler, bin_edges=bin_edges,
    )

    print("\n--- [CatBoost] 블렌딩용 CatBoost 모델 학습 ---")
    X_train_raw, y_train_raw = train_split[features], train_split[target_col].values
    X_val_raw, y_val_raw = val_split[features], val_split[target_col].values
    catboost_model, catboost_best_iteration = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=True)
    print(f"[CatBoost] 학습 완료 (best_iteration={catboost_best_iteration})")

    mlp_val_preds = predict_bundle(mlp_bundle, X_val_raw, device=device)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    w_cat, w_mlp, intercept, blend_score, blend_brier = fit_meta_model(cat_val_preds, mlp_val_preds, y_val_raw)
    meta_model = {"w_cat": w_cat, "w_mlp": w_mlp, "intercept": intercept}
    print(f"[Blend] 스태킹 메타모델 w_cat={w_cat:.3f} w_mlp={w_mlp:.3f} intercept={intercept:.3f} | 블렌드 Val Score: {blend_score:.2f}")

    bundle = make_blend_bundle(catboost_model, mlp_bundle, meta_model)
    bundle["catboost_best_iteration"] = catboost_best_iteration

    os.makedirs("./open/temp", exist_ok=True)
    with open("./open/temp/latest_model.pkl", 'wb') as f:
        pickle.dump(bundle, f)
    print(f"✅ Model saved to ./open/temp/latest_model.pkl (mlp best_epoch_avg={mlp_bundle['best_epoch_']}, catboost best_iteration={catboost_best_iteration}, meta_model={meta_model})")

if __name__ == "__main__":
    main()
