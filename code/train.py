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

def process_trackman_features_safe(df_main, df_trm, is_train_split=True):
    """타임 리크가 차단된 10-Key 상황 지문 기반 트랙맨 전처리 결합 엔진"""
    df_main_copy = df_main.copy()
    df_trm_copy = df_trm.copy()

    if is_train_split:
        max_season = df_main_copy['season'].max()
        max_month = df_main_copy[df_main_copy['season'] == max_season]['game_month'].max()
        future_mask = (df_trm_copy['season'] > max_season) | \
                      ((df_trm_copy['season'] == max_season) & (df_trm_copy['game_month'] > max_month))
        df_trm_copy = df_trm_copy[~future_mask].reset_index(drop=True)
        print(f"⏳ [Time Filter] {max_season}년 {max_month}월 이전의 트랙맨 데이터만 잘라내어 피처를 산출합니다.")

    match_cols = [dfc for dfc in df_main_copy.columns if (dfc in df_trm_copy.columns) and dfc != 'row_id']

    grouped = df_trm_copy.groupby(match_cols + ['pitch_type_group', 'auto_pitch_type'])
    grouped_phase1 = grouped[['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break', 'extension', 'rel_height', 'rel_side', 'zone_speed']].agg(['mean', 'std'])
    grouped_phase2 = grouped_phase1.reset_index()

    std_cols = [col for col in grouped_phase2.columns if 'std' in col]
    grouped_phase2[std_cols] = grouped_phase2[std_cols].fillna(0)
    grouped_phase2.columns = ['_'.join(col).strip('_') for col in grouped_phase2.columns]

    grouped_phase3 = grouped_phase2.drop(columns='auto_pitch_type')
    grouped_phase3 = grouped_phase3.groupby(match_cols + ['pitch_type_group']).agg(['mean'])

    pivoted = grouped_phase3.unstack(level='pitch_type_group')
    pivoted.columns = [f"{col[0]}_{col[1]}_{col[2]}" for col in pivoted.columns]
    tm_final = pivoted.reset_index()
    tm_final = tm_final.fillna(0)

    df_main_copy['top_bottom'] = df_main_copy['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    tm_final['top_bottom'] = tm_final['top_bottom'].map({'Top': 0, 'Bottom': 1}).astype(np.int64)

    for col in ['batter_hand', 'pitcher_hand']:
        if col in tm_final.columns:
            tm_final[col] = tm_final[col].map({'Left': 1, 'Right': 2}).astype(np.int64)

    tr_final = pd.merge(df_main_copy, tm_final, on=match_cols, how='left')
    new_feature_cols = [col for col in tm_final.columns if col not in match_cols]
    tr_final[new_feature_cols] = tr_final[new_feature_cols].fillna(tr_final[new_feature_cols].mean())

    return tr_final, match_cols

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
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"))

    tr_final, match_cols = process_trackman_features_safe(df, df_trm, is_train_split=True)

    train_df = tr_final.dropna(subset=[target_col]).reset_index(drop=True)

    train_mask = train_df['season'] < 2024
    val_mask = train_df['season'] == 2024

    league_success_mean = train_df.loc[train_mask, target_col].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = ['row_id', target_col]
    features = [col for col in train_df.columns if col not in drop_cols]
    num_cols = [c for c in features if c not in CAT_COLS]

    train_split = train_df.loc[train_mask, features + [target_col]].reset_index(drop=True)
    val_split = train_df.loc[val_mask, features + [target_col]].reset_index(drop=True)
    print(f"훈련 데이터 (2019~2023): {len(train_split)} 행 | 검증 데이터 (2024): {len(val_split)} 행")

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
