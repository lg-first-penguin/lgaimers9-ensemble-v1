# script.py
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ID_COL = "row_id"
TARGET_COL = "control_success"
# code/mlp_model.py::CAT_COLS 와 동일 (수동 동기화 유지)
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
]


class TabularMLP(nn.Module):
    """code/mlp_model.py::TabularMLP 와 동일한 구조 (submit.zip에는 code/ 패키지가
    포함되지 않으므로 대회 서버에서 독립 실행 가능하도록 그대로 복제해 둡니다)."""

    def __init__(self, num_numeric_feats, cat_dims, embed_dims, hidden1=128, hidden2=64, dropout=0.3):
        super().__init__()
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        total_input_dim = sum(self.embed_dims) + num_numeric_feats

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        embeds = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embeds, dim=1)
        x_all = torch.cat([x_embed, x_num], dim=1)
        return self.sigmoid(self.mlp(x_all)).squeeze(-1)


def add_engineered_features(df, league_success_mean):
    """asof_* 및 카운트 정보를 조합한 파생 피처를 추가합니다. (code/train.py와 수동 동기화 유지)"""
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


def apply_preprocessing(df, cat_cols, num_cols, cat_encoder, num_imputer, num_scaler):
    """code/mlp_model.py::apply_preprocessing 와 동일 로직 (수동 동기화 유지)"""
    df = df.copy()
    df[cat_cols] = df[cat_cols].astype(str)
    df[cat_cols] = cat_encoder.transform(df[cat_cols]) + 1
    df[num_cols] = num_imputer.transform(df[num_cols])
    df[num_cols] = num_scaler.transform(df[num_cols])
    return df


def main():
    # 대회 서빙 환경 표준 경로 정의
    DATA_DIR = "./data"
    MODEL_PATH = "./model/final_retained_model.pkl"
    OUTPUT_DIR = "./output"

    # 1. 필수 입력 데이터 로드
    test_path = os.path.join(DATA_DIR, "test.csv")
    sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
    trackman_path = os.path.join(DATA_DIR, "trackman_history.csv")
    train_path = os.path.join(DATA_DIR, "train.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ 필수 입력 파일이 없습니다: {test_path}")

    df_test = pd.read_csv(test_path, encoding="utf-8-sig")
    df_sub = pd.read_csv(sample_sub_path, encoding="utf-8-sig")
    df_trm = pd.read_csv(trackman_path, encoding="utf-8-sig")

    # final_retained_model.pkl과 동일하게, 전체 train.csv 기준 리그 평균 성공률 계산
    df_train_raw = pd.read_csv(train_path, encoding="utf-8-sig")
    league_success_mean = df_train_raw[TARGET_COL].mean()

    # 2. 저장된 최종 통합 완습 모델 번들 로드
    # (dict: {"catboost_model": CatBoostClassifier, "mlp_bundle": {...}, "alpha": float})
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 제출 구조 내 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    mlp_bundle = bundle["mlp_bundle"]

    # 3. 파이프라인 무결성 유지를 위한 트랙맨 전처리 수행 (train.py의 전처리 구조를 직접 복제)
    df_main_copy = df_test.copy()
    df_trm_copy = df_trm.copy()

    # 전처리 결합 기준 컬럼 추출
    match_cols = [dfc for dfc in df_main_copy.columns if (dfc in df_trm_copy.columns) and dfc != 'row_id']

    # 과거 트랙맨 로그 기반 통계량 산출 (추론 시점이므로 과거 전체 데이터 활용)
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

    # 데이터 타입 정밀 매칭 및 인코딩
    df_main_copy['top_bottom'] = df_main_copy['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
    tm_final['top_bottom'] = tm_final['top_bottom'].map({'Top': 0, 'Bottom': 1}).astype(np.int64)

    for col in ['batter_hand', 'pitcher_hand']:
        if col in tm_final.columns:
            tm_final[col] = tm_final[col].map({'Left': 1, 'Right': 2}).astype(np.int64)

    tr_final = pd.merge(df_main_copy, tm_final, on=match_cols, how='left')
    new_feature_cols = [col for col in tm_final.columns if col not in match_cols]

    # 훈련 시점 피처 통계 기반 결측치 보정 (추론 시점의 결측값은 0으로 예외 처리 방어 조치)
    tr_final[new_feature_cols] = tr_final[new_feature_cols].fillna(0)

    # 3.5 asof_* 및 카운트 정보를 조합한 파생 피처 추가 (code/train.py와 동일 정의)
    tr_final = add_engineered_features(tr_final, league_success_mean)

    # 4. 모델 입력 데이터 정렬
    drop_cols = [ID_COL, TARGET_COL]
    features = [col for col in tr_final.columns if col not in drop_cols]
    X_test = tr_final[features]

    # 5. CatBoost + Tabular MLP 앙상블 블렌드 확률 추론 수행
    # 5a. CatBoost (원본 dtype 그대로 입력 — game_type/base_state는 문자열로 자체 처리)
    cat_preds = bundle["catboost_model"].predict_proba(X_test)[:, 1]

    # 5b. Tabular MLP 앙상블 (여러 시드로 학습된 멤버들의 예측 평균)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_proc = apply_preprocessing(
        X_test, mlp_bundle["cat_cols"], mlp_bundle["num_cols"],
        mlp_bundle["cat_encoder"], mlp_bundle["num_imputer"], mlp_bundle["num_scaler"],
    )
    X_cat = torch.tensor(X_proc[mlp_bundle["cat_cols"]].values.astype(np.float32)).to(device)
    X_num = torch.tensor(X_proc[mlp_bundle["num_cols"]].values.astype(np.float32)).to(device)

    preds_list = []
    with torch.no_grad():
        for member in mlp_bundle["members"]:
            model = TabularMLP(
                num_numeric_feats=len(mlp_bundle["num_cols"]),
                cat_dims=mlp_bundle["cat_dims"],
                embed_dims=mlp_bundle["embed_dims"],
            ).to(device)
            model.load_state_dict(member["state_dict"])
            model.eval()
            preds_list.append(model(X_cat, X_num).cpu().numpy())
    mlp_preds = np.mean(preds_list, axis=0)

    # 5c. alpha * CatBoost + (1-alpha) * MLP 앙상블 가중 평균 블렌드
    alpha = bundle["alpha"]
    preds = alpha * cat_preds + (1 - alpha) * mlp_preds

    # 6. 제출 서식 동기화 및 저장
    df_sub[TARGET_COL] = preds

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    df_sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ 추론 및 제출용 파일 저장 완료: {out_path}")

if __name__ == "__main__":
    main()
