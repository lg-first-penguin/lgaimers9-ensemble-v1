# script.py
import math
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


class QuantileEmbedding(nn.Module):
    """code/mlp_model.py::QuantileEmbedding 와 동일 (수동 동기화 유지). 수치형 피처별
    quantile 기반 piecewise-linear 인코딩(PLE) + 피처별 독립 Linear + ReLU."""

    def __init__(self, bin_edges, d_embed=8, use_relu=True):
        super().__init__()
        self.register_buffer("edges", bin_edges)  # (num_numeric, n_bins+1), 학습 안 함
        num_numeric, n_bins_plus1 = bin_edges.shape
        n_bins = n_bins_plus1 - 1
        self.num_numeric = num_numeric
        self.n_bins = n_bins
        self.use_relu = use_relu
        self.weight = nn.Parameter(torch.empty(num_numeric, n_bins, d_embed))
        self.bias = nn.Parameter(torch.zeros(num_numeric, d_embed))
        bound = 1.0 / math.sqrt(n_bins)
        nn.init.uniform_(self.weight, -bound, bound)

    def encode(self, x_num):
        left = self.edges[:, :-1].unsqueeze(0)
        right = self.edges[:, 1:].unsqueeze(0)
        x = x_num.unsqueeze(-1)
        width = (right - left).clamp_min(1e-6)
        frac = (x - left) / width
        return frac.clamp(0.0, 1.0)

    def forward(self, x_num):
        p = self.encode(x_num)
        e = torch.einsum("bnf,nfd->bnd", p, self.weight) + self.bias
        if self.use_relu:
            e = torch.relu(e)
        return e.reshape(e.shape[0], -1)


class TabularMLP(nn.Module):
    """code/mlp_model.py::TabularMLP 와 동일한 구조 (submit.zip에는 code/ 패키지가
    포함되지 않으므로 대회 서버에서 독립 실행 가능하도록 그대로 복제해 둡니다).
    수치형 입력은 QuantileEmbedding(PLE, n_bins=24, d=8 — EXPERIMENTS.md §15.3)으로
    인코딩합니다."""

    def __init__(self, cat_dims, embed_dims, bin_edges, quantile_d=8, hidden1=128, hidden2=64, dropout=0.3):
        super().__init__()
        self.embed_dims = list(embed_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 2, embedding_dim=edim) for dim, edim in zip(cat_dims, self.embed_dims)
        ])
        self.quantile = QuantileEmbedding(bin_edges, d_embed=quantile_d)
        total_input_dim = sum(self.embed_dims) + bin_edges.shape[0] * quantile_d

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
        x_quantile = self.quantile(x_num)
        x_all = torch.cat([x_embed, x_quantile], dim=1)
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
    train_path = os.path.join(DATA_DIR, "train.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ 필수 입력 파일이 없습니다: {test_path}")

    df_test = pd.read_csv(test_path, encoding="utf-8-sig")
    df_sub = pd.read_csv(sample_sub_path, encoding="utf-8-sig")
    df_test['top_bottom'] = df_test['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)

    # final_retained_model.pkl과 동일하게, 전체 train.csv 기준 리그 평균 성공률 계산
    df_train_raw = pd.read_csv(train_path, encoding="utf-8-sig")
    league_success_mean = df_train_raw[TARGET_COL].mean()

    # 2. 저장된 최종 통합 완습 모델 번들 로드
    # (dict: {"catboost_model": CatBoostClassifier, "mlp_bundle": {...}, "meta_model": {"w_cat", "w_mlp", "intercept"}})
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 제출 구조 내 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    mlp_bundle = bundle["mlp_bundle"]

    # 3. asof_* 및 카운트 정보를 조합한 파생 피처 추가 (code/train.py와 동일 정의)
    tr_final = add_engineered_features(df_test, league_success_mean)

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
                cat_dims=mlp_bundle["cat_dims"],
                embed_dims=mlp_bundle["embed_dims"],
                bin_edges=mlp_bundle["bin_edges"],
                quantile_d=mlp_bundle.get("quantile_d", 8),
            ).to(device)
            model.load_state_dict(member["state_dict"])
            model.eval()
            preds_list.append(model(X_cat, X_num).cpu().numpy())
    mlp_preds = np.mean(preds_list, axis=0)

    # 5c. 스태킹 메타모델(로지스틱 회귀) 기반 CatBoost+MLP 비선형 결합
    # code/blend_model.py::predict_meta 와 동일 로직 (수동 동기화 유지)
    meta = bundle["meta_model"]
    z = meta["w_cat"] * cat_preds + meta["w_mlp"] * mlp_preds + meta["intercept"]
    preds = 1.0 / (1.0 + np.exp(-z))

    # 6. 제출 서식 동기화 및 저장
    df_sub[TARGET_COL] = preds

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    df_sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ 추론 및 제출용 파일 저장 완료: {out_path}")

if __name__ == "__main__":
    main()
