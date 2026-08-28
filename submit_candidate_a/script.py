# script.py -- candidate A(손수연 레시피+TrackA+F1) 전용 추론 스크립트
#
# CatBoost는 손수연 레시피(build_sooyun_features 계열, 8-categorical, 트랙맨 std5/gap4,
# season-1 앵커, TE-residual 6개) + F1필터 + 3-seed 배깅. MLP는 이 repo 기존 프로덕션
# (submit/model/final_retained_model.pkl)의 mlp_bundle을 손대지 않고 그대로 재사용한다
# (candidate A가 검증한 그대로 -- 재학습 없음). 2입력 로지스틱 메타모델로 블렌드.
#
# code/ 패키지에 의존하지 않고 완전히 독립 실행되도록 필요한 부분을 전부 인라인 복제했다
# (기존 submit/script.py와 동일한 관례).
import math
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ID_COL = "row_id"
TARGET_COL = "control_success"

# --- MLP 쪽(우리 기존 프로덕션, 손 안 댐) -- code/mlp_model.py 와 동일 ---
MLP_CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
]


class QuantileEmbedding(nn.Module):
    def __init__(self, bin_edges, d_embed=8, use_relu=True):
        super().__init__()
        self.register_buffer("edges", bin_edges)
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


def apply_mlp_preprocessing(df, cat_cols, num_cols, cat_encoder, num_imputer, num_scaler):
    df = df.copy()
    df[cat_cols] = df[cat_cols].astype(str)
    df[cat_cols] = cat_encoder.transform(df[cat_cols]) + 1
    df[num_cols] = num_imputer.transform(df[num_cols])
    df[num_cols] = num_scaler.transform(df[num_cols])
    return df


SEASON_PROGRESSION_SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
]


def add_season_progression(df, lookup):
    """code/train.py::apply_season_progression_features 와 수식 동일 (손수연 레시피의
    add_season_progress_features도 동일 계산 -- experiment_sooyun_recipe.py 문서 참고).
    CatBoost(손수연)/MLP(우리 기존) 양쪽에 이 하나의 함수+lookup을 공용으로 쓴다."""
    df = df.copy()
    for role, id_col, n_col, rate_col in SEASON_PROGRESSION_SPECS:
        lut = lookup.loc[lookup["role"] == role, ["id", "season", "end_n", "end_rate", "end_success"]]
        merged = df[[id_col, "season"]].merge(
            lut, left_on=[id_col, "season"], right_on=["id", "season"], how="left",
        )
        pre_n = (merged["end_n"] + 1).fillna(0).values
        pre_success = ((merged["end_n"] * merged["end_rate"]).round().fillna(0) + merged["end_success"].fillna(0)).values

        cum_n = df[n_col].values
        cum_success = np.round(df[n_col].values * df[rate_col].values)
        season_n = np.maximum(cum_n - pre_n, 0)
        season_success = np.maximum(cum_success - pre_success, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            season_rate = np.where(season_n > 0, season_success / season_n, np.nan)

        df[f"{role}_season_n"] = season_n
        df[f"{role}_season_success_count"] = season_success
        df[f"{role}_season_success_rate"] = season_rate
        df[f"{role}_season_rate_gap"] = season_rate - df[rate_col].values
    return df


def add_engineered_features_ours(df, league_success_mean):
    """code/train.py 와 동일(MLP 쪽 전용, 손수연 레시피와 무관)."""
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


def apply_same_hand_ours(df):
    df = df.copy()
    df['same_hand'] = (df['pitcher_hand'] == df['batter_hand']).astype(np.int64)
    df['same_hand_advantage'] = df['pitcher_relative_success'] * df['same_hand']
    return df


# --- CatBoost 쪽(손수연 레시피, candidate A) -- code/experiment_sooyun_recipe.py 와 동일 ---
SOOYUN_MATCH_COLS_6 = ["inning", "top_bottom", "balls_before", "strikes_before", "pitcher_hand", "batter_hand"]

SOOYUN_CAT_COLS = [
    "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "base_state", "game_type", "top_bottom",
    "hand_matchup",
]

TE_K_SMOOTH = 200
TE_AXES = [
    ("te_p_cnt", ["pitcher_id", "balls_before", "strikes_before"], "p_main"),
    ("te_p_bhand", ["pitcher_id", "batter_hand"], "p_main"),
    ("te_p_run", ["pitcher_id", "num_runners_on"], "p_main"),
    ("te_p_inn", ["pitcher_id", "inning"], "p_main"),
    ("te_b_cnt", ["batter_id", "balls_before", "strikes_before"], "b_main"),
]
TE_MAIN_AXES = [("p_main", ["pitcher_id"]), ("b_main", ["batter_id"])]
TE_RESIDUAL_COLS = [f"{name}_res" for name, *_ in TE_AXES] + ["te_covered"]


def add_merge_features(df, match_table, match_cols, time_col="season"):
    df = df.copy()
    n_before = len(df)
    df["match_season"] = df[time_col] - 1
    merged = df.merge(match_table, on=["match_season"] + match_cols, how="left")
    merged = merged.drop(columns=["match_season"])
    assert len(merged) == n_before, "merge로 행 개수가 변했습니다"
    return merged


def causal_smoothed_te_encode(source_df, query_df, group_cols, prior, k=TE_K_SMOOTH):
    agg = source_df.groupby(group_cols + ["season"])[TARGET_COL].agg(["sum", "count"]).reset_index()
    agg = agg.sort_values("season")
    agg["cum_sum"] = agg.groupby(group_cols)["sum"].cumsum()
    agg["cum_n"] = agg.groupby(group_cols)["count"].cumsum()
    agg = agg[group_cols + ["season", "cum_sum", "cum_n"]].sort_values("season")

    query = query_df[group_cols + ["season"]].reset_index()
    query_sorted = query.sort_values("season")
    merged = pd.merge_asof(
        query_sorted, agg, on="season", by=group_cols,
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.sort_values("index")
    cum_sum = merged["cum_sum"].fillna(0.0).values
    cum_n = merged["cum_n"].fillna(0.0).values
    enc = (cum_sum + prior * k) / (cum_n + k)
    covered = (cum_n > 0).astype(np.int64)
    return enc, covered


def apply_te_residual_features(source_df, query_df, prior):
    query_df = query_df.copy()
    mains = {}
    covered_any = np.zeros(len(query_df), dtype=np.int64)
    for name, group_cols in TE_MAIN_AXES:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        mains[name] = enc
        covered_any = np.maximum(covered_any, covered)
    for name, group_cols, main_key in TE_AXES:
        enc, covered = causal_smoothed_te_encode(source_df, query_df, group_cols, prior)
        query_df[f"{name}_res"] = enc - mains[main_key]
        covered_any = np.maximum(covered_any, covered)
    query_df["te_covered"] = covered_any
    return query_df


def build_sooyun_features_inference(df, league_mean_dict, season_end_lookup, std_table, gap_table):
    """code/experiment_sooyun_recipe.py::build_sooyun_features 와 동일 계산을, 학습
    데이터 자기참조 없이(모두 정적 lookup으로) test.csv 단일 행에도 안전하게 재현한다.
    df는 원본 그대로(top_bottom='T'/'B' 문자열, pitcher_id/batter_id 보존) 넘겨야 한다."""
    df = df.copy()
    df = df.drop(columns=[c for c in ["asof_pitcher_pitchmix_n"] if c in df.columns])

    df["prev_season_league_mean"] = df["season"].map(lambda s: league_mean_dict.get(int(s) - 1, np.nan))
    df["pitcher_relative_success"] = df["asof_pitcher_success_rate"] - df["prev_season_league_mean"]
    df["pitcher_recent1_gap"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["pitcher_recent3_gap"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]
    df["pitcher_recent5_gap"] = df["asof_pitcher_prev5_game_success_rate"] - df["asof_pitcher_success_rate"]

    df["count_diff"] = df["balls_before"] - df["strikes_before"]
    df["is_full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)
    df["pitcher_count_advantage_raw"] = df["asof_pitcher_success_rate"] * df["count_diff"]
    df["pitcher_count_advantage_rel"] = df["pitcher_relative_success"] * df["count_diff"]
    df["pitcher_trend"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]
    df["pitcher_consistency"] = (
        df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_prev5_game_success_rate"]
    ).abs()
    df["matchup"] = df["asof_pitcher_success_rate"] - df["asof_batter_success_rate"]
    df["count_pressure"] = df["pitcher_relative_success"] * df["is_full_count"]

    df["same_hand"] = (df["pitcher_hand"].astype(str) == df["batter_hand"].astype(str)).astype(int)
    df["same_hand_advantage"] = df["pitcher_relative_success"] * df["same_hand"]

    df = add_merge_features(df, std_table, SOOYUN_MATCH_COLS_6)
    df = add_merge_features(df, gap_table, SOOYUN_MATCH_COLS_6)

    df = add_season_progression(df, season_end_lookup)

    df["hand_matchup"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
    return df


def main():
    DATA_DIR = "./data"
    MODEL_DIR = "./model"
    MODEL_PATH = "./model/final_retained_model.pkl"
    OUTPUT_DIR = "./output"

    test_path = os.path.join(DATA_DIR, "test.csv")
    sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
    train_path = os.path.join(DATA_DIR, "train.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"필수 입력 파일이 없습니다: {test_path}")

    df_test_raw = pd.read_csv(test_path, encoding="utf-8-sig")
    df_sub = pd.read_csv(sample_sub_path, encoding="utf-8-sig")
    df_train_raw = pd.read_csv(train_path, encoding="utf-8-sig")

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    mlp_bundle = bundle["mlp_bundle"]

    # ---- MLP 쪽(우리 기존 프로덕션, top_bottom 0/1 매핑 후 처리) ----
    df_mlp = df_test_raw.copy()
    df_mlp["top_bottom"] = df_mlp["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    league_success_mean = df_train_raw[TARGET_COL].mean()

    season_end_lookup = pd.read_csv(os.path.join(MODEL_DIR, "season_end_lookup.csv"))
    df_mlp = add_season_progression(df_mlp, season_end_lookup)
    df_mlp = add_engineered_features_ours(df_mlp, league_success_mean)
    df_mlp = apply_same_hand_ours(df_mlp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_mlp_proc = apply_mlp_preprocessing(
        df_mlp, mlp_bundle["cat_cols"], mlp_bundle["num_cols"],
        mlp_bundle["cat_encoder"], mlp_bundle["num_imputer"], mlp_bundle["num_scaler"],
    )
    X_mlp_cat = torch.tensor(X_mlp_proc[mlp_bundle["cat_cols"]].values.astype(np.float32)).to(device)
    X_mlp_num = torch.tensor(X_mlp_proc[mlp_bundle["num_cols"]].values.astype(np.float32)).to(device)

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
            preds_list.append(model(X_mlp_cat, X_mlp_num).cpu().numpy())
    mlp_preds = np.mean(preds_list, axis=0)

    # ---- CatBoost 쪽(손수연 레시피, candidate A, top_bottom 문자열 그대로) ----
    std_table = pd.read_csv(os.path.join(MODEL_DIR, "trackman_match_table.csv"))
    gap_table = pd.read_csv(os.path.join(MODEL_DIR, "trackman_match_table_gap.csv"))
    league_mean_dict = bundle["league_mean_dict"]

    df_cb = build_sooyun_features_inference(df_test_raw, league_mean_dict, season_end_lookup, std_table, gap_table)

    te_source = pd.read_csv(os.path.join(MODEL_DIR, "te_source.csv"))
    te_prior = bundle["te_prior"]
    df_cb = apply_te_residual_features(te_source, df_cb, te_prior)

    df_cb = df_cb.drop(columns=["pitcher_id", "batter_id"])
    for c in SOOYUN_CAT_COLS:
        df_cb[c] = df_cb[c].astype(str)

    cat_feature_cols = bundle["cat_feature_cols"]
    X_cb = df_cb[cat_feature_cols]
    cat_preds = np.mean([m.predict_proba(X_cb)[:, 1] for m in bundle["catboost_models"]], axis=0)

    # ---- 블렌드 (2입력 로지스틱 메타모델) ----
    meta = bundle["meta_model"]
    z = meta["w_cat"] * cat_preds + meta["w_mlp"] * mlp_preds + meta["intercept"]
    preds = 1.0 / (1.0 + np.exp(-z))

    df_sub[TARGET_COL] = preds
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    df_sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"추론 및 제출용 파일 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
