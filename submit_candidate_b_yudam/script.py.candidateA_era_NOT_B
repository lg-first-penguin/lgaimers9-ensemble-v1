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


# ---- 트랙맨 pitcher-std 피처 (code/trackman_pitcher_features.py 와 동일, 수동 동기화 유지) ----
# code/ 패키지 없이 독립 실행되어야 하므로 인라인 복제. 2026-08-17 세션 결론: tier A(투수x
# 구종군)->MLP만 채택, tier B(+압박)/C(+타자손)는 재검증 결과 기각(모듈 docstring 참고).
# 2026-08-18 세션: A+pitchmix 조합이 실전 950.81(-31.41)로 확인되어 tier A를 뺀
# pitchmix-only로 전환 (code/train.py의 TRACKMAN_TIER_FEED 주석 참고).
TRACKMAN_METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
                     "extension", "rel_height", "rel_side", "zone_speed"]
TRACKMAN_TIER_SPECS = {
    "a": ([], "trkstdA_"),
}
TRACKMAN_TIER_FEED = {}

# coarse pitchmix(볼카운트x손 조합별 구종 비중, ->CatBoost). dopip.py Full Retrain 단계에서
# 미리 계산해 model/pitchmix_lookup.csv로 동봉해 두므로(pitcher_map.csv와 동일한 관례),
# 추론 시에는 재계산 없이 그대로 병합만 한다.
COARSE_COLS = ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]
PITCHMIX_COLS = ["coarse_pitchmix_fastball", "coarse_pitchmix_breaking",
                  "coarse_pitchmix_offspeed", "coarse_pitchmix_other"]


def clean_trackman(df):
    # extension/zone_speed/rel_speed 결측(NaN) 행을 `x>0`/`x<=y` 비교의 False 부작용으로
    # 통째로 삭제하지 않도록 결측은 통과시킨다(2026-08-17 버그 수정, code/trackman_pitcher_
    # features.py::clean_trackman 참고).
    mask = (
        (df["inning"] >= 1)
        & df["balls_before"].between(0, 3)
        & df["strikes_before"].between(0, 2)
        & df["outs_before"].between(0, 2)
        & (df["extension"].isna() | (df["extension"] > 0))
        & (df["zone_speed"].isna() | df["rel_speed"].isna() | (df["zone_speed"] <= df["rel_speed"]))
    )
    df = df[mask].copy()
    dup_cols = [c for c in df.columns if c != "trackman_id"]
    df = df.drop_duplicates(subset=dup_cols)

    hand_counts = df.groupby(["pitcher_trackman_id", "pitcher_hand"]).size().reset_index(name="n")
    majority_hand = hand_counts.sort_values("n", ascending=False).drop_duplicates("pitcher_trackman_id")
    df = df.merge(majority_hand[["pitcher_trackman_id", "pitcher_hand"]],
                  on=["pitcher_trackman_id", "pitcher_hand"], how="inner")
    return df


def add_coarse_pitchmix(df_test, model_dir):
    """model/pitchmix_lookup.csv(Full Retrain 시 트랙맨 전체로 미리 계산된 정적 테이블)를
    볼카운트x손 조합으로 병합한다. df_test의 pitcher_hand/batter_hand는 이미 train.csv와
    같은 정수코드(1/2)라 별도 매핑 없이 바로 조인된다. 결측 조합(사실상 없음 — 179만 트랙맨
    행에서 나온 조합 전체를 담고 있음)은 테이블 전체 평균으로 채운다."""
    lookup = pd.read_csv(os.path.join(model_dir, "pitchmix_lookup.csv"))
    fallback = lookup[PITCHMIX_COLS].mean()
    merged = pd.merge(df_test, lookup, on=COARSE_COLS, how="left")
    for c in PITCHMIX_COLS:
        merged[c] = merged[c].fillna(fallback[c])
    return merged


def build_pitcher_lookup(df_trm_clean, pitcher_map, tier):
    merged = df_trm_clean.merge(pitcher_map[["pitcher_trackman_id", "pitcher_id"]],
                                 on="pitcher_trackman_id", how="inner")
    extra_keys, prefix = TRACKMAN_TIER_SPECS[tier]
    if "pressure" in extra_keys:
        merged["pressure"] = ((merged["balls_before"] >= 3) | (merged["strikes_before"] >= 2)).astype(np.int64)
    pivot_levels = extra_keys + ["pitch_type_group"]
    group_cols = ["pitcher_id"] + pivot_levels

    g = merged.groupby(group_cols)[TRACKMAN_METRICS].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    std_cols = [c for c in g.columns if c.endswith("_std")]
    g[std_cols] = g[std_cols].fillna(0.0)

    pivoted = g.set_index(group_cols).unstack(level=pivot_levels)
    pivoted.columns = ["_".join(str(x) for x in c) for c in pivoted.columns]
    pivoted = pivoted.reset_index().fillna(0.0)
    rename = {c: prefix + c for c in pivoted.columns if c != "pitcher_id"}
    return pivoted.rename(columns=rename)


def add_trackman_features(df_test, data_dir, model_dir):
    """실전 추론 전용: test 시즌(2025)은 트랙맨 파일에 아예 없으므로 asof 루프 없이 전체
    트랙맨(2019~2024)으로 만든 lookup 하나만 붙이면 된다(look-ahead 걱정이 구조적으로 없음).
    tier(투수 정체성 크로스워크 기반)와 coarse pitchmix(상황 기반, 정적 CSV) 둘 다 병합한다."""
    pitcher_map = pd.read_csv(os.path.join(model_dir, "pitcher_map.csv"))
    df_trm = pd.read_csv(os.path.join(data_dir, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    df = df_test
    trk_mlp_cols, trk_cat_cols = [], []
    for tier, feed in TRACKMAN_TIER_FEED.items():
        lookup = build_pitcher_lookup(df_trm_clean, pitcher_map, tier)
        cols = [c for c in lookup.columns if c != "pitcher_id"]
        df = pd.merge(df, lookup, on="pitcher_id", how="left")
        df[cols] = df[cols].fillna(0.0)
        (trk_mlp_cols if feed == "mlp" else trk_cat_cols).extend(cols)

    df = add_coarse_pitchmix(df, model_dir)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS
    return df, trk_mlp_cols, trk_cat_cols


# 투수/타자 "시즌 진행분" 8개 (code/train.py::apply_season_progression_features 와
# 동일, 수동 동기화 유지). test.csv(season=2025)는 과거 시즌 행도 정답도 없어 lookup을
# 자체 계산할 수 없으므로, dopip.py Full Retrain 단계에서 train.csv 전체(2019~2024)로
# 미리 계산해 model/season_end_lookup.csv로 동봉한 정적 테이블을 그대로 병합만 한다
# (pitcher_map.csv/pitchmix_lookup.csv와 동일한 관례).
SEASON_PROGRESSION_SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    ("batter", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
]


def add_season_progression(df, model_dir):
    lookup = pd.read_csv(os.path.join(model_dir, "season_end_lookup.csv"))
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


def apply_f1_filter(df):
    """code/train.py::apply_f1_filter 와 동일 (수동 동기화 유지). 2022년 이하 시즌의
    game_type=='F' 행을 target-encoding 인코딩 소스에서 제거한다 — 그러지 않으면
    F1 오염이 인코딩 통계로 재유입된다(팀원이 처음 겪은 버그와 동일 함정)."""
    return df[~((df['game_type'] == 'F') & (df['season'] <= 2022))].reset_index(drop=True)


# target-encoding 잔차 6개(팀원 제보 Track A, ->CatBoost 전용, code/train.py::
# apply_te_residual_features 와 동일, 수동 동기화 유지). test.csv(season=2025)는
# 항상 학습 데이터의 모든 시즌보다 뒤이므로, 시즌-causal 인코딩(merge_asof
# backward, allow_exact_matches=False)이 자연스럽게 "학습 데이터 전체 누적"을
# 반환한다 — 정적 lookup CSV가 따로 필요 없고, train.csv(평가 서버 data/ 에도
# 동봉됨, league_success_mean과 동일한 근거)로 매 실행 재계산한다.
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
    MODEL_DIR = "./model"
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

    # 2b. 트랙맨 pitcher-std(tier A) + coarse pitchmix 피처 병합 (test 시즌=2025는 트랙맨
    # 파일에 아예 없으므로 look-ahead 걱정 없이 전체 트랙맨 2019~2024 lookup만 붙이면 된다)
    df_test, _, _ = add_trackman_features(df_test, DATA_DIR, MODEL_DIR)

    # 2c. 투수/타자 시즌 진행분 8개 (정적 lookup 병합, add_season_progression 문서 참고)
    df_test = add_season_progression(df_test, MODEL_DIR)

    # 3. asof_* 및 카운트 정보를 조합한 파생 피처 추가 (code/train.py와 동일 정의)
    tr_final = add_engineered_features(df_test, league_success_mean)

    # 3b. target-encoding 잔차 6개(->CatBoost 전용). 인코딩 소스는 F1 필터가 적용된
    # train.csv 전체 — test.csv(season=2025)는 항상 그보다 뒤라 causal_smoothed_te_encode가
    # 자연히 "학습 데이터 전체 누적"을 반환한다(정적 lookup 불필요, apply_te_residual_features
    # 문서 참고). features 목록에는 안 넣어 MLP는 이 6개를 못 보게 유지한다.
    te_train_source = apply_f1_filter(df_train_raw)
    te_prior = te_train_source[TARGET_COL].mean()
    tr_final = apply_te_residual_features(te_train_source, tr_final, te_prior)

    # 4. 모델 입력 데이터 정렬
    drop_cols = [ID_COL, TARGET_COL]
    features = [col for col in tr_final.columns if col not in drop_cols and col not in TE_RESIDUAL_COLS]
    X_test = tr_final[features + TE_RESIDUAL_COLS]

    # 5. CatBoost + Tabular MLP 앙상블 블렌드 확률 추론 수행
    # 5a. CatBoost (원본 dtype 그대로 입력 — game_type/base_state는 문자열로 자체 처리)
    # coarse pitchmix만 학습에 썼으므로(tier A는 MLP 전용), 저장된 컬럼 목록으로 서브셋한다.
    cat_feature_cols = bundle.get("cat_feature_cols")
    X_test_cat = X_test[cat_feature_cols] if cat_feature_cols is not None else X_test
    cat_preds = bundle["catboost_model"].predict_proba(X_test_cat)[:, 1]

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
