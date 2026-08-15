# code/experiment_mlp_subfeature.py
"""MLP 전용 서브피처 스크리닝 실험.

§14.2에서 CatBoost 단독으로만 테스트했던 "asof_batter_* 강화"(7개)/"pitch-mix 교차"(7개)
14개 피처가 CatBoost에선 전부 손해였다(-43.89 / -39.06) — CatBoost는 트리 분기로
상호작용을 스스로 찾아내므로 명시적 교차항이 대체로 중복이었다는 게 §14.2/§25.4/§26에서
반복 확인된 패턴이다. 반면 MLP(concat + deep layer)는 상호작용을 층으로 근사해야 하는
구조라, 같은 피처가 CatBoost와 반대로 도움이 될 수 있다는 가설을 이 스크립트로 검증한다.

원본 스크립트가 리포지토리에 남아있지 않아(PROJECT_HISTORY.md/EXPERIMENTS.md엔 이름만
기록, 정확한 수식 없음) 이름으로부터 재구성했다. `fastball_rate_x_situational_speed`/
`breaking_rate_x_situational_break` 2개는 이름상 트랙맨의 상황별 물리 지표에 의존한
것으로 보이는데, 트랙맨은 §25에서 최종 종결됐으므로 원안 그대로는 재현 불가능해
순수 상황 변수(`li`/`num_runners_on`) 기반으로 대체했다.

`code/experiment_attention.py::build_split`(트랙맨 없음, F1 필터 적용, season==2024
기본 홀드아웃)을 그대로 재사용한다. §14.3/§15와 동일한 관례로 7-seed 풀 앙상블 대신
3-seed 스크리닝으로 방향성만 먼저 확인한다 — 델타가 시드 간 변동폭보다 뚜렷하게 커야
7-seed 재검증으로 넘어갈 가치가 있다고 판단한다.

사용법:
  python -m code.experiment_mlp_subfeature --holdout 2024
"""
import argparse

import numpy as np

from code.mlp_model import (
    CAT_COLS, ENSEMBLE_SEEDS, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble,
    to_tensors, train_ensemble,
)
from code.experiment_attention import build_split

SCREEN_SEEDS = [42, 123, 7]  # §14.3/§15와 동일한 3-seed 스크리닝 세트

BATTER_FEATURES = [
    "batter_relative_success", "batter_relative_middle", "batter_experience_log",
    "pitcher_experience_log", "matchup_confidence", "batter_pressure",
    "batter_middle_vs_pitcher_middle",
]
PITCHMIX_FEATURES = [
    "pitchmix_confidence", "fastball_pressure", "breaking_fullcount",
    "offspeed_ahead", "mix_entropy", "fastball_rate_x_li", "breaking_rate_x_runners",
]
SUBFEATURES = BATTER_FEATURES + PITCHMIX_FEATURES


def add_subfeatures(df, league_success_mean, league_middle_mean):
    """§14.2 "배터 강화"/"pitch-mix 교차" 14개 피처의 재구성 버전을 추가합니다.
    `df`는 이미 `code/train.py::add_engineered_features`를 거쳐 `matchup`/`count_diff`/
    `is_full_count`가 존재한다고 가정합니다 (`build_split`이 이를 보장)."""
    df = df.copy()
    eps = 1e-9

    pressure_signal = df["li"] * ((df["strikes_before"] >= 2) | (df["balls_before"] >= 3)).astype(np.int64)

    # 배터 강화 (투수 쪽 pitcher_relative_success/matchup/count_pressure와 대칭 구조)
    df["batter_relative_success"] = df["asof_batter_success_rate"] - league_success_mean
    df["batter_relative_middle"] = df["asof_batter_middle_rate"] - league_middle_mean
    df["batter_experience_log"] = np.log1p(df["asof_batter_n"])
    df["pitcher_experience_log"] = np.log1p(df["asof_pitcher_n"])
    df["matchup_confidence"] = df["matchup"] * np.log1p(np.minimum(df["asof_pitcher_n"], df["asof_batter_n"]))
    df["batter_pressure"] = df["batter_relative_success"] * pressure_signal
    df["batter_middle_vs_pitcher_middle"] = df["asof_batter_middle_rate"] - df["asof_pitcher_middle_rate"]

    # pitch-mix 교차 (마지막 2개는 트랙맨 미의존 대체)
    df["pitchmix_confidence"] = np.log1p(df["asof_pitcher_pitchmix_n"])
    df["fastball_pressure"] = df["asof_pitcher_fastball_rate"] * pressure_signal
    df["breaking_fullcount"] = df["asof_pitcher_breaking_rate"] * df["is_full_count"]
    df["offspeed_ahead"] = df["asof_pitcher_offspeed_rate"] * (df["count_diff"] > 0).astype(np.int64)
    fb, br, off = df["asof_pitcher_fastball_rate"], df["asof_pitcher_breaking_rate"], df["asof_pitcher_offspeed_rate"]
    df["mix_entropy"] = -(fb * np.log(fb + eps) + br * np.log(br + eps) + off * np.log(off + eps))
    df["fastball_rate_x_li"] = df["asof_pitcher_fastball_rate"] * df["li"]
    df["breaking_rate_x_runners"] = df["asof_pitcher_breaking_rate"] * df["num_runners_on"]

    return df


def run_variant(train_split, val_split, num_cols, device, seeds=SCREEN_SEEDS):
    tr_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, num_cols)
    va_proc = apply_preprocessing(val_split, CAT_COLS, num_cols, cat_encoder, num_imputer, num_scaler)

    X_tr_cat, X_tr_num, y_tr = to_tensors(tr_proc, CAT_COLS, num_cols, "control_success")
    X_va_cat, X_va_num, y_va = to_tensors(va_proc, CAT_COLS, num_cols, "control_success")

    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr, cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_va_cat, X_val_num=X_va_num, y_val=y_va,
        seeds=seeds, device=device, verbose=False,
    )
    preds = predict_ensemble(
        members, cat_dims, len(num_cols), embed_dims, X_va_cat, X_va_num,
        bin_edges=bin_edges, device=device,
    )
    return compute_bss(preds, y_va.numpy())[2], preds


# `code/train.py::add_engineered_features`가 만드는 12개 파생 피처. §14.2가 CatBoost
# 기준으로 "일부는 기존 피처의 단조변환이라 새 정보가 거의 없다"고 진단한 그룹 — MLP에서도
# 같은지(제거해도 안 변하거나 오히려 나아지는지) 확인 대상.
ENGINEERED_FEATURES = [
    "pitcher_recent1_gap", "pitcher_recent3_gap", "pitcher_recent5_gap",
    "pitcher_relative_success", "count_diff", "is_full_count",
    "pitcher_count_advantage_raw", "pitcher_count_advantage_rel",
    "pitcher_trend", "pitcher_consistency", "matchup", "count_pressure",
]


def run(holdout):
    train_split, val_split, features, num_cols = build_split(holdout, apply_f1=True)

    league_success_mean = train_split["control_success"].mean()
    league_middle_mean = train_split["asof_pitcher_middle_rate"].mean()

    device = get_device()
    print(f"[Device] {device}")

    baseline_score, _ = run_variant(train_split, val_split, num_cols, device)
    print(f"[baseline] Val Score: {baseline_score:.2f}")

    tr_sub = add_subfeatures(train_split, league_success_mean, league_middle_mean)
    va_sub = add_subfeatures(val_split, league_success_mean, league_middle_mean)
    sub_score, _ = run_variant(tr_sub, va_sub, num_cols + SUBFEATURES, device)
    print(f"[+subfeatures(14)] Val Score: {sub_score:.2f}")

    print(f"\nDelta: {sub_score - baseline_score:+.2f}")


def run_removal_blend(holdout, seeds=SCREEN_SEEDS):
    """`add_engineered_features`의 12개를 MLP 입력에서만 제거했을 때, MLP 단독 점수가
    아니라 CatBoost+MLP 스태킹 블렌드 점수가 어떻게 바뀌는지 확인한다 (CatBoost는 두
    변형에서 항상 동일한 피처를 쓰므로 한 번만 학습해 재사용). solo MLP 점수만으로는
    프로덕션이 실제로 최적화하는 지표(블렌드)를 놓칠 수 있다는 지적을 반영한 설계."""
    from code.catboost_model import train_catboost, predict_catboost
    from code.blend_model import fit_meta_model

    train_split, val_split, features, num_cols = build_split(holdout, apply_f1=True)
    y_val = val_split["control_success"].values

    device = get_device()
    print(f"[Device] {device}")

    X_train_raw = train_split[features]
    y_train_raw = train_split["control_success"].values
    X_val_raw = val_split[features]
    catboost_model, _ = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val, verbose=False)
    cat_val_preds = predict_catboost(catboost_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val)[2]
    print(f"[CatBoost(고정)] Val Score: {cat_score:.2f}\n")

    variants = {
        "baseline": num_cols,
        "-engineered(12)": [c for c in num_cols if c not in ENGINEERED_FEATURES],
    }

    results = {}
    for name, cols in variants.items():
        mlp_score, mlp_val_preds = run_variant(train_split, val_split, cols, device, seeds=seeds)
        w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_val_preds, mlp_val_preds, y_val)
        results[name] = (mlp_score, blend_score)
        print(f"[{name}] MLP solo: {mlp_score:.2f} | Blend: {blend_score:.2f} (w_cat={w_cat:.3f} w_mlp={w_mlp:.3f})")

    (base_mlp, base_blend), (rm_mlp, rm_blend) = results["baseline"], results["-engineered(12)"]
    print(f"\nMLP solo Delta: {rm_mlp - base_mlp:+.2f}")
    print(f"Blend Delta: {rm_blend - base_blend:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024)
    parser.add_argument("--experiment", choices=["add_subfeatures", "remove_engineered_blend"], default="remove_engineered_blend")
    parser.add_argument("--ensemble", action="store_true", default=False, help="7-seed(ENSEMBLE_SEEDS)로 재검증 (기본은 3-seed 스크리닝)")
    args = parser.parse_args()
    if args.experiment == "add_subfeatures":
        run(args.holdout)
    else:
        seeds = ENSEMBLE_SEEDS if args.ensemble else SCREEN_SEEDS
        run_removal_blend(args.holdout, seeds=seeds)
