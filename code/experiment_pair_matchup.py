# code/experiment_pair_matchup.py
"""사용자 제안(2026-08-22 세션): 투수x타자 "페어" 단위 상대 전적 as-of 피처를 MLP에
먹여본다. 시즌진행분(§45)이 유일하게 CatBoost/MLP 양쪽·양쪽 레짐 다 크게 통했던 이유는
"행 자신의 원본 컬럼만으로는 유도 불가능한, 조인이 있어야만 나오는 새 정보"였기
때문이라는 가설 아래, asof_pitcher_success_rate/asof_batter_success_rate 같은 단독
한계분포에는 없는 "이 투수가 이 특정 타자를 상대할 때"의 이력을 시도한다.

구현은 시즌진행분과 동일한 두 갈래 컨벤션을 따른다:
  - train.csv 자체(스플릿 이전)에 대해서는 row_id 시간순 진짜 as-of 확장 통계
    (그 행 이전까지 이 페어가 만난 횟수/성공수 — pandas groupby.cumcount/cumsum이
    그 행 자신을 자동으로 배제, 시즌 경계 없이 순수 시간순)를 쓴다 — asof_pitcher_n
    등 공식 피처와 동일한 성격의 통계라 self-leakage가 없다.
  - test.csv(2025, 페어 자체 역사가 없음) 추론 시에는 train.csv 전체의 "최종" 페어별
    누적치를 정적 lookup으로 얼려서 병합만 한다(season_end_lookup/pitchmix_lookup과
    동일 컨벤션) — test.csv 행끼리 서로 다른 갱신값을 못 갖게 해 평가 규칙(행별 독립
    추론)을 지킨다.

5개 컬럼: pair_n, pair_success_count, pair_success_rate, pair_rate_gap_pitcher(=
pair_success_rate - asof_pitcher_success_rate), pair_rate_gap_batter(같은 방식,
타자 기준). 대부분의 페어는 통산 몇 번밖에 안 만나 표본이 매우 희박할 것으로
예상되므로(792명x830명 조합), 학습 전 커버리지부터 점검한다.

thirdmodel_common.build_split 위에 이 5개를 추가해 ->MLP 전용 / ->CatBoost 전용
두 가지로 dual-feed 스크리닝(3-seed MLP, CatBoost 전체 재학습, cutoff7 단일 레짐 1차
스크리닝 — 통과하는 쪽이 있으면 season==2023 재검증으로 확장).

사용법: python -m code.experiment_pair_matchup
"""
import argparse
import os

import numpy as np
import pandas as pd

from code.mlp_model import (
    CAT_COLS, QUANTILE_D, apply_preprocessing, compute_bss, embed_dim_for_cardinality,
    fit_preprocessing, fit_quantile_edges, get_device, predict_ensemble, to_tensors, train_ensemble,
)
from code.catboost_model import train_catboost, predict_catboost
from code.blend_model import fit_meta_model
from code.train import (
    TE_RESIDUAL_COLS, TRACKMAN_TIER_FEED, add_engineered_features, apply_f1_filter,
    apply_te_residual_features,
)
from code.trackman_pitcher_features import add_all_tiers, clean_trackman, merge_coarse_pitchmix, PITCHMIX_COLS

DATA_DIR = "./open/data"
TARGET_COL = "control_success"
SCREEN_SEEDS = [42, 123, 7]

PAIR_COLS = ["pair_n", "pair_success_count", "pair_success_rate",
             "pair_rate_gap_pitcher", "pair_rate_gap_batter"]

# 프로덕션 7-seed cutoff7 레퍼런스 (code/experiment_thirdmodel_base.py 캐시, 다른 스크리닝
# 실험들과 동일한 비교 기준선)
REF_CAT = 706.56
REF_MLP7 = 738.55
REF_BLEND = 753.37


def apply_pair_matchup_rowlevel(df):
    """row_id 시간순으로 (pitcher_id, batter_id) 페어의 "그 행 이전까지" 누적 통계를
    계산해 붙인다. control_success가 필요하므로 train.csv 전체(스플릿 이전)에서만
    호출한다."""
    df = df.sort_values("row_id").reset_index(drop=True)
    g = df.groupby(["pitcher_id", "batter_id"])["control_success"]
    cum_n = g.cumcount().values.astype(np.float64)  # 이 행 이전까지 만난 횟수(자기 제외)
    cum_success = (g.cumsum() - df["control_success"]).values  # 이전까지 누적 성공수(자기 제외)
    with np.errstate(invalid="ignore", divide="ignore"):
        pair_rate = np.where(cum_n > 0, cum_success / cum_n, np.nan)
    df["pair_n"] = cum_n
    df["pair_success_count"] = cum_success
    df["pair_success_rate"] = pair_rate
    df["pair_rate_gap_pitcher"] = pair_rate - df["asof_pitcher_success_rate"].values
    df["pair_rate_gap_batter"] = pair_rate - df["asof_batter_success_rate"].values
    return df


def build_split_with_pair(cutoff7=True, holdout=2024, apply_f1=True):
    """thirdmodel_common.build_split과 동일하되 pair-matchup 5개 컬럼을 추가로 계산해
    mlp_num_cols/cat_feature_cols에는 아직 넣지 않고 별도로 반환한다(호출부에서
    feed-to 선택)."""
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df["top_bottom"] = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # pair-matchup as-of 통계는 전체 시간순으로 한 번만 계산(스플릿 이전) — asof_pitcher_n과
    # 동일한 성격(행 자신 이전 정보만 사용)이라 나중에 train/val로 나눠도 안전하다.
    df = apply_pair_matchup_rowlevel(df)

    if cutoff7:
        train_mask = (df["season"] < 2024) | ((df["season"] == 2024) & (df["game_month"] < 7))
        val_mask = (df["season"] == 2024) & (df["game_month"] >= 7)
        trk_holdout = 2024
    else:
        train_mask = df["season"] < holdout
        val_mask = df["season"] == holdout
        trk_holdout = holdout

    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    df, trk_tier_cols = add_all_tiers(df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=trk_holdout)
    trk_mlp_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "mlp" for c in cols]
    trk_cat_cols = [c for tier, cols in trk_tier_cols.items() if TRACKMAN_TIER_FEED[tier] == "cat" for c in cols]

    df = merge_coarse_pitchmix(df, df_trm, holdout=trk_holdout)
    trk_cat_cols = trk_cat_cols + PITCHMIX_COLS

    league_success_mean = df.loc[train_mask, TARGET_COL].mean()
    df = add_engineered_features(df, league_success_mean)

    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in df.columns if c not in drop_cols]
    # pair 컬럼은 base mlp_num_cols/cat_feature_cols에서 일단 빼둔다(호출부에서 feed-to별로 추가)
    mlp_num_cols = [c for c in features if c not in CAT_COLS and c not in trk_cat_cols
                    and c not in trk_mlp_cols and c not in PAIR_COLS]
    cat_feature_cols = [c for c in features if c not in trk_mlp_cols and c not in PAIR_COLS]

    train_split = df.loc[train_mask, features + [TARGET_COL]].reset_index(drop=True)
    val_split = df.loc[val_mask, features + [TARGET_COL]].reset_index(drop=True)
    if apply_f1:
        before = len(train_split)
        train_split = apply_f1_filter(train_split)
        print(f"[F1 필터] {before} -> {len(train_split)}행")

    te_prior = train_split[TARGET_COL].mean()
    train_split = apply_te_residual_features(train_split, train_split, te_prior)
    val_split = apply_te_residual_features(train_split, val_split, te_prior)
    cat_feature_cols = cat_feature_cols + TE_RESIDUAL_COLS

    print(f"[build_split_with_pair] train={len(train_split)} val={len(val_split)} "
          f"mlp_num_cols={len(mlp_num_cols)} cat_feature_cols={len(cat_feature_cols)}")
    return train_split, val_split, mlp_num_cols, cat_feature_cols


def report_coverage(train_split, val_split):
    tr_cov = (train_split["pair_n"] > 0).mean()
    val_cov = (val_split["pair_n"] > 0).mean()
    print(f"[커버리지] train: pair_n>0 비율={tr_cov:.4f} ({(train_split['pair_n']>0).sum()}/{len(train_split)}), "
          f"pair_n 분포(>0인 행만)=\n{train_split.loc[train_split['pair_n']>0,'pair_n'].describe()}")
    print(f"[커버리지] val: pair_n>0 비율={val_cov:.4f} ({(val_split['pair_n']>0).sum()}/{len(val_split)})")
    return tr_cov, val_cov


def run_one(name, train_split, val_split, mlp_num_cols, cat_feature_cols):
    print(f"\n{'='*20} 실험: {name} {'='*20}")
    print(f"mlp_num_cols={len(mlp_num_cols)} cat_feature_cols={len(cat_feature_cols)}")

    X_train_raw, y_train_raw = train_split[cat_feature_cols], train_split[TARGET_COL].values
    X_val_raw, y_val_raw = val_split[cat_feature_cols], val_split[TARGET_COL].values
    cat_model, cat_best_iter = train_catboost(X_train_raw, y_train_raw, X_val_raw, y_val_raw, verbose=False)
    cat_val_preds = predict_catboost(cat_model, X_val_raw)
    cat_score = compute_bss(cat_val_preds, y_val_raw)[2]
    print(f"[CatBoost] best_iter={cat_best_iter} score={cat_score:.2f} (ref={REF_CAT:.2f}, delta={cat_score-REF_CAT:+.2f})")

    train_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_split, CAT_COLS, mlp_num_cols)
    val_proc = apply_preprocessing(val_split, CAT_COLS, mlp_num_cols, cat_encoder, num_imputer, num_scaler)
    X_tr_cat, X_tr_num, y_tr = to_tensors(train_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    X_val_cat, X_val_num, y_val_t = to_tensors(val_proc, CAT_COLS, mlp_num_cols, TARGET_COL)
    bin_edges = fit_quantile_edges(X_tr_num)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    device = get_device()

    members = train_ensemble(
        X_tr_cat, X_tr_num, y_tr,
        cat_dims=cat_dims, embed_dims=embed_dims, bin_edges=bin_edges,
        X_val_cat=X_val_cat, X_val_num=X_val_num, y_val=val_proc[TARGET_COL].values,
        seeds=SCREEN_SEEDS, device=device,
    )
    ens_pred = predict_ensemble(
        members, cat_dims, len(mlp_num_cols), embed_dims,
        X_val_cat, X_val_num, bin_edges=bin_edges, quantile_d=QUANTILE_D, device=device,
    )
    mlp_score = compute_bss(ens_pred, y_val_raw)[2]
    print(f"[MLP {len(SCREEN_SEEDS)}-seed] score={mlp_score:.2f} (ref 7-seed={REF_MLP7:.2f}, delta={mlp_score-REF_MLP7:+.2f})")

    w_cat, w_mlp, intercept, blend_score, _ = fit_meta_model(cat_val_preds, ens_pred, y_val_raw)
    print(f"[2-way blend] score={blend_score:.2f} (ref={REF_BLEND:.2f}, delta={blend_score-REF_BLEND:+.2f}) "
          f"weights cat={w_cat:.3f} mlp={w_mlp:.3f} intercept={intercept:.3f}")
    return dict(name=name, cat_score=cat_score, mlp_score=mlp_score, blend_score=blend_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()

    regime = "cutoff7" if args.cutoff7 else f"season=={args.holdout}"
    train_split, val_split, base_mlp_cols, base_cat_cols = build_split_with_pair(
        cutoff7=args.cutoff7, holdout=args.holdout,
    )
    report_coverage(train_split, val_split)

    results = []
    results.append(run_one(
        "baseline(pair 미포함)", train_split, val_split, base_mlp_cols, base_cat_cols,
    ))
    results.append(run_one(
        "pair -> MLP", train_split, val_split, base_mlp_cols + PAIR_COLS, base_cat_cols,
    ))
    results.append(run_one(
        "pair -> CatBoost", train_split, val_split, base_mlp_cols, base_cat_cols + PAIR_COLS,
    ))
    results.append(run_one(
        "pair -> both", train_split, val_split, base_mlp_cols + PAIR_COLS, base_cat_cols + PAIR_COLS,
    ))

    print(f"\n{'='*20} 요약 ({regime}) {'='*20}")
    print(f"참고 7-seed cutoff7 프로덕션 레퍼런스: CatBoost={REF_CAT:.2f} MLP={REF_MLP7:.2f} blend={REF_BLEND:.2f} "
          f"(현재 레짐={regime}이라 직접 비교 불가, 같은 실행 내 baseline 대비 delta를 봐야 함)")
    base = results[0]
    for r in results:
        d_cat, d_mlp, d_blend = r["cat_score"] - base["cat_score"], r["mlp_score"] - base["mlp_score"], r["blend_score"] - base["blend_score"]
        print(f"{r['name']:20s}: CatBoost={r['cat_score']:.2f}({d_cat:+.2f}) "
              f"MLP(3-seed)={r['mlp_score']:.2f}({d_mlp:+.2f}) blend={r['blend_score']:.2f}({d_blend:+.2f})")


if __name__ == "__main__":
    main()
