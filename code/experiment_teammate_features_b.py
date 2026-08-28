# code/experiment_teammate_features_b.py
"""팀원이 공유한 "CatBoost 단독 피처" 문서(A~H) 중, 코드 검색으로 확인한 결과 우리
프로덕션에 아직 없는 5가지 후보를 CatBoost 단독 dual-regime(cutoff7 + season==2023)로
스크리닝한다. (A/C/G는 이미 add_engineered_features/apply_season_progression_features에
동일하게 있음.)

**2026-08-23 재개 배경**: B'/D/H 3개를 단독/조합으로 먼저 스크리닝했을 때 dual-regime
부호가 반전(cutoff7 전부 마이너스, season2023 전부 큰 플러스)해 기각했었다. 그런데
팀원은 이 피처셋(A~H 전체) 그대로 CatBoost 단독으로 **실전 리더보드 1,000점대**를
받았다고 확인됨(우리 CatBoost 단독의 마지막 실전 기록은 878.73, 그마저도 F1필터/
cutoff7/시즌진행분/TE-residual 전부 없던 EDA 1라운드 시절 — 최근 CatBoost 단독 실전
제출 자체가 없어 직접 비교 불가). 사용자 가설: E/F(트랙맨 coarse std/gap)까지 포함한
**전체 조합의 상호작용**에서 나온 효과일 수 있다 — B'/D/H만 따로 뗀 이전 스크리닝은
이 상호작용을 못 볼 수 있다. 이번 재개에서 E/F를 팀원 스펙 그대로(6키+season-1 매칭,
std/gap) 새로 구현해 추가하고, 개별 + 전체 조합(B'+D+H+E+F) 둘 다 확인한다.

후보 1 (B', season-relative baseline): 우리 기존 `pitcher_relative_success`는
`asof_pitcher_success_rate - league_success_mean`에서 league_success_mean이 학습기간
전체 평균 스칼라 하나로 고정돼 있다. 팀원 버전은 시즌별로(season-1 anchor) 달라지는
리그 평균을 쓴다 — ABS 레짐 시프트로 시즌마다 성공률 자체가 바뀐다는 이 프로젝트의 핵심
발견(§35)을 감안하면 정적 스칼라보다 나을 수 있다는 가설. lookup은 train_split(source)의
시즌별 평균을 season+1로 시프트해서 만든다(season s의 baseline = season s-1의 실제 평균,
이미 완결된 과거 시즌이라 리크 없음). 첫 시즌(2019, prior season 없음)은 train_split
전체 평균으로 폴백.

후보 2 (D, same_hand): `pitcher_hand == batter_hand` 단순 일치 여부 + 기존
`pitcher_relative_success`(정적 버전, B' 테스트와 독립적으로 유지)와의 곱.

후보 3 (H, hand_matchup): `pitcher_hand + "_" + batter_hand` 4분류 범주형 (CatBoost
cat_features에 추가).

후보 4 (E, trackman std5): 팀원 스펙 — 매칭 키 6개(inning, top_bottom, balls_before,
strikes_before, pitcher_hand, batter_hand), **season-1 매칭**(우리 기존 coarse_phys/
pitchmix의 "cumulative all-history" 관례와 다름 — 트랙맨의 "그 행 시즌의 바로 전 시즌"
만 사용). 물리 지표 5개(rel_speed/spin_rate/induced_vert_break/horz_break/extension)의
표준편차. 시즌-1 트랙맨 커버리지가 없는 행(2019시즌 등)은 전체 트랙맨 기준 global
fallback으로 채운다.

후보 5 (F, trackman gap_fast_break): 같은 6키+season-1, fastball vs breaking 구종군
간 평균 차이 4개(speed/ivb/hbreak/spin).

사용법:
  python -m code.experiment_teammate_features_b --cutoff7
  python -m code.experiment_teammate_features_b --holdout 2023
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.mlp_model import compute_bss
from code.thirdmodel_common import DATA_DIR, build_split, TARGET_COL
from code.trackman_pitcher_features import clean_trackman, _HAND_CODE

SEASON_REL_COL = "pitcher_relative_success_v2_seasonrel"
HAND_COLS = ["same_hand", "same_hand_advantage"]
HAND_MATCHUP_COL = "hand_matchup"

_TOPBOTTOM_CODE = {"Top": 0, "Bottom": 1}
MATCH_KEYS6 = ["inning", "top_bottom", "balls_before", "strikes_before", "pitcher_hand", "batter_hand"]
STD5_RENAME = {
    "rel_speed": "tm_rel_speed_std", "spin_rate": "tm_spin_rate_std",
    "induced_vert_break": "tm_ivb_std", "horz_break": "tm_hb_std", "extension": "tm_extension_std",
}
STD5_COLS = list(STD5_RENAME.values())
GAP4_RENAME = {
    "rel_speed": "speed_gap_fast_break", "induced_vert_break": "ivb_gap_fast_break",
    "horz_break": "hbreak_gap_fast_break", "spin_rate": "spin_gap_fast_break",
}
GAP4_COLS = list(GAP4_RENAME.values())
GAP4_METRICS = list(GAP4_RENAME.keys())


def prep_trackman_for_ef(df_trm_clean):
    df = df_trm_clean.copy()
    df["pitcher_hand"] = df["pitcher_hand"].map(_HAND_CODE).astype(np.int64)
    df["batter_hand"] = df["batter_hand"].map(_HAND_CODE).astype(np.int64)
    df["top_bottom"] = df["top_bottom"].map(_TOPBOTTOM_CODE).astype(np.int64)
    return df


def build_std5_table(trm_prepped):
    std5_metrics = list(STD5_RENAME.keys())
    g = trm_prepped.groupby(["season"] + MATCH_KEYS6)[std5_metrics].std().reset_index()
    g[std5_metrics] = g[std5_metrics].fillna(0.0)
    g = g.rename(columns=STD5_RENAME)
    g["season"] = g["season"] + 1
    fallback = trm_prepped[std5_metrics].std().rename(index=STD5_RENAME).fillna(0.0)
    return g, fallback


def build_gap4_table(trm_prepped):
    sub = trm_prepped[trm_prepped["pitch_type_group"].isin(["fastball", "breaking"])]
    pv = sub.groupby(["season"] + MATCH_KEYS6 + ["pitch_type_group"])[GAP4_METRICS].mean().unstack("pitch_type_group")
    pv.columns = [f"{m}_{grp}" for m, grp in pv.columns]
    pv = pv.reset_index()
    for metric, outcol in GAP4_RENAME.items():
        fb_col, br_col = f"{metric}_fastball", f"{metric}_breaking"
        pv[outcol] = pv.get(fb_col) - pv.get(br_col) if fb_col in pv.columns and br_col in pv.columns else np.nan
    pv = pv[["season"] + MATCH_KEYS6 + GAP4_COLS]
    pv["season"] = pv["season"] + 1

    global_means = sub.groupby("pitch_type_group")[GAP4_METRICS].mean()
    fallback = pd.Series({
        outcol: global_means.loc["fastball", metric] - global_means.loc["breaking", metric]
        for metric, outcol in GAP4_RENAME.items()
    })
    return pv, fallback


def apply_table(df, table, fallback, cols):
    df = df.copy()
    merged = df.merge(table, on=["season"] + MATCH_KEYS6, how="left")
    for c in cols:
        df[c] = merged[c].fillna(fallback[c]).values
    return df


def build_prev_season_league_table(train_split, target_col=TARGET_COL):
    season_mean = train_split.groupby("season")[target_col].mean().rename("prev_season_league_mean")
    table = season_mean.reset_index()
    table["season"] = table["season"] + 1
    return table


def apply_season_relative(df, table, fallback_mean):
    df = df.copy()
    merged = df[["season"]].merge(table, on="season", how="left")
    prev_mean = merged["prev_season_league_mean"].fillna(fallback_mean).values
    df[SEASON_REL_COL] = df["asof_pitcher_success_rate"].values - prev_mean
    return df


def apply_hand_features(df):
    df = df.copy()
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    df["same_hand_advantage"] = df["pitcher_relative_success"] * df["same_hand"]
    return df


def apply_hand_matchup(df):
    df = df.copy()
    df[HAND_MATCHUP_COL] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)
    return df


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def run_regime(cutoff7, holdout):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} ===\n{'='*70}")

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split(cutoff7=cutoff7, holdout=holdout)

    season_table = build_prev_season_league_table(train_split)
    fallback_mean = train_split[TARGET_COL].mean()

    print("[트랙맨 E/F] trackman_history.csv 로드 + 클렌징 + season-1 테이블 구축 중...")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    df_trm_clean = clean_trackman(df_trm)
    trm_prepped = prep_trackman_for_ef(df_trm_clean)
    std5_table, std5_fallback = build_std5_table(trm_prepped)
    gap4_table, gap4_fallback = build_gap4_table(trm_prepped)

    variants = {
        "baseline(현재 프로덕션)": ([], []),
        "+season_relative(B', 1)": (["season_rel"], [SEASON_REL_COL]),
        "+same_hand(D, 2)": (["hand"], HAND_COLS),
        "+hand_matchup(H, 1, cat)": (["hand_matchup"], [HAND_MATCHUP_COL]),
        "+trackman_std5(E, 5)": (["std5"], STD5_COLS),
        "+trackman_gap4(F, 4)": (["gap4"], GAP4_COLS),
        "+E+F(9)": (["std5", "gap4"], STD5_COLS + GAP4_COLS),
        "+B'+D+H+E+F 전부(13)": (["season_rel", "hand", "hand_matchup", "std5", "gap4"],
                                 [SEASON_REL_COL] + HAND_COLS + [HAND_MATCHUP_COL] + STD5_COLS + GAP4_COLS),
    }

    results = {}
    for tag, (kinds, new_cols) in variants.items():
        ts, vs = train_split, val_split
        if "season_rel" in kinds:
            ts = apply_season_relative(ts, season_table, fallback_mean)
            vs = apply_season_relative(vs, season_table, fallback_mean)
        if "hand" in kinds:
            ts = apply_hand_features(ts)
            vs = apply_hand_features(vs)
        if "hand_matchup" in kinds:
            ts = apply_hand_matchup(ts)
            vs = apply_hand_matchup(vs)
        if "std5" in kinds:
            ts = apply_table(ts, std5_table, std5_fallback, STD5_COLS)
            vs = apply_table(vs, std5_table, std5_fallback, STD5_COLS)
        if "gap4" in kinds:
            ts = apply_table(ts, gap4_table, gap4_fallback, GAP4_COLS)
            vs = apply_table(vs, gap4_table, gap4_fallback, GAP4_COLS)

        cat_cols = cat_feature_cols + new_cols
        cat_features = CAT_FEATURES + ([HAND_MATCHUP_COL] if HAND_MATCHUP_COL in new_cols else [])

        X_train, y_train = ts[cat_cols], ts[TARGET_COL].values
        X_val, y_val = vs[cat_cols], vs[TARGET_COL].values

        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, cat_features)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"[{tag}] Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_cols)})")
        results[tag] = score

    base = results["baseline(현재 프로덕션)"]
    print(f"\n--- {label} 요약 --- baseline={base:.2f}")
    for tag, score in results.items():
        if tag == "baseline(현재 프로덕션)":
            continue
        print(f"  {tag}: {score:.2f} (delta={score-base:+.2f})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(args.cutoff7, holdout)


if __name__ == "__main__":
    main()
