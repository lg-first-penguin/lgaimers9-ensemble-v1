# code/experiment_asof_rate_decomposition.py
"""season-progression(성공률 시즌분해)/prev-game-gap 패턴을 아직 안 써본 asof_* rate
컬럼들에 확장 적용해 CatBoost 단독 dual-regime으로 1차 스크리닝한다.

세 그룹, 전부 "행 자신의 상황/시점 기준으로 기존 asof_* 컬럼을 쪼갠다"는 이미 검증된
패턴(시즌진행분 success_rate, TE-residual)을 따른다:

- A' (prev-game middle gap): asof_pitcher_prev{1,3,5}_game_middle_rate를 이미 production에
  있는 asof_pitcher_middle_rate와 비교. success_rate 버전(pitcher_recent{1,3,5}_gap 등)은
  이미 add_engineered_features에 있어 재검증 불필요 — middle_rate 버전만 신규.
  이미 기각된 p_middle_season(시즌 경계 lookup 방식)과 메커니즘이 다르다(레이블 근사가
  아니라 이미 공식 제공되는 prev-game 컬럼끼리의 순수 뺄셈이라 근사 오차 자체가 없음).

- B (reverse/ball/strike 시즌분해): asof_pitcher_reverse_rate/ball_rate/strike_rate에
  code/experiment_season_progression_middle.py와 동일한 "+1 보정 없는" 시즌분해를 적용.
  middle_rate와 달리 season_n을 그 컬럼 고유의 n_col로 독립 계산한다(공유 재사용 아님 —
  아래 build_rate_lookup/apply_rate_features 참고, 원본 미들 버전은 n_col이
  asof_pitcher_n으로 성공률과 같아서 재사용이 우연히 맞았을 뿐).

- C (pitchmix rate 시즌분해): asof_pitcher_fastball/breaking/offspeed_rate, n_col은
  asof_pitcher_pitchmix_n(성공률/미들과 다른 표본수 컬럼) — B와 같은 함수, 다른 n_col.

사용법:
  python -m code.experiment_asof_rate_decomposition --cutoff7
  python -m code.experiment_asof_rate_decomposition --holdout 2023
"""
import argparse
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from code.catboost_model import CAT_FEATURES, CATBOOST_PARAMS
from code.experiment_season_progression import TARGET_COL, build_split
from code.mlp_model import compute_bss
from code.train import apply_f1_filter

# ---- Group A': prev-game middle gap (lookup 불필요, 공식 컬럼끼리 뺄셈만) ----
GROUP_A_COLS = [
    "pitcher_recent1_middle_gap", "pitcher_recent3_middle_gap", "pitcher_recent5_middle_gap",
    "pitcher_middle_trend", "pitcher_middle_consistency",
]


def add_group_a(df):
    df = df.copy()
    df["pitcher_recent1_middle_gap"] = df["asof_pitcher_prev1_game_middle_rate"] - df["asof_pitcher_middle_rate"]
    df["pitcher_recent3_middle_gap"] = df["asof_pitcher_prev3_game_middle_rate"] - df["asof_pitcher_middle_rate"]
    df["pitcher_recent5_middle_gap"] = df["asof_pitcher_prev5_game_middle_rate"] - df["asof_pitcher_middle_rate"]
    df["pitcher_middle_trend"] = df["asof_pitcher_prev1_game_middle_rate"] - df["asof_pitcher_prev5_game_middle_rate"]
    df["pitcher_middle_consistency"] = df[[
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]].std(axis=1)
    return df


# ---- Group B/C: "+1 보정 없는" 시즌분해 (라벨 없는 rate 컬럼 공통 처리, n_col 독립 계산) ----
# (role, id_col, n_col, rate_col, tag)
GROUP_B_SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate", "reverse"),
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate", "ball"),
    ("pitcher", "pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate", "strike"),
]
GROUP_C_SPECS = [
    ("pitcher", "pitcher_id", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate", "fastball"),
    ("pitcher", "pitcher_id", "asof_pitcher_pitchmix_n", "asof_pitcher_breaking_rate", "breaking"),
    ("pitcher", "pitcher_id", "asof_pitcher_pitchmix_n", "asof_pitcher_offspeed_rate", "offspeed"),
]


def build_rate_lookup(df, specs):
    tables = []
    for role, id_col, n_col, rate_col, tag in specs:
        idx = df.groupby([id_col, "season"])[n_col].idxmax()
        season_end = df.loc[idx, [id_col, "season", n_col, rate_col]].copy()
        season_end.columns = ["id", "season", "end_n", "end_rate"]
        season_end["season"] = season_end["season"] + 1
        season_end.insert(0, "role_tag", f"{role}_{tag}")
        tables.append(season_end)
    return pd.concat(tables, ignore_index=True)


def apply_rate_features(df, lookup, specs):
    df = df.copy()
    for role, id_col, n_col, rate_col, tag in specs:
        key = f"{role}_{tag}"
        lut = lookup.loc[lookup["role_tag"] == key, ["id", "season", "end_n", "end_rate"]]
        merged = df[[id_col, "season"]].merge(
            lut, left_on=[id_col, "season"], right_on=["id", "season"], how="left",
        )
        pre_n = merged["end_n"].fillna(0).values
        pre_count = (merged["end_n"] * merged["end_rate"]).round().fillna(0).values

        cum_n = df[n_col].values
        cum_count = np.round(cum_n * df[rate_col].values)
        season_n = np.maximum(cum_n - pre_n, 0)
        season_count = np.maximum(cum_count - pre_count, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            season_rate = np.where(season_n > 0, season_count / season_n, np.nan)

        df[f"{key}_season_rate"] = season_rate
        df[f"{key}_season_gap"] = season_rate - df[rate_col].values
    return df


def cols_for(specs, tag):
    role, *_ , t = next(s for s in specs if s[4] == tag)
    key = f"{role}_{t}"
    return [f"{key}_season_rate", f"{key}_season_gap"]


ALL_VARIANTS = [
    ("baseline", []),
    ("A' +middle_prevgame(5)", GROUP_A_COLS),
    ("B +reverse_season(2)", cols_for(GROUP_B_SPECS, "reverse")),
    ("B +ball_season(2)", cols_for(GROUP_B_SPECS, "ball")),
    ("B +strike_season(2)", cols_for(GROUP_B_SPECS, "strike")),
    ("C +fastball_season(2)", cols_for(GROUP_C_SPECS, "fastball")),
    ("C +breaking_season(2)", cols_for(GROUP_C_SPECS, "breaking")),
    ("C +offspeed_season(2)", cols_for(GROUP_C_SPECS, "offspeed")),
]
GROUP_TAGS = {"a": ["A'"], "b": ["B"], "c": ["C"]}
ALL_NEW_COLS = GROUP_A_COLS + [c for _, cols in ALL_VARIANTS[2:] for c in cols]


def train_catboost_custom(X_train, y_train, X_val, y_val, cat_features):
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = 1500
    params["early_stopping_rounds"] = 50
    model = CatBoostClassifier(**params)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model, int(model.get_best_iteration())


def run_regime(holdout, cutoff7, group):
    label = "cutoff7" if cutoff7 else f"holdout={holdout}"
    print(f"\n{'='*70}\n=== 레짐: {label} (그룹={group}) ===\n{'='*70}")

    variants = [ALL_VARIANTS[0]] + [v for v in ALL_VARIANTS[1:] if v[0].split(" ")[0] in GROUP_TAGS[group]]

    df, train_mask, val_mask, trk_mlp_cols, trk_cat_cols = build_split(holdout, cutoff7)
    if group == "a":
        df = add_group_a(df)
    else:
        specs = GROUP_B_SPECS if group == "b" else GROUP_C_SPECS
        lookup = build_rate_lookup(df, specs)
        df = apply_rate_features(df, lookup, specs)

    drop_cols = ["row_id", TARGET_COL]
    results = {}
    for tag, keep_cols in variants:
        exclude_cols = [c for c in ALL_NEW_COLS if c not in keep_cols]
        base_features = [
            c for c in df.columns
            if c not in drop_cols and c not in trk_mlp_cols and c not in trk_cat_cols and c not in exclude_cols
        ]
        cat_features = base_features + trk_cat_cols

        train_split = df.loc[train_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        val_split = df.loc[val_mask, cat_features + [TARGET_COL]].reset_index(drop=True)
        train_split = apply_f1_filter(train_split)

        X_train, y_train = train_split[cat_features], train_split[TARGET_COL].values
        X_val, y_val = val_split[cat_features], val_split[TARGET_COL].values

        t0 = time.time()
        model, best_iter = train_catboost_custom(X_train, y_train, X_val, y_val, CAT_FEATURES)
        preds = model.predict_proba(X_val)[:, 1]
        score = compute_bss(preds, y_val)[2]
        print(f"[{tag}] Val Score={score:.2f} (best_iteration={best_iter}, {time.time()-t0:.1f}s, n_features={len(cat_features)})")
        results[tag] = score

    base = results["baseline"]
    print(f"\n--- {label} 요약 (vs baseline) ---")
    for tag, _ in variants:
        print(f"  {tag}: {results[tag]:.2f} ({results[tag]-base:+.2f})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2023, choices=[2023, 2024])
    parser.add_argument("--cutoff7", action="store_true")
    parser.add_argument("--group", choices=["a", "b", "c"], required=True)
    args = parser.parse_args()
    holdout = 2024 if args.cutoff7 else args.holdout
    run_regime(holdout, args.cutoff7, args.group)


if __name__ == "__main__":
    main()
