# code/trackman_conditioned_features_v2.py
"""code/trackman_conditioned_features.py의 재실행판 -- row-level 매칭 소스만
53.39%(code/pitcher_crosswalk.py 내부 게임정렬 재사용)에서 86.73%
(open/temp/train_with_trackman_match_info.csv, code/build_trackman_rowmatch.py로
2026-08-24 세션에 별도 재현)로 교체한다. 나머지 파이프라인(프로필 구축, 의사라벨링,
투수x구종군 성공-실패 조건화 피처 압축)은 원본과 동일하다.

사용법(모듈, 직접 실행 없음): code/experiment_trackman_conditioned_screen_v2.py에서 사용.
"""
import pandas as pd

from code.trackman_conditioned_features import (  # noqa: F401 (재노출)
    INFORMATIVE_METRICS, MIN_GROUP_N, COND_PREFIX, DATA_DIR,
    build_reference_profiles, pseudo_label_unmatched, build_conditioned_lookup,
)
from code.trackman_pitcher_features import METRICS, clean_trackman

ROWMATCH_PATH = "./open/temp/train_with_trackman_match_info.csv"


def build_row_level_labels():
    """86.73% 커버리지 매칭(open/temp/train_with_trackman_match_info.csv)에서
    trackman_match_status != 'unmatched'인 행만 남기고 train.csv의
    pitcher_id/season/control_success를 조인한다. 원본 build_row_level_labels()와
    동일한 반환 스키마(row_id, trackman_id, pitcher_id, season, control_success)."""
    rowmatch = pd.read_csv(ROWMATCH_PATH, dtype={"trackman_id": "Int64"})
    matched = rowmatch[rowmatch["trackman_match_status"] != "unmatched"][["row_id", "trackman_id"]]

    label_df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                            usecols=["row_id", "pitcher_id", "season", "control_success"])
    out = matched.merge(label_df, on="row_id", how="inner")
    out["trackman_id"] = out["trackman_id"].astype("int64")
    return out


def build_labeled_trackman(row_labels, trm_all, trm_metrics, pitcher_map, cutoff_season):
    """원본과 동일 -- 다만 row_labels가 86.73% 매칭 소스."""
    real = row_labels[row_labels["season"] <= cutoff_season].merge(
        trm_all[["trackman_id", "pitch_type_group"] + METRICS], on="trackman_id", how="inner")
    real = real[["trackman_id", "pitcher_id", "season", "control_success", "pitch_type_group"] + METRICS]

    profiles, rookie_profile = build_reference_profiles(row_labels, trm_metrics, cutoff_season)
    pseudo = pseudo_label_unmatched(trm_all, pitcher_map, row_labels, profiles, rookie_profile, cutoff_season)

    return pd.concat([real, pseudo], ignore_index=True)


def merge_trackman_conditioned_v2(df_main, holdout, metrics=None, pitch_types=None):
    """merge_trackman_conditioned과 동일한 asof 관례, row_labels만 86.73% 소스."""
    row_labels = build_row_level_labels()
    trm_all = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")
    trm_clean = clean_trackman(trm_all)
    trm_metrics = trm_clean[["trackman_id"] + METRICS]
    pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")

    pieces = []
    feature_cols = None
    seasons = sorted(df_main["season"].unique())
    for season in seasons:
        cutoff_season = 2024 if holdout is None else min(season, holdout - 1)
        labeled_trm = build_labeled_trackman(row_labels, trm_clean, trm_metrics, pitcher_map, cutoff_season)
        lookup = build_conditioned_lookup(labeled_trm, metrics=metrics, pitch_types=pitch_types)
        if feature_cols is None:
            feature_cols = [c for c in lookup.columns if c != "pitcher_id"]
        rows = df_main[df_main["season"] == season]
        merged = pd.merge(rows, lookup, on="pitcher_id", how="left")
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True)
    result[feature_cols] = result[feature_cols].fillna(0.0)
    return result, feature_cols
