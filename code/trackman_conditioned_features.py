# code/trackman_conditioned_features.py
"""사용자 제안(2026-08-22): train<->trackman 행 단위(1:1) 매칭을 실제로 만들어서
(code/pitcher_crosswalk.py가 선수 크로스워크를 만들 때 내부적으로 쓰고 버리는, 게임
시퀀스 검증까지 마친 포지션 정렬을 재사용 — code/experiment_trackman_rowlevel_diagnostic.py
에서 진단), control_success=1/0 그룹 간 물리량(rel_speed 등) std 차이를 확인했다.

진단 결과(2026-08-22): 53.39%의 train 행이 검증된 행 단위 매칭을 얻음(기존의 "상황
지문 근사 조인"과 달리 게임 전체 시퀀스 >=95% 일치를 확인한 진짜 매칭). 매칭된 행에서
success 그룹의 std가 fail 그룹보다 대부분 작다(rel_speed -3~5%, spin_rate -3%,
induced_vert_break -6.5%, zone_speed -2.5%; horz_break/rel_height/rel_side는 노이즈
수준). 투수x시즌 단위(표본 20+ 양쪽)로 봐도 방향성이 우연 수준을 넘는다(induced_vert_break
는 1706개 투수x시즌 조합 중 71.3%에서 success std < fail std). 이건 기존에 4번 독립
기각된 "상황 지문 근사 조인" 라인과 메커니즘이 다르다(근사 조인이 아니라 검증된 실제
매칭이라 노이즈 발생 경로가 다름) — 그래서 재시도할 가치가 있다고 판단해 다음 단계로
진행한다.

이 모듈은 그 신호를 실제 피처로 만든다:
  1. 실제 매칭된 행(진짜 라벨)으로 투수x시즌별 물리 지표 mean/std 프로필(성공군/실패군)을
     추정한다(양쪽 표본 20개 이상만 채택).
  2. 매칭되지 않은 트랙맨 행(크로스워크로 pitcher_id는 알지만 train 행과 연결은 안 되는
     행)은, 그 투수x시즌 프로필과의 (표준화)거리로 성공/실패를 의사 라벨링한다. 그 투수x
     시즌 프로필이 없으면(표본 부족/신인) "그 시즌까지의 모든 신인(그 투수 자신의 프로필
     데이터 안에서 최초로 등장한 시즌) 프로필 평균"을 대체 임계값으로 쓴다.
  3. (실제+의사) 라벨이 붙은 트랙맨 데이터로, tier A와 같은 투수x구종군 단위지만 성공/
     실패로 조건화한 mean/std를 만들고, "성공군-실패군" 차이만 남겨 tier A와 같은 컬럼
     수로 압축한다(64컬럼) — 리크 방지 asof 클램프(`cutoff_season=min(season,holdout-1)`)
     는 `merge_asof_pitcher_std`와 동일한 관례를 따른다.

사용법(모듈, 직접 실행 없음): code/experiment_trackman_conditioned_screen.py에서 사용.
"""
import numpy as np
import pandas as pd

from code.pitcher_crosswalk import (
    game_summary_train, game_summary_trackman, load_train, load_trackman,
    match_games_by_pitch_count, segment_train_games, validate_sequence,
)
from code.trackman_pitcher_features import METRICS, clean_trackman

INFORMATIVE_METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "zone_speed"]
MIN_GROUP_N = 20
COND_PREFIX = "trkcond_"
DATA_DIR = "./open/data"


def build_row_level_labels():
    """게임 구조 기반 검증(1회, 시즌 무관)으로 train row_id <-> trackman_id 실제 매칭 +
    control_success 라벨 테이블을 만든다. code/experiment_trackman_rowlevel_diagnostic.py와
    동일 로직."""
    train = load_train()
    trk = load_trackman()
    train_seg = segment_train_games(train)
    g_train = game_summary_train(train_seg)
    g_trk = game_summary_trackman(trk)
    pairs = match_games_by_pitch_count(g_train, g_trk)
    valid_pairs = validate_sequence(train_seg, trk, pairs)

    seq_cols = ["inning", "balls_before", "strikes_before", "outs_before"]
    train_sub = train_seg[train_seg["game_uid"].isin(set(valid_pairs["game_uid"]))]
    trk_sub = trk[trk["trackman_game_id"].isin(set(valid_pairs["trackman_game_id"]))]
    train_groups = {k: g.sort_values("row_id") for k, g in train_sub.groupby("game_uid")}
    trk_groups = {k: g.sort_values("pitch_no") for k, g in trk_sub.groupby("trackman_game_id")}

    pieces = []
    for _, row in valid_pairs.iterrows():
        tg = train_groups.get(row["game_uid"])
        kg = trk_groups.get(row["trackman_game_id"])
        if tg is None or kg is None:
            continue
        n = min(len(tg), len(kg))
        if n == 0:
            continue
        tg_n, kg_n = tg.iloc[:n], kg.iloc[:n]
        match_mask = np.all([tg_n[c].to_numpy() == kg_n[c].to_numpy() for c in seq_cols], axis=0)
        pieces.append(pd.DataFrame({
            "row_id": tg_n["row_id"].to_numpy()[match_mask],
            "trackman_id": kg_n["trackman_id"].to_numpy()[match_mask],
        }))
    row_map = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["row_id", "trackman_id"])

    label_df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                            usecols=["row_id", "pitcher_id", "season", "control_success"])
    return row_map.merge(label_df, on="row_id", how="inner")


def build_reference_profiles(row_labels, trm_metrics, cutoff_season):
    """season<=cutoff_season인 실제 매칭 라벨로 (pitcher_id, season)별 성공/실패군
    mean/std 프로필(표본 20+ 양쪽) + '신인 프로필'(각 투수의 season<=cutoff_season 내
    최초 등장 시즌들을 풀링한 전역 평균)을 만든다."""
    merged = row_labels[row_labels["season"] <= cutoff_season].merge(trm_metrics, on="trackman_id", how="inner")
    rows = []
    for (pid, season), sub in merged.groupby(["pitcher_id", "season"]):
        g1, g0 = sub[sub["control_success"] == 1], sub[sub["control_success"] == 0]
        if len(g1) < MIN_GROUP_N or len(g0) < MIN_GROUP_N:
            continue
        rec = {"pitcher_id": pid, "season": season, "n1": len(g1), "n0": len(g0)}
        for m in INFORMATIVE_METRICS:
            rec[f"{m}_mean1"], rec[f"{m}_std1"] = g1[m].mean(), max(g1[m].std(), 1e-6)
            rec[f"{m}_mean0"], rec[f"{m}_std0"] = g0[m].mean(), max(g0[m].std(), 1e-6)
        rows.append(rec)
    profiles = pd.DataFrame(rows)
    if len(profiles) == 0:
        rookie_profile = None
    else:
        debut = profiles.groupby("pitcher_id")["season"].min().rename("debut_season")
        rookie_rows = profiles.merge(debut, on="pitcher_id").query("season == debut_season")
        rookie_profile = {f"{m}_{stat}{g}": rookie_rows[f"{m}_{stat}{g}"].mean()
                           for m in INFORMATIVE_METRICS for stat in ("mean", "std") for g in ("1", "0")}
    return profiles, rookie_profile


def pseudo_label_unmatched(trm_all, pitcher_map, row_labels, profiles, rookie_profile, cutoff_season):
    """season<=cutoff_season 트랙맨 행 중 실제 매칭 안 된(하지만 pitcher_id 크로스워크는
    되는) 행에 (pitcher_id, season) 프로필과의 표준화 거리로 성공/실패 의사 라벨을 매긴다."""
    trm_cut = trm_all[trm_all["season"] <= cutoff_season].copy()
    matched_ids = set(row_labels.loc[row_labels["season"] <= cutoff_season, "trackman_id"])
    trm_unmatched = trm_cut[~trm_cut["trackman_id"].isin(matched_ids)]
    trm_unmatched = trm_unmatched.merge(pitcher_map[["pitcher_trackman_id", "pitcher_id"]],
                                         on="pitcher_trackman_id", how="inner")
    if len(trm_unmatched) == 0 or rookie_profile is None:
        return pd.DataFrame(columns=["trackman_id", "pitcher_id", "season", "control_success"] + METRICS)

    merged = trm_unmatched.merge(profiles, on=["pitcher_id", "season"], how="left", suffixes=("", "_prof"))
    has_own = merged["n1"].notna()

    dist1 = np.zeros(len(merged))
    dist0 = np.zeros(len(merged))
    for m in INFORMATIVE_METRICS:
        mean1 = np.where(has_own, merged[f"{m}_mean1"], rookie_profile[f"{m}_mean1"])
        std1 = np.where(has_own, merged[f"{m}_std1"], rookie_profile[f"{m}_std1"])
        mean0 = np.where(has_own, merged[f"{m}_mean0"], rookie_profile[f"{m}_mean0"])
        std0 = np.where(has_own, merged[f"{m}_std0"], rookie_profile[f"{m}_std0"])
        val = merged[m].to_numpy()
        valid = ~np.isnan(val)
        z1 = np.where(valid, ((val - mean1) / std1) ** 2, 0.0)
        z0 = np.where(valid, ((val - mean0) / std0) ** 2, 0.0)
        dist1 += z1
        dist0 += z0

    pseudo_label = (dist1 < dist0).astype(np.int64)
    out = merged[["trackman_id", "pitcher_id", "season", "pitch_type_group"] + METRICS].copy()
    out["control_success"] = pseudo_label
    return out


def build_labeled_trackman(row_labels, trm_all, trm_metrics, pitcher_map, cutoff_season):
    """season<=cutoff_season 범위에서 (실제+의사) 라벨이 붙은 트랙맨 테이블을 만든다."""
    real = row_labels[row_labels["season"] <= cutoff_season].merge(
        trm_all[["trackman_id", "pitch_type_group"] + METRICS], on="trackman_id", how="inner")
    real = real[["trackman_id", "pitcher_id", "season", "control_success", "pitch_type_group"] + METRICS]

    profiles, rookie_profile = build_reference_profiles(row_labels, trm_metrics, cutoff_season)
    pseudo = pseudo_label_unmatched(trm_all, pitcher_map, row_labels, profiles, rookie_profile, cutoff_season)

    return pd.concat([real, pseudo], ignore_index=True)


def build_conditioned_lookup(labeled_trm, metrics=None, pitch_types=None):
    """투수x구종군별 성공군/실패군 mean/std를 만들고, 성공-실패 차이만 남긴다(tier A와
    동일한 컬럼 수로 압축). metrics/pitch_types를 좁히면(예: induced_vert_break만,
    breaking만) 컬럼 수도 그만큼 줄어든다 -- 2026-08-22 세션, t-test 효과크기 분석에서
    induced_vert_break x breaking 조합만 상대적으로 두드러져서(Cohen's d~0.13, 나머지는
    전부 <0.07) 이 조합만 좁혀 재검증하기 위해 추가."""
    metrics = metrics if metrics is not None else METRICS
    if pitch_types is not None:
        labeled_trm = labeled_trm[labeled_trm["pitch_type_group"].isin(pitch_types)]
    g = labeled_trm.groupby(["pitcher_id", "pitch_type_group", "control_success"])[metrics].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    std_cols = [c for c in g.columns if c.endswith("_std")]
    g[std_cols] = g[std_cols].fillna(0.0)

    p1 = g[g["control_success"] == 1].drop(columns="control_success").set_index(["pitcher_id", "pitch_type_group"])
    p0 = g[g["control_success"] == 0].drop(columns="control_success").set_index(["pitcher_id", "pitch_type_group"])
    diff = (p1 - p0).dropna(how="all").fillna(0.0).reset_index()

    pivoted = diff.set_index(["pitcher_id", "pitch_type_group"]).unstack(level="pitch_type_group")
    pivoted.columns = ["_".join(str(x) for x in c) for c in pivoted.columns]
    pivoted = pivoted.reset_index().fillna(0.0)
    rename = {c: COND_PREFIX + c for c in pivoted.columns if c != "pitcher_id"}
    return pivoted.rename(columns=rename)


def merge_trackman_conditioned(df_main, holdout, metrics=None, pitch_types=None):
    """merge_asof_pitcher_std와 동일한 asof/리크방지 관례: holdout 행(및 그 이후)의
    자기 시즌 트랙맨은 못 보게 cutoff_season=min(season, holdout-1)로 클램프.
    holdout=None이면 전체 히스토리(실전 추론/Full Retrain 전용, build_lookup_full_history와
    동일 관례). metrics/pitch_types는 build_conditioned_lookup으로 그대로 전달."""
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
