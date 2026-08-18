# code/pitcher_crosswalk.py
"""train.csv의 pitcher_id/batter_id <-> trackman_history.csv의
pitcher_trackman_id/batter_trackman_id 크로스워크를 시퀀스 재구성으로 복원한다.

두 파일의 선수 ID 공간은 겹치지 않는다(overlap 0, EXPERIMENTS.md 14.1/25.1).
하지만 두 파일 모두 "투구 하나 = 한 행"이고, 같은 경기 안에서는 원본 저장 순서가
시간순이다(train: row_id 순, trackman: pitch_no 순). 외부 데이터 없이 두 공식 CSV의
컬럼만으로 아래 4단계를 거쳐 유효한 선수 단위 크로스워크를 복원한다:

  1) 경기 짝짓기: (season, game_month, game_dayofweek) 버킷 안에서, train 쪽 경기와
     trackman 쪽 경기(trackman_game_id)를 투구수(count)로 1:1 유일 대응되는 경우만 채택.
  2) 팀명 매핑: 1)에서 얻은 확정 게임 쌍으로부터 team_id(정수) <-> team(문자열) 대응을
     다수결로 복원.
  3) 시퀀스 검증: 짝지어진 두 경기의 inning/balls_before/strikes_before/outs_before
     시퀀스 전체가 정렬 후 일치하는지 대조. 일치율이 낮으면 그 경기 쌍은 폐기.
  4) 선수 다수결: 검증을 통과한 경기들에서 같은 위치의 pitcher_id/pitcher_trackman_id
     쌍을 누적, 확신도(최다득표 비율) 0.9 이상 & 총 투구수 100 이상만 채택.

사용법:
  python -m code.pitcher_crosswalk
출력:
  ./open/temp/pitcher_map.csv   (pitcher_id, pitcher_trackman_id, confidence, n_pitch)
  ./open/temp/batter_map.csv    (동일 구조)
"""
import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = "./open/data"
OUT_DIR = "./open/temp"


def load_train():
    cols = ["row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom",
            "balls_before", "strikes_before", "outs_before", "pitcher_id", "batter_id",
            "pitcher_team_id", "batter_team_id"]
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=cols)
    return df.sort_values("row_id").reset_index(drop=True)


def load_trackman():
    cols = ["trackman_id", "season", "game_month", "game_dayofweek", "trackman_game_id",
            "pitch_no", "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
            "pitcher_trackman_id", "batter_trackman_id", "pitcher_team", "batter_team"]
    df = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig", usecols=cols)
    return df.sort_values(["trackman_game_id", "pitch_no"]).reset_index(drop=True)


def segment_train_games(train):
    """train.csv를 원본(row_id) 순서로 훑으며 경기 경계를 탐지해 game_uid를 부여한다."""
    home = np.where(train["top_bottom"] == "T", train["pitcher_team_id"], train["batter_team_id"])
    away = np.where(train["top_bottom"] == "T", train["batter_team_id"], train["pitcher_team_id"])
    train = train.copy()
    train["home_team"] = home
    train["away_team"] = away

    pair_change = (train["home_team"] != train["home_team"].shift()) | (train["away_team"] != train["away_team"].shift())
    inning_reset = (train["inning"] < train["inning"].shift()) & ~pair_change
    new_game = (pair_change | inning_reset).values
    new_game[0] = True
    train["game_uid"] = np.cumsum(new_game)
    return train


def game_summary_train(train_seg):
    g = train_seg.groupby("game_uid").agg(
        season=("season", "first"), game_month=("game_month", "first"),
        game_dayofweek=("game_dayofweek", "first"),
        home_team=("home_team", "first"), away_team=("away_team", "first"),
        n_pitch=("row_id", "count"),
    ).reset_index()
    return g


def game_summary_trackman(trk):
    home = np.where(trk["top_bottom"] == "Top", trk["pitcher_team"], trk["batter_team"])
    away = np.where(trk["top_bottom"] == "Top", trk["batter_team"], trk["pitcher_team"])
    trk = trk.copy()
    trk["home_team"] = home
    trk["away_team"] = away
    g = trk.groupby("trackman_game_id").agg(
        season=("season", "first"), game_month=("game_month", "first"),
        game_dayofweek=("game_dayofweek", "first"),
        home_team=("home_team", "first"), away_team=("away_team", "first"),
        n_pitch=("trackman_id", "count"),
    ).reset_index()
    return g


def match_games_by_pitch_count(g_train, g_trk):
    """(season, game_month, game_dayofweek) 버킷 안에서 투구수가 양쪽 다 유일한 경우만 매칭."""
    bucket_cols = ["season", "game_month", "game_dayofweek"]
    pairs = []
    for key, tr_grp in g_train.groupby(bucket_cols):
        trk_grp = g_trk[(g_trk["season"] == key[0]) & (g_trk["game_month"] == key[1]) & (g_trk["game_dayofweek"] == key[2])]
        if len(trk_grp) == 0:
            continue
        tr_counts = tr_grp["n_pitch"].value_counts()
        trk_counts = trk_grp["n_pitch"].value_counts()
        unique_tr = set(tr_counts[tr_counts == 1].index)
        unique_trk = set(trk_counts[trk_counts == 1].index)
        common = unique_tr & unique_trk
        for n in common:
            tr_row = tr_grp[tr_grp["n_pitch"] == n].iloc[0]
            trk_row = trk_grp[trk_grp["n_pitch"] == n].iloc[0]
            pairs.append((tr_row["game_uid"], trk_row["trackman_game_id"], n))
    return pd.DataFrame(pairs, columns=["game_uid", "trackman_game_id", "n_pitch"])


def build_group_arrays(df, key_col, sort_col, needed_keys, value_cols):
    """needed_keys에 해당하는 행만 추려 key_col별로 정렬된 numpy 배열 dict를 만든다.
    반복 boolean-mask 필터링(O(n) x 매칭 쌍 수) 대신 단일 groupby로 O(n) 한 번에 끝낸다."""
    sub = df[df[key_col].isin(needed_keys)].sort_values([key_col, sort_col])
    out = {}
    for k, g in sub.groupby(key_col, sort=False):
        out[k] = {c: g[c].to_numpy() for c in value_cols}
    return out


def validate_sequence(train_seg, trk, pairs):
    """짝지어진 경기들의 이닝/카운트 시퀀스를 대조해 일치율 낮은 쌍은 버린다."""
    seq_cols = ["inning", "balls_before", "strikes_before", "outs_before"]
    train_groups = build_group_arrays(train_seg, "game_uid", "row_id", set(pairs["game_uid"]), seq_cols)
    trk_groups = build_group_arrays(trk, "trackman_game_id", "pitch_no", set(pairs["trackman_game_id"]), seq_cols)

    valid_pairs = []
    match_rates = []
    for _, row in pairs.iterrows():
        tr_g = train_groups.get(row["game_uid"])
        trk_g = trk_groups.get(row["trackman_game_id"])
        if tr_g is None or trk_g is None:
            continue
        n = min(len(tr_g["inning"]), len(trk_g["inning"]))
        if n == 0:
            continue
        tr_seq = np.stack([tr_g[c][:n] for c in seq_cols], axis=1)
        trk_seq = np.stack([trk_g[c][:n] for c in seq_cols], axis=1)
        match_rate = (tr_seq == trk_seq).all(axis=1).mean()
        match_rates.append(match_rate)
        if match_rate >= 0.95:
            valid_pairs.append(row["game_uid"])
    result = pairs[pairs["game_uid"].isin(valid_pairs)].reset_index(drop=True)
    print(f"  시퀀스 검증: {len(pairs)}쌍 중 {len(result)}쌍 통과 (>=95% 일치), "
          f"평균 일치율 {np.mean(match_rates) if match_rates else 0:.4f}")
    return result


def accumulate_votes(train_seg, trk, valid_pairs, id_col_train, id_col_trk):
    train_groups = build_group_arrays(train_seg, "game_uid", "row_id", set(valid_pairs["game_uid"]), [id_col_train])
    trk_groups = build_group_arrays(trk, "trackman_game_id", "pitch_no", set(valid_pairs["trackman_game_id"]), [id_col_trk])

    votes = {}
    for _, row in valid_pairs.iterrows():
        tr_g = train_groups.get(row["game_uid"])
        trk_g = trk_groups.get(row["trackman_game_id"])
        if tr_g is None or trk_g is None:
            continue
        n = min(len(tr_g[id_col_train]), len(trk_g[id_col_trk]))
        tr_ids = tr_g[id_col_train][:n]
        trk_ids = trk_g[id_col_trk][:n]
        for a, b in zip(tr_ids, trk_ids):
            votes.setdefault(a, {}).setdefault(b, 0)
            votes[a][b] += 1
    return votes


def resolve_votes(votes, min_conf=0.9, min_n=100):
    rows = []
    for main_id, counter in votes.items():
        total = sum(counter.values())
        if total < min_n:
            continue
        best_id, best_n = max(counter.items(), key=lambda kv: kv[1])
        conf = best_n / total
        if conf >= min_conf:
            rows.append({"id": main_id, "trackman_id": best_id, "confidence": conf, "n_pitch": total})
    return pd.DataFrame(rows)


def main():
    print("[1/5] 데이터 로드")
    train = load_train()
    trk = load_trackman()
    print(f"  train: {len(train)}행, trackman: {len(trk)}행")

    print("[2/5] 경기 경계 탐지 (train)")
    train_seg = segment_train_games(train)
    g_train = game_summary_train(train_seg)
    g_trk = game_summary_trackman(trk)
    print(f"  train 경기 수: {len(g_train)}, trackman 경기 수: {len(g_trk)}")

    print("[3/5] 경기 짝짓기 (투구수 유일 매칭)")
    pairs = match_games_by_pitch_count(g_train, g_trk)
    print(f"  후보 경기 쌍: {len(pairs)} / train 경기의 {len(pairs) / len(g_train) * 100:.1f}%")

    print("[4/5] 시퀀스 검증")
    valid_pairs = validate_sequence(train_seg, trk, pairs)

    print("[5/5] 선수 다수결")
    os.makedirs(OUT_DIR, exist_ok=True)

    pitcher_votes = accumulate_votes(train_seg, trk, valid_pairs, "pitcher_id", "pitcher_trackman_id")
    pitcher_map = resolve_votes(pitcher_votes)
    pitcher_map.rename(columns={"id": "pitcher_id", "trackman_id": "pitcher_trackman_id"}, inplace=True)
    pitcher_map.to_csv(os.path.join(OUT_DIR, "pitcher_map.csv"), index=False)
    print(f"  투수 매핑: {len(pitcher_map)}명 확정, 평균 확신도 {pitcher_map['confidence'].mean():.4f}, "
          f"train pitcher_id 커버리지 {len(pitcher_map) / train['pitcher_id'].nunique() * 100:.1f}%")

    batter_votes = accumulate_votes(train_seg, trk, valid_pairs, "batter_id", "batter_trackman_id")
    batter_map = resolve_votes(batter_votes)
    batter_map.rename(columns={"id": "batter_id", "trackman_id": "batter_trackman_id"}, inplace=True)
    batter_map.to_csv(os.path.join(OUT_DIR, "batter_map.csv"), index=False)
    print(f"  타자 매핑: {len(batter_map)}명 확정, 평균 확신도 {batter_map['confidence'].mean():.4f}, "
          f"train batter_id 커버리지 {len(batter_map) / train['batter_id'].nunique() * 100:.1f}%")

    dup_p = pitcher_map["pitcher_trackman_id"].duplicated().sum()
    dup_b = batter_map["batter_trackman_id"].duplicated().sum()
    print(f"  1:1 중복 체크: pitcher {dup_p}건, batter {dup_b}건 (0이어야 정상)")


if __name__ == "__main__":
    main()
