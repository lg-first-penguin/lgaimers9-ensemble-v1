# code/experiment_walk4_pitcherfix.py
"""사용자가 지적한 정밀화: 기존 시나리오1/2(자동고의사구 후보 / 순수 4구 연속 볼)
탐지에 pitcher_id가 그대로 유지되는지를 빠뜨렸었다. 이 스크립트는 두 시나리오를
pitcher_id 동일 조건을 추가해 다시 계산한다.

시나리오1 (투구수 0 증가, "투수는 안 던졌는데 타자만 바뀜"): 같은 게임+이닝+아웃카운트
+투수id, 타자id만 바뀌고, 주자 상황이 강제진루 공식과 정확히 일치하는 인접 행 쌍.
(직전 행 자신이 balls_before==3인 "본인 4구째 볼넷"은 시나리오2와 겹치므로 제외)

시나리오2 (투구수 4 증가, "순수 4구 연속 볼"): 같은 타자+같은 게임 내에서
balls_before가 0->1->2->3으로 이어지고 strikes_before가 계속 0으로 유지된 4개 행
전부가 같은 pitcher_id인 경우만 인정.

사용법: python -m code.experiment_walk4_pitcherfix
"""
import os

import pandas as pd
import numpy as np

from code.pitcher_crosswalk import segment_train_games

DATA_DIR = "./open/data"


def load():
    cols = ["row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom",
            "balls_before", "strikes_before", "outs_before",
            "runner_on_1b", "runner_on_2b", "runner_on_3b",
            "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id", "control_success"]
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=cols)
    df = df.sort_values("row_id").reset_index(drop=True)
    seg = segment_train_games(df)
    seg = seg.sort_values(["game_uid", "row_id"]).reset_index(drop=True)
    return seg


def scenario1(seg):
    nxt_r1 = seg["runner_on_1b"].shift(-1)
    nxt_r2 = seg["runner_on_2b"].shift(-1)
    nxt_r3 = seg["runner_on_3b"].shift(-1)
    nxt_outs = seg["outs_before"].shift(-1)
    nxt_batter = seg["batter_id"].shift(-1)
    nxt_pitcher = seg["pitcher_id"].shift(-1)
    nxt_inning = seg["inning"].shift(-1)
    nxt_tb = seg["top_bottom"].shift(-1)
    same_game = seg["game_uid"] == seg["game_uid"].shift(-1)

    exp_r1 = 1
    exp_r2 = np.where(seg["runner_on_1b"] == 1, 1, seg["runner_on_2b"])
    exp_r3 = np.where((seg["runner_on_1b"] == 1) & (seg["runner_on_2b"] == 1), 1, seg["runner_on_3b"])

    forced_match = (nxt_r1 == exp_r1) & (nxt_r2 == exp_r2) & (nxt_r3 == exp_r3)
    outs_same = nxt_outs == seg["outs_before"]
    new_batter = nxt_batter != seg["batter_id"]
    same_pitcher = nxt_pitcher == seg["pitcher_id"]
    same_inning = (nxt_inning == seg["inning"]) & (nxt_tb == seg["top_bottom"])
    actually_changed = (seg["runner_on_1b"] != exp_r1) | (seg["runner_on_2b"] != exp_r2) | (seg["runner_on_3b"] != exp_r3)
    not_own_walk = ~((seg["balls_before"] == 3) & (seg["strikes_before"] < 3))

    cand = same_game & same_inning & same_pitcher & outs_same & new_batter & forced_match & actually_changed & not_own_walk
    return seg[cand].copy()


def scenario2(seg):
    b = seg
    same_batter = [b["batter_id"] == b["batter_id"].shift(k) for k in (1, 2, 3)]
    same_pitcher = [b["pitcher_id"] == b["pitcher_id"].shift(k) for k in (1, 2, 3)]
    same_game = [b["game_uid"] == b["game_uid"].shift(k) for k in (1, 2, 3)]
    same_inning = [(b["inning"] == b["inning"].shift(k)) & (b["top_bottom"] == b["top_bottom"].shift(k)) for k in (1, 2, 3)]
    is_ball4 = (
        (b["balls_before"] == 3) & (b["strikes_before"] == 0)
        & (b["balls_before"].shift(1) == 2) & (b["strikes_before"].shift(1) == 0)
        & same_batter[0] & same_pitcher[0] & same_game[0] & same_inning[0]
        & (b["balls_before"].shift(2) == 1) & (b["strikes_before"].shift(2) == 0)
        & same_batter[1] & same_pitcher[1] & same_game[1] & same_inning[1]
        & (b["balls_before"].shift(3) == 0) & (b["strikes_before"].shift(3) == 0)
        & same_batter[2] & same_pitcher[2] & same_game[2] & same_inning[2]
    )
    return b[is_ball4].copy()


def main():
    seg = load()

    s1 = scenario1(seg)
    print(f"[시나리오1: 투구수 0증가, pitcher_id 동일 추가] 총 {len(s1)}건")
    print(s1["control_success"].value_counts(dropna=False))
    sub00 = s1[(s1["balls_before"] == 0) & (s1["strikes_before"] == 0)]
    print(f"  그 중 직전 행이 0/0 카운트: {len(sub00)}건")
    print(sub00["control_success"].value_counts(dropna=False))

    print()
    s2 = scenario2(seg)
    print(f"[시나리오2: 투구수 4증가(순수 연속볼), pitcher_id 동일 추가] 총 {len(s2)}건")
    print(s2["control_success"].value_counts(dropna=False))

    # pitcher_id 동일 조건이 실제로 몇 건을 걸러냈는지 (이전 버전과 비교용)
    b = seg
    same_batter = [b["batter_id"] == b["batter_id"].shift(k) for k in (1, 2, 3)]
    same_game = [b["game_uid"] == b["game_uid"].shift(k) for k in (1, 2, 3)]
    is_ball4_no_pitcher_check = (
        (b["balls_before"] == 3) & (b["strikes_before"] == 0)
        & (b["balls_before"].shift(1) == 2) & (b["strikes_before"].shift(1) == 0) & same_batter[0] & same_game[0]
        & (b["balls_before"].shift(2) == 1) & (b["strikes_before"].shift(2) == 0) & same_batter[1] & same_game[1]
        & (b["balls_before"].shift(3) == 0) & (b["strikes_before"].shift(3) == 0) & same_batter[2] & same_game[2]
    )
    print(f"\n[비교] pitcher_id 동일 조건 없이(기존): {is_ball4_no_pitcher_check.sum()}건 "
          f"-> pitcher_id 동일 추가 후: {len(s2)}건 (차이 {is_ball4_no_pitcher_check.sum() - len(s2)}건은 "
          f"PA 도중 투수 교체가 낀 경우로 추정)")


if __name__ == "__main__":
    main()
