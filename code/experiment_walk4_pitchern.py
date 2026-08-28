# code/experiment_walk4_pitchern.py
"""code/experiment_walk4_countcarry.py 수정판. "카운트가 동일"을 balls_before/
strikes_before(현재 타석 볼카운트) 대신 asof_pitcher_n(해당 투구 직전까지 그 투수의
누적 실제 투구 수, 공식 as-of 피처)으로 검사한다 — 이게 진짜 "그 투수가 실제로 몇 개
던졌는지"를 보여주는 컬럼이라 훨씬 직접적인 신호다.

같은 게임/이닝/아웃카운트, pitcher_id 동일, 타자만 i->i+1에서 교체, 1루로 주자 진출
조건은 그대로 두고, asof_pitcher_n 비교를 두 가지로 나눠서 찾는다:
  A) row i+1의 asof_pitcher_n == row i의 asof_pitcher_n (그 투수 누적 투구수가 전혀
     안 늘어남 -> 실제 투구가 없었다는 뜻)
  B) row i+1의 asof_pitcher_n == row i의 asof_pitcher_n + 4 (행 하나만에 누적
     투구수가 4 증가 -> 중간에 기록 안 된 투구 4개가 있었다는 뜻)

사용법: python -m code.experiment_walk4_pitchern
"""
import os

import pandas as pd

from code.pitcher_crosswalk import segment_train_games

DATA_DIR = "./open/data"


def load():
    cols = ["row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom",
            "balls_before", "strikes_before", "outs_before",
            "runner_on_1b", "runner_on_2b", "runner_on_3b",
            "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
            "asof_pitcher_n", "control_success"]
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=cols)
    df = df.sort_values("row_id").reset_index(drop=True)
    seg = segment_train_games(df)
    seg = seg.sort_values(["game_uid", "row_id"]).reset_index(drop=True)
    return seg


def base_candidates(seg):
    nxt_r1 = seg["runner_on_1b"].shift(-1)
    nxt_outs = seg["outs_before"].shift(-1)
    nxt_batter = seg["batter_id"].shift(-1)
    nxt_pitcher = seg["pitcher_id"].shift(-1)
    nxt_inning = seg["inning"].shift(-1)
    nxt_tb = seg["top_bottom"].shift(-1)
    same_game = seg["game_uid"] == seg["game_uid"].shift(-1)

    same_inning = (nxt_inning == seg["inning"]) & (nxt_tb == seg["top_bottom"])
    same_pitcher = nxt_pitcher == seg["pitcher_id"]
    same_outs = nxt_outs == seg["outs_before"]
    new_batter = nxt_batter != seg["batter_id"]
    runner_to_1b = nxt_r1 == 1

    mask = same_game & same_inning & same_pitcher & same_outs & new_batter & runner_to_1b
    nxt_pn = seg["asof_pitcher_n"].shift(-1)
    return mask, nxt_pn


def main():
    seg = load()
    mask, nxt_pn = base_candidates(seg)
    print(f"[기본 조건] 같은게임/이닝/아웃/투수, 타자 교체, 1루 진출: {mask.sum()}건")

    diff = nxt_pn - seg["asof_pitcher_n"]

    same_n = seg[mask & (diff == 0)].copy()
    print(f"\n=== A) asof_pitcher_n 동일(실투구 0) : {len(same_n)}건 ===")
    print(same_n["control_success"].value_counts(dropna=False))
    if len(same_n) > 0:
        print(same_n[["row_id", "season", "inning", "asof_pitcher_n",
                       "pitcher_id", "batter_id", "control_success"]].head(20).to_string(index=False))

    plus4 = seg[mask & (diff == 4)].copy()
    print(f"\n=== B) asof_pitcher_n이 행 하나만에 +4 : {len(plus4)}건 ===")
    print(plus4["control_success"].value_counts(dropna=False))
    if len(plus4) > 0:
        print(plus4[["row_id", "season", "inning", "asof_pitcher_n",
                      "pitcher_id", "batter_id", "control_success"]].head(20).to_string(index=False))

    print("\n=== 참고: diff 전체 분포(기본 조건 만족 행들) ===")
    print(diff[mask].value_counts().sort_index().head(20))


if __name__ == "__main__":
    main()
