# code/experiment_walk4_countcarry.py
"""사용자 지시: 같은 경기/이닝/아웃카운트, pitcher_id 동일, 타자id만 i->i+1 사이에
바뀌고, "투구카운트가 i행과 i+1행에서 동일한 값"(balls_before, strikes_before가
정확히 같음)이면서 1루로 주자가 나가는(강제진루) 케이스를 찾는다.

정상적인 타석은 항상 0-0에서 시작하므로, 새 타자(i+1행)의 카운트가 0-0이 아닌데도
직전 행(i행)의 카운트와 정확히 같다면 이례적인 신호다 — 실제 투구 없이 이전 상태가
그대로 "복사"된 것처럼 보이는 행일 수 있다(자동고의사구처럼 실제 투구 시퀀스 없이
발생하는 이벤트의 아티팩트 가능성).

사용법: python -m code.experiment_walk4_countcarry
"""
from code.experiment_walk4_pitcherfix import load


def main():
    seg = load()
    nxt_balls = seg["balls_before"].shift(-1)
    nxt_strikes = seg["strikes_before"].shift(-1)
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
    count_carried = (nxt_balls == seg["balls_before"]) & (nxt_strikes == seg["strikes_before"])
    runner_to_1b = nxt_r1 == 1

    cand = same_game & same_inning & same_pitcher & same_outs & new_batter & count_carried & runner_to_1b
    result = seg[cand].copy()
    print(f"총 {len(result)}건 (카운트가 i행과 i+1행에서 정확히 동일 + 1루로 주자 진출 + 타자만 교체)")
    print("\n카운트 분포 (row i 기준, = row i+1과 동일):")
    print(result.groupby(["balls_before", "strikes_before"]).size().sort_values(ascending=False))

    nontrivial = result[~((result["balls_before"] == 0) & (result["strikes_before"] == 0))]
    print(f"\n--- 그 중 0-0이 아닌(=이례적인) 케이스: {len(nontrivial)}건 ---")
    if len(nontrivial) > 0:
        print(nontrivial["control_success"].value_counts(dropna=False))
        print(nontrivial[["row_id", "season", "inning", "balls_before", "strikes_before",
                           "pitcher_id", "batter_id", "control_success"]].head(20).to_string(index=False))
    else:
        print("(해당 없음)")

    trivial00 = result[(result["balls_before"] == 0) & (result["strikes_before"] == 0)]
    print(f"\n--- 0-0인(=자연스러운 새 타석 시작) 케이스: {len(trivial00)}건 ---")
    print(trivial00["control_success"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
