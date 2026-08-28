# code/experiment_trackman_rowlevel_diagnostic.py
"""사용자 제안 진단: train<->trackman 진짜 행 단위(1:1) 매칭이 실제로 존재하는가,
그리고 매칭된 행에서 control_success=1/0 그룹 간 물리량 std가 실제로 다른가.

`code/pitcher_crosswalk.py`는 이미 검증된 게임 단위 시퀀스 정렬(inning/balls/strikes/outs
전체 시퀀스 일치율 >=95%)로 train game_uid <-> trackman_game_id 쌍을 만들고, 그 안에서
같은 위치(포지션)의 pitcher_id<->pitcher_trackman_id를 다수결로 뽑아 선수 크로스워크를
만든다. 이 포지션 정렬 자체가 사실상 "행 단위 1:1 매칭"인데, 기존 코드는 이걸 선수
다수결에만 쓰고 버린다. 이 스크립트는 그 정렬을 재사용해 (train row_id <-> trackman_id)
행 단위 매핑을 실제로 만들고, control_success별 물리량 std 차이가 있는지 확인한다.

사용법: python -m code.experiment_trackman_rowlevel_diagnostic
"""
import os

import numpy as np
import pandas as pd

from code.pitcher_crosswalk import (
    game_summary_train, game_summary_trackman, load_train, load_trackman,
    match_games_by_pitch_count, segment_train_games, validate_sequence,
)
from code.trackman_pitcher_features import METRICS, clean_trackman

DATA_DIR = "./open/data"


def build_row_level_map(train_seg, trk, valid_pairs):
    """검증된 게임 쌍 안에서, 정렬된 포지션끼리 train row_id <-> trackman_id를 짝짓는다."""
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
        # 시퀀스(inning/balls/strikes/outs) 위치별 완전 일치하는 행만 채택 (부분 오정렬 제거)
        match_mask = np.all(
            [tg_n[c].to_numpy() == kg_n[c].to_numpy() for c in seq_cols], axis=0
        )
        piece = pd.DataFrame({
            "row_id": tg_n["row_id"].to_numpy()[match_mask],
            "trackman_id": kg_n["trackman_id"].to_numpy()[match_mask],
        })
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["row_id", "trackman_id"])


def main():
    print("[1/6] train/trackman 로드 (crosswalk 전용 컬럼)")
    train = load_train()
    trk = load_trackman()
    print(f"  train: {len(train)}행, trackman: {len(trk)}행")

    print("[2/6] 게임 경계 탐지 + 짝짓기 + 시퀀스 검증 (pitcher_crosswalk.py 재사용)")
    train_seg = segment_train_games(train)
    g_train = game_summary_train(train_seg)
    g_trk = game_summary_trackman(trk)
    pairs = match_games_by_pitch_count(g_train, g_trk)
    print(f"  후보 경기 쌍: {len(pairs)} / train 경기의 {len(pairs) / len(g_train) * 100:.1f}%")
    valid_pairs = validate_sequence(train_seg, trk, pairs)

    print("[3/6] 행 단위(포지션 정렬) 매핑 생성")
    row_map = build_row_level_map(train_seg, trk, valid_pairs)
    print(f"  행 단위 매핑: {len(row_map)}쌍 (train 전체의 {len(row_map) / len(train) * 100:.2f}%)")

    print("[4/6] 라벨(control_success) + 물리량 로드")
    label_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig",
                            usecols=["row_id", "pitcher_id", "season", "control_success"])
    trm_full = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    trm_clean = clean_trackman(trm_full)
    trm_metrics = trm_clean[["trackman_id"] + METRICS]

    merged = row_map.merge(label_df, on="row_id", how="inner").merge(trm_metrics, on="trackman_id", how="inner")
    print(f"  클렌징 후 최종 매칭: {len(merged)}행 (control_success=1: {merged['control_success'].mean():.4f})")

    print("\n[5/6] 전체 풀링 기준 control_success별 물리량 std 비교")
    for m in METRICS:
        s1 = merged.loc[merged["control_success"] == 1, m].std()
        s0 = merged.loc[merged["control_success"] == 0, m].std()
        n1 = merged.loc[merged["control_success"] == 1, m].notna().sum()
        n0 = merged.loc[merged["control_success"] == 0, m].notna().sum()
        print(f"  {m:22s} std(success)={s1:.4f} (n={n1}) | std(fail)={s0:.4f} (n={n0}) | ratio={s1/s0 if s0 else float('nan'):.4f}")

    print("\n[6/6] 투수x시즌 단위 (표본 20+ 양쪽 그룹 필수) std 비교 요약")
    grp = merged.groupby(["pitcher_id", "season", "control_success"])[METRICS].agg(["std", "count"])
    rows = []
    for (pid, season), sub in merged.groupby(["pitcher_id", "season"]):
        g1 = sub[sub["control_success"] == 1]
        g0 = sub[sub["control_success"] == 0]
        if len(g1) < 20 or len(g0) < 20:
            continue
        for m in METRICS:
            rows.append({"pitcher_id": pid, "season": season, "metric": m,
                         "std1": g1[m].std(), "std0": g0[m].std(),
                         "n1": len(g1), "n0": len(g0)})
    ps_df = pd.DataFrame(rows)
    if len(ps_df) == 0:
        print("  표본 부족: 투수x시즌x양쪽그룹 20+ 조건을 만족하는 조합이 없음")
    else:
        n_pairs = ps_df.groupby(["pitcher_id", "season"]).ngroups
        print(f"  조건 만족 투수x시즌 조합: {n_pairs}개")
        for m in METRICS:
            sub = ps_df[ps_df["metric"] == m]
            diff = (sub["std1"] - sub["std0"])
            frac_lower = (diff < 0).mean()
            print(f"  {m:22s} mean(std1-std0)={diff.mean():+.4f} | success가 더 작은 비율={frac_lower:.2%} (n={len(sub)})")


if __name__ == "__main__":
    main()
