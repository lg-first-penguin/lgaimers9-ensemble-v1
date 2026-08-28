# code/build_trackman_rowmatch.py
"""train.csv <-> trackman_history.csv row-level 매칭 재구현 (2026-08-24, 순수 실험용).

`train-trackman_연결/train_trackman_matching_complete_report.md`에 문서화된 팀원의
방법론(경기 전체 10컬럼 exact unique 매칭 -> 삽입/삭제 허용 정렬 -> 안정 equal 구간)을
텍스트 스펙만 보고 독립적으로 재구현한 것이다. 원본 산출물(train_with_trackman_match_info.csv)은
공유되지 않았고, 여기서는 코드로 직접 다시 만든다.

원칙(보고서 5번과 동일하게 유지): 애매하면 매칭하지 않는다 — 틀린 ID를 붙이는 것이
빠뜨리는 것보다 더 나쁜 오류.

출력:
  open/temp/train_with_trackman_match_info.csv (row_id, trackman_id, trackman_match_status,
  trackman_match_basis, game_block_id)

사용법: python -m code.build_trackman_rowmatch
"""
import os
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

DATA_DIR = "./open/data"
OUT_DIR = "./open/temp"

_HAND_CODE = {"Left": 1, "Right": 2}
MATCH_COLS = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
              "balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand"]


def encode_tokens(df, top_bottom_map=None):
    """10개 공통 컬럼을 하나의 정수 토큰으로 결합한다 (mixed-radix, 문자열 결합보다 빠름)."""
    season = df["season"].astype(np.int64) - 2019
    month = df["game_month"].astype(np.int64) - 1
    dow = df["game_dayofweek"].astype(np.int64)
    inning = np.minimum(df["inning"].astype(np.int64), 29)
    if top_bottom_map is not None:
        tb = df["top_bottom"].map(top_bottom_map).astype(np.int64)
    else:
        tb = df["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)
    balls = df["balls_before"].astype(np.int64)
    strikes = df["strikes_before"].astype(np.int64)
    outs = df["outs_before"].astype(np.int64)
    phand = (df["pitcher_hand"].astype(np.int64) - 1)
    bhand = (df["batter_hand"].astype(np.int64) - 1)

    code = season
    code = code * 13 + month
    code = code * 8 + dow
    code = code * 30 + inning
    code = code * 2 + tb
    code = code * 4 + balls
    code = code * 3 + strikes
    code = code * 3 + outs
    code = code * 2 + phand
    code = code * 2 + bhand
    return code.values


def load_trackman():
    usecols = ["trackman_id", "trackman_game_id", "pitch_no"] + MATCH_COLS
    trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig", usecols=usecols)
    trm["top_bottom"] = trm["top_bottom"].map({"Top": "T", "Bottom": "B"})
    trm["pitcher_hand"] = trm["pitcher_hand"].map(_HAND_CODE)
    trm["batter_hand"] = trm["batter_hand"].map(_HAND_CODE)
    return trm


def filter_eligible_games(trm):
    row_ok = (
        (trm["inning"] >= 1)
        & trm["top_bottom"].isin(["T", "B"])
        & trm["balls_before"].between(0, 3)
        & trm["strikes_before"].between(0, 2)
        & trm["outs_before"].between(0, 2)
        & trm["pitcher_hand"].isin([1, 2])
        & trm["batter_hand"].isin([1, 2])
    )
    game_all_ok = row_ok.groupby(trm["trackman_game_id"]).transform("all")
    n_total_games = trm["trackman_game_id"].nunique()
    eligible = trm[game_all_ok].copy()
    n_eligible_games = eligible["trackman_game_id"].nunique()
    print(f"[트랙맨] 전체 경기 {n_total_games}개, 적격 경기 {n_eligible_games}개")
    eligible = eligible.sort_values(["trackman_game_id", "pitch_no", "trackman_id"]).reset_index(drop=True)
    eligible["token"] = encode_tokens(eligible)
    return eligible


def load_train():
    usecols = ["row_id", "season", "game_month", "game_dayofweek", "game_type",
               "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
               "pitcher_hand", "batter_hand"]
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig", usecols=usecols)
    df = df.sort_values("row_id").reset_index(drop=True)
    df["token"] = encode_tokens(df)
    return df


def build_game_blocks(df):
    season_chg = df["season"].ne(df["season"].shift(1))
    month_chg = df["game_month"].ne(df["game_month"].shift(1))
    dow_chg = df["game_dayofweek"].ne(df["game_dayofweek"].shift(1))
    type_chg = df["game_type"].ne(df["game_type"].shift(1))
    inning_dec = df["inning"].astype(np.int64).lt(df["inning"].astype(np.int64).shift(1))
    is_first_top = (df["inning"] == 1) & (df["top_bottom"] == "T")
    prev_first_top = is_first_top.shift(1, fill_value=False)
    new_first_top = is_first_top & (~prev_first_top)

    is_new_game = season_chg | month_chg | dow_chg | type_chg | inning_dec | new_first_top
    is_new_game.iloc[0] = True
    game_block_id = is_new_game.cumsum()
    df = df.copy()
    df["game_block_id"] = game_block_id.values
    n_blocks = df["game_block_id"].nunique()
    print(f"[Train] 경기 블록 {n_blocks}개")
    return df


def build_sequences(df, group_col):
    """group_col 기준으로 정렬된 순서의 token 튜플을 만든다. 반환: {group_id: tuple(tokens)}, {group_id: [row_index,...]}"""
    seqs = {}
    idxs = {}
    for gid, grp in df.groupby(group_col, sort=False):
        seqs[gid] = tuple(grp["token"].tolist())
        idxs[gid] = grp.index.tolist()
    return seqs, idxs


def exact_unique_match(train_seqs, trm_seqs):
    """양쪽에서 완전히 같은 토큰 시퀀스를 갖는 게임이 정확히 하나씩일 때만 매칭."""
    train_by_seq = defaultdict(list)
    for gid, seq in train_seqs.items():
        train_by_seq[seq].append(gid)
    trm_by_seq = defaultdict(list)
    for gid, seq in trm_seqs.items():
        trm_by_seq[seq].append(gid)

    matches = {}  # train_block_id -> trackman_game_id
    for seq, train_gids in train_by_seq.items():
        if len(train_gids) != 1:
            continue
        trm_gids = trm_by_seq.get(seq)
        if trm_gids is None or len(trm_gids) != 1:
            continue
        matches[train_gids[0]] = trm_gids[0]
    return matches


def align_remaining(train_seqs, train_idxs, trm_seqs, trm_idxs, trm_gid_to_ids, train_meta,
                     trm_meta, used_trm_games, min_run_len=3):
    """exact 매칭 안 된 나머지 train 블록에 대해 삽입/삭제 허용 정렬 수행.
    후보는 (season, game_month, game_dayofweek) 메타데이터가 같은, 아직 사용되지 않은
    trackman 게임으로 제한한다. 반환: row_matches = {train_row_index: trackman_id}."""
    # 메타데이터 버킷: (season, month, dow) -> [trm_game_id...]
    bucket = defaultdict(list)
    for gid in trm_seqs:
        if gid in used_trm_games:
            continue
        meta = trm_meta.get(gid)
        if meta is None:
            continue
        bucket[meta].append(gid)

    row_matches = {}
    used_trackman_ids = set()
    n_aligned_games = 0
    n_no_candidate = 0

    total_blocks = len(train_seqs)
    for progress_i, block_id in enumerate(train_seqs):
        if progress_i % 200 == 0:
            print(f"  [정렬 진행] {progress_i}/{total_blocks}")
        seq_a = train_seqs[block_id]
        meta = train_meta.get(block_id)
        candidates = bucket.get(meta, [])
        candidates = [c for c in candidates if c not in used_trm_games]
        if not candidates:
            n_no_candidate += 1
            continue

        best_gid, best_ratio, best_sm = None, -1.0, None
        for cand in candidates:
            seq_b = trm_seqs[cand]
            sm = SequenceMatcher(a=seq_a, b=seq_b, autojunk=False)
            r = sm.quick_ratio()
            if r < 0.5:
                continue
            r2 = sm.ratio()
            if r2 > best_ratio:
                best_ratio, best_gid, best_sm = r2, cand, sm

        if best_gid is None or best_ratio < 0.85:
            continue

        opcodes = best_sm.get_opcodes()
        train_row_idx = train_idxs[block_id]
        trm_row_idx = trm_idxs[best_gid]
        trm_id_seq = [None] * len(trm_row_idx)  # placeholder, filled via trm_ids lookup below

        conflict = False
        pending = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag != "equal":
                continue
            run_len = i2 - i1
            if run_len < min_run_len:
                continue
            pending.append((i1, i2, j1, j2))

        if not pending:
            continue

        # trackman_id 실제 값 조회
        gid_trackman_ids = trm_gid_to_ids.get(best_gid)
        if gid_trackman_ids is None:
            continue

        local_assign = {}
        ok = True
        for i1, i2, j1, j2 in pending:
            for offset in range(i2 - i1):
                tid = gid_trackman_ids[j1 + offset]
                if tid in used_trackman_ids:
                    ok = False
                    break
                local_assign[train_row_idx[i1 + offset]] = tid
            if not ok:
                break
        if not ok:
            continue

        row_matches.update(local_assign)
        used_trackman_ids.update(local_assign.values())
        used_trm_games.add(best_gid)
        n_aligned_games += 1

    print(f"[정렬] 정렬 매칭 게임 {n_aligned_games}개, 후보 없음 {n_no_candidate}개")
    return row_matches


def main():
    print("[1/6] 트랙맨 로드 및 정규화")
    trm = load_trackman()
    eligible = filter_eligible_games(trm)

    print("[2/6] Train 로드 및 경기 블록 생성")
    train = load_train()
    train = build_game_blocks(train)

    print("[3/6] 시퀀스 구성")
    train_seqs, train_idxs = build_sequences(train, "game_block_id")
    trm_seqs, trm_idxs = build_sequences(eligible, "trackman_game_id")

    trm_gid_to_ids = {}
    for gid, rows in trm_idxs.items():
        trm_gid_to_ids[gid] = eligible.loc[rows, "trackman_id"].tolist()

    train_block_meta = train.groupby("game_block_id")[["season", "game_month", "game_dayofweek"]].first()
    train_meta_dict = {gid: (row.season, row.game_month, row.game_dayofweek) for gid, row in train_block_meta.iterrows()}

    trm_block_meta = eligible.groupby("trackman_game_id")[["season", "game_month", "game_dayofweek"]].first()
    trm_meta_dict = {gid: (row.season, row.game_month, row.game_dayofweek) for gid, row in trm_block_meta.iterrows()}

    print("[4/6] 경기 전체 exact unique 매칭")
    exact_matches = exact_unique_match(train_seqs, trm_seqs)
    print(f"[exact] {len(exact_matches)}개 train 블록이 exact unique 매칭됨")

    row_trackman_id = {}
    used_trm_games = set(exact_matches.values())
    matched_train_blocks = set(exact_matches.keys())
    n_exact_rows = 0
    for block_id, trm_gid in exact_matches.items():
        train_rows = train_idxs[block_id]
        trm_ids = trm_gid_to_ids[trm_gid]
        for r, tid in zip(train_rows, trm_ids):
            row_trackman_id[r] = tid
        n_exact_rows += len(train_rows)
    print(f"[exact] {n_exact_rows}행 연결")

    print("[5/6] 나머지 경기 삽입/삭제 허용 정렬")
    remaining_train_seqs = {b: s for b, s in train_seqs.items() if b not in matched_train_blocks}
    remaining_train_idxs = {b: train_idxs[b] for b in remaining_train_seqs}
    align_matches = align_remaining(
        remaining_train_seqs, remaining_train_idxs, trm_seqs, trm_idxs, trm_gid_to_ids,
        train_meta_dict, trm_meta_dict, used_trm_games,
    )
    row_trackman_id.update(align_matches)
    print(f"[정렬] {len(align_matches)}행 추가 연결")

    print("[6/6] 결과 조립 및 검증")
    train["trackman_id"] = train.index.map(row_trackman_id.get)
    train["trackman_match_status"] = np.where(
        train.index.isin(exact_matches_row_index(exact_matches, train_idxs)), "exact_unique",
        np.where(train["trackman_id"].notna(), "aligned", "unmatched"),
    )

    total = len(train)
    matched = train["trackman_id"].notna().sum()
    print(f"전체 {total}행, 연결 {matched}행 ({matched/total*100:.2f}%)")

    dup_count = train["trackman_id"].dropna().duplicated().sum()
    print(f"trackman_id 중복 사용: {dup_count}건")

    # 공통 10컬럼 재검증
    trm_lookup = eligible.set_index("trackman_id")[MATCH_COLS]
    matched_train = train[train["trackman_id"].notna()].copy()
    joined = matched_train.merge(trm_lookup, left_on="trackman_id", right_index=True, suffixes=("", "_trm"))
    mismatch = 0
    for c in MATCH_COLS:
        if c == "top_bottom":
            trm_tb = joined[f"{c}_trm"]
            mismatch += (joined[c] != trm_tb).sum()
        elif c in ("pitcher_hand", "batter_hand"):
            mismatch += (joined[c] != joined[f"{c}_trm"]).sum()
        else:
            mismatch += (joined[c] != joined[f"{c}_trm"]).sum()
    print(f"공통 10컬럼 불일치: {mismatch}건")

    # pitch_no 역전 검사 (경기 블록 내부)
    pitch_no_map = eligible.set_index("trackman_id")["pitch_no"]
    matched_train["pitch_no"] = matched_train["trackman_id"].map(pitch_no_map)
    rev = 0
    for gid, grp in matched_train.groupby("game_block_id"):
        pn = grp["pitch_no"].values
        if len(pn) > 1 and (np.diff(pn) < 0).any():
            rev += 1
    print(f"경기 내부 pitch_no 역전 발생 블록: {rev}개")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = train[["row_id", "game_block_id", "trackman_id", "trackman_match_status"]]
    out.to_csv(os.path.join(OUT_DIR, "train_with_trackman_match_info.csv"), index=False)
    print(f"저장 완료: {OUT_DIR}/train_with_trackman_match_info.csv")


def exact_matches_row_index(exact_matches, train_idxs):
    idx = set()
    for block_id in exact_matches:
        idx.update(train_idxs[block_id])
    return idx


if __name__ == "__main__":
    main()
