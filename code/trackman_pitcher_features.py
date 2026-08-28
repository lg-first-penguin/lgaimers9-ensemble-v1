# code/trackman_pitcher_features.py
"""진짜 투수 정체성(pitcher_crosswalk.py) x 구종군별 트랙맨 물리 지표(평균+표준편차) 피처,
그리고 볼카운트x손 조합별 구종 비중(coarse pitchmix) 피처 생성 — 프로덕션 파이프라인
(train.py/test.py/dopip.py/submit/script.py)에서 재사용하기 위해 분리.

2026-08-17 세션 경과(중요, 아래 결론이 이 파일의 현재 상태를 결정함):
  1) tier A(투수x구종군)/B(+압박)/C(+타자손)를 cutoff=7 스플릿으로 검증해 A/B/C 전부
     프로덕션에 반영했으나(§38), 실제 리더보드 869.52로 대폭 하락(982.22 대비 -112.70).
  2) 사후조사로 `clean_trackman()`의 실제 버그를 발견 — extension/zone_speed/rel_speed
     결측(NaN)행을 `x>0`/`x<=y` 비교의 False 부작용으로 "이상치"로 오인해 8,119행을
     통째로 삭제하고 있었음(진짜 이상치는 102행뿈). 버그를 고치고 2023(pre-ABS)+cutoff7
     (ABS) 듀얼체크로 재검증한 결과:
       tier A -> MLP만 피딩:      2023 +7.58  / cutoff7 +32.31  (통과)
       tier B -> CatBoost만 피딩: 2023 +16.33 / cutoff7 +28.64  (개별 통과이나 아래 3 참고)
       tier C -> CatBoost만 피딩: 2023 -14.64 (탈락, 제외)
  3) 그러나 B는 사실 CatBoost 단독 스코어 자체를 크게 깎아먹고(-55.72) 메타모델이 CatBoost
     비중을 낮춰 블렌드만 구제한 가짜 신호였음(pitchmix는 CatBoost 단독도 진짜로 개선).
     A+B+coarse pitchmix를 한꺼번에 합치자 CatBoost가 다시 무너지고 이번엔 MLP도 못
     구해줘서 블렌드가 2023에서 -25.51까지 떨어짐 — **B는 최종 제외**.
  4) 팀원이 제보한 coarse pitchmix(볼카운트x손 4축, trackman_history.csv 전체 조인 —
     투수 정체성도 season도 조인 키로 쓰지 않아 크로스워크 커버리지 문제와 season 구조적
     불일치 문제를 둘 다 피해감)를 우리 파이프라인으로 독립 검증: CatBoost만 피딩,
     2023 +12.84 / cutoff7 +13.05 (둘 다 통과, CatBoost 단독 점수도 진짜로 개선).
  5) 최종 채택: **A(->MLP) + coarse pitchmix(->CatBoost)** 조합만 프로덕션에 반영.
     A+pitchmix combined: 2023 +9.57 / cutoff7 +34.43 (둘 다 통과, 서로 잡아먹지 않음
     확인 완료). B/C/F는 전부 제외.

크로스워크는 code/pitcher_crosswalk.py가 만든 ./open/temp/pitcher_map.csv를 쓴다
(외부 데이터 아님 — train.csv/trackman_history.csv 두 공식 CSV의 컬럼만으로 시퀀스 재구성).
coarse pitchmix는 크로스워크 없이 trackman_history.csv 전체를 상황(볼카운트x손) 축으로만
집계하므로 pitcher_map.csv가 필요 없다.
"""
import numpy as np
import pandas as pd

METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
           "extension", "rel_height", "rel_side", "zone_speed"]

TIER_SPECS = {
    "a": ([], "trkstdA_"),
    "b": (["pressure"], "trkstdB_"),
    "c": (["batter_hand"], "trkstdC_"),
    "f": (["batter_hand", "pressure"], "trkstdF_"),
}


def _drop_generation_bug_fragments(df):
    """게임 ID 생성버그로 생긴 파편 게임(팀원 EDA 항목 8, 2026-08-22 세션에서 재확인:
    예 `20190528-NCDinosMajors-2`~`-33` 32개, 전부 1투구짜리이고 진짜 경기인
    `20190528-NCDinosMajors-1`(328투구)의 특정 투구 하나씩과 완전히 겹친다)을 제거한다.
    소규모(<=10행) `trackman_game_id`의 행이 `(game_date, pitch_no, pitcher_trackman_id,
    batter_trackman_id)` 기준으로 다른(더 큰) game_id에도 이미 존재하면 파편으로 간주해
    제거하고 원본(더 큰 game_id) 쪽만 남긴다. `match_games_by_pitch_count`(투구수 유일
    매칭)는 이 파편들이 전부 n_pitch=1이라 애초에 진짜 경기와 매칭될 수 없어 구조적으로
    안전하지만(2026-08-22 세션 확인), 파편 행 자체는 pitchmix/물리지표 집계에 여전히
    섞여 들어가므로 여기서 제거한다."""
    game_sizes = df.groupby("trackman_game_id").size()
    small_ids = game_sizes[game_sizes <= 10].index
    if len(small_ids) == 0:
        return df
    key_cols = ["game_date", "pitch_no", "pitcher_trackman_id", "batter_trackman_id"]
    is_small = df["trackman_game_id"].isin(small_ids)
    big_keys = set(map(tuple, df.loc[~is_small, key_cols].dropna().to_numpy().tolist()))
    small = df.loc[is_small]
    is_fragment = small[key_cols].apply(tuple, axis=1).isin(big_keys)
    fragment_ids = set(small.loc[is_fragment, "trackman_id"])
    if fragment_ids:
        before = len(df)
        df = df[~df["trackman_id"].isin(fragment_ids)]
        print(f"  game_id 생성버그 파편 제거: {before} -> {len(df)}행 ({before - len(df)}행 제거)")
    return df


def clean_trackman(df):
    """수동 검토(사람이 판정한 이상치) 결과를 반영한 클렌징: inning<1, balls/strikes/outs_before
    범위 밖 행을 제거하고, extension<=0/zone_speed>rel_speed는 해당 컬럼만 NaN 처리하며(행은
    유지), game_id 생성버그 파편 행과 trackman_id 제외 완전 중복 행을 제거하고, 손(pitcher_hand)이
    여러 값으로 기록된 투수는 다수 기록된 손만 남긴다.

    extension/zone_speed/rel_speed는 결측(NaN)인 행이 각각 7,716/7,921/7,617행 있는데,
    `x > 0`/`x <= y` 형태의 조건은 NaN 피연산자에서 항상 False가 되어 결측 행까지 "이상치"로
    오인해 통째로 삭제해버리는 버그가 있었다(2026-08-17 세션에서 발견 — 실제 진짜 이상치는
    extension<=0 3행, zone_speed>rel_speed 1행뿐인데 8,119행이 지워지고 있었음). 결측은
    이상치가 아니므로 `.isna()`로 통과시킨다 — groupby(...).agg(["mean","std"])는 어차피
    컬럼별로 NaN을 자동으로 건너뛰므로(skipna=True 기본값), 한 컬럼이 결측이라고 그 행의
    다른 멀쩡한 지표(rel_speed, spin_rate 등)까지 버릴 이유가 없다.

    2026-08-22 세션 갱신: 팀원 EDA와 대조해, extension<=0(3행)/zone_speed>rel_speed(1행)도
    같은 논리로 "행 전체 제거"가 아니라 "해당 컬럼만 NaN"으로 바꿨다(다른 물리량은 정상일
    수 있으므로) — 기존엔 이 4행만 행째로 지우고 있었다. game_id 생성버그 파편(32행,
    `_drop_generation_bug_fragments`)도 추가했다. pitch_no 이슈(한화 시작번호 오프셋,
    연속이라 정렬 순서엔 영향 없음/중복 pitch_no 2건)는 기존 완전-중복 제거로 이미
    커버되는 것을 확인해 별도 처리를 추가하지 않았다."""
    before = len(df)
    row_mask = (
        (df["inning"] >= 1)
        & df["balls_before"].between(0, 3)
        & df["strikes_before"].between(0, 2)
        & df["outs_before"].between(0, 2)
    )
    df = df[row_mask].copy()

    ext_bad = df["extension"].notna() & (df["extension"] <= 0)
    df.loc[ext_bad, "extension"] = np.nan
    speed_bad = df["zone_speed"].notna() & df["rel_speed"].notna() & (df["zone_speed"] > df["rel_speed"])
    df.loc[speed_bad, ["zone_speed", "rel_speed"]] = np.nan

    df = _drop_generation_bug_fragments(df)

    dup_cols = [c for c in df.columns if c != "trackman_id"]
    df = df.drop_duplicates(subset=dup_cols)

    hand_counts = df.groupby(["pitcher_trackman_id", "pitcher_hand"]).size().reset_index(name="n")
    majority_hand = hand_counts.sort_values("n", ascending=False).drop_duplicates("pitcher_trackman_id")
    df = df.merge(majority_hand[["pitcher_trackman_id", "pitcher_hand"]],
                  on=["pitcher_trackman_id", "pitcher_hand"], how="inner")

    after = len(df)
    print(f"  트랙맨 클렌징: {before} -> {after}행 ({before - after}행 제거)")
    return df


def build_pitcher_lookup(df_trm_clean, pitcher_map, tier):
    """트랙맨 로그를 진짜 pitcher_id로 크로스워크 매칭한 뒤, tier에 정의된 추가 그룹핑 축
    (압박/타자손) x 구종군별 mean/std를 wide 포맷으로 pivot한다."""
    merged = df_trm_clean.merge(pitcher_map[["pitcher_trackman_id", "pitcher_id"]],
                                 on="pitcher_trackman_id", how="inner")
    extra_keys, prefix = TIER_SPECS[tier]
    if "pressure" in extra_keys:
        merged["pressure"] = ((merged["balls_before"] >= 3) | (merged["strikes_before"] >= 2)).astype(np.int64)
    pivot_levels = extra_keys + ["pitch_type_group"]
    group_cols = ["pitcher_id"] + pivot_levels

    g = merged.groupby(group_cols)[METRICS].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    std_cols = [c for c in g.columns if c.endswith("_std")]
    g[std_cols] = g[std_cols].fillna(0.0)

    pivoted = g.set_index(group_cols).unstack(level=pivot_levels)
    pivoted.columns = ["_".join(str(x) for x in c) for c in pivoted.columns]
    pivoted = pivoted.reset_index().fillna(0.0)
    rename = {c: prefix + c for c in pivoted.columns if c != "pitcher_id"}
    return pivoted.rename(columns=rename)


def merge_asof_pitcher_std(df_main, df_trm_clean, pitcher_map, tier, holdout):
    """holdout 시즌 자체(및 그 이후)의 트랙맨은 절대 쓰지 않는다 — 실제 배포(season=2025)는
    트랙맨 커버리지(~2024)가 목표 시즌보다 항상 한 시즌 앞서 끊겨 있어 목표 시즌 자기 자신의
    트랙맨을 절대 볼 수 없기 때문. holdout 행은 cutoff=holdout-1로 클램프해 이 간극을 재현한다
    (안 그러면 시즌 내 미래 경기 트랙맨이 검증에 새어 들어가는 within-season leak이 생김).

    실전 최종 학습(holdout=2025로 호출, 진짜 test season 2025는 트랙맨에 아예 없음)에서는
    모든 학습 행(season<=2024)이 자기 시즌까지의 트랙맨을 그대로 보게 되어(cutoff=min(season,2024)
    =season), 서빙 시점(전체 2019~2024를 보는 season=2025 test 행)과 분포가 가장 가까워진다."""
    pieces = []
    feature_cols = None
    for season in sorted(df_main["season"].unique()):
        cutoff_season = min(season, holdout - 1)
        trm_cut = df_trm_clean[df_trm_clean["season"] <= cutoff_season]
        lookup = build_pitcher_lookup(trm_cut, pitcher_map, tier)
        if feature_cols is None:
            feature_cols = [c for c in lookup.columns if c != "pitcher_id"]
        rows = df_main[df_main["season"] == season]
        merged = pd.merge(rows, lookup, on="pitcher_id", how="left")
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True)
    result[feature_cols] = result[feature_cols].fillna(0.0)
    return result, feature_cols


def add_all_tiers(df_main, df_trm_clean, pitcher_map, tiers, holdout):
    """여러 tier를 순서대로 병합해 각 tier의 접두사(trkstdA_/trkstdB_/trkstdC_/trkstdF_)가
    붙은 피처들을 df_main에 모두 추가한다. 반환: (병합된 df, {tier: [feature_cols]})."""
    tier_cols = {}
    df = df_main
    for tier in tiers:
        df, cols = merge_asof_pitcher_std(df, df_trm_clean, pitcher_map, tier, holdout)
        tier_cols[tier] = cols
    return df, tier_cols


def build_lookup_full_history(df_trm_clean, pitcher_map, tier):
    """실전 추론(submit/script.py) 전용: test 시즌(2025)은 트랙맨에 전혀 없으므로 asof 루프 없이
    전체 트랙맨(2019~2024)으로 만든 단일 lookup 하나만 필요하다."""
    return build_pitcher_lookup(df_trm_clean, pitcher_map, tier)


# ---- coarse pitchmix (팀원 제보, 2026-08-17 세션에서 독립 검증 후 채택) ----
COARSE_COLS = ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]
PITCHMIX_COLS = ["coarse_pitchmix_fastball", "coarse_pitchmix_breaking",
                  "coarse_pitchmix_offspeed", "coarse_pitchmix_other"]
# train.csv는 pitcher_hand/batter_hand를 정수코드(1/2)로 익명화하지만 trackman_history.csv는
# "Right"/"Left" 문자열이다. pitcher_map.csv/batter_map.csv 크로스워크로 방향을 역산했다:
# train 1<->Left, 2<->Right (투수 568명 중 1명, 타자 506명 중 1명만 불일치 — 노이즈로 간주).
_HAND_CODE = {"Left": 1, "Right": 2}


def compute_coarse_pitchmix(df_trm):
    """(balls_before, strikes_before, pitcher_hand, batter_hand) 조합별 pitch_type_group
    비율을 계산해 wide 포맷으로 반환한다. 투수 정체성도 season도 조인 키가 아니므로
    pitcher_map.csv 크로스워크가 필요 없고, 실전(season=2025)과 학습 양쪽에서 항상 같은
    48~50개 조합만 나와 look-ahead/커버리지 문제가 구조적으로 생기지 않는다. 결측 조합
    대비 전체 평균(global fallback)도 함께 반환한다."""
    df_trm = df_trm.copy()
    df_trm["pitcher_hand"] = df_trm["pitcher_hand"].map(_HAND_CODE)
    df_trm["batter_hand"] = df_trm["batter_hand"].map(_HAND_CODE)

    counts = df_trm.groupby(COARSE_COLS + ["pitch_type_group"]).size().reset_index(name="n")
    totals = counts.groupby(COARSE_COLS)["n"].transform("sum")
    counts["ratio"] = counts["n"] / totals
    pivoted = counts.pivot_table(
        index=COARSE_COLS, columns="pitch_type_group", values="ratio", fill_value=0.0,
    ).reset_index()
    pivoted = pivoted.rename(columns={
        "fastball": "coarse_pitchmix_fastball", "breaking": "coarse_pitchmix_breaking",
        "offspeed": "coarse_pitchmix_offspeed", "other": "coarse_pitchmix_other",
    })
    for c in PITCHMIX_COLS:
        if c not in pivoted.columns:
            pivoted[c] = 0.0

    global_ratio = df_trm["pitch_type_group"].value_counts(normalize=True)
    fallback = {
        "coarse_pitchmix_fastball": global_ratio.get("fastball", 0.0),
        "coarse_pitchmix_breaking": global_ratio.get("breaking", 0.0),
        "coarse_pitchmix_offspeed": global_ratio.get("offspeed", 0.0),
        "coarse_pitchmix_other": global_ratio.get("other", 0.0),
    }
    return pivoted[COARSE_COLS + PITCHMIX_COLS], fallback


def merge_coarse_pitchmix(df_main, df_trm, holdout):
    """holdout이 정수면 그 시즌(및 그 이후)의 트랙맨은 테이블 계산에서 제외한다(검증 시
    val 시즌 자기 자신이 자기 피처에 섞여드는 걸 방지). holdout=None이면 트랙맨 전체
    (2019~2024)로 계산한다 — 실전 추론(test season=2025는 트랙맨에 아예 없음)과 Full
    Retrain(전체 데이터로 제출 모델을 만드는 단계)에 쓴다. df_trm은 clean_trackman()을
    거치지 않은 원본이어도 된다 — pitchmix는 balls/strikes_before/hand/pitch_type_group만
    쓰고, clean_trackman이 걸러내는 물리 지표 이상치(extension/zone_speed 등)와 무관하다."""
    trm_cut = df_trm if holdout is None else df_trm[df_trm["season"] < holdout]
    lookup, fallback = compute_coarse_pitchmix(trm_cut)
    merged = pd.merge(df_main, lookup, on=COARSE_COLS, how="left")
    for c in PITCHMIX_COLS:
        merged[c] = merged[c].fillna(fallback[c])
    return merged


# ---- coarse physical metrics (신규 후보, 2026-08-20 세션) ----
# tier A(투수 identity 크로스워크 x 구종군별 물리 지표)는 실전 -31.41로 기각됐다
# (PROJECT_HISTORY.md §41) — 원인으로 유력한 건 크로스워크 커버리지가 train.csv
# 검증구간에서 낙관적으로 편향된 것(핵심 교훈 #21, 실제 test.csv 샘플 5개 중 4개가
# pitcher_map.csv에 없었음). coarse pitchmix는 투수 identity 대신 (balls_before,
# strikes_before, pitcher_hand, batter_hand) 축만 써서 이 커버리지 편향을 구조적으로
# 피해갔고 실제로 채택됐다. 같은 non-identity 축을 물리 지표(rel_speed 등)에도 적용해본
# 적은 아직 없다 — coarse_pitchmix가 "이 카운트/손 조합에서 어떤 구종을 던지는 경향"을
# 담듯, coarse_phys는 "이 카운트/손 조합에서 던지는 공의 평균 물리 특성(구속/무브먼트 등)"을
# 담는다(예: 2스트라이크 카운트는 변화구 비중이 높아 평균 구속이 낮아지는 식의 상황적 신호).
COARSE_PHYS_COLS = [f"coarse_phys_{m}" for m in METRICS]


def compute_coarse_physmetrics(df_trm_clean):
    """(balls_before, strikes_before, pitcher_hand, batter_hand) 조합별 물리 지표 평균
    8개를 계산해 wide 포맷으로 반환한다. df_trm_clean은 clean_trackman()을 거친 데이터여야
    한다 — 비율(pitchmix)과 달리 실제 물리값 평균이라 이상치(결측 오필터링 버그가 고쳐진
    버전) 처리가 필요하다."""
    df_trm_clean = df_trm_clean.copy()
    df_trm_clean["pitcher_hand"] = df_trm_clean["pitcher_hand"].map(_HAND_CODE)
    df_trm_clean["batter_hand"] = df_trm_clean["batter_hand"].map(_HAND_CODE)

    g = df_trm_clean.groupby(COARSE_COLS)[METRICS].mean().reset_index()
    g = g.rename(columns={m: f"coarse_phys_{m}" for m in METRICS})

    fallback = {f"coarse_phys_{m}": v for m, v in df_trm_clean[METRICS].mean().items()}
    return g, fallback


def merge_coarse_physmetrics(df_main, df_trm_clean, holdout):
    """merge_coarse_pitchmix와 동일한 holdout/누출방지 컨벤션. df_trm_clean은 이미
    clean_trackman()을 적용한 데이터를 넘겨야 한다(compute_coarse_physmetrics 문서 참고)."""
    trm_cut = df_trm_clean if holdout is None else df_trm_clean[df_trm_clean["season"] < holdout]
    lookup, fallback = compute_coarse_physmetrics(trm_cut)
    merged = pd.merge(df_main, lookup, on=COARSE_COLS, how="left")
    for c in COARSE_PHYS_COLS:
        merged[c] = merged[c].fillna(fallback[c])
    return merged
