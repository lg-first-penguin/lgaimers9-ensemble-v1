# code/experiment_walk4_filter_nocnt.py
"""code/experiment_walk4_filter.py / code/experiment_walk4_flip.py의 후속.

두 시도(행 제거, 라벨 뒤집기) 모두 balls_before==3 & strikes_before==0 조합이
walk4(순수 4구 연속 볼)와 사실상 동일 집합이라 "그 카운트=한쪽 라벨"이라는 인위적
완전분리를 만들어 CatBoost/MLP가 통째로 붕괴했다(둘 다 점수 0.00). 이번엔 그 원인이 된
카운트 관련 피처 전부를 모델에서 제외하고 나서 다시 "control_success=1인 walk4 행만
제거 / =0인 행만 제거"를 시도한다(사용자가 선택한 원안 재시도, CLAUDE.md 대화 참고).

제외 대상(COUNT_LEAK_COLS): balls_before/strikes_before 원본 + 이를 직접 쓰는
파생피처(count_diff, is_full_count, pitcher_count_advantage_raw/rel, count_pressure,
code/train.py::add_engineered_features 정의) + 카운트로 그룹핑되는 TE-residual 2개
(te_p_cnt_res, te_b_cnt_res, code/train.py::TE_AXES) + coarse pitchmix
(PITCHMIX_COLS, code/trackman_pitcher_features.py — (balls_before,strikes_before,
pitcher_hand,batter_hand)로 조인되므로 카운트의 간접 프록시가 될 수 있어 함께 제외).

baseline도 "행 제거 전, 같은 축소 피처셋"으로 새로 잡아야 공정한 비교가 된다 — 원래
REF_CAT/REF_MLP7/REF_BLEND(78개 피처 기준)와 그대로 비교하면 안 됨.

사용법: python -m code.experiment_walk4_filter_nocnt
"""
from code.experiment_walk4_filter import build_split_with_rowid, find_walk4_rowids, run_one
from code.trackman_pitcher_features import PITCHMIX_COLS

COUNT_LEAK_COLS = [
    "balls_before", "strikes_before",
    "count_diff", "is_full_count",
    "pitcher_count_advantage_raw", "pitcher_count_advantage_rel",
    "count_pressure",
    "te_p_cnt_res", "te_b_cnt_res",
] + PITCHMIX_COLS


def main():
    rowids_cs1, rowids_cs0 = find_walk4_rowids()

    train_split, val_split, mlp_num_cols, cat_feature_cols = build_split_with_rowid(cutoff7=True)
    mlp_num_cols2 = [c for c in mlp_num_cols if c not in COUNT_LEAK_COLS]
    cat_feature_cols2 = [c for c in cat_feature_cols if c not in COUNT_LEAK_COLS]
    print(f"[nocnt] mlp_num_cols {len(mlp_num_cols)} -> {len(mlp_num_cols2)} "
          f"| cat_feature_cols {len(cat_feature_cols)} -> {len(cat_feature_cols2)}")
    missing = [c for c in COUNT_LEAK_COLS if c not in cat_feature_cols and c not in mlp_num_cols]
    if missing:
        print(f"[경고] COUNT_LEAK_COLS 중 원본 피처셋에 없던 이름: {missing}")

    results = []
    results.append(run_one("baseline(카운트 피처 제외, 행 제거 없음)", train_split, val_split,
                            mlp_num_cols2, cat_feature_cols2, set()))
    results.append(run_one("cs=1 제외(카운트 피처 제외)", train_split, val_split,
                            mlp_num_cols2, cat_feature_cols2, rowids_cs1))
    results.append(run_one("cs=0 제외(카운트 피처 제외)", train_split, val_split,
                            mlp_num_cols2, cat_feature_cols2, rowids_cs0))

    print(f"\n{'='*20} 요약 {'='*20}")
    for r in results:
        print(f"{r['name']:32s} (n_removed={r['n_removed']:6d}): CatBoost={r['cat_score']:.2f} "
              f"MLP(3-seed)={r['mlp_score']:.2f} blend={r['blend_score']:.2f}")
    base, a, b = results
    print(f"\ncs=1 제외 vs baseline: CatBoost {a['cat_score']-base['cat_score']:+.2f} | "
          f"MLP {a['mlp_score']-base['mlp_score']:+.2f} | blend {a['blend_score']-base['blend_score']:+.2f}")
    print(f"cs=0 제외 vs baseline: CatBoost {b['cat_score']-base['cat_score']:+.2f} | "
          f"MLP {b['mlp_score']-base['mlp_score']:+.2f} | blend {b['blend_score']-base['blend_score']:+.2f}")


if __name__ == "__main__":
    main()
