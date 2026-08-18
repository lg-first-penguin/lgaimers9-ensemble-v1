# code/experiment_trackman_season_fallback.py
"""트랙맨 매칭 'season' 등호 문제 검증 실험.

발견(치명적): `process_trackman_features_safe`의 match_cols(10개)에 'season'이 포함돼
있는데, 실제 평가 데이터(test.csv)는 항상 season==2025이고 trackman_history.csv는
2019~2024만 있다. 따라서 실제 제출 시 64개 트랙맨 파생 피처가 구조적으로 100% 매칭
실패 -> `submit/script.py`의 fillna(0) 방어 처리로 전부 상수 0이 된다. 로컬 홀드아웃
(season==2024)은 트랙맨이 2024까지 있어서 이 문제를 재현하지 못했다 (val 시즌이
트랙맨 커버리지 안에 있으므로 실제로 매칭이 됨).

이 스크립트는 로컬 val(season==2024) 피처를 만들 때 trackman_history에서
season==2024 데이터를 통째로 제거해 "실제 2025 상황"을 흉내 낸 뒤, 두 매칭 방식을
비교한다:
  - broken   : 현재 프로덕션 그대로(season 포함 10-key) -> 트랙맨이 val 시즌을 못 보므로
               전부 미매칭 -> fillna(0). 실제 제출이 지금 겪고 있는 상황을 재현.
  - fallback : season 등호 매칭 실패 시에만 season을 뺀 나머지 9-key로 재매칭.
               평가 시점의 season은 trackman_history의 모든 시즌보다 항상 미래이므로
               (2025 > 2019~2024) 리크 위험 없이 안전하게 과거 데이터를 재사용 가능.
               학습 시점(season < 2024)에는 이 fallback을 적용하지 않는다 — season이
               섞이면 이전 시즌 학습 행이 이후 시즌 트랙맨 데이터를 보게 되는 진짜
               리크가 생기기 때문 (현재 season 등호가 사실상 유일한 리크 방지 장치).

재학습 없이 이미 학습된 open/reference/best_model.pkl(CatBoost+MLP+메타모델)을 그대로
불러와 두 버전의 val 피처로 추론만 수행해 비교한다 (code/experiment_3way_stack.py와
동일한 관례 — 무거운 재학습을 반복하지 않기 위한 지름길).

사용법:
  python -m code.experiment_trackman_season_fallback
"""
import pickle

import numpy as np
import pandas as pd

from code.blend_model import predict_meta
from code.catboost_model import predict_catboost
from code.mlp_model import compute_bss, get_device, predict_bundle
from code.train import add_engineered_features

DATA_DIR = "./open/data"
REFERENCE_BUNDLE = "./open/reference/best_model.pkl"
TARGET_COL = "control_success"


def build_lookup(df_trm, cols):
    """process_trackman_features_safe와 동일한 집계/피벗 로직. cols가 곧 groupby 매칭 키."""
    grouped = df_trm.groupby(cols + ["pitch_type_group", "auto_pitch_type"])
    g1 = grouped[["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
                  "extension", "rel_height", "rel_side", "zone_speed"]].agg(["mean", "std"])
    g2 = g1.reset_index()
    std_cols = [c for c in g2.columns if "std" in c]
    g2[std_cols] = g2[std_cols].fillna(0)
    g2.columns = ["_".join(c).strip("_") for c in g2.columns]
    g3 = g2.drop(columns="auto_pitch_type")
    g3 = g3.groupby(cols + ["pitch_type_group"]).agg(["mean"])
    pivoted = g3.unstack(level="pitch_type_group")
    pivoted.columns = [f"{c[0]}_{c[1]}_{c[2]}" for c in pivoted.columns]
    tm_final = pivoted.reset_index().fillna(0)
    if "top_bottom" in tm_final.columns:
        tm_final["top_bottom"] = tm_final["top_bottom"].map({"Top": 0, "Bottom": 1}).astype(np.int64)
    for col in ["batter_hand", "pitcher_hand"]:
        if col in tm_final.columns:
            tm_final[col] = tm_final[col].map({"Left": 1, "Right": 2}).astype(np.int64)
    return tm_final


def merge_trackman(df_main, df_trm, match_cols, drop_season_fallback=False):
    df_main_copy = df_main.copy()
    df_main_copy["top_bottom"] = df_main_copy["top_bottom"].map({"T": 0, "B": 1}).astype(np.int64)

    tm_primary = build_lookup(df_trm, match_cols)
    result = pd.merge(df_main_copy, tm_primary, on=match_cols, how="left")
    new_feature_cols = [c for c in tm_primary.columns if c not in match_cols]

    if drop_season_fallback and "season" in match_cols:
        fallback_cols = [c for c in match_cols if c != "season"]
        tm_fallback = build_lookup(df_trm, fallback_cols)
        fallback = pd.merge(df_main_copy, tm_fallback, on=fallback_cols, how="left")
        missing = result[new_feature_cols].isna().all(axis=1)
        result.loc[missing, new_feature_cols] = fallback.loc[missing, new_feature_cols].values
        print(f"  -> fallback으로 복구된 행: {missing.sum()}/{len(result)} "
              f"({missing.sum()/len(result)*100:.1f}%)")

    result[new_feature_cols] = result[new_feature_cols].fillna(0)
    return result, new_feature_cols


def score_variant(bundle, val_df, y_val, label, device):
    drop_cols = ["row_id", TARGET_COL]
    features = [c for c in val_df.columns if c not in drop_cols]
    X = val_df[features]

    cat_preds = predict_catboost(bundle["catboost_model"], X)
    mlp_preds = predict_bundle(bundle["mlp_bundle"], X, device=device)
    meta = bundle["meta_model"]
    blend = predict_meta(meta["w_cat"], meta["w_mlp"], meta["intercept"], cat_preds, mlp_preds)

    cat_score = compute_bss(cat_preds, y_val)[2]
    mlp_score = compute_bss(mlp_preds, y_val)[2]
    blend_score = compute_bss(blend, y_val)[2]
    print(f"[{label}] CatBoost={cat_score:.2f} | MLP={mlp_score:.2f} | Blend={blend_score:.2f}")
    return blend_score


def main():
    df = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    df_trm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig")

    val_mask = df["season"] == 2024
    league_success_mean = df.loc[df["season"] < 2024, TARGET_COL].mean()

    match_cols = [c for c in df.columns if c in df_trm.columns and c != "row_id"]
    print(f"match_cols ({len(match_cols)}): {match_cols}")

    df_trm_blind = df_trm[df_trm["season"] != 2024].reset_index(drop=True)
    print(f"[sim] 트랙맨에서 season==2024 제거(2025 test처럼 val 시즌을 못 보는 상황 흉내): "
          f"{len(df_trm)} -> {len(df_trm_blind)}행 "
          f"(남은 시즌 범위: {df_trm_blind['season'].min()}~{df_trm_blind['season'].max()})")

    val_df = df.loc[val_mask].reset_index(drop=True)
    y_val = val_df[TARGET_COL].values
    print(f"val 행 수: {len(val_df)}\n")

    print("[broken] 현재 프로덕션과 동일(season 포함 10-key), 트랙맨이 val 시즌을 못 보므로 전부 미매칭 예상...")
    val_broken, feat_cols = merge_trackman(val_df, df_trm_blind, match_cols, drop_season_fallback=False)
    n_dead = (val_broken[feat_cols] == 0).all(axis=1).sum()
    print(f"  -> 트랙맨 피처 전부 0인 행: {n_dead}/{len(val_broken)} ({n_dead/len(val_broken)*100:.1f}%)")

    print("\n[fallback] season 등호 실패 시 season 제외 9-key로 재매칭...")
    val_fallback, _ = merge_trackman(val_df, df_trm_blind, match_cols, drop_season_fallback=True)
    n_dead_fb = (val_fallback[feat_cols] == 0).all(axis=1).sum()
    print(f"  -> 트랙맨 피처 전부 0인 행: {n_dead_fb}/{len(val_fallback)} ({n_dead_fb/len(val_fallback)*100:.1f}%)")

    val_broken = add_engineered_features(val_broken, league_success_mean)
    val_fallback = add_engineered_features(val_fallback, league_success_mean)

    print(f"\n[참고] 현재 로컬 val 그대로(트랙맨 2024 포함, season 매칭 정상 작동) 기준값도 함께 계산합니다.")
    val_asis, _ = merge_trackman(val_df, df_trm, match_cols, drop_season_fallback=False)
    n_dead_asis = (val_asis[feat_cols] == 0).all(axis=1).sum()
    print(f"  -> 트랙맨 피처 전부 0인 행: {n_dead_asis}/{len(val_asis)} ({n_dead_asis/len(val_asis)*100:.1f}%)")
    val_asis = add_engineered_features(val_asis, league_success_mean)

    with open(REFERENCE_BUNDLE, "rb") as f:
        bundle = pickle.load(f)
    for key in ("catboost_model", "mlp_bundle", "meta_model"):
        if key not in bundle:
            raise RuntimeError(f"{REFERENCE_BUNDLE}가 현재 블렌드 번들 스키마가 아닙니다 ('{key}' 없음).")
    device = get_device()

    print()
    s_asis = score_variant(bundle, val_asis, y_val, "as-is (참고, 트랙맨 2024 포함 — 실제 배포와 다른 조건)", device)
    s_broken = score_variant(bundle, val_broken, y_val, "broken (실제 2025 제출 상황 재현)", device)
    s_fallback = score_variant(bundle, val_fallback, y_val, "fallback (season 제외 재매칭)", device)

    print(f"\ndelta (fallback - broken): {s_fallback - s_broken:+.2f}")
    print(f"delta (as-is - broken): {s_asis - s_broken:+.2f}  (트랙맨이 val 시즌을 볼 수 있었다면 얻었을 이론적 상한)")


if __name__ == "__main__":
    main()
