# dopip.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import shutil
import subprocess
import pickle
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

# 전체 데이터 재학습 시 early stopping으로 찾은 best_epoch/best_iteration에 얼마나 여유(buffer)를 더 줄지
FULL_RETRAIN_EPOCH_BUFFER = 5
DEFAULT_FULL_RETRAIN_EPOCHS = 20
CATBOOST_ITERATION_BUFFER = 50

def get_numbered_path(base_dest_path):
    if not os.path.exists(base_dest_path):
        return base_dest_path
    base, ext = os.path.splitext(base_dest_path)
    counter = 1
    while True:
        new_path = f"{base}_v{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def run_script(script_path):
    process = subprocess.Popen([sys.executable, script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"❌ {script_path} 실행 중 오류가 발생했습니다.")

def main():
    print("[Pipeline] 통합 드라이버 파이프라인(dopip.py) 실행중...")

    os.makedirs("./open/temp", exist_ok=True)
    os.makedirs("./open/reference", exist_ok=True)
    os.makedirs("./open/former_model", exist_ok=True)
    os.makedirs("./submit/model", exist_ok=True)

    print("\n--- [Step 1] 모델 학습 프로세스 가동 (code/train.py) ---")
    run_script("./code/train.py")

    print("\n--- [Step 2] 성능 검증 및 Reference 비교 프로세스 가동 (code/test.py) ---")
    run_script("./code/test.py")

    result_flag_path = "./open/temp/compare_result.txt"
    if not os.path.exists(result_flag_path):
        print("❌ 비교 결과 플래그를 찾을 수 없습니다.")
        return

    with open(result_flag_path, "r") as f:
        result = f.read().strip()

    latest_model_src = "./open/temp/latest_model.pkl"
    former_model_dir = "./open/former_model"
    ref_model_path = "./open/reference/best_model.pkl"

    base_former_dest = os.path.join(former_model_dir, "former_latest_model.pkl")
    former_model_dest = get_numbered_path(base_former_dest)
    base_ref_dest = os.path.join(former_model_dir, "former_best_model.pkl")
    former_ref_dest = get_numbered_path(base_ref_dest)

    if os.path.exists(latest_model_src):
        shutil.copy2(latest_model_src, former_model_dest)
        print(f"최신 훈련 모델 백업 완료 -> {former_model_dest}")

    if result == "NEW_BEST":
        print("\n[Result] 신규 모델이 기존 Reference보다 높은 점수. ref model 교체를 진행...")
        if os.path.exists(ref_model_path):
            shutil.move(ref_model_path, former_ref_dest)
            print(f"기존 Reference 모델을 백업함 -> {former_ref_dest}")
        shutil.move(latest_model_src, ref_model_path)
        print(f"신규 모델을 Reference로 등록 완료 -> {ref_model_path}")
    else:
        print("\n[Result] 기존 Reference 모델의 성능이 더 우수하거나 동일합니다. 기준 모델을 유지합니다.")
        if os.path.exists(latest_model_src):
            os.remove(latest_model_src)
            print("임시 버퍼(temp) 내 최신 모델을 비웠습니다.")

    # =========================================================================
    # Step 3. 최종 확정된 Reference 모델(bundle) 기반 전체 데이터(2019~2024) 통합 완습 (Full Retrain) 후 submit 구조 구축
    # =========================================================================
    print("\n--- [Step 3] 최종 검증 완료본 기반 전체 시즌 데이터 통합 완습 (Full Retrain) ---")
    if not os.path.exists(ref_model_path):
        print("⚠ 학습된 Reference 모델이 없어 전체 재학습을 진행할 수 없습니다.")
        return

    with open(ref_model_path, 'rb') as f:
        best_bundle = pickle.load(f)

    # 2026-08-27: 유담님 파이프라인 이식(전면교체) — Full Retrain도 code/train.py와
    # 동일한 피처 빌드 경로(트랙맨64 물리조인 복원, reverse_rate 시즌진행분 신규,
    # quantile PLE 제거, CatBoost v2 하이퍼파라미터, MLP/CatBoost 시드 7/5개)로
    # 다시 작성했다.
    from code.train import (
        process_trackman_features_safe, apply_f1_filter, add_engineered_features,
        apply_te_residual_features, TE_RESIDUAL_COLS, build_season_end_lookup,
        build_rate_end_lookup, apply_same_hand, SAME_HAND_COLS, is_trackman64,
        YUDAM_ENSEMBLE_SEEDS, YUDAM_CATBOOST_SEEDS,
    )
    from code.mlp_model import CAT_COLS, embed_dim_for_cardinality, fit_preprocessing, to_tensors, train_mlp, make_bundle, get_device
    from code.catboost_model import train_catboost_ensemble, DEFAULT_FULL_RETRAIN_ITERATIONS
    from code.blend_model import make_blend_bundle
    from code.trackman_pitcher_features import merge_coarse_pitchmix

    # reference 번들이 현재 유담 레시피 포맷(MLP 7-seed)이면 MLP 멤버별 best_epoch,
    # CatBoost 시드별 best_iteration, 메타모델 가중치를 그대로 재사용하고, 구버전/
    # 스키마가 다른 레퍼런스(전면교체 직후 첫 실행 등)면 기본값으로 재학습합니다.
    is_blend_ref = (
        isinstance(best_bundle, dict) and "mlp_bundle" in best_bundle and "catboost_model" in best_bundle
        and "meta_model" in best_bundle
        and len(best_bundle["mlp_bundle"].get("members", [])) == len(YUDAM_ENSEMBLE_SEEDS)
    )
    if is_blend_ref:
        per_seed_epochs = [m.get("best_epoch") or DEFAULT_FULL_RETRAIN_EPOCHS for m in best_bundle["mlp_bundle"]["members"]]
        if best_bundle.get("catboost_best_iterations"):
            ref_catboost_best_iterations = best_bundle["catboost_best_iterations"]
        else:
            single = best_bundle.get("catboost_best_iteration") or DEFAULT_FULL_RETRAIN_ITERATIONS
            ref_catboost_best_iterations = [single] * len(YUDAM_CATBOOST_SEEDS)
        blend_meta_model = best_bundle["meta_model"]
    else:
        per_seed_epochs = [DEFAULT_FULL_RETRAIN_EPOCHS] * len(YUDAM_ENSEMBLE_SEEDS)
        ref_catboost_best_iterations = [DEFAULT_FULL_RETRAIN_ITERATIONS] * len(YUDAM_CATBOOST_SEEDS)
        blend_meta_model = {"w_cat": 1.0, "w_mlp": 1.0, "intercept": 0.0}
        print("⚠ Reference 번들이 유담 레시피 포맷(MLP 7-seed)이 아닙니다 -- 기본값으로 전체 재학습합니다.")

    DATA_DIR = "./open/data"
    train_df_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")

    # 트랙맨 상황(10-key) 물리지표 조인. is_train_split=False -> 시간 필터 없이
    # trackman_history.csv 전체(2019~2024)를 그대로 씀 -- 실전 서빙(test season=2025,
    # 트랙맨엔 아예 없는 시즌)과 가장 가까운 분포(유담님 full_retrain_blend_f1.py와 동일).
    tr_final, match_cols, trackman_cols = process_trackman_features_safe(train_df_raw, df_trm, is_train_split=False)
    train_df = tr_final.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=None)
    print(f"[Full Retrain][트랙맨] 물리지표 조인 {len(trackman_cols)}개 컬럼, coarse pitchmix 4개 컬럼")

    # 투수/타자 시즌 진행분 + reverse_rate 시즌 진행분 정적 lookup 저장 (test.csv는
    # season=2025뿐이라 과거 시즌 행이 없으므로, 이 테이블들을 한 번 계산해 정적으로
    # 동봉 -- pitchmix_lookup.csv와 동일 관례).
    season_end_lookup = build_season_end_lookup(train_df)
    season_end_lookup_path = "./submit/model/season_end_lookup.csv"
    season_end_lookup.to_csv(season_end_lookup_path, index=False)
    rate_end_lookup = build_rate_end_lookup(train_df)
    rate_end_lookup_path = "./submit/model/rate_end_lookup.csv"
    rate_end_lookup.to_csv(rate_end_lookup_path, index=False)
    print(f"[Full Retrain] 시즌 진행분 lookup 저장 완료 -> {season_end_lookup_path} ({len(season_end_lookup)}행), "
          f"{rate_end_lookup_path} ({len(rate_end_lookup)}행)")

    league_success_mean = train_df[TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)
    train_df = apply_same_hand(train_df)
    train_df = apply_f1_filter(train_df)

    # Track A(target-encoding 잔차 6개, ->CatBoost 전용). te_source(F1 필터 적용된
    # train_df 자신)를 submit/model/te_source.csv로 정적 동봉 -- submit/script.py가
    # test.csv(2025)에 적용할 때도 이 소스로 causal_smoothed_te_encode를 재사용한다.
    te_prior = train_df[TARGET_COL].mean()
    te_source_cols = ["pitcher_id", "batter_id", "balls_before", "strikes_before",
                       "batter_hand", "num_runners_on", "inning", "season", TARGET_COL]
    te_source = train_df[te_source_cols].copy()
    te_source_path = "./submit/model/te_source.csv"
    te_source.to_csv(te_source_path, index=False)
    print(f"[Full Retrain] TE 소스 테이블 저장 완료 -> {te_source_path} ({len(te_source)}행)")
    train_df = apply_te_residual_features(te_source, train_df, te_prior)

    drop_cols = [ID_COL, TARGET_COL]
    all_cols = [c for c in train_df.columns if c not in drop_cols]
    # 트랙맨64 제거 (code/train.py 와 동일 — 실전 1117.03 레시피). 컬럼은 df 에 남기고 모델 입력에서만 제외.
    cat_feature_cols = [c for c in all_cols if c not in SAME_HAND_COLS and not is_trackman64(c)]
    num_cols = [c for c in all_cols
                if c not in CAT_COLS and c not in TE_RESIDUAL_COLS and not is_trackman64(c)]

    print(f"[Full Retrain] 총 {len(train_df)}행, CatBoost {len(cat_feature_cols)}개/MLP {len(num_cols)+len(CAT_COLS)}개 피처, "
          f"{len(YUDAM_ENSEMBLE_SEEDS)}개 시드 MLP 앙상블 재학습. (시드별 epoch: {[e + FULL_RETRAIN_EPOCH_BUFFER for e in per_seed_epochs]})")

    full_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_df, CAT_COLS, num_cols)
    X_full_cat, X_full_num, y_full = to_tensors(full_proc, CAT_COLS, num_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]
    # 유담님 순정 레시피: quantile PLE 없음(raw concat) -- bin_edges=None.

    device = get_device()
    full_members = []
    for seed, base_epoch in zip(YUDAM_ENSEMBLE_SEEDS, per_seed_epochs):
        full_epochs = max(base_epoch, 1) + FULL_RETRAIN_EPOCH_BUFFER
        model, _ = train_mlp(
            X_full_cat, X_full_num, y_full,
            cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims, bin_edges=None,
            max_epochs=full_epochs, device=device, seed=seed,
        )
        full_members.append({
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": full_epochs,
            "seed": seed,
        })
        print(f"[Full Retrain] seed={seed} {full_epochs} epoch 학습 완료")

    final_mlp_bundle = make_bundle(
        full_members, CAT_COLS, num_cols, cat_dims, embed_dims,
        cat_encoder, num_imputer, num_scaler, bin_edges=None,
    )

    import json
    with open("./teammate/yudam/model_py311/best_catboost_hparams_v2.json") as f:
        _tuned = json.load(f)["best_params"]
    yudam_catboost_params = dict(
        depth=_tuned["depth"], learning_rate=_tuned["learning_rate"], l2_leaf_reg=_tuned["l2_leaf_reg"],
        random_strength=_tuned["random_strength"], bagging_temperature=_tuned["bagging_temperature"],
        border_count=_tuned["border_count"], min_data_in_leaf=_tuned["min_data_in_leaf"],
        bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="BrierScore",
    )
    catboost_full_iterations = [it + CATBOOST_ITERATION_BUFFER for it in ref_catboost_best_iterations]
    print(f"[Full Retrain] CatBoost(v2 HP) {len(YUDAM_CATBOOST_SEEDS)}-seed 재학습, iteration={catboost_full_iterations} "
          f"(reference best_iterations={ref_catboost_best_iterations} + buffer {CATBOOST_ITERATION_BUFFER})")
    X_full_raw, y_full_raw = train_df[cat_feature_cols], train_df[TARGET_COL].values
    catboost_results = train_catboost_ensemble(
        X_full_raw, y_full_raw, seeds=YUDAM_CATBOOST_SEEDS,
        per_seed_iterations=catboost_full_iterations, verbose=True, params=yudam_catboost_params,
    )
    final_catboost_models = [m for m, _ in catboost_results]

    final_bundle = make_blend_bundle(final_catboost_models, final_mlp_bundle, blend_meta_model, cat_feature_cols=cat_feature_cols)
    final_bundle["catboost_best_iteration"] = catboost_full_iterations[0]
    final_bundle["catboost_best_iterations"] = catboost_full_iterations
    print(f"[Full Retrain] 최종 블렌드 번들 구성 완료 (meta_model={blend_meta_model})")

    FINAL_SUBMIT_MODEL_PATH = "./submit/model/final_retained_model.pkl"
    os.makedirs(os.path.dirname(FINAL_SUBMIT_MODEL_PATH), exist_ok=True)
    with open(FINAL_SUBMIT_MODEL_PATH, 'wb') as f:
        pickle.dump(final_bundle, f)

    print(f"[Pipeline 완료] 대형 파이프라인 무결성 통과. 제출용 모델 생성 성공: {FINAL_SUBMIT_MODEL_PATH}")

if __name__ == "__main__":
    main()
