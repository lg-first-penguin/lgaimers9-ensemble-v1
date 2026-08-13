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
    process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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

    # [수정 완료 지점] train.py에서 선언한 무결성 타임필터 전처리 함수를 그대로 수입하여 1줄로 결합 완수
    from code.train import process_trackman_features_safe, add_engineered_features
    from code.mlp_model import CAT_COLS, ENSEMBLE_SEEDS, embed_dim_for_cardinality, fit_preprocessing, to_tensors, train_mlp, make_bundle, get_device
    from code.catboost_model import train_catboost, DEFAULT_FULL_RETRAIN_ITERATIONS
    from code.blend_model import make_blend_bundle

    # reference 번들이 현재 스태킹 포맷("catboost_model"+"mlp_bundle"+"meta_model")이면 MLP
    # 멤버별 best_epoch, CatBoost best_iteration, 메타모델 가중치를 그대로 재사용하고,
    # 구버전/레거시 포맷(alpha 가중평균 시절 포함)이면 전부 기본값으로 재학습합니다.
    is_blend_ref = (
        isinstance(best_bundle, dict) and "mlp_bundle" in best_bundle and "catboost_model" in best_bundle
        and "meta_model" in best_bundle
        and len(best_bundle["mlp_bundle"].get("members", [])) == len(ENSEMBLE_SEEDS)
    )
    if is_blend_ref:
        per_seed_epochs = [m.get("best_epoch") or DEFAULT_FULL_RETRAIN_EPOCHS for m in best_bundle["mlp_bundle"]["members"]]
        ref_catboost_best_iteration = best_bundle.get("catboost_best_iteration") or DEFAULT_FULL_RETRAIN_ITERATIONS
        blend_meta_model = best_bundle["meta_model"]
    else:
        per_seed_epochs = [DEFAULT_FULL_RETRAIN_EPOCHS] * len(ENSEMBLE_SEEDS)
        ref_catboost_best_iteration = DEFAULT_FULL_RETRAIN_ITERATIONS
        # 폴백 기본값 (거의 사용되지 않음 — NEW_BEST가 항상 우선 승격되므로 이 분기는
        # reference가 이미 호환 포맷일 때만 도달하지 않고, 다음 dopip.py 실행에서
        # 재학습된 메타모델로 즉시 갱신됨)
        blend_meta_model = {"w_cat": 1.0, "w_mlp": 1.0, "intercept": 0.0}

    DATA_DIR = "./open/data"
    train_df_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    df_trm_raw = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")

    tr_final, match_cols = process_trackman_features_safe(train_df_raw, df_trm_raw, is_train_split=False)

    train_df = tr_final.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    league_success_mean = train_df[TARGET_COL].mean()
    train_df = add_engineered_features(train_df, league_success_mean)

    drop_cols = [ID_COL, TARGET_COL]
    full_features = [col for col in train_df.columns if col not in drop_cols]
    num_cols = [c for c in full_features if c not in CAT_COLS]

    print(f"[Full Retrain] 총 {len(train_df)}행 전체 데이터에 대해 {len(ENSEMBLE_SEEDS)}개 시드 앙상블을 재학습합니다. (시드별 epoch: {[e + FULL_RETRAIN_EPOCH_BUFFER for e in per_seed_epochs]})")

    full_proc, cat_encoder, num_imputer, num_scaler, cat_dims = fit_preprocessing(train_df, CAT_COLS, num_cols)
    X_full_cat, X_full_num, y_full = to_tensors(full_proc, CAT_COLS, num_cols, TARGET_COL)
    embed_dims = [embed_dim_for_cardinality(d) for d in cat_dims]

    device = get_device()
    full_members = []
    for seed, base_epoch in zip(ENSEMBLE_SEEDS, per_seed_epochs):
        full_epochs = max(base_epoch, 1) + FULL_RETRAIN_EPOCH_BUFFER
        model, _ = train_mlp(
            X_full_cat, X_full_num, y_full,
            cat_dims=cat_dims, num_numeric_feats=len(num_cols), embed_dims=embed_dims,
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
        cat_encoder, num_imputer, num_scaler,
    )

    catboost_full_iterations = ref_catboost_best_iteration + CATBOOST_ITERATION_BUFFER
    print(f"[Full Retrain] CatBoost {catboost_full_iterations} iteration 재학습 (reference best_iteration={ref_catboost_best_iteration} + buffer {CATBOOST_ITERATION_BUFFER})")
    X_full_raw, y_full_raw = train_df[full_features], train_df[TARGET_COL].values
    final_catboost_model, _ = train_catboost(X_full_raw, y_full_raw, iterations=catboost_full_iterations, verbose=True)

    final_bundle = make_blend_bundle(final_catboost_model, final_mlp_bundle, blend_meta_model)
    final_bundle["catboost_best_iteration"] = catboost_full_iterations
    print(f"[Full Retrain] 최종 블렌드 번들 구성 완료 (meta_model={blend_meta_model})")

    FINAL_SUBMIT_MODEL_PATH = "./submit/model/final_retained_model.pkl"
    os.makedirs(os.path.dirname(FINAL_SUBMIT_MODEL_PATH), exist_ok=True)
    with open(FINAL_SUBMIT_MODEL_PATH, 'wb') as f:
        pickle.dump(final_bundle, f)

    print(f"[Pipeline 완료] 대형 파이프라인 무결성 통과. 제출용 모델 생성 성공: {FINAL_SUBMIT_MODEL_PATH}")

if __name__ == "__main__":
    main()
