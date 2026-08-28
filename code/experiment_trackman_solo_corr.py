# code/experiment_trackman_solo_corr.py
"""트랙맨-only solo 모델(code/experiment_trackman_solo.py)의 예측이 기존 프로덕션
CatBoost/MLP/블렌드 예측과 얼마나 상관돼 있는지 확인 (cutoff7 레짐, row_id로 정렬).
추론만 하므로(재학습 없음) 가볍다."""
import os
import pickle

import numpy as np
import pandas as pd

from code.train import add_engineered_features, TRACKMAN_TIER_FEED
from code.mlp_model import predict_bundle
from code.blend_model import predict_meta
from code.catboost_model import predict_catboost
from code.trackman_pitcher_features import clean_trackman, add_all_tiers, merge_coarse_pitchmix

ID_COL = "row_id"
TARGET_COL = "control_success"
DATA_DIR = "./open/data"

df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df['top_bottom'] = df['top_bottom'].map({'T': 0, 'B': 1}).astype(np.int64)
train_df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

train_mask = (train_df['season'] < 2024) | ((train_df['season'] == 2024) & (train_df['game_month'] < 7))

pitcher_map = pd.read_csv("./open/temp/pitcher_map.csv")
df_trm = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
df_trm_clean = clean_trackman(df_trm)
train_df, _ = add_all_tiers(train_df, df_trm_clean, pitcher_map, list(TRACKMAN_TIER_FEED), holdout=2024)
train_df = merge_coarse_pitchmix(train_df, df_trm, holdout=2024)

league_success_mean = train_df.loc[train_mask, TARGET_COL].mean()
train_df = add_engineered_features(train_df, league_success_mean)

features = [c for c in train_df.columns if c not in [ID_COL, TARGET_COL]]
val_split = train_df[(train_df['season'] == 2024) & (train_df['game_month'] >= 7)].reset_index(drop=True)
X_val, y_val = val_split[features], val_split[TARGET_COL].values

with open("open/reference/best_model.pkl", "rb") as f:
    bundle = pickle.load(f)

cat_feature_cols = bundle.get("cat_feature_cols")
cat_df = X_val[cat_feature_cols] if cat_feature_cols is not None else X_val
cat_preds = predict_catboost(bundle["catboost_model"], cat_df)
mlp_preds = predict_bundle(bundle["mlp_bundle"], X_val)
meta = bundle["meta_model"]
blend_preds = predict_meta(meta["w_cat"], meta["w_mlp"], meta["intercept"], cat_preds, mlp_preds)

prod_df = pd.DataFrame({
    "row_id": val_split["row_id"].values,
    "cat_pred": cat_preds, "mlp_pred": mlp_preds, "blend_pred": blend_preds, "y": y_val,
})

trk = np.load("./open/temp/experiment_trackman_solo/cutoff7_preds.npz", allow_pickle=True)
trk_df = pd.DataFrame({"row_id": trk["row_id"], "trk_pred": trk["preds"]})

merged = prod_df.merge(trk_df, on="row_id", how="inner")
print(f"매칭된 val 행 수: {len(merged)} / 프로덕션 val {len(prod_df)} / 트랙맨-only val {len(trk_df)}")

print("\n=== 상관관계 (cutoff7 val) ===")
for col in ["cat_pred", "mlp_pred", "blend_pred"]:
    r = np.corrcoef(merged["trk_pred"], merged[col])[0, 1]
    print(f"corr(trackman_solo, {col}) = {r:.4f}")

# 참고: 기존에 실패로 판정난 3rd-model 후보들의 상관관계 구간(0.82~0.999)과 비교하기 위한 출력
