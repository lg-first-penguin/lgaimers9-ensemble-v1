# code/experiment_reliability_cutoff7_subgroups.py
"""cutoff7 검증셋(2024 7~10월, 프로덕션 best_model.pkl) 안에서 season 자체는
못 가르지만(단일 시즌) game_month(=cutoff 이후 경과 개월수 proxy), game_type,
cold-start(asof_pitcher_n) 서브그룹별 REL/gap을 본다. 재학습 없이 기존 번들의
예측값만 재사용하므로 빠르다.
"""
import pickle

import numpy as np
import pandas as pd

from code.experiment_reliability_diagram import build_val_split
from code.mlp_model import predict_bundle, compute_bss
from code.catboost_model import predict_catboost
from code.blend_model import predict_meta


def subgroup_table(preds, y, group_key, name):
    df = pd.DataFrame({"pred": preds, "y": y, "g": group_key})
    out = []
    for g, sub in df.groupby("g"):
        n = len(sub)
        pbar = sub["pred"].mean()
        obar = sub["y"].mean()
        _, _, score = compute_bss(sub["pred"].values, sub["y"].values)
        out.append({name: g, "n": n, "mean_pred": pbar, "actual_rate": obar, "gap": pbar - obar, "score": score})
    return pd.DataFrame(out)


def main():
    print("cutoff7 검증셋 재구성 중...")
    X_val, y_val = build_val_split()

    with open("./open/reference/best_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    cat_feature_cols = bundle.get("cat_feature_cols")
    cat_df = X_val[cat_feature_cols] if cat_feature_cols is not None else X_val
    cat_preds = predict_catboost(bundle["catboost_model"], cat_df)
    mlp_preds = predict_bundle(bundle["mlp_bundle"], X_val)
    meta = bundle["meta_model"]
    blend_preds = predict_meta(meta["w_cat"], meta["w_mlp"], meta["intercept"], cat_preds, mlp_preds)

    print(f"\n표본수={len(y_val)}, ō={y_val.mean():.4f}")
    print(f"game_type 분포:\n{X_val['game_type'].value_counts()}")

    print("\n-- game_month별 (블렌드) --")
    print(subgroup_table(blend_preds, y_val, X_val["game_month"].values, "month").to_string(index=False, formatters={
        "mean_pred": "{:.4f}".format, "actual_rate": "{:.4f}".format, "gap": "{:+.4f}".format, "score": "{:.1f}".format}))

    print("\n-- game_type별 (블렌드) --")
    print(subgroup_table(blend_preds, y_val, X_val["game_type"].values, "game_type").to_string(index=False, formatters={
        "mean_pred": "{:.4f}".format, "actual_rate": "{:.4f}".format, "gap": "{:+.4f}".format, "score": "{:.1f}".format}))

    n_pitcher = X_val["asof_pitcher_n"].values
    q = pd.qcut(n_pitcher, 4, labels=["Q1(콜드)", "Q2", "Q3", "Q4(웜)"], duplicates="drop")
    print("\n-- asof_pitcher_n 4분위(콜드스타트, 블렌드) --")
    print(subgroup_table(blend_preds, y_val, q, "n_bucket").to_string(index=False, formatters={
        "mean_pred": "{:.4f}".format, "actual_rate": "{:.4f}".format, "gap": "{:+.4f}".format, "score": "{:.1f}".format}))

    cold_mask = n_pitcher <= np.quantile(n_pitcher, 0.25)
    cold_gap = blend_preds[cold_mask].mean() - y_val[cold_mask].mean()
    warm_gap = blend_preds[~cold_mask].mean() - y_val[~cold_mask].mean()
    print(f"\n콜드(하위25%) gap={cold_gap:+.4f} (n={cold_mask.sum()})  vs  웜(상위75%) gap={warm_gap:+.4f} (n={(~cold_mask).sum()})")

    # CatBoost 단독 / MLP 단독에서도 콜드스타트 방향을 따로 확인 (블렌드가 상쇄시켰는지 보려고)
    for name, preds in [("CatBoost 단독", cat_preds), ("MLP 단독", mlp_preds)]:
        cg = preds[cold_mask].mean() - y_val[cold_mask].mean()
        wg = preds[~cold_mask].mean() - y_val[~cold_mask].mean()
        print(f"  [{name}] 콜드 gap={cg:+.4f}  웜 gap={wg:+.4f}")


if __name__ == "__main__":
    main()
