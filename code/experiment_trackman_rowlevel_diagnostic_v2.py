# code/experiment_trackman_rowlevel_diagnostic_v2.py
"""code/experiment_trackman_rowlevel_diagnostic.py의 재현판 -- 53.39% 매칭 대신
86.73% 매칭(open/temp/train_with_trackman_match_info.csv)을 사용. std 비교에 더해
Cohen's d(metric x pitch_type_group)도 직접 계산한다(예전엔 팀원이 별도 Gemini 세션에서
계산, 최대값 induced_vert_break x breaking d~=0.133).

사용법: python -m code.experiment_trackman_rowlevel_diagnostic_v2
"""
import os

import numpy as np
import pandas as pd

from code.trackman_conditioned_features_v2 import build_row_level_labels
from code.trackman_pitcher_features import METRICS, clean_trackman

DATA_DIR = "./open/data"


def cohens_d(g1, g0):
    n1, n0 = len(g1), len(g0)
    if n1 < 2 or n0 < 2:
        return np.nan
    v1, v0 = g1.var(ddof=1), g0.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return (g1.mean() - g0.mean()) / pooled


def main():
    print("[1/5] row-level 매칭(86.73%) 로드")
    row_labels = build_row_level_labels()
    print(f"  매칭 행: {len(row_labels)} (train 전체의 {len(row_labels) / 1_475_092 * 100:.2f}%)")

    print("[2/5] 트랙맨 물리량 로드 + 클렌징")
    trm_full = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"), encoding="utf-8-sig")
    trm_clean = clean_trackman(trm_full)
    trm_metrics = trm_clean[["trackman_id", "pitch_type_group"] + METRICS]

    merged = row_labels.merge(trm_metrics, on="trackman_id", how="inner")
    print(f"  클렌징 후 최종 매칭: {len(merged)}행 (control_success=1: {merged['control_success'].mean():.4f})")

    print("\n[3/5] 전체 풀링 기준 control_success별 물리량 std 비교")
    for m in METRICS:
        s1 = merged.loc[merged["control_success"] == 1, m].std()
        s0 = merged.loc[merged["control_success"] == 0, m].std()
        n1 = merged.loc[merged["control_success"] == 1, m].notna().sum()
        n0 = merged.loc[merged["control_success"] == 0, m].notna().sum()
        print(f"  {m:22s} std(success)={s1:.4f} (n={n1}) | std(fail)={s0:.4f} (n={n0}) | ratio={s1/s0 if s0 else float('nan'):.4f}")

    print("\n[4/5] 투수x시즌 단위 (표본 20+ 양쪽 그룹 필수) std 비교 요약")
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
        print("  표본 부족")
    else:
        n_pairs = ps_df.groupby(["pitcher_id", "season"]).ngroups
        print(f"  조건 만족 투수x시즌 조합: {n_pairs}개 (구버전 53.39%: 1706개)")
        for m in METRICS:
            sub = ps_df[ps_df["metric"] == m]
            diff = (sub["std1"] - sub["std0"])
            frac_lower = (diff < 0).mean()
            print(f"  {m:22s} mean(std1-std0)={diff.mean():+.4f} | success가 더 작은 비율={frac_lower:.2%} (n={len(sub)})")

    print("\n[5/5] Cohen's d (metric x pitch_type_group, mean 차이 기준)")
    results = []
    for ptg in merged["pitch_type_group"].dropna().unique():
        sub = merged[merged["pitch_type_group"] == ptg]
        g1 = sub[sub["control_success"] == 1]
        g0 = sub[sub["control_success"] == 0]
        for m in METRICS:
            d = cohens_d(g1[m].dropna(), g0[m].dropna())
            results.append({"pitch_type_group": ptg, "metric": m, "cohens_d": d,
                             "n1": g1[m].notna().sum(), "n0": g0[m].notna().sum()})
    d_df = pd.DataFrame(results).sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)
    print(d_df.head(15).to_string(index=False))
    print(f"\n  최대 |Cohen's d|: {d_df['cohens_d'].abs().max():.4f} (구버전 최대값: induced_vert_break x breaking, d~=0.133)")
    top = d_df.iloc[0]
    print(f"  최대 조합: {top['metric']} x {top['pitch_type_group']} (d={top['cohens_d']:.4f}, n1={top['n1']}, n0={top['n0']})")


if __name__ == "__main__":
    main()
