# code/experiment_fr_filterblend_season2023.py
"""code/experiment_fr_gameroute_and_filterblend.py 아이디어 1(F1필터 적용/미적용 50:50
블렌드)이 cutoff7에서 +7.74로 사전 우려와 달리 플러스가 나왔다. season==2023으로
dual-regime 재검증 — 아이디어 2(game_type 라우팅)는 F subgroup −51.02로 이미 명확히
기각돼 재검증 대상에서 제외한다.

사용법: python -m code.experiment_fr_filterblend_season2023
"""
from code.experiment_fr_gameroute_and_filterblend import load_base, build_split, run_idea1_filter_blend


def main():
    train_df, df_trm = load_base()
    train_split, val_split, features = build_split(
        train_df, df_trm, holdout=2023,
        train_mask_fn=lambda df: df['season'] < 2023,
        val_mask_fn=lambda df: df['season'] == 2023,
    )
    print(f"[season2023] n_train(필터전)={len(train_split)} n_val={len(val_split)}")
    run_idea1_filter_blend(train_split, val_split, features)


if __name__ == "__main__":
    main()
