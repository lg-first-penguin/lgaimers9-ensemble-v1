# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A private team pipeline for the LG Aimers / DACON competition (야구 투구 제어 성공 예측, `control_success` binary target, evaluated via Brier Skill Score). Built for Ubuntu 24.04 (WSL2), Python 3.11.

**Current production model**: CatBoost (re-tuned) + Tabular MLP ensemble (quantile/PLE numerical-feature embedding), combined via a stacking meta-model (`sigmoid(w_cat*catboost_pred + w_mlp*mlp_pred + intercept)`). Training data has a filter (`code/train.py::apply_f1_filter`) removing stale pre-2023 `game_type=='F'` rows; validation/inference data is untouched. Train/val split includes 2024 head months in training (`season<2024` + `season==2024 & game_month<7`, validated on `season==2024 & game_month>=7`) — adopted after the ABS regime-shift investigation (see split-convention section below). Feature set includes `code/train.py::apply_season_progression_features` (2026-08-18, see below) — this is currently the single biggest, most reliably-validated feature win in the project's history.

**Trackman tier A was tried, real-leaderboard-tested, and reverted (as of 2026-08-18)**: tier A (`code/trackman_pitcher_features.py`, real pitcher-identity crosswalk × pitch-type-group physical-metric mean/std, fed to MLP only) + coarse pitchmix (count×handedness pitch-mix ratios, fed to CatBoost only, no identity/season join — see below) was submitted and scored **950.81** on the real leaderboard, a **−31.41 regression** from the prior best 982.22. Local dual-regime checks had been ambiguous (cutoff=7 favored tier A, season==2023 favored dropping it) — the real submission result sided with the season==2023 direction, so **tier A was removed** (`TRACKMAN_TIER_FEED = {}` in both `code/train.py` and `submit/script.py`); coarse pitchmix (→CatBoost only) is kept — it passed both regimes cleanly on its own (2023 +9.57 / cutoff7 +34.43, unaffected by tier A's removal) and has no tier-A-style pitcher-identity-crosswalk dependency. Everything else tried (tiers B/C/F, asof9key, situational-fingerprint matching) stayed rejected. If you see stale references elsewhere to "trackman fully removed" or to tier A being active, they predate this change — trust this paragraph and the "Trackman history" section below over older prose.

**Investor/teammate-sourced feature — "season progression" (2026-08-18, adopted)**: `code/train.py::apply_season_progression_features` (paired with `build_season_end_lookup`) adds 8 columns (`{pitcher,batter}_season_n`, `_success_count`, `_success_rate`, `_rate_gap`) that decompose the official `asof_{pitcher,batter}_success_rate` (a career-cumulative average) into "this season only" performance vs. a pre-season baseline (the pitcher's/batter's last row of the *previous* season). Unlike every trackman experiment in this project, it improved **both** CatBoost and MLP on **both** regimes: CatBoost cutoff7 +34.66/season==2023 +150.76, production blend cutoff7 +35.90/season==2023 +164.90 (`EXPERIMENTS.md` §45). Implementation is split into a lookup-builder (needs `control_success`, so train.csv-only) and an applier (safe on any row, no dependency on other rows) — this split matters because a naive single-function version would have made `submit/script.py` build the "previous season" lookup from `test.csv` itself, which has no history and no labels, i.e. exactly the kind of other-row-dependent feature the competition rules forbid. The lookup table is precomputed once from all of train.csv during `dopip.py`'s Full Retrain step and bundled as `submit/model/season_end_lookup.csv` (same static-CSV convention as `pitcher_map.csv`/`pitchmix_lookup.csv`).

**Current status (2026-08-19)**: submitted and confirmed. Local cutoff7 blend score 748.87 (pitchmix-only trackman + season progression) scored **1027.54** on the real leaderboard — a **+45.32** improvement over the prior best 982.22, and +76.73 over the tier-A version (950.81). This is now the confirmed best real-leaderboard result and both open questions from the prior session are resolved by it: tier A's removal was the right call (its local-only ambiguity is moot now), and the season-progression feature's local gains translate to a real, large improvement. Ranking-contention threshold is reportedly ~1150 — next work should aim to close that ~122-point gap. **Superseded by the 2026-08-20 TE-residual adoption below and, provisionally, by the 2026-08-21 entry further down — read those before trusting this paragraph's 1027.54 as "current."**

**Update (2026-08-20, confirmed)**: adopted target-encoding residual features (Track A, teammate-sourced, →CatBoost only — see `code/train.py::TE_RESIDUAL_COLS`/`apply_te_residual_features`). Real leaderboard **1041.40** (+13.86 over 1027.54). `open/reference/best_model.pkl` and `submit/model/final_retained_model.pkl` reflected this 2-way (CatBoost+MLP) configuration as of that confirmation. Ranking-contention gap ~108.6 points at that point. Full story: `teammate_catboost_mlp_track_983.md` memory file, `EXPERIMENTS.md` (TE-residual sections).

**Update (2026-08-21, PENDING — not yet confirmed on the real leaderboard)**: a long session re-opened the "3rd-model stacking" line (previously closed) and tuned four architecturally-different candidates (DeepFM/EBM/BART/NAM) chosen to have a genuinely different error mechanism from CatBoost/MLP rather than just lower correlation. Evidence stayed noise-level throughout — most notably, DeepFM's 3-way stacking delta flipped sign (+1.87 → −7.26) on a plain rerun of the identical config, directly proving the original "improvement" was run-to-run noise (see `EXPERIMENTS.md` §67, `PROJECT_HISTORY.md` §67 and 핵심 교훈 #33/#34). Despite this, the user chose to submit anyway (to get a real-leaderboard data point) — `submit/model/final_retained_model.pkl` was overwritten with a **3-way DeepFM blend** (CatBoost+MLP unchanged/reused, DeepFM added as a 3rd stacking input, `meta_model` now has a `w_fm` key). The old 2-way bundle is backed up at `submit/model/final_retained_model_2way_backup.pkl`. **`open/reference/best_model.pkl` was intentionally left untouched (still 2-way)** — this means the next `dopip.py` run's Full Retrain step will rebuild `submit/model/final_retained_model.pkl` from the 2-way reference and silently discard the 3-way DeepFM addition. **Do not run `dopip.py` until the real-leaderboard result for this 3-way submission is known and a decision is made to keep or roll back** — check `EXPERIMENTS.md` §67.7 "중단점" first. As of session end, the real-leaderboard score for this submission is unknown.

**Full history**: every architecture change, feature-engineering attempt, and why each was adopted or rejected — from the original RandomForest baseline through today's 1027.54 — is chronologically documented in **`PROJECT_HISTORY.md`**, including a "핵심 교훈" (key lessons) section worth reading before proposing a new experiment (several intuitive-seeming directions were already tried and failed for specific, non-obvious reasons — e.g. don't embed columns that get new values at inference time, don't trust a single-season holdout, GBDT and bagging models respond oppositely to loss-function swaps). Raw experiment numbers, tables, and reproduction commands are in **`EXPERIMENTS.md`** (§-numbered, referenced throughout this file and `PROJECT_HISTORY.md`). Neither file is loaded automatically — read them on demand when historical context or exact numbers are needed.

Competition data link: https://dacon.io/competitions/official/236743/data

## Competition's rule

[배경] 
최근 스포츠 현장에서는 선수의 경기력 분석과 전략 수립에 데이터 기반 의사결정의 중요성이 빠르게 커지고 있습니다. 특히 야구에서 투구의 제구력은 실점 억제, 볼카운트 운영, 타자 대응 전략에 직접적인 영향을 주는 핵심 요소입니다.

기존에는 투수의 제구력을 평균자책점, 볼넷 수, 스트라이크 비율 등 경기 후 집계 지표로 평가하는 경우가 많았습니다. 그러나 실제 경기에서는 매 투구 직전의 볼카운트, 주자 상황, 타자·투수 특성, 과거 투구 이력 등 다양한 정보가 복합적으로 작용합니다.

따라서 단순한 결과 통계가 아니라, 투구가 이루어지기 전까지 확인 가능한 정보만을 바탕으로 해당 투구가 원하는 제구 범위에 성공할 가능성을 예측하는 AI 모델링이 중요해지고 있습니다.

이번 해커톤은 이러한 문제의식을 바탕으로, 야구 경기 데이터를 중심으로 투구 직전의 상황과 과거 이력을 활용하여 제구 성공 확률을 예측하는 실전형 AI 문제를 수행하게 됩니다. 

트랙맨(Trackman) 데이터는 2019~2024년의 과거 투구 특성을 참고할 수 있는 보조 데이터로 제공됩니다.



### [주제]
투구 단위의 제구 성공 확률 예측 AI 모델 개발



### [설명]
이번 온라인 해커톤(Phase 2)은 투구 직전까지 확인 가능한 경기 상황, 선수 정보, 주자 상황, 과거 이력을 바탕으로 각 투구의 제구 성공 확률을 예측하는 AI 모델을 개발하는 것을 목표로 합니다.

참가자는 제공된 데이터를 활용하여 테스트 데이터의 각 투구에 대해 control_success 의 확률값을 예측할 수 있어야 합니다. 

학습 데이터의 control_success는 제구 성공을 1, 제구 실패를 0으로 정의한 학습용 Target이며, 예측해야하는 control_success는 제구 성공 가능성을 나타내는 확률입니다.

또한 참가자는 투구 이전 시점에서 활용 가능한 정보만을 바탕으로 예측 모델을 설계할 수 있어야합니다.



온라인 해커톤(Phase 2)에서의 제구 성공은 각 투구의 공 위치를 기준으로 정의합니다.

아래의 3가지 경우는 제구 실패에 해당하며, 그 외 유효한 투구는 제구 성공에 해당합니다.

1) 스트라이크존 가운데 부근으로 들어간 공

2) 스트라이크존에서 크게 벗어난 공

3) 포수의 요구 방향과 반대로 들어간 공



온라인 해커톤(Phase2)에서 교육생들의 문제 해결 능력을 검증하여 오프라인 해커톤(Phase3)에 진출자(약 100명)를 선발하기 위한 과정입니다.

오프라인 해커톤(Phase 3)은 1박 2일간 오프라인으로 진행되며, 세부 과제는 추후 안내될 예정이며, 온라인 해커톤(Phase 2)과 동일하게 야구 데이터를 기반으로 한 AI 문제로 진행될 예정입니다.



[코드 제출 대회]

본 대회는 submit.zip 업로드 방식의 코드 제출 형식 대회로 진행됩니다.

전체 추론 실행 시간 ≤ 10분 (245,789개 샘플 추론)
패키지(라이브러리) 설치 시간 ≤ 10분
제출 파일 용량 ≤ 10GB (*압축해제 후 최대 32GB)
오프라인 환경 실행 (패키지 설치 외 인터넷 연결 불가능)
6 vCPU, 28GB RAM, L4 GPU 22.4GiB VRAM 환경에서 실행
자세한 사항은 평가 탭과 코드 제출 가이드를 반드시 참고하여 진행하시길 바랍니다.

1. 리더 보드

평가 산식 : Brier Skill Score
본 대회는 각 투구의 control_success = 1일 확률을 예측하는 확률 예측 과제입니다. 추론 확률이 실제 정답에 가까울수록 높은 점수를 받습니다.
Score = max(0, 100000 × (1 - Brier Score / 평균 제구율 Brier Score))
Brier Score = mean((p_i - y_i)^2)
r = mean(y_i)
평균 제구율 Brier Score = r × (1 - r)
p_i : i번째 샘플의 제구 성공 예측 확률
y_i : i번째 샘플의 실제 정답 (0, 1)
r : 전체 평가 데이터의 평균 제구 성공률 (비공개 수치)


Public Score : 전체 테스트 데이터 100%
Private Score : 대회 종료 시점의 Public Score


2. 평가 방식

LG Aimers 수료 조건
Phase1을 이수하고 Phase2의 Public Score (LB: 549.51) 이상
기준 점수는 운영진이 제공한 베이스라인 추론 코드를 운영진 평가 환경에서 실행했을 때의 점수를 기준으로 측정
1차 평가 : 리더보드 Private Score 100%
동점자의 경우, 기존 리더보드 순위 산정 방식을 따름 [링크]의 '리더보드 점수' 부분을 참고
2차 평가 : 오프라인 해커톤(Phase3) 진출을 희망하는 팀은 코드 제출 후 코드 검증
Private 리더보드 상위팀(약 100명)은 코드 및 PPT 필수 제출 대상
코드 및 PPT 제출과 검증를 모두 통과한 Private 리더보드 상위팀(약 100명)이 오프라인 해커톤(Phase3) 진출


3. 코드 제출 대회 가이드

본 대회는 submit.zip 파일을 제출하는 방식의 '코드 제출 대회'로 진행됩니다. (기본 가이드 문서)

참가자는 아래와 같은 구조로 submit.zip을 구성하여 제출해야 합니다.

아래의 구조와 동일하고 디렉토리 명과 파일 명을 모두 일치 시켜야합니다.

제출 파일 구조 (submit.zip)

submit.zip
├── model/        # 모델 가중치 파일을 저장하는 디렉토리
│      └── (예: model.pt 등)
├── script.py       # 실제 추론이 수행되는 실행 코드
└── requirements.txt   # 필요한 패키지 및 버전 명시
script.py는 submit.zip을 제출 시 평가 서버에서 자동으로 실행됩니다.
requirements.txt는 pip install -r requirements.txt 명령어로 설치 가능한 형태여야 하며, 추론 시 필요한 모든 패키지를 포함해야 합니다.
submit.zip 내 구조는 반드시 일치해야하며, 추가 최상위 폴더가 zip 구조 내 존재하는 경우 등 구조가 불일치하는 경우 설치 오류가 발생합니다.


평가 서버에서 추가되는 항목

제출 시, 평가 서버에서 참가자가 제출한 submit.zip 파일에는 아래 항목이 자동으로 추가됩니다.

submit.zip
├── model/        # 참가자 구성
├── script.py       # 참가자 구성
├── requirements.txt   # 참가자 구성
├── data/         # 평가에 사용될 테스트 데이터 (디렉토리 자동 생성)
└── output/submission.csv        # 참가자 추론 결과가 저장되는 경로 (디렉토리 자동 생성)
data/ 디렉토리는 실제 평가 데이터를 포함한 경진대회 데이터가 포함되며, 읽기전용으로 쓰기 및 수정이 불가능한 디렉토리입니다.
output/ 디렉토리는 참가자의 script.py 실행 결과로 생성된 예측 결과 파일이 저장되는 디렉토리이며, 해당 디렉토리 내에 반드시 submission.csv으로 생성될 수 있어야합니다.


 제출 파일 용량 제한

제출 파일(zip) 용량 제한: 최대 10GB 이내 (*압축해제 후 최대 32GB)


⏱ 실행 시간 제한
패키지 설치 시간: 최대 10분 이내 (시간 초과 시 설치 오류)
추론 코드 실행 시간: 최대 10분 이내 (시간 초과 시 제출 오류)

 평가 서버 사양
OS : Ubuntu 22.04.5 LTS
GPU : NVIDIA L4 (VRAM 22.4GiB)
CPU: 6 vCPU
CPU RAM: 28GB
Python : 3.11.15
인터넷 접속:  비활성화 (패키지 설치 외 외부 서버 연결 및 다운로드 불가)
CUDA : 12.8


 평가 서버 기본 설치 패키지(라이브러리) 목록

아래의 패키지(라이브러리)는 평가 서버에 기본적으로 설치되어 있으며, 버전이 명시된 아래의 패키지(라이브러리)에 한해서는 다른 버전을 사용할 때 설치 에러가 발생할 수 있으므로 가급적 평가 서버에 기본 설치된 패키지(라이브러리)를 활용하고 제출하는 requirements.txt에는 포함하지 않는 것을 권장드립니다. 
라이브러리 설치 에러가 발생하면 설치 오류에 해당하며, 일일 제출 횟수에는 반영되지 않습니다.


1) 주요 설치 패키지(라이브러리)

torch==2.7.1+cu128
pandas==2.0.3
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.8.0
joblib==1.5.3
threadpoolctl==3.6.0
narwhals==2.21.2
transformers==4.46.3
accelerate==1.9.0
sentencepiece==0.1.99
regex==2023.12.25
tqdm==4.66.4
loguru==0.7.2
pyyaml==6.0.1
rich==13.7.1


2) 주요 설치 시스템 패키지﻿

git
build-essential
python3.11
python3.11-dev
python3.11-venv
python3-pip
libffi-dev
libblas3
liblapack3
libomp-dev
tzdata
unzip
p7zip-full
gfortran
libatlas-base-dev
default-jre-headless
cmake
pkg-config
ninja-build
libgl1
libglib2.0-0


유의사항
제출 시 발생하는 오류의 종류는 두 가지로 정의되며, 일일 제출 횟수 반영에 대한 기준이 다르므로 반드시 숙지하여 진행해야 합니다.
1) 설치 오류 : 제출하는 submit.zip 내부 구조가 불일치한 경우, 패키지 설치 오류 -> 일일 제출 횟수 반영되지 않음
2) 제출 오류 : script.py 코드 실행 후 발생하는 모든 오류 -> 일일 제출 횟수 반영됨
script.py 내에서 open/ 디렉토리의 데이터를 로드하고, output/ 디렉토리에 예측 결과를 반드시 submission.csv의 파일명으로 저장되어야 합니다.
평가 서버 환경은 인터넷 접속이 불가능하므로, 패키지 설치 이후 외부 다운로드가 필요한 코드나 모델은 작동하지 않습니다.

2. 대회 규칙

1) 사전학습모델 및 가중치 사용 가능 범위

공식적으로 누구에게나 가중치가 공개되었으며, 최소한 비상업적 이용이 허용된 라이선스(MIT, Apache 2.0 등) 하에 배포된 모델 및 가중치만 사용 가능합니다.
해당 조건을 만족하지 않는 모델 및 가중치는 사용할 수 없습니다.
2) 외부 API 사용 제한

원격 서버 기반의 API 형태로만 접근 가능한 모델(OpenAI API, Gemini API 등)은 사용이 불가합니다.
모든 작업은 로컬 환경에서 직접 코드로 실행 및 재현 가능해야 하며, 외부 서버에 의존하는 방식은 제한됩니다.
3) 외부 데이터 사용 금지

온라인 해커톤(Phase 2)에서 제공하는 공식 데이터 외의 외부 데이터는 사용할 수 없습니다.
		   4) 추론 코드의 평가 데이터 예측 원칙

평가 데이터(test.csv)의 각 행은 하나의 독립적인 예측 대상입니다.
참가자는 각 행에 포함된 입력 변수와 주최 측이 제공한 공식 학습 데이터만을 이용하여 해당 행의 예측값(제구 성공 확률)을 추론해야 합니다. 
평가 데이터의 다른 행이나 전체 평가 데이터의 분포를 이용해 특정 행의 예측값을 보정하거나 생성하는 방식은 정상적인 추론 절차로 인정되지 않습니다.



## 제출 모델 만들기 설명서

[Baseline] RandomForest — 학습
KBO 투구 하나가 제구 성공 투구일 확률을 예측하는 베이스라인입니다.

입력: test.csv 의 47개 컬럼 (경기 상황, 투수·타자의 직전까지 누적 기록 등)
출력: 제구 성공 확률 (0 이상 1 이하의 실수)
평가지표: Brier Skill Score
trackman_history.csv 는 이 베이스라인에서 사용하지 않습니다. 2019~2024 과거 로그 179만 행이 그대로 남아 있으니 직접 활용해 보세요.

이 노트북은 모델을 학습하여 ./model/rf.pkl 로 저장합니다. 저장한 모델은 추론용 script.py 와 함께 baseline_submit.zip 으로 묶어 제출합니다.

1. 라이브러리 불러오기
데이터 처리(pandas)와 모델 학습(scikit-learn)에 필요한 라이브러리를 불러옵니다. joblib 은 학습한 모델을 파일로 저장할 때 사용합니다.

```
import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
```

2. 데이터 불러오기
train.csv 는 2019~2024 시즌이고 평가 데이터는 2025 시즌입니다.

사용할 피처 목록은 test.csv 가 정합니다. train.csv 에만 있는 컬럼을 학습에 넣으면 평가 시점에 그 컬럼이 없어 추론이 실패하기 때문입니다.

```
test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                        encoding="utf-8-sig", nrows=0).columns
FEATURES = [c for c in test_cols if c != ID]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                    encoding="utf-8-sig", usecols=FEATURES + [TARGET])

print("train:", train.shape, "| 피처:", len(FEATURES),
      f"(범주형 {len(CAT_COLS)}, 수치형 {len(NUM_COLS)})")
print("시즌:", train["season"].min(), "~", train["season"].max())
print(f"제구 성공률: {train[TARGET].mean():.4f}")
```

3. 전처리 정의
범주형 3개(top_bottom, game_type, base_state)는 정수로 바꾸고, 수치형 44개의 결측값은 중앙값으로 채웁니다.

두 변환을 ColumnTransformer 로 묶어 모델 파이프라인 안에 넣습니다. 이렇게 하면 추론할 때도 같은 변환이 자동으로 따라가므로, 전처리를 빠뜨려 생기는 실수를 막을 수 있습니다. handle_unknown="use_encoded_value" 는 학습 때 보지 못한 범주가 평가 데이터에 나타나면 -1 로 처리하라는 뜻입니다.

```
preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                           unknown_value=-1), CAT_COLS),
    ("num", SimpleImputer(strategy="median"), NUM_COLS),
])
```

4. 모델 정의와 학습
트리 깊이를 10, 잎 노드의 최소 샘플 수를 200으로 제한해 얕게 두었습니다. 학습이 1분 안에 끝나고 모델 파일도 4MB 정도로 가볍습니다. random_state 를 고정했고 GPU 를 사용하지 않으므로, 같은 데이터와 같은 패키지 버전이면 어느 환경에서 실행해도 결과가 같습니다.

```
model = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=200,
        n_jobs=-1,
        random_state=42,
    )),
])
```

2024 시즌을 검증용으로 떼어 두고 2019~2023 으로 학습합니다.

```
is_val = train["season"] == 2024
X_train, y_train = train.loc[~is_val, FEATURES], train.loc[~is_val, TARGET]
X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]
print("train:", len(X_train), "| val:", len(X_val))

t = time.time()
model.fit(X_train, y_train)
print(f"학습 완료 :: {time.time() - t:.1f}s")
```

5. 검증 — Brier Skill Score
학습 데이터에서 떼어 둔 2024 시즌으로 검증 점수를 계산합니다.

Brier 는 예측 확률과 실제값(0/1) 차이의 제곱 평균이고, 이를 상수 예측의 Brier 인 r(1-r) 로 나누어 Brier Skill Score 를 구합니다. 검증 분할 방식은 참가자가 자유롭게 바꿀 수 있습니다.

```
val_pred = model.predict_proba(X_val)[:, 1]

r = y_val.mean()
brier = ((val_pred - y_val) ** 2).mean()
baseline_brier = r * (1 - r)
score = max(0, 100000 * (1 - brier / baseline_brier))

print(f"Brier: {brier:.6f} | 기준선 r(1-r): {baseline_brier:.6f}")
print(f"Validation Score: {score:.2f}")
```

6. 전체 데이터로 재학습 & 모델 저장
검증으로 성능을 확인했으니 이제 전체 학습 데이터로 다시 학습합니다.

학습한 파이프라인을 ./model/rf.pkl 로 저장합니다. 이 파일을 추론용 script.py, requirements.txt 와 함께 baseline_submit.zip 으로 묶으면 제출 준비가 끝납니다.

```
t = time.time()
model.fit(train[FEATURES], train[TARGET])
print(f"재학습 완료 :: {time.time() - t:.1f}s")

os.makedirs("./model", exist_ok=True)
joblib.dump(model, "./model/rf.pkl", compress=3)
print("저장 완료: ./model/rf.pkl")
```

## 데이터 설명서

### [배포용 데이터 구조]

```text
open.zip

baseline_submit.zip : 베이스라인 코드 기반 리더보드 제출 파일(zip) 예시 (참고용)
data/
  - train.csv : 학습 입력 및 정답 데이터 (1,475,092행 x 49컬럼)

  - test.csv : 평가 입력 데이터 (형식 확인용 5건 샘플, 48컬럼)

  - sample_submission.csv : 제출 양식 파일 (형식 확인용 5행 x 2컬럼)

  - trackman_history.csv : 2019~2024년 Trackman 과거 로그 (1,793,078행 x 30컬럼)

data_description.md : 데이터 설명서
```

※ `test.csv`는 배포본에 형식 확인용 5건만 포함됩니다. 실제 평가 데이터는 비공개이며, 제출용 파일(zip)을 리더보드 평가에 제출하면 평가 서버에서 동일한 경로와 동일한 컬럼 구조의 실제 평가 데이터로 교체되어 처리됩니다.

※ `sample_submission.csv` 역시 형식 확인용 5건 샘플입니다. 실제 평가 시에는 평가 서버의 `test.csv` 행 수와 동일한 제출 양식을 생성해 제출해야 합니다.

※ 모든 데이터 파일은 CSV 형식입니다. `trackman_history.csv`는 메인 학습/평가 데이터와 1:1로 결합하는 정답 테이블이 아니라, 참가자가 과거 이력 기반 피처를 만들 때 참고할 수 있는 로그 데이터입니다.

### [세부 설명]

#### 1) 학습/추론 데이터: `train.csv` · `test.csv`

각 행은 한 개의 투구 시점 상태를 나타냅니다. 참가자는 투구 직전까지 알 수 있는 경기 상황, 선수/팀 정보, 과거 이력 피처를 바탕으로 해당 투구의 제구 성공 확률을 예측합니다.

`train.csv`와 `test.csv`의 입력 피처 구조는 동일합니다. 단, `train.csv`에는 학습 정답인 `control_success`가 포함되고, `test.csv`에는 정답 컬럼이 포함되지 않습니다.

##### 기본 식별자 및 경기 정보

| 컬럼 | 설명 |
| --- | --- |
| `row_id` | 샘플 고유 식별자입니다. 제출 파일과 매칭하는 데 사용합니다. |
| `season` | 시즌 연도입니다. |
| `game_month` | 경기 월입니다. |
| `game_dayofweek` | 경기 요일입니다. 월요일은 0, 일요일은 6입니다. |
| `inning` | 투구 직전 이닝입니다. |
| `top_bottom` | 초/말 구분입니다. `T`는 초, `B`는 말을 의미합니다. |
| `game_type` | 경기 유형 코드입니다. |

##### 투구 직전 카운트 및 점수 상황

| 컬럼 | 설명 |
| --- | --- |
| `balls_before` | 투구 직전 볼 카운트입니다. |
| `strikes_before` | 투구 직전 스트라이크 카운트입니다. |
| `outs_before` | 투구 직전 아웃 카운트입니다. |
| `run_top_before` | 투구 직전 초 공격 팀의 점수입니다. |
| `run_bot_before` | 투구 직전 말 공격 팀의 점수입니다. |
| `run_total_before` | 투구 직전 양 팀 합산 점수입니다. |
| `score_diff_home` | 투구 직전 홈 팀 기준 점수 차입니다. |
| `score_diff_pitcher_team` | 투구 직전 투수 소속 팀 기준 점수 차입니다. |

##### 주자 및 상황 중요도

| 컬럼 | 설명 |
| --- | --- |
| `runner_on_1b` | 투구 직전 1루 주자 여부입니다. `1`은 있음, `0`은 없음을 의미합니다. |
| `runner_on_2b` | 투구 직전 2루 주자 여부입니다. `1`은 있음, `0`은 없음을 의미합니다. |
| `runner_on_3b` | 투구 직전 3루 주자 여부입니다. `1`은 있음, `0`은 없음을 의미합니다. |
| `num_runners_on` | 투구 직전 출루 주자 수입니다. |
| `base_state` | 투구 직전 주자 상황입니다. `___`=주자 없음, `1__`=1루, `_2_`=2루, `__3`=3루, `12_`=1/2루, `1_3`=1/3루, `_23`=2/3루, `123`=만루입니다. |
| `home_win_expectancy` | 투구 직전 경기 상황에서 홈 팀의 기대 승률입니다. 0~100 범위의 값입니다. |
| `away_win_expectancy` | 투구 직전 경기 상황에서 원정 팀의 기대 승률입니다. 0~100 범위의 값입니다. |
| `li` | 투구 직전 상황 중요도 지표입니다. 값이 클수록 경기 흐름에 미치는 영향이 큰 상황을 의미합니다. |

##### 선수 및 팀 정보

| 컬럼 | 설명 |
| --- | --- |
| `pitcher_id` | 투수 익명 ID입니다. |
| `batter_id` | 타자 익명 ID입니다. |
| `pitcher_hand` | 투수의 좌우 유형 코드입니다. |
| `batter_hand` | 타자의 좌우 유형 코드입니다. |
| `pitcher_team_id` | 투수 소속 팀 ID입니다. |
| `batter_team_id` | 타자 소속 팀 ID입니다. |

##### 투구 직전 기준 과거 이력 피처

`asof_*` 컬럼은 해당 행의 투구 직전까지 확인 가능한 과거 기록으로 사전 계산된 피처입니다. 현재 투구 이후에 확정되는 정보는 사용하지 않았습니다.

| 컬럼 | 설명 |
| --- | --- |
| `asof_pitcher_n` | 해당 투구 직전까지 해당 투수의 누적 투구 수입니다. |
| `asof_pitcher_success_rate` | 해당 투구 직전까지 해당 투수의 제구 성공률입니다. |
| `asof_pitcher_reverse_rate` | 해당 투구 직전까지 해당 투수의 의도 반대성 투구 비율입니다. |
| `asof_pitcher_middle_rate` | 해당 투구 직전까지 해당 투수의 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_ball_rate` | 해당 투구 직전까지 해당 투수의 볼성 결과 비율입니다. |
| `asof_pitcher_strike_rate` | 해당 투구 직전까지 해당 투수의 스트라이크성 결과 비율입니다. |
| `asof_pitcher_prev1_game_success_rate` | 해당 투수의 직전 1경기 제구 성공률입니다. |
| `asof_pitcher_prev3_game_success_rate` | 해당 투수의 직전 3경기 제구 성공률입니다. |
| `asof_pitcher_prev5_game_success_rate` | 해당 투수의 직전 5경기 제구 성공률입니다. |
| `asof_pitcher_prev1_game_middle_rate` | 해당 투수의 직전 1경기 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_prev3_game_middle_rate` | 해당 투수의 직전 3경기 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_prev5_game_middle_rate` | 해당 투수의 직전 5경기 가운데 또는 위험 코스 비율입니다. |
| `asof_batter_n` | 해당 투구 직전까지 해당 타자가 상대한 누적 투구 수입니다. |
| `asof_batter_success_rate` | 해당 투구 직전까지 해당 타자가 상대한 투구의 제구 성공률입니다. |
| `asof_batter_middle_rate` | 해당 투구 직전까지 해당 타자가 상대한 투구의 가운데 또는 위험 코스 비율입니다. |
| `asof_pitcher_pitchmix_n` | 해당 투구 직전까지 해당 투수의 구종 사용 이력 표본 수입니다. |
| `asof_pitcher_fastball_rate` | 해당 투구 직전까지 해당 투수의 fastball 계열 사용 비율입니다. |
| `asof_pitcher_breaking_rate` | 해당 투구 직전까지 해당 투수의 breaking 계열 사용 비율입니다. |
| `asof_pitcher_offspeed_rate` | 해당 투구 직전까지 해당 투수의 offspeed 계열 사용 비율입니다. |

※ 표본 수가 0인 경우 일부 rate 컬럼은 결측값일 수 있습니다. 이런 cold-start 상황의 결측 처리, smoothing, fallback 전략은 참가자가 자유롭게 설계할 수 있습니다.

#### 2) 학습 정답 데이터: `train.csv`의 `control_success`

`train.csv`에는 아래 정답 컬럼이 포함됩니다.

| 컬럼 | 설명 |
| --- | --- |
| `control_success` | 예측 대상입니다. `1`은 제구 성공, `0`은 제구 실패를 의미합니다. |

`control_success`는 운영 기준에 따라 산출된 제구 성공 여부입니다. Target 산출에 사용되는 현재 투구의 사후 정보는 입력 피처로 제공되지 않습니다.

#### 3) 과거 Trackman 로그: `trackman_history.csv`

`trackman_history.csv`는 2019~2024년 Trackman 과거 로그입니다. 2025년 Trackman 데이터는 제공되지 않습니다.

참가자는 이 파일을 이용해 과거 투구 특성, 구종 특성, 투수 단위 요약값 등 추가 피처를 만들 수 있습니다. 단, 이 파일은 `train.csv` 또는 `test.csv`와 1:1로 직접 결합되는 테이블이 아니며, 평가 시점 이후 정보를 포함하는 방식으로 사용할 수 없습니다.

| 컬럼 | 설명 |
| --- | --- |
| `trackman_id` | Trackman 과거 로그의 행 식별자입니다. |
| `season` | Trackman 투구가 속한 시즌입니다. 2019~2024만 포함됩니다. |
| `game_date` | Trackman 투구의 경기 날짜입니다. |
| `game_month` | Trackman 투구의 경기 월입니다. |
| `game_dayofweek` | Trackman 투구의 경기 요일입니다. 월요일은 0, 일요일은 6입니다. |
| `trackman_game_id` | Trackman 기준 경기 ID입니다. 메인 데이터의 `row_id`와 직접 대응하지 않습니다. |
| `pitch_no` | Trackman 기준 경기 내 투구 번호입니다. |
| `inning` | Trackman 로그의 이닝입니다. |
| `top_bottom` | Trackman 로그의 초/말 표기입니다. |
| `balls_before` | 투구 직전 볼 카운트입니다. |
| `strikes_before` | 투구 직전 스트라이크 카운트입니다. |
| `outs_before` | 투구 직전 아웃 카운트입니다. |
| `pitch_of_pa` | 해당 타석에서의 투구 순번입니다. |
| `pitcher_trackman_id` | Trackman 기준 투수 ID입니다. |
| `batter_trackman_id` | Trackman 기준 타자 ID입니다. |
| `pitcher_hand` | 투수의 좌우 유형 코드입니다. |
| `batter_hand` | 타자의 좌우 유형 코드입니다. |
| `pitcher_team` | 투수 소속 팀입니다. |
| `batter_team` | 타자 소속 팀입니다. |
| `tagged_pitch_type` | 수동 또는 태깅 기반 구종명입니다. |
| `auto_pitch_type` | 자동 분류 기반 구종명입니다. |
| `pitch_type_group` | 구종을 `fastball`, `breaking`, `offspeed`, `other`로 단순화한 구종군입니다. |
| `rel_speed` | 릴리스 시점의 구속입니다. |
| `spin_rate` | 투구 회전수입니다. |
| `induced_vert_break` | 유도 수직 무브먼트입니다. |
| `horz_break` | 수평 무브먼트입니다. |
| `extension` | 릴리스 확장 거리입니다. |
| `rel_height` | 릴리스 높이입니다. |
| `rel_side` | 릴리스 좌우 위치입니다. |
| `zone_speed` | 홈플레이트 근처 구속입니다. |

#### 4) 모델 추론 결과 양식 파일: `sample_submission.csv`

| 컬럼 | 설명 |
| --- | --- |
| `row_id` | 평가 데이터 `test.csv`의 샘플 식별자입니다. |
| `control_success` | 예측값입니다. 해당 투구가 제구에 성공할 확률을 0 이상 1 이하의 실수로 입력합니다. |

※ 제출 시 `row_id` 값은 평가 서버에서 제공되는 `test.csv`와 정확히 일치해야 합니다.

#### 5) 평가 데이터 예측 원칙

평가 데이터의 각 행은 독립적으로 예측해야 합니다. 평가 서버에서 실제 `test.csv` 전체가 주어지더라도, 참가자는 `test.csv`의 다른 행을 이용해 현재 행의 피처를 만들 수 없습니다.

금지되는 예시는 다음과 같습니다.

- `test.csv` 내부 행들을 이용한 선수별, 팀별, 월별 누적 통계
- `test.csv` 내부 빈도값 또는 분포 통계
- `test.csv` 내부 target encoding
- `test.csv` 행 순서 기반 rolling 또는 expanding feature
- 평가 데이터 전체를 보고 만든 사후 보정값

운영 측에서 제공한 `asof_*` 컬럼은 각 행의 투구 직전 시점까지의 과거 기록만으로 계산된 공식 입력 피처이므로 사용할 수 있습니다.

#### 6) 사용 금지 정보

공정한 평가를 위해 다음 정보는 입력으로 사용할 수 없습니다.

- 현재 투구 이후에 확정되는 모든 정보
- 현재 투구의 실제 위치 또는 코스 정보
- 현재 투구의 실제 판정, 결과, 제구 성공 여부
- 현재 투구의 실제 구종
- 현재 투구의 Trackman 측정값
- 2025년 Trackman 데이터
- 평가 데이터 내부의 다른 행을 이용해 만든 누적, 빈도, 분포, rolling, target encoding 피처

제공된 `train.csv`, 평가 환경의 `test.csv`, 2019~2024년 `trackman_history.csv`, 그리고 대회 규칙상 허용되는 외부 데이터만 사용할 수 있습니다.

1) 반드시 지켜야 할 핵심 원칙
'평가 데이터의 각 행은 독립적으로 예측되어야 합니다.'

즉, 특정 행 A의 예측값은 아래 정보만으로 생성되어야 합니다.

- 행 A에 포함된 입력 변수
- 행 A의 입력 변수만을 이용해 생성한 파생변수
- 주최 측이 제공한 공식 학습 데이터
- 공식 학습 데이터만을 이용해 생성한 통계·모델·파생변수


2) 허용되지 않는 방식
아래와 같은 방식은 정상적인 추론 절차로 인정되지 않습니다.

test.csv 내 다른 행을 이용한 누적 통계 생성
test.csv 내 다른 행을 이용한 rolling / lag feature 생성
test.csv 전체의 평균, 분포, 빈도, 순위 등을 이용한 예측값 보정
같은 선수, 팀, 월, 경기 단위로 평가 데이터 내 다른 행을 집계하는 방식
평가 데이터상 시점이 더 과거로 보이는 행을 이용해 현재 행의 피처를 생성하는 방식
평가 데이터에 시점 정보가 포함되어 있더라도,

같은 test.csv 안의 다른 행은 현재 예측 대상 행의 추론에 사용할 수 없습니다.



3) 쉽게 이해할 수 있는 예시
어떤 행의 예측값은 아래 두 경우 모두 동일해야 합니다.

test.csv에 해당 행 1개만 있는 경우
test.csv에 전체 평가 데이터가 함께 있는 경우
두 경우의 예측값이 달라진다면, 평가 데이터의 다른 행이 추론에 영향을 준 것으로 볼 수 있습니다.

## Commands

There is no build/lint/test tooling — this is a data science pipeline run as plain scripts.

```bash
# Run the full pipeline: train -> validate against reference -> full retrain -> submit artifact
python dopip.py

# Run pieces individually
python code/train.py   # trains, writes ./open/temp/latest_model.pkl
python code/test.py    # evaluates latest_model.pkl vs open/reference/best_model.pkl, writes open/temp/compare_result.txt
```

Data must be manually downloaded into `open/data/` (`train.csv`, `test.csv`, `trackman_history.csv`, `sample_submission.csv`) — this directory is gitignored. `trackman_history.csv` is part of the official dataset and **is** read by the main pipeline (`code/train.py`, `code/test.py`, `dopip.py`, `submit/script.py`) as of 2026-08-17 — see "Trackman history" under "Pipeline flow" below for what's used (tier A + coarse pitchmix) versus what stays rejected.

`torch`, `pandas`, `numpy`, `scikit-learn` are all pre-installed on the competition eval server (see package list above) — `submit/requirements.txt` intentionally stays empty rather than pinning versions that could conflict. Locally, use the project's `.venv` (has `torch` with CUDA support already installed).

## Architecture

### Production model: CatBoost + MLP ensemble blend (`code/blend_model.py`, `code/catboost_model.py`)

The model actually shipped in `submit/` is **not** the MLP ensemble alone — it's a stacked combination of two independently-trained models: CatBoost (hyperparameters hardcoded in `code/catboost_model.py::CATBOOST_PARAMS`, from the team's own Optuna search via `code/tune.py` — most recently re-run on the current trackman-free/F1-filtered feature set, EXPERIMENTS.md §21) and the Tabular MLP ensemble described below. Final prediction = `sigmoid(w_cat * catboost_pred + w_mlp * mlp_ensemble_pred + intercept)`, where `w_cat`/`w_mlp`/`intercept` are fit per-run by a 2-feature `LogisticRegression` (`code/blend_model.py::fit_meta_model`) trained on `[catboost_pred, mlp_pred] -> y` over the validation split. This stacking approach replaced an earlier fixed weighted-average blend (rolling-origin CV showed it consistently wins, EXPERIMENTS.md §9) — see `PROJECT_HISTORY.md` §6-§8 for that transition, and §14-§19 for the later trackman-removal/F1-filter/re-tuning changes that got the underlying CatBoost and feature set to their current state. "What this is" above has current scores.

- **Bundle schema**: `{"catboost_model": <CatBoostClassifier>, "mlp_bundle": {...the MLP ensemble bundle below...}, "meta_model": {"w_cat": float, "w_mlp": float, "intercept": float}, "catboost_best_iteration": int}`. `catboost_model` is a plain library object (not project-defined), so it unpickles fine anywhere `catboost` is installed — no `code/` package dependency, same reasoning as the MLP bundle and `meta_model` being plain dicts (the latter stores raw floats, not a pickled `sklearn.linear_model.LogisticRegression` instance, so inference never needs `sklearn.linear_model` — just the sigmoid formula in `code/blend_model.py::predict_meta`).
- `code/catboost_model.py::train_catboost(X_train, y_train, X_val=None, y_val=None)` mirrors `train_mlp`'s early-stopping pattern: with a validation set it early-stops on `BrierScore` (patience 50) and returns `(model, best_iteration)`; without one (full-retrain case) it trains a fixed `iterations` count.
- CatBoost's categorical features are just `['game_type', 'base_state']` — `top_bottom` is mapped to int 0/1 directly in `code/train.py::main()`/`dopip.py`/`submit/script.py::main()` before CatBoost ever sees it, so CatBoost doesn't need it declared as categorical (confirmed via `model.get_cat_feature_indices()` on the old tuned reference). This mapping used to live inside the now-removed `process_trackman_features_safe` (it was needed there to join against trackman's `Top`/`Bottom` string convention); after trackman was dropped (EXPERIMENTS.md §17.3) it moved to each entrypoint's `main()`.
- `submit/requirements.txt` pins `catboost==1.2.10` (matching the locally-installed/trained version, to avoid cross-version pickle issues) — this is *not* in the eval server's pre-installed package list, unlike torch/pandas/numpy/scikit-learn.

### MLP ensemble component (`code/mlp_model.py`)

The model is a small PyTorch net (`TabularMLP`): an `nn.Embedding` per low-cardinality categorical column, concatenated with the numeric columns after those go through a **QuantileEmbedding** (quantile-based piecewise-linear encoding, "PLE"/Q-LR from Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep Learning": each numeric feature is bucketed into `n_bins=24` quantile bins fit on the training split, encoded as a piecewise-linear position-in-bin vector, then passed through a per-feature independent `Linear(24 → 8) + ReLU` to get an 8-dim embedding per numeric feature), concatenated with the categorical embeddings and fed through `Linear(128) → BatchNorm → ReLU → Dropout(0.3) → Linear(64) → BatchNorm → ReLU → Linear(1) → Sigmoid`. Trained with `BCELoss` + `AdamW` (lr 0.003, weight_decay 0.01), batch size 4096. This replaced the earlier "standardize + concat raw numeric values" scheme after a 7-seed paired-seed grid sweep over `n_bins` (4/12/16/24/32) showed `n_bins=16`–`24` winning 7/7 seeds with delta ≈ +50–56 pts (well outside the ~17–24 pt seed-to-seed std), vs. earlier `n_bins=8` and a sin/cos "periodic" embedding variant that both washed out to noise at 7-seed re-check — see EXPERIMENTS.md §15 (§15.3 for the adoption). `code/mlp_model.py::TabularMLP` still accepts a `bin_edges=None` fallback that reproduces the old raw-concat behavior, kept for backward compatibility with the archived embedding-contrast scripts (`code/periodic_mlp_model.py`/`code/experiment_periodic_embed.py`, `code/quantile_mlp_model.py`/`code/experiment_quantile_embed.py` — still untracked, not part of the `dopip.py` main path, kept for future re-sweeps at other `n_bins`/`d`).

- `CAT_COLS = ['top_bottom', 'game_type', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id']` (defined in `code/mlp_model.py`) are embedded, sized per-column via `embed_dim_for_cardinality` (fastai-style heuristic, `min(50, (cardinality+1)//2)`, floor 4). Every other feature column is numeric (median-imputed, standardized, then quantile-embedded as described above). Two categorical columns were tried and explicitly reverted after real-data testing — see the `CAT_COLS` comment in `code/mlp_model.py` and EXPERIMENTS.md for the full story:
  - `pitcher_id`/`batter_id` (cardinality 792/830): embedding these overfit within ~3 epochs and tanked validation score (660 → 304). Too high-cardinality relative to ~1.2M training rows for a raw learned embedding — unlike CatBoost's regularized target-statistic handling of high-cardinality categoricals. Their signal is already present indirectly via the `asof_pitcher_*`/`asof_batter_*` running-rate features, so dropping them costs little.
  - `season`: at the time this was tested, validation was *entirely* season==2024, which never appeared in training. Embedding it means every validation row shares one untrained/random embedding vector — a systematic miscalibration (660 → 351). As a numeric feature the model can at least extrapolate smoothly to an unseen season value, which matters because the real eval `test.csv` is season 2025 — a value that *never* appears in `train.csv` (2019–2024) under any split convention, including the current cutoff=7 one (see split-convention section below), so this reasoning still holds even though train now includes some season==2024 rows. Any feature guaranteed to take on new values at inference time should stay numeric, not become an embedded category.
- **Ensemble, not a single model.** A single `TabularMLP` run has high epoch-to-epoch validation variance (same config: Val Score swings ~500–740 across epochs). `code/mlp_model.py::train_ensemble` trains one model per seed in `ENSEMBLE_SEEDS` (7 seeds) and predictions are averaged (`predict_ensemble`/`predict_bundle`). This reliably beat any single-model score in real testing (best single model ~715–740 vs. 7-seed ensemble 780–790); 15 seeds and a wider network (256/128) were also tried and gave no further improvement, so 7 was kept. See EXPERIMENTS.md for the full sweep.
- Models/preprocessors are **not** pickled as class instances. `code/mlp_model.py::make_bundle` packages everything needed for inference into a plain `dict` — `members` (a list of `{state_dict, best_epoch, seed}` per ensemble member), `cat_cols`/`num_cols`, `cat_dims`, `embed_dims`, the fitted `OrdinalEncoder`/`SimpleImputer`/`StandardScaler`, `bin_edges` (the `(num_numeric, n_bins+1)` quantile-boundary tensor fit on the training split via `fit_quantile_edges`, consumed by `QuantileEmbedding`) + `quantile_d`, and `best_epoch_` (mean best epoch across members, informational). This is deliberate: `submit.zip` does not include the `code/` package, so a pickled instance of a project-defined class would fail to unpickle on the eval server. A plain dict of tensors + sklearn objects does not have this problem. (Older bundles predating the quantile-embedding adoption simply lack the `bin_edges`/`quantile_d` keys — `predict_bundle` uses `.get()` and falls back to the old raw-concat inference path for those.)
- `code/mlp_model.py::train_mlp` does the training loop for a single member (now takes a `seed` param). If given validation tensors it early-stops on validation Brier score (patience 7, up to 60 epochs) and returns the best-epoch weights + `best_epoch`; without validation tensors (full-retrain case) it just runs `max_epochs` and returns the final weights. `train_ensemble` calls it once per seed.

### Pipeline flow (`dopip.py`)

`dopip.py` is the orchestrator, run end-to-end:
1. Runs `code/train.py` as a subprocess → trains the 7-seed MLP ensemble *and* a CatBoost model on the same train/val split, fits the stacking `meta_model` (`code/blend_model.py::fit_meta_model`) on the validation predictions, and produces `open/temp/latest_model.pkl` (a blend bundle dict — see "Production model" above).
2. Runs `code/test.py` as a subprocess → rebuilds the same val split, scores `latest_model.pkl`'s **blended prediction** (`predict_blend_bundle`) against `open/reference/best_model.pkl` (the current best), writes `NEW_BEST` or `KEEP_REF` to `open/temp/compare_result.txt`. If the reference file isn't a current-format blend bundle (i.e. it's a leftover CatBoost-only/MLP-only pickle, or an older alpha-weighted-average blend bundle from before the stacking upgrade — detected by the absence of *any* of `"catboost_model"`, `"mlp_bundle"`, or `"meta_model"` keys), `test.py` logs a warning and treats it as "no reference" rather than crashing — the new model wins by default and gets promoted, which is how the repo self-migrates off stale reference formats the first time `dopip.py` is run after a bundle-schema change.
3. Based on that flag: backs up the previous latest/reference model into `open/former_model/` (auto-numbered `_v1`, `_v2`, ... via `get_numbered_path`), and if `NEW_BEST`, promotes `latest_model.pkl` to `open/reference/best_model.pkl`.
4. **Full retrain**: regardless of step 3's outcome, reloads whichever bundle is now `open/reference/best_model.pkl`. If it's a compatible blend bundle with an MLP ensemble of the same size as `ENSEMBLE_SEEDS` *and* a `"meta_model"` key, it reuses: each MLP seed's own `best_epoch` (+ 5-epoch buffer) as that seed's full-retrain epoch budget, CatBoost's `catboost_best_iteration` (+ 50-iteration buffer via `CATBOOST_ITERATION_BUFFER`) as its full-retrain iteration count, and the reference's `meta_model` weights as-is (not re-fit — there's no validation set at this stage). Both the MLP ensemble and CatBoost are retrained on the **entire** dataset (`is_train_split=False`, no held-out split, so no early stopping is possible), and the resulting blend bundle is written to `submit/model/final_retained_model.pkl`. I.e. the reference/eval model only decides *how many epochs/iterations to use, which preprocessing stats, and what meta-model weights* — the actual submitted model is always retrained on all available data.

`submit/script.py` is the standalone inference entrypoint used by the competition server — it does **not** import from `code/`. It duplicates the `TabularMLP` class definition, `add_engineered_features`, `apply_preprocessing`, and the blend inference logic (CatBoost `predict_proba` + MLP ensemble-averaging loop over `bundle["mlp_bundle"]["members"]`, combined via the sigmoid of `bundle["meta_model"]`'s `w_cat`/`w_mlp`/`intercept`) inline. It does *not* need to duplicate any CatBoost training/class code — `CatBoostClassifier` unpickles directly as a fully-trained object, `submit/script.py` just calls `.predict_proba()` on it. Keep the duplicated pieces in sync manually if `code/mlp_model.py`, `code/catboost_model.py`, `code/blend_model.py`, `code/train.py`, or `code/trackman_pitcher_features.py` change — this is the same manual-sync convention the CatBoost-era pipeline used for `add_engineered_features`. It loads `trackman_history.csv` (coarse pitchmix only as of 2026-08-18, see "Trackman history" below) and static CSVs bundled alongside the model: `pitcher_map.csv` (unused now that tier A is off, harmless leftover), `pitchmix_lookup.csv`, and `season_end_lookup.csv` (season-progression feature, see "What this is" above).

**Trackman history — dropped, then two features re-added (as of 2026-08-17)**: an earlier version joined `trackman_history.csv` via a `match_cols` fingerprint that included `season`, which structurally never matched at real inference (test.csv is always season 2025, trackman only covers 2019–2024) — silently zero for all 64 derived columns on a real submission before anyone noticed. Chasing a fix eventually showed that version of trackman carried no reliable signal under proper multi-season validation (2023+2024 average delta ≈0 for every matching strategy tried), corroborated by an independent teammate investigation, and it was fully removed for a long stretch (`code/train.py::process_trackman_features_safe` no longer exists). Full story: `PROJECT_HISTORY.md` §14-§16.3, `EXPERIMENTS.md` §16-§18.

A later session (§37/§38 in `PROJECT_HISTORY.md`) re-tried pitcher-identity-crosswalk-based trackman features (tiers A/B/C/F, `code/trackman_pitcher_features.py`) and shipped a version (A+B+C) that scored 869.52 on the real leaderboard — a −112.70 regression from 982.22. Root-causing that regression (next session) found a real bug in `clean_trackman()` (a NaN-vs-anomaly filtering bug that silently discarded ~8,000 legitimate trackman rows, `code/trackman_pitcher_features.py::clean_trackman`'s docstring has the details) and, after fixing it and re-validating on both a pre-ABS (season==2023) and ABS-regime (cutoff=7) holdout, found tier B's apparent gain was fake (it degraded CatBoost's own score and was only rescued by the stacking meta-model's reweighting — see PROJECT_HISTORY.md "핵심 교훈" #23) and tier C stayed negative on 2023 either way. A teammate-reported **coarse pitchmix** feature (count×handedness pitch-mix ratios from `trackman_history.csv`, joined only on `(balls_before, strikes_before, pitcher_hand, batter_hand)` — no pitcher identity, no season, so it structurally can't hit either of the two failure modes above) was independently validated and, combined with tier A only, passed both regimes cleanly (2023 +9.57 / cutoff=7 +34.43) with real per-submodel improvement (not meta-model reweighting tricks) on both CatBoost and MLP. That A+pitchmix combination was later submitted and scored 950.81 on the real leaderboard — a further −31.41 regression from 982.22 — and tier A was removed as a result (2026-08-18 session; local dual-regime checks were ambiguous, the real-leaderboard result broke the tie). **Current state: coarse pitchmix (→CatBoost) only; tier A and tiers B/C/F and the old fingerprint/asof9key approaches all stay rejected.** `TRACKMAN_TIER_FEED = {}` in both `code/train.py` and `submit/script.py`. Full story with numbers: `PROJECT_HISTORY.md` §37-§38, §45, `EXPERIMENTS.md` §38, §45.

**F1 filter (`code/train.py::apply_f1_filter`)**: `train = train[~((train['game_type'] == 'F') & (train['season'] <= 2022))]`, training data only — never touches `code/test.py`/`submit/script.py`'s validation/inference data. `game_type`'s F(퓨처스/2군)/R(1군) success-rate relationship inverted starting 2023, and `game_type` is CatBoost's #1 feature importance, so training on the pre-2023 relationship actively hurts (most visible on a season==2023 holdout — the unfiltered model collapses near zero there even though season==2024 alone looks fine). Story and numbers: `PROJECT_HISTORY.md` §16, `EXPERIMENTS.md` §18.

### Train/eval split convention

`code/train.py` and `code/test.py` share the same split logic and must be kept consistent with each other — `test.py` reconstructs the identical validation set from scratch to score against the reference model, it does not receive it from `train.py`.

**Current convention (cutoff=7, adopted after the ABS regime-shift investigation)**: `(train_df['season'] < 2024) | ((train_df['season'] == 2024) & (train_df['game_month'] < 7))` for training, `(train_df['season'] == 2024) & (train_df['game_month'] >= 7)` for validation — i.e. 2024's March–June rows are now training data, and only July–October 2024 is held out. This replaced the earlier **season == 2024 entirely held out** convention (`train_df['season'] < 2024` / `== 2024`), which itself had replaced an earlier chronological 80/20 positional split (`sort_values(['season','game_month','game_dayofweek'])` + `iloc[:split_idx]`) that, in practice, put 100% of 2024 plus part of Sept–Oct 2023 into validation. If touching the split again, verify actual row-count/season composition with a quick pandas check rather than trusting the code's comments — the "time filter" logic elsewhere in this codebase has a documented gap (see below).

**Why cutoff=7**: KBO adopted ABS (automatic ball-strike system) league-wide starting the 2024 season (previously piloted only in the Futures/F league 2020–2023) — a real rule change that plausibly shifts what `control_success` (a strike-zone-position label) means, so 2019–2023 (pre-ABS) data alone may not represent 2024/2025 (ABS-era) well. Including some 2024 data in training tests this directly. A cutoff/weight sweep (cutoff∈{4..10}, weight∈{1,5}) found weight=1 with cutoff=7 gives the largest, most reliable blend gain (+58.03 in 3-seed screening; +60.85 in the real 7-seed production run that got promoted) — see below for full validation including a real-leaderboard confirmation. Weight=5 (emphasizing 2024 rows) was tested and rejected — unstable across cutoffs and reversed on a larger out-of-sample check (2023's October, `EXPERIMENTS.md` §35.4, §35.6).

**Known reliability gap, deliberately not fully fixed**: a single-season-derived validation window is still sensitive to that window's league-wide baseline volatility, not just model quality (see `apply_f1_filter` above) — every experiment result computed on a single season/window alone should be read with that caveat. A true dual-season fix (`train<2023`, validate on `{2023,2024}`) was considered and rejected for the routine promotion decision: it would force `apply_f1_filter` to strip *all* F rows from the remaining 2019–2022 training data, destroying the exact benefit F1 exists for. So the dual-season check stays a manual audit tool (`code/experiment_f1_filter.py --holdout {2023,2024}`, `code/experiment_trackman_asof9key.py --holdout {2023,2024}`, `code/experiment_abs_regime.py`/`code/experiment_abs_mlp.py --val-cutoff N`) for judging individual candidate changes, not the routine promotion decision `code/test.py`/`dopip.py` make every run. Full reasoning: `PROJECT_HISTORY.md` §18, §34, §35, `EXPERIMENTS.md` §18.1, §20, §35–§37.

**Real-leaderboard confirmation of cutoff=7**: local blend score under the new split went from 644.49 (old-convention reference, re-scored on the new 2024-Jul–Oct val set) to 705.34 (+60.85) after a full production run (7-seed MLP + re-tuned CatBoost). Submitted to the real leaderboard: **982.22**, up from the prior best of 971 (**+11.22**) — this is the third "did a local win hold up on the real leaderboard" check this project has run; unlike the two prior cases (MLP hyperparameter retune, CatBoost GPU retune — both reverted), this one held and was kept. `EXPERIMENTS.md` §37.

Alternate split strategies were prototyped and are kept as untracked/parallel scripts rather than merged in:
- `code/train_x30.py` / `code/test_x30.py`: train on all seasons < 2024 + first 70% of 2024 (season-ordered), validate on the last 30% of 2024.
- `code/train_rd30.py` / `code/test_rd30.py`: fully random 30% holdout across all seasons (stratified `train_test_split`), ignoring time ordering entirely.

These variants also diverge structurally from `train.py`/`test.py`: they read `test.csv`'s columns to determine `base_features`, and merge in trackman via a much lighter `build_pitcher_consistency_features` (per-pitcher/pitch-type-group std of break/speed metrics) rather than `process_trackman_features_safe`'s full situational join — closer to what `submit/script.py` does.

**Not migrated to Tabular MLP**: `code/train_x30.py`, `code/test_x30.py`, `code/train_rd30.py`, `code/test_rd30.py`, and `code/train.last.py` are all still CatBoost-based and were left untouched by the MLP migration — they're experimental/historical, not part of the `dopip.py` main path; they also still reference a lighter, never-migrated trackman join of their own (`build_pitcher_consistency_features`), separate from (and untouched by) the trackman removal described above. `code/tune.py` (Optuna search for `CATBOOST_PARAMS`) is CatBoost-only for the same reason (it doesn't train the MLP side) but *is* kept in sync with current feature engineering — it was updated this session to drop trackman and apply the F1 filter (EXPERIMENTS.md §21).

### Model storage convention

All `.pkl` files below are now **blend bundle dicts** with `"catboost_model"` + `"mlp_bundle"` (the latter containing a `"members"` list) — see "Production model" above, not raw model objects and not an MLP-only bundle.

- `open/temp/latest_model.pkl` — most recent `train.py` output, transient.
- `open/reference/best_model.pkl` — current best-scoring bundle, source of truth for CatBoost's `catboost_best_iteration`, each MLP member's `best_epoch`, and the stacking `meta_model` weights, all reused in full retrain. This file may still be a leftover raw `CatBoostClassifier` pickle, an MLP-only bundle, an older single-model MLP bundle, or an older alpha-weighted-average blend bundle (missing `"catboost_model"`/`"mlp_bundle"`/`"meta_model"` keys) until the next `dopip.py` run promotes a new stacking blend bundle over it; `test.py` handles that transition gracefully (see pipeline flow above).
- `open/former_model/` — auto-numbered backups of superseded latest/reference models, written by `dopip.py` every run (this is also where old CatBoost-only/MLP-only/pre-blend references end up once superseded).
- `submit/model/final_retained_model.pkl` — the actual artifact submitted to the competition server, always a full-data retrain of both CatBoost and the 7-member MLP ensemble.

