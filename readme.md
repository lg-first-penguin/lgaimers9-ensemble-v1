CatBoost + Tabular MLP 앙상블 블렌드 파이프라인입니다. (v3)

ubuntu 24.04.4 LTS를 Windows WSL 기능을 사용해 만들었습니다.

환경세팅(패키지나 라이브러리)은 대회 서버 기본 내장 패키지로 맞추었습니다.

## environments

```
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.4 LTS
Release:        24.04
Codename:       noble

python 3.11,

"torch==2.7.1+cu128" \
"pandas==2.0.3" \
"numpy==1.26.4" \
"scipy==1.15.3" \
"scikit-learn==1.8.0" \
"joblib==1.5.3" \
"transformers==4.46.3" \
"accelerate==1.9.0" \
"tqdm==4.66.4" \
"loguru==0.7.2" \

```

## 파이프라인 시스템 구조

현재 프로덕션 모델은 **CatBoost + Tabular MLP(7-seed 앙상블) 블렌드**입니다.
두 모델을 각각 학습한 뒤, 검증 데이터에서 alpha(0.01 단위 그리드 서치)로 예측 확률을 가중평균해 블렌딩합니다. (`alpha ≈ 0.59`, CatBoost 쪽 가중치)

```

lgaimers9 (루트 폴더)
├── submit/
│   ├── model/
│   │   └── final_retained_model.pkl (dopip.py 실행 시 전체 데이터 재학습 후 자동 생성 — CatBoost+MLP 블렌드 번들)
│   ├── script.py (대회 서버 전용 추론 파일, code/ 미의존 — TabularMLP/전처리/블렌드 로직 자체 복제)
│   └── requirements.txt (catboost==1.2.10만 명시 — torch/pandas/numpy/sklearn은 서버 기본 설치라 비워둠)
├── open/
│   ├── data/ (참가자 데이터 폴더 - train.csv, test.csv, trackman_history.csv)
│   ├── former_model/ (이전 temp 및 reference 후보 저장 버퍼, 자동 넘버링 _v1, _v2 ...)
│   ├── reference/ (현재 최고 스코어를 기록한 기준 블렌드 번들 저장)
│   └── temp/ (방금 학습을 마친 따끈따끈한 최신 블렌드 번들 및 비교 결과 저장)
├── code/
│   ├── mlp_model.py (TabularMLP 정의, 7-seed 앙상블 학습/추론, 번들 패키징)
│   ├── catboost_model.py (튜닝된 CatBoost 하이퍼파라미터, 조기종료 학습)
│   ├── blend_model.py (alpha 그리드 서치, CatBoost+MLP 블렌드 추론)
│   ├── train.py (트랙맨 피처 병합 + 두 모델 학습 + 블렌드 번들 생성)
│   ├── test.py (season==2024 검증 데이터로 블렌드 BSS 점수 산출 및 reference 모델과 비교)
│   ├── tune.py (CatBoost Optuna 하이퍼파라미터 탐색, 파이프라인 미포함)
│   └── train_x30.py / test_x30.py, train_rd30.py / test_rd30.py, train.last.py
│       (다른 분할 전략 프로토타입 — 아직 CatBoost 단독 기반, MLP/블렌드로 미이관, dopip.py 메인 경로 아님)
└── dopip.py (전체 파이프라인 제어 오케스트레이터 및 리트레인 수행 드라이버)

```

### `dopip.py` 실행 흐름

1. `code/train.py`를 서브프로세스로 실행 — 7-seed MLP 앙상블과 CatBoost를 동일한 train/val(season==2024 홀드아웃) 분할로 학습하고, 검증 예측으로 alpha를 스윕해 `open/temp/latest_model.pkl`(블렌드 번들)을 생성합니다.
2. `code/test.py`를 서브프로세스로 실행 — 동일한 검증 분할을 재구성해 `latest_model.pkl`의 블렌드 예측을 `open/reference/best_model.pkl`과 비교하고, `open/temp/compare_result.txt`에 `NEW_BEST`/`KEEP_REF`를 기록합니다. reference가 구버전(CatBoost 단독/MLP 단독) 포맷이면 자동으로 "참조 없음"으로 간주하고 새 모델을 승격시킵니다.
3. 비교 결과에 따라 이전 latest/reference 모델을 `open/former_model/`로 백업하고, `NEW_BEST`면 `latest_model.pkl`을 `open/reference/best_model.pkl`로 승격합니다.
4. 결과와 무관하게 최신 `open/reference/best_model.pkl`을 다시 불러와, 각 MLP seed의 `best_epoch`(+5 버퍼), CatBoost의 `catboost_best_iteration`(+50 버퍼), reference의 `alpha`를 그대로 재사용해 **전체 데이터(2019~2024 전부)로 재학습**하고 `submit/model/final_retained_model.pkl`을 생성합니다.

## 세팅 및 다운

#### 0. ubuntu 리눅스에서

(Windows면)

```bash

wsl -d ubuntu

cd ~/

```

#### 1. 원격 Private 레포지토리 코드를 내 컴퓨터로 복제 
```
git clone https://github.com/kau-newbie/lgaimers9
```

#### 2. 복제된 프로젝트 폴더 내부로 이동
```
cd lgaimers9
```
#### 3. venv 환경 세팅

[여기](https://tropical-boa-e17.notion.site/3b526ae03b6380709318d6a9c98a33d6?source=copy_link)

## <주의사항>

- (v2) CatBoost에서 PyTorch 기반 Tabular MLP(+ 7-seed 앙상블)로 모델을 교체했습니다. 처음엔 코랩에서 간단한 피처셋으로 1,000점대가 나와서 전환했으나, 그 1,000점은 코랩 코드의 무작위 20% 분할 폴백(의도한 `season==2024` 홀드아웃이 아님)으로 인한 선수-정체성 리크로 밝혀져 무효였습니다. 실제 파이프라인(트랙맨 병합 + 파생 피처 + `season==2024` 홀드아웃)으로 재검증한 실측 점수는 **CatBoost 818.54 vs. MLP 7-seed 앙상블 789.58**로, 로컬 검증에서는 MLP가 CatBoost를 넘지 못했습니다.

- (v3) 그럼에도 두 모델의 예측을 가중평균으로 **블렌딩**했더니 로컬 검증 851.19로 CatBoost/MLP 단독보다 모두 높게 나왔고(두 모델 예측 상관관계가 0.9991로 매우 높은데도), 실제 대회 리더보드 제출에서는 CatBoost가 MLP보다 낮았던 로컬 순위와 반대로 **MLP 쪽이 실측에서 더 높게** 나오는 등 로컬 검증과 실측 리더보드 순위가 어긋나는 현상이 반복 관찰되었습니다. 최종적으로 **CatBoost+MLP 블렌드(`code/blend_model.py`)가 실제 대회 대시보드에서 924점**을 기록해 현재 프로덕션 모델로 채택되어 있습니다. 자세한 튜닝/블렌딩 과정은 `PROJECT_HISTORY.md`(특히 4장, 6장) 및 `EXPERIMENTS.md` 참고. 수료 기준(549.51)은 여유 있게 통과.

- open/data/ 아래 직접 데이터 다운받아서 넣어주셔야 합니다. (아래 대회 데이터다운링크)
> https://dacon.io/competitions/official/236743/data

- `open/reference/best_model.pkl`에 예전 CatBoost 단독 모델이나 MLP 단독 번들이 남아있어도 문제 없습니다. `code/test.py`가 `"catboost_model"`/`"mlp_bundle"` 키가 둘 다 없는 구버전 포맷이면 자동으로 "참조 없음"으로 간주하고, `dopip.py`를 한 번 돌리면 새 블렌드 모델로 자동 교체되면서 예전 파일은 `open/former_model/`로 백업됩니다.

- MLP 모델 구조/하이퍼파라미터는 `code/mlp_model.py`에 있습니다 (`TabularMLP` 클래스, epoch/batch size/lr 등). CatBoost 하이퍼파라미터는 `code/catboost_model.py`의 `CATBOOST_PARAMS`에 하드코딩되어 있습니다. `submit/script.py`는 `code/`를 import하지 않는 독립 실행 파일이라 `TabularMLP` 클래스, 전처리 로직, 블렌드 추론 로직이 그대로 복제되어 있습니다 — `mlp_model.py`, `catboost_model.py`, `blend_model.py`, `train.py`를 고치면 `submit/script.py`도 손으로 맞춰줘야 합니다. (단, `CatBoostClassifier`는 라이브러리 객체라 그대로 unpickle되므로 CatBoost 학습/클래스 코드까지 복제할 필요는 없습니다.)

- epoch 수(`code/mlp_model.py`의 `MAX_EPOCHS`, `PATIENCE`)가 하드웨어따라 너무 버거울 수도 있습니다. 직접 알맞게 줄이시면 되겠습니다.

- `submit/requirements.txt`에는 `catboost==1.2.10`만 명시되어 있습니다. torch/pandas/numpy/scikit-learn은 대회 서버에 기본 설치되어 있어 버전 충돌을 피하려 일부러 비워둔 것이니, 새 의존성을 추가할 때 이 관례를 참고하세요.

## <코드 설명>

`dopip.py`로 실행하면 알아서 훈련(CatBoost+MLP 동시 학습 및 블렌드), 테스트(reference 블렌드 모델과의 1:1비교), 전체 데이터로 재훈련 후

submit 제출파일 안 model에 블렌드 .pkl 모델 파일을 넣어줍니다.


`code/` 폴더 아래 `train.py`와 `test.py`가 메인 파이프라인(`dopip.py`)이 실제로 사용하는 학습/테스트 코드입니다.
- `train.py` : 트랙맨 피처 병합 + `mlp_model.py`(MLP 7-seed 앙상블) + `catboost_model.py`(CatBoost) 학습 + `blend_model.py`(alpha 스윕)로 블렌드 번들 생성
- `test.py` : season==2024 검증 데이터로 블렌드 예측의 BSS 점수를 산출해 reference 모델과 비교
- `mlp_model.py` / `catboost_model.py` / `blend_model.py` : 각 모델 정의·학습·블렌드 로직을 담은 모듈, `train.py`/`test.py`/`dopip.py`에서 import해서 사용
- `tune.py` : CatBoost 하이퍼파라미터 Optuna 탐색 스크립트 (파이프라인 미포함, 실험용)
- 그 외 코드 파일들 (아직 CatBoost 단독 기반, MLP/블렌드로 미이관 — `dopip.py` 메인 경로 아님)
	- train/test_x30 : 24년도의 제일 마지막 경기부터 30%(24년도의)를 잘라 테스트 데이터로 쓴 파일들입니다.
	- train/test_rd30 : 19-24년 전체 데이터 중 30%를 랜덤하게 잘라 테스트 데이터로 쓴 파일들입니다.

자세한(?) 설명은 : [여기](https://tropical-boa-e17.notion.site/3b526ae03b6380709318d6a9c98a33d6?source=copy_link)
