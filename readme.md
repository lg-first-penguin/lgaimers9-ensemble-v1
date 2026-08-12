우리팀용 비공개 파이프라인입니다. (v1)

ubuntu 24.04.4 LTS를 Windows WSL 기능을 사용해 만들었습니다.

환경세팅(패키지나 라이브러리)은 대회 서버 기본 내장 패키지로 맞추었습니다.

## environments

```

> wsl -l -v
  NAME              STATE           VERSION
* Ubuntu            Running         2
  docker-desktop    Stopped         2

> wsl -d Ubuntu -e lsb_release -a
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

```

lgaimers9 (루트 폴더)
├── submit/
│   ├── model/
│   │   └── final_retained_model.pkl (dopip.py 실행 시 전체 데이터 재학습 후 자동 생성)
│   ├── script.py (대회 서버 전용 추론 파일)
│   └── requirements.txt (필수 의존성 명시)
├── open/
│   ├── data/ (참가자 데이터 폴더 - train.csv, test.csv, trackman_history.csv) 
│   ├── former_model/ (이전 temp 및 reference 후보 저장 버퍼)
│   ├── reference/ (현재 최고 스코어를 기록한 기준 모델 저장)
│   └── temp/ (방금 학습을 마친 따끈따끈한 최신 모델 저장)
├── code/
│   ├── train.py (모델 학습 및 피처 엔지니어링 수행)
│   └── test.py (2024 검증 데이터를 통한 BSS 점수 산출 및 최고 모델 비교)
└── dopip.py (전체 파이프라인 제어 오케스트레이터 및 리트레인 수행 드라이버)

```

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

- (v2) CatBoost에서 PyTorch 기반 Tabular MLP(+ 7-seed 앙상블)로 모델을 교체했습니다. 처음엔 코랩에서 간단한 피처셋으로 1,000점대가 나와서 전환했으나, 그 1,000점은 코랩 코드의 무작위 20% 분할 폴백(의도한 `season==2024` 홀드아웃이 아님)으로 인한 선수-정체성 리크로 밝혀져 무효였습니다. 실제 파이프라인(트랙맨 병합 + 파생 피처 + `season==2024` 홀드아웃)으로 재검증한 실측 점수는 **CatBoost 818.54 vs. MLP 7-seed 앙상블 789.58**로, MLP가 CatBoost를 아직 넘지 못했습니다. 자세한 튜닝 과정(임베딩 실패 사례, 앙상블 도입 등)은 `EXPERIMENTS.md` 6장 참고. 그럼에도 MLP 전환의 이점(속도/구조 단순함)을 유지하기로 팀에서 결정해 MLP를 계속 사용 중입니다 — 수료 기준(549.51)은 여유 있게 통과.

- open/data/ 아래 직접 데이터 다운받아서 넣어주셔야 합니다. (아래 대회 데이터다운링크)
> https://dacon.io/competitions/official/236743/data

- `open/reference/best_model.pkl`에 예전 CatBoost 모델이 남아있어도 문제 없습니다. `code/test.py`가 포맷이 다르면 자동으로 무시하고, `dopip.py`를 한 번 돌리면 새 MLP 모델로 자동 교체되면서 예전 파일은 `open/former_model/`로 백업됩니다.

- 모델 구조/하이퍼파라미터는 `code/mlp_model.py`에 있습니다 (`TabularMLP` 클래스, epoch/batch size/lr 등). `submit/script.py`는 `code/`를 import하지 않는 독립 실행 파일이라 `TabularMLP` 클래스와 전처리 로직이 그대로 복제되어 있습니다 — `mlp_model.py`를 고치면 `submit/script.py`도 손으로 맞춰줘야 합니다.

- epoch 수(`code/mlp_model.py`의 `MAX_EPOCHS`, `PATIENCE`)가 하드웨어따라 너무 버거울 수도 있습니다. 직접 알맞게 줄이시면 되겠습니다.

## <코드 설명>

`dopip.py`로 실행하면 알아서 훈련, 테스트(reference model과의 1:1비교), 전체 데이터로 재훈련 후 

submit 제출파일 안 model에 .pkl 모델 파일을 넣어줍니다.


`code/` 폴더 아래 `train.py`와 `test.py`를 직접 작성하시면 되겠습니다.
- 각각 훈련 코드와 테스트 코드 입니다.
- 그 외 코드 파일들
	- train/test_x30 : 24년도의 제일 마지막 경기부터 30%(24년도의)를 잘라 테스트 데이터로 쓴 파일들입니다.
	- train/test_rd30 : 19-24년 전체 데이터 중 30%를 랜덤하게 잘라 테스트 데이터로 쓴 파일들입니다.

자세한(?) 설명은 : [여기](https://tropical-boa-e17.notion.site/3b526ae03b6380709318d6a9c98a33d6?source=copy_link)
