# Trackman 활용 전략 v4: regime·reliability·calibration 계획

작성일: 2026-08-22  
상태: Phase 7 완료, Trackman 제출 피처 보류  
기준 제출 후보: `0822(std21)`  
기준 모델: 0821_2 고정 레시피

## 1. v4의 핵심 방향

v1~v3에서 과거 투수별 Trackman 평균·표준편차·MAD·구종 geometry·상황별 profile·시즌 변화량을 기존 모델에 직접 추가했지만 안정적인 개선을 확인하지 못했다.

v4에서는 Trackman을 투수의 고정된 제구 능력값으로 취급하지 않는다. 먼저 다음 세 가지 가능성을 분리한다.

1. Trackman 물리값이 기존 `pitcher_id`·`asof_*`와 독립적인 투구 단위 정보를 제공하는가
2. Trackman이 실제 능력보다 연도·game_type·측정 환경의 분포 이동을 나타내는가
3. Trackman 매칭 표본 수·최근성·매칭 등급이 baseline 예측을 얼마나 신뢰할 수 있는지를 알려주는가

새 raw Trackman 피처를 대량으로 추가하는 것은 v4의 기본 전략이 아니다. 독립 신호가 확인된 경우에도 우선 baseline의 calibration 또는 shrinkage만 제한적으로 조정한다.

## 2. 실패 부검에서 고정한 사실

### 2.1 target regime가 크게 변한다

Train의 `control_success` 비율은 다음과 같다.

| 연도 | Futures | Regular |
|---|---:|---:|
| 2022 | 0.7087 | 0.5037 |
| 2023 | 0.4729 | 0.5031 |
| 2024 | 0.4593 | 0.4897 |

특히 Futures는 2022년에서 2023년 사이에 약 23.6%p 하락했다. 물리 프로필이 이 변화를 설명한다고 가정하지 않는다. v4는 모든 결과를 연도×game_type별로 분리해 확인한다.

### 2.2 기존 baseline은 이미 투수·상황 정보를 많이 사용한다

0821_2의 63개 피처에는 `pitcher_id`, `batter_id`, 투수·타자 이력, 최근 경기, 구종 비율, game_type, 경기 상황이 포함된다. 따라서 Trackman profile의 추가 효과는 전체 평균 성능이 아니라 baseline 잔차에 대한 작은 독립 신호여야 한다.

### 2.3 현재 반복성 profile은 투수×구종군의 과거 pooled summary다

`build_repeatability_features.py`의 기본 profile은 여러 과거 시즌을 합치며, 기본 반복성 피처는 game_type·월·현재 시즌 상태를 직접 분리하지 않는다. 76개 또는 21개 컬럼을 독립적인 76개·21개 신호로 해석하지 않는다.

### 2.4 최종 투구 매칭과 profile 매핑은 다르다

최종 투구 ID 연결, 선수 ID 매핑, 반복성 profile 사용 가능 행은 서로 다른 coverage다. 각 실험은 다음을 별도로 기록한다.

- 최종 `trackman_id` 행 매칭 여부
- 선수 ID 매핑 방식과 등급
- 과거 Trackman profile의 표본 수·최근성
- 실제 사용할 피처의 결측·부분 결측 여부

`confirmed_global_exact_fallback`을 exact strict와 동일한 신뢰도로 취급하지 않는다.

### 2.5 2024 단일 소폭 개선은 선택 편향과 구분해야 한다

`std21`의 2024 개선은 약 `6e-6`이며, paired loss 차이의 표준오차보다 작다. 2024 결과를 보고 후보·gate·offset을 고르는 방식을 금지한다.

## 3. 규칙과 정보 시점

공식 `대회 규칙.md`, `평가 데이터 독립 예측 원칙 위반 사례.md`, `data_description.md`를 우선한다.

허용:

- 현재 행에 포함된 pre-pitch 입력
- 공식 train으로 학습한 통계·모델·보정
- 공식 2019~2024 Trackman의 목표 행 시즌 이전 자료
- 목표 행의 입력값과 학습 데이터만으로 생성한 결정론적 피처

금지:

- 현재 투구의 실제 Trackman 측정값·구종·위치·결과
- 2025년 Trackman 측정값
- 외부 데이터·외부 API
- 다른 test 행을 이용한 집계·분포·순위·보정
- 평가 시점 이후 정보를 과거 행에 섞는 집계

최종 제출 후보는 test 행을 하나만 남겼을 때와 전체 test를 넣었을 때 예측값이 같아야 한다.

## 4. 고정 baseline과 검증 원칙

모든 후보는 다음 0821_2 recipe와 동일한 행·seed·시간 cutoff로 비교한다.

- 기존 63개 pre-pitch 피처
- Full 모델 + F1-filtered 모델 50:50
- 각 모델 내부 CatBoost 70% + 3층 MLP 30%
- offset `-0.01114`
- 2019~2021 → 2022
- 2019~2022 → 2023
- 2019~2023 → 2024

단, v4의 진단 단계에서는 offset이 개선을 숨기는지 확인하기 위해 raw prediction과 고정 offset prediction을 모두 저장한다. 최종 후보 비교에서 offset을 다시 최적화하지 않는다.

### 4.1 nested 선택 규칙

- Phase 0~1: 2022·2023만 사용해 가설과 설정을 정한다.
- Phase 2 이후의 고정 holdout: 2024는 설정을 바꾸지 않고 한 번 평가한다.
- 2024 target을 보고 피처, threshold, lambda, gate, calibration map을 선택하지 않는다.
- 2024에서만 좋아지는 후보는 채택하지 않는다.

### 4.2 채택 기준

다음 조건을 모두 만족해야 제출 후보로 검토한다.

- 정상 Trackman이 적절한 negative control보다 우수
- 2022·2023에서 동일한 방향의 개선
- 2024에서 설정 고정 후 같은 방향
- 전체 Brier뿐 아니라 Regular·Futures 모두에서 치명적 악화가 없음
- 개선이 calibration shift 하나에만 의존하지 않거나, calibration 방식 자체가 개발 fold에서 안정적
- 충분한 coverage와 매칭 신뢰도
- test 행 간 의존성 없음

### 4.3 중단 기준

- Trackman이 shuffled·stratified null과 구분되지 않음
- season/game_type을 통제하면 신호가 사라짐
- 특정 연도 또는 소수 매칭 행에서만 개선
- Trackman 추가로 예측 평균만 이동하고 ranking/resolution이 개선되지 않음
- 2022·2023과 2024 방향이 반전
- raw 피처를 늘려도 동일한 실패가 반복

## 5. Phase 0. 실험·피처 lineage와 coverage 감사

새 피처를 만들기 전에 어떤 Trackman 산출물이 실제 모델 입력으로 들어갔는지부터 확인한다.

### 산출물

- feature family별 생성 코드·입력 파일·cutoff 표
- final row linkage와 player profile linkage의 차이
- strict/global fallback/conflict/unmapped별 행·투수 수
- season×game_type×pitcher_hand×pitch_type_group별 profile coverage
- 표본 수, 마지막 관측 시즌, 과거 시즌 수, 부분 결측률
- train 행에서 동일 profile이 몇 번 반복되는지

### 성공 기준

후속 Phase에서 사용되는 모든 컬럼이 다음 항목을 갖는다.

```text
source file
mapping key
time cutoff
aggregation level
missing rule
confidence grade
```

lineage가 확인되지 않은 기존 피처는 새 후보에 포함하지 않는다.

## 6. Phase 1. target regime와 baseline calibration 부검

Trackman보다 먼저 baseline이 어디서 실패하는지 확인한다.

### 분석

- 연도×game_type별 target rate, mean prediction, residual, Brier
- Full/F1 각각의 calibration과 blend 후 calibration
- baseline uncertainty와 target rate의 관계
- Trackman available/missing별 baseline residual
- `std21`과 0821_2의 paired loss 차이를 연도×game_type별 분해
- raw prediction과 offset 적용 prediction의 차이

### 목적

`Trackman이 직접 제구를 설명하는가`와 `Trackman 후보가 단순히 예측 평균을 이동시키는가`를 분리한다. 후자라면 Trackman physical feature가 아니라 regime calibration 문제로 분류한다.

## 7. Phase 2. 올바른 negative control과 독립 신호 검정

v3의 전체 행 독립 shuffle 하나만으로 결론을 내리지 않는다. 다음 대조군을 같은 모델·같은 cutoff로 비교한다.

1. 정상 Trackman
2. 같은 season×game_type×pitcher_hand 층에서 player profile을 교환한 Trackman
3. 같은 season×game_type 층에서 profile을 교환한 Trackman
4. Trackman 물리값을 제거하고 표본 수·최근성·missing만 사용
5. missing·coverage·mapping grade만 사용
6. Trackman feature column의 season/game_type 표준화값만 사용

### 판정

정상 profile이 2·3번 대조군보다 우수해야 물리적 player-specific signal을 인정한다. 4·5번만 정상 profile과 비슷하면, 물리값이 아니라 selection/reliability 신호로 분류한다.

## 8. Phase 3. Trackman 분포 이동과 표준화 검증

절대적인 구속·무브먼트·표준편차를 바로 사용하지 않는다.

### 후보 변환

- season×game_type×pitcher_hand×pitch_type_group 내 percentile
- 같은 그룹 내 robust z-score `(x - median) / MAD`
- 투수 내부의 과거 시즌 percentile
- 이전 시즌과 career profile의 rank 변화
- Trackman 측정값의 season-to-season distribution shift

표준화 파라미터는 목표 시즌 이전의 학습 Trackman으로만 계산한다. 전체 test 분포로 표준화하지 않는다.

### 판정

표준화 후에도 2022·2023·2024에서 방향이 유지되는 피처가 없으면 raw physical signal은 종료한다.

## 9. Phase 4. Trackman-as-regime·reliability meta features

Trackman을 baseline에 직접 입력하지 않고, baseline을 얼마나 믿을지 결정하는 보조 변수로만 사용한다.

### 후보

- 과거 Trackman 표본 수와 유효 구종군 수
- 마지막 관측 시즌과 목표 시즌 사이의 간격
- strict/global fallback/conflict 여부
- profile의 season-to-season 안정성
- Trackman과 공식 `asof_pitcher_pitchmix`의 coverage·분포 차이
- Trackman profile이 존재하지만 공식 이력이 부족한 정도
- Full/F1 prediction gap과 Trackman reliability의 결합

### 적용

전체 행에 raw profile을 추가하지 않는다. 다음 두 가지를 별도로 검증한다.

1. baseline prediction에 작은 calibration shift만 적용
2. baseline과 league/game_type prior 사이의 shrinkage weight만 조정

gate 밖의 행과 매칭 불확실 행은 baseline을 그대로 사용한다.

## 10. Phase 5. 제한적 calibration 모델

Phase 1~4에서 Trackman이 calibration 또는 reliability와 관련된다는 근거가 있을 때만 실행한다.

### 모델

```text
p_base = 0821_2 prediction
q = calibration_model(p_base, game_type, season, reliability_features)
p_final = clip(q, 0, 1)
```

우선순위:

1. 학습 데이터 기반의 고정 game_type calibration
2. 얕은 logistic/Ridge calibration
3. Trackman reliability가 높은 구간만의 shrinkage
4. 마지막으로만 residual correction

CatBoost에 Trackman raw 피처를 다시 대량 추가하지 않는다.

### 시간 안전 학습

- 2023 calibration은 2022 이하 잔차로만 학습
- 2024 calibration은 2022·2023으로만 학습
- calibration parameter는 validation target을 보고 재선택하지 않음

## 11. Phase 6. row-level matched Trackman의 제한적 진단

최종 row-level 매칭을 직접 평가 피처로 사용하는 단계가 아니다. 현재 투구의 Trackman은 최종 추론에 사용할 수 없으므로 다음만 진단한다.

- row-level Trackman이 historical player profile보다 target과 가까운 정보를 갖는지
- current-pitch privileged teacher가 baseline의 어떤 error regime를 설명하는지
- 그 error regime가 pre-pitch의 game_type·season·asof 정보로 재현 가능한지

teacher/auxiliary/distillation 결과는 운영진 규칙 검토 전까지 제출 후보로 사용하지 않는다. 재현 가능한 pre-pitch 학생 모델로 변환되지 않으면 진단 결과로만 보관한다.

## 12. Phase 7. 최종 검증과 제출 판단

후보가 생기면 다음을 고정한다.

- 2022에서 발견한 설정을 2023에서 확인
- 2023까지의 설정으로 2024를 단 한 번 평가
- 전체·Regular·Futures Brier
- calibration slope/intercept와 mean prediction
- paired loss 차이와 bootstrap 또는 paired standard error
- coverage·mapping grade별 성능
- single-row test 실행과 전체 test 실행 동일성
- submission ZIP 구조·추론 시간·확률 범위

최종 조건을 만족하지 못하면 `0822(std21)`을 유지하고 Trackman을 제출에 사용하지 않는다.

## 13. 권장 실행 순서

1. Phase 0: feature lineage·coverage 감사
2. Phase 1: target regime·baseline calibration 부검
3. Phase 2: 층화 negative control
4. Phase 3: season/game_type 표준화 signal 검정
5. Phase 4: reliability·regime meta feature 검정
6. Phase 5: 제한적 calibration/shrinkage
7. Phase 6: row-level teacher는 진단으로만 확인
8. Phase 7: nested rolling·제출 검증

Phase 0~3에서 물리적 player-specific signal이 없으면 Phase 4만 제한적으로 진행한다. Phase 4에서도 안정적인 calibration/reliability 신호가 없으면 Trackman은 최종적으로 폐기한다.

이번 실행에서는 Phase 0~7을 모두 완료했다. Phase 2의 층화 null, Phase 3의 표준화, Phase 4의 reliability metadata, Phase 5의 shrinkage 모두 2023에서 안정적인 개선을 만들지 못했다. Phase 6에서 current-pitch teacher만 matched 행에서 개선했지만 pre-pitch student가 재현하지 못했으므로, Trackman은 진단 자료로 보존하고 제출 모델에는 넣지 않는다.

## 14. 산출물

원본 실험 코드는 `code/`, Trackman 생성 코드는 `tools/trackman/`, 결과는 다음 위치에 저장한다.

```text
open/experiments/trackman_strategy_v4/
```

각 Phase는 다음을 남긴다.

- `config.json`
- feature lineage와 cutoff
- coverage·mapping grade 표
- 연도×game_type별 Brier·calibration
- 정상·negative control 비교
- paired loss 차이와 불확실성
- `report.md`

`0822(std21).zip`, `submit/`, `open/champion/`, `reference`는 후보가 최종 채택되기 전 변경하지 않는다.

## 15. 최종 성공 기준

v4의 성공은 Trackman 피처 수가 늘어나는 것이 아니다.

다음 중 하나를 시간 안전하게 입증하면 성공으로 본다.

1. 표준화된 Trackman profile이 층화 permutation보다 안정적으로 우수하다.
2. Trackman reliability가 baseline calibration error를 2022·2023·2024에서 같은 방향으로 줄인다.
3. Trackman을 사용하지 않을 때보다 제한적 calibration/shrinkage가 최종 holdout에서 개선된다.

셋 모두 입증하지 못하면 다음 결론으로 종료한다.

> 이 대회의 pre-pitch 정보와 현재 target regime에서는 historical Trackman이 독립적인 제출 신호가 아니다. Trackman은 진단 자료로만 보존하고, 제출 모델은 공식 pre-pitch baseline을 사용한다.

## 16. 진행 상태

- [x] Phase 0: 실험·피처 lineage와 coverage 감사
  - Train `1,475,092`행, Trackman history `1,793,078`행 확인
  - 선수 ID 기반 profile 매핑 가능 행은 `1,464,612`행(`99.29%`)
  - strict 매핑 `1,424,897`행, global fallback `39,715`행, unmapped/other `10,480`행
  - 최종 투구 행 `trackman_id` 연결은 `1,275,274`행(`86.45%`)이지만, 기존 repeatability·context·geometry·season_change 4개 family는 모두 최종 row linkage를 직접 사용하지 않고 player profile lookup으로 생성됨
  - 4개 feature family는 각각 76·28·23·51개 피처와 Train 전체 1,475,092행을 산출했으며, 공통 cutoff는 목표 시즌보다 이전 시즌임
  - 2024 profile 전체 구종군 coverage는 Futures 좌투수 `51.35%`, Futures 우투수 `82.31%`, Regular 좌투수 `73.99%`, Regular 우투수 `83.86%`
  - 2024 Futures 좌투수에서는 profile partial 행이 `989`행으로, 단순 available/missing flag보다 구종군별 부분 결측을 분리해야 함
  - 2019년은 이전 시즌 Trackman이 없어 profile coverage가 0이며, 2020년 이후에도 game_type·pitcher_hand별 coverage 편차가 큼
  - 결론: 지금까지 Trackman 실패는 raw 물리값 자체뿐 아니라 profile coverage·mapping grade·최종 row linkage를 하나의 available flag로 뭉친 영향일 수 있음. Phase 1은 target regime와 baseline calibration을 먼저 분해하고, 이후 strict/global/partial을 분리해 검증한다.
  - 상세 결과: `open/experiments/trackman_strategy_v4/phase0_lineage_20260822/`
- [x] Phase 1: target regime·baseline calibration 부검
  - 0821_2 exact baseline과 `std21`을 raw/fixed offset으로 분리해 비교
  - 2022 Futures에서 `std21` Brier `+0.002650` 악화, 2023 Futures에서 `+0.001203` 악화, 2024 Futures에서만 `-0.000045` 개선
  - 2024 Regular는 `-0.000001` 수준으로 사실상 변화 없음
  - 전체 2024는 고정 offset에서만 `-0.000006` 개선되고 raw prediction에서는 `+0.000004` 악화
  - 후보의 예측 평균 이동 방향도 Futures에서 연도별로 반전: 2022 `-0.0147`, 2023 `+0.0083`, 2024 `-0.0001`
  - strict mapping보다 global fallback에서 악화폭이 큰 경향이 있으나, fallback 표본 자체의 selection 효과와 분리되지 않음
  - profile complete/missing/partial 세그먼트에서도 연도별 방향이 안정적이지 않음
  - 결론: 2024 소폭 개선은 물리적 Trackman 독립 신호보다 연도×game_type calibration 상호작용에 가깝다. Phase 2에서는 player-specific 물리값과 mapping/profile selection 효과를 층화 permutation으로 분리한다.
  - 상세 결과: `open/experiments/trackman_strategy_v4/phase1_regime_audit_20260822/`
- [x] Phase 2: 층화 negative control
  - `season×game_type×pitcher_hand` 또는 `season×game_type` 내부에서 profile 벡터 전체를 교환해 physical player identity를 제거
  - 2023 residual correction 기준 정상 physical std21 delta `+0.000089`, 층화 null delta `+0.000105~+0.000106`으로 모두 악화
  - 2024 정상 physical std21 delta `-0.000002`, `season×game_type×pitcher_hand` null `-0.000003`, `season×game_type` null `-0.0000005`로 null이 정상 profile보다 나쁘지 않음
  - reliability-only와 physical+reliability도 정상 physical profile과 거의 같은 결과
  - 결론: 정상 Trackman profile이 player-specific 물리 signal로 null을 안정적으로 이기지 못함. 2024 소폭 개선은 selection/regime/calibration 효과일 가능성이 더 높으며, Phase 3에서는 연도·game_type 표준화 후에도 남는 신호가 있는지만 확인
  - 상세 결과: `open/experiments/trackman_strategy_v4/phase2_stratified_null_20260822/`
- [x] Phase 3: season/game_type 표준화 signal 검정
  - 목표 시즌 이전 자료만 사용해 `game_type×pitcher_hand`별 robust median/MAD z-score를 계산하고, 원시 std21 residual correction과 비교
  - 2023: 표준화 후보 전체 delta `+0.000090`, raw std21 reference `+0.000089`로 둘 다 악화
  - 2024: 표준화 후보 전체 delta `-0.0000012`, raw std21 reference `-0.0000018`로 사실상 동일하며 표준화가 더 낫지 않음
  - 2023 Futures는 표준화·원시 모두 약 `+0.00136` 악화했고, Regular만 약 `-0.00006` 개선
  - 결론: 시즌·게임타입·투수손별 robust 표준화로도 독립적인 physical signal이 나타나지 않음. raw 값의 단위나 분포 이동만의 문제라는 가설은 지지되지 않으며, Phase 4에서는 물리값이 아닌 reliability·regime meta feature만 제한적으로 검정한다.
  - 상세 결과: `open/experiments/trackman_strategy_v4/phase3_standardized_signal_20260822/`
- [x] Phase 4: reliability·regime meta feature 검정
  - raw 물리값을 제외하고 매핑 등급, profile complete/partial, 구종군별 표본 수, 전체 표본 수, profile 최근성·연속 시즌 수만 사용
  - 2023 전체 delta: `regime_only +0.000104`, `mapping_only +0.000096`, `profile_only +0.000108`, `history_reliability +0.000103`으로 모두 악화
  - 2024 전체 delta: 네 변형 모두 약 `-0.000001` 수준으로, 2023 개선 없이 사실상 동일
  - 2024 Futures에서 `history_reliability`가 `-0.000004`였지만 paired SE가 `0.000019`로 크고 2023에서 악화되어 채택 근거가 아님
  - 결론: Trackman의 profile 신뢰도·매핑 메타정보도 baseline 잔차를 시간 안정적으로 설명하지 못함. Phase 5는 Trackman 효과를 주장하기 위한 단계가 아니라, 제한적 shrinkage/calibration이 개발 연도에서 재현되는지 확인하는 마지막 진단으로 진행한다.
  - 상세 결과: `open/experiments/trackman_strategy_v4/phase4_reliability_meta_20260822/`
- [x] Phase 5: 제한적 calibration/shrinkage
  - source 연도의 game_type target prior를 baseline과 5%·10%·20%로 shrinkage하고, strict+complete+profile 표본 500개 이상 gate에서도 같은 강도를 적용
  - 2023 전체는 전체 shrinkage `+0.000116~+0.000611`, 고신뢰 gate `+0.000044~+0.000246`으로 모든 변형 악화
  - 2024 전체도 전체 shrinkage `+0.000021~+0.000215`, 고신뢰 gate `+0.000008~+0.000106`으로 모두 악화
  - 결론: Trackman reliability로 baseline과 source-year prior 사이의 가중치를 조절하는 방식도 검증연도에 재현되지 않음. Phase 6은 제출 후보를 만들지 않고 row-level Trackman을 privileged teacher로만 진단한다.
  - 상세 결과: `open/experiments/trackman_strategy_v4/phase5_limited_calibration_20260822/`
- [x] Phase 6: row-level teacher 진단
  - 정확 경기 수열 매칭 행만 사용한 privileged teacher의 검증 coverage는 2022 `44.97%`, 2023 `56.25%`, 2024 `57.51%`
  - matched 행 전체에서 teacher Brier는 2022 `0.247469→0.246154`(`-0.001315`), 2023 `0.248571→0.248173`(`-0.000398`), 2024 `0.247696→0.247138`(`-0.000558`)
  - 다만 Futures teacher 효과는 2023 `+0.012646` 악화로 방향이 크게 반전했고, teacher는 현재 투구의 Trackman을 직접 본 privileged 정보임
  - teacher 보정량을 pre-pitch 입력만으로 Ridge가 복원한 결과는 2023 baseline 대비 `+0.000388`, 2024 `+0.000615` 악화. teacher 보정량 상관도 2023 `0.384`, 2024 `0.246`에 그침
  - 결론: 현재 투구 Trackman에는 matched 행에서 유용한 정보가 있지만, 그 효과는 pre-pitch 입력만으로 시간 안정적으로 복원되지 않음. 이는 제출용 Trackman feature가 아니라 “평가 시점에 관측할 수 없는 privileged signal”이라는 진단이다.
  - 상세 결과: `open/experiments/trackman_strategy_v4/phase6_teacher_diagnostic_20260822/`
- [x] Phase 7: nested rolling·제출 검증
  - 고정된 `0822(std21).zip`의 구성·필수 모델 파일·`std_lookup.csv`를 확인하고, 검증 과정에서 ZIP을 수정하지 않음
  - 로컬 test 5행 full 실행과 `TEST_000001` 단일행 실행에서 해당 예측값이 완전히 일치(`abs diff=0.0`)
  - full 출력 행 수 `5`, row_id 순서 일치, 확률 범위 `0.41214079~0.53796450`
  - 제출 스크립트의 `test_df.groupby`, `rolling`, `shift`, `expanding` 기반 평가행 간 의존성 정적 검사 모두 미검출
  - 결론: 제출 패키지의 행 독립성·출력 형식은 통과했지만, Trackman 신규 후보의 성능 채택 근거는 없음. 기존 `0822(std21)`은 그대로 보존하고 v4 Trackman 피처는 제출에 추가하지 않는다.
  - 상세 결과: `open/experiments/trackman_strategy_v4/phase7_final_verification_20260822/`

## 17. 현재 가설 우선순위

| 우선순위 | 가설 | 예상 가능성 | 검증 방식 |
|---:|---|---|---|
| 1 | Trackman은 physical signal보다 regime/reliability signal이다 | 높음 | Phase 1·4 |
| 2 | season·game_type 표준화 전에는 측정·집단 효과가 섞인다 | 중간 | Phase 3 |
| 3 | 매핑 grade와 profile 표본 수가 raw 값보다 유용하다 | 중간 | Phase 0·4 |
| 4 | row-level teacher가 pre-pitch residual regime를 알려준다 | 낮음~중간 | Phase 6 진단 |
| 5 | 새로운 raw Trackman 피처가 직접 Brier를 개선한다 | 낮음 | 추가 raw sweep 금지 |
