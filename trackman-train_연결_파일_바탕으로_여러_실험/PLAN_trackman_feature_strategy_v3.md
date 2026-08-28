# Trackman 활용 전략 v3 실험 계획

작성일: 2026-08-22  
상태: Phase 0~3 신호 감사 완료, 후속 Trackman 피처 개발 중단  
기준 제출 후보: `0822(std21)`  
기준 모델: 0821_2 고정 레시피

## 1. 목적

Trackman 전략 v1·v2에서 반복성, 상황별 예상 구종, anomaly, residual, 시즌 변화량을 검증했지만 2024를 포함한 rolling fold에서 안정적인 개선을 확인하지 못했다.

v3의 목표는 새로운 Trackman 피처를 계속 추가하는 것이 아니다. 먼저 Trackman이 기존 `asof_*` 피처와 독립적인 정보를 제공하는지 확인하고, 실제 신호가 확인된 경우에만 다음 세 가지 방식으로 제한적으로 사용한다.

1. 기존 투수 통계의 신뢰도와 shrinkage 조정
2. Trackman 프로필과 공식 pre-pitch 이력의 불일치 표현
3. cold-start·고불확실성 행에 한정한 보정

기존 제출 후보 `0822(std21)`은 실험 중 수정하지 않는다.

## 2. v1·v2 실패에 대한 핵심 가설

### 2.1 투구 단위 target과 투수 프로필의 시간·단위 불일치

`control_success`는 투구 단위 target이지만 Trackman 피처 대부분은 `투수 × 구종군 × 과거 시즌`의 고정 프로필이다. 같은 투수의 여러 행에 같은 값이 반복되어 카운트, 타자손, 최근 상태, 경기 상황에 따른 투구별 변동을 설명하기 어렵다.

### 2.2 기존 공식 피처와의 중복

기존 모델은 투수·타자의 누적 이력, 최근 경기, 구종 사용률, 상황 정보를 이미 사용한다. Trackman 평균 구속·구종 비율·장기 profile은 새로운 정보가 아니라 `pitcher_id`와 `asof_*`로 이미 간접 표현된 정보를 반복할 가능성이 높다.

### 2.3 Trackman coverage와 표본이 무작위가 아닐 가능성

Trackman 연결 여부, 과거 투구 수, 구종군 coverage, 마지막 관측 시즌은 선수·시즌·game_type에 따라 다르다. 모델이 물리적 특성 대신 “Trackman이 잘 연결되는 행”의 선택 효과를 학습했을 수 있다.

### 2.4 물리적 불안정성과 제구 실패의 관계가 단순하지 않음

릴리스·구속·무브먼트 변동성이 크다고 항상 제구 실패가 되는 것은 아니다. 의도된 구종 설계, 타자 대응, 포수·상황 효과가 함께 작용하므로 `불안정성 → 실패`라는 단일 방향 가정이 틀릴 수 있다.

### 2.5 강한 baseline에 약한 신호를 직접 주입

0821_2는 이미 강한 모델이므로 약한 Trackman 신호를 Full/F1 CatBoost와 MLP에 동시에 넣으면 신호보다 결측·측정 오차가 커질 수 있다. Trackman 후보가 2022·2023에서 좋아도 2024에서 반복적으로 반전된 것은 이 문제와 시간 분포 변화가 결합된 결과일 가능성이 있다.

## 3. 규칙·정보 시점 원칙

공식 원문인 `대회 규칙.md`, `평가 데이터 독립 예측 원칙 위반 사례.md`, `data_description.md`를 우선한다.

허용:

- 현재 평가행에 포함된 pre-pitch 입력
- 공식 `train.csv`로 계산한 통계·모델·파생변수
- 공식 2019~2024 Trackman 과거 로그
- 목표 행의 시즌보다 이전 Trackman만 이용한 요약

금지:

- 현재 투구의 실제 구종·위치·판정·결과·Trackman 측정값
- 2025년 Trackman 측정값
- 외부 데이터·외부 API
- 다른 test 행을 이용한 집계·rolling·분포·target encoding·사후 보정
- 평가 시점 이후 정보를 과거 행에 섞는 집계

모든 제출 추론은 test 행을 한 개만 남겨 실행해도 전체 test와 같은 결과가 나와야 한다.

## 4. 고정 기준과 검증 규칙

모든 후보는 다음 0821_2 기준과 동일한 행·seed·시간 분할로 비교한다.

- 기존 63개 pre-pitch 피처
- Full 모델 + F1-filtered 모델 50:50
- 각 모델 내부 CatBoost 70% + 3층 MLP 30%
- 최종 offset `-0.01114`
- 기본 rolling:
  - 2019~2021 → 2022
  - 2019~2022 → 2023
  - 2019~2023 → 2024

2024만 보고 피처·gate·lambda를 선택하지 않는다. 2022·2023은 개발 fold, 2024는 최종 holdout으로 취급한다.

### 채택 기준

- 2022·2023 개발 fold에서 같은 방향의 개선
- 개발 fold에서 정한 설정을 바꾸지 않고 2024도 개선
- 전체 Brier 악화 없이 coverage가 충분한 구간 개선
- 평균 개선폭 최소 `2e-5` 또는 calibration·segment 근거가 일관됨
- 공식 train과 pre-pitch 정보만 사용
- test 행 간 의존성 없음

### 보류·기각 기준

- 2024에서만 개선
- 2022·2023과 2024의 개선 방향 반전
- permutation 대조군과 성능이 비슷함
- Trackman 결측 여부만으로 성능이 좋아짐
- 소수 매칭 행에서만 큰 개선
- 기존 `asof_*`와 거의 같은 정보만 표현
- gate·lambda를 2024 target을 보고 선택해야 함

## 5. Phase 0. 데이터와 신호 감사

새 피처를 만들기 전에 Trackman 자체가 기존 모델에 독립적인 정보를 제공하는지 확인한다.

### 산출물

- 시즌·game_type·투수손·구종군별 coverage
- 투수별 Trackman 투구 수·경기 수·마지막 관측 시즌
- 매칭 여부와 `asof_pitcher_n`의 관계
- Trackman available/missing별 target rate와 baseline Brier
- 표본 수 구간별 baseline residual
- 매칭 품질과 2022·2023·2024 분포 비교

### 핵심 판단

Trackman coverage가 특정 시즌·game_type·투수 유형에 치우쳐 있고, available flag만으로 baseline residual이 설명된다면 raw 물리 피처 개발보다 selection·reliability 처리부터 수행한다.

## 6. Phase 1. Permutation·negative control 검증

Trackman 피처가 실제로 pitcher-level 정보를 갖는지 확인한다. 모델 구조는 간단한 CatBoost와 Ridge를 사용해 빠르게 screening한다.

### 대조군

1. 정상 Trackman 매핑
2. 같은 pitcher 내부 무작위 행 교환
3. 같은 시즌 전체 무작위 행 교환
4. game_type·pitcher_hand를 보존한 층화 무작위 교환
5. Trackman missing flag만 사용
6. Trackman 표본 수·coverage만 사용

정상 Trackman과 무작위 대조군의 residual 예측력이 비슷하면 현재 physical profile에는 실질적인 독립 신호가 약하다고 판단한다.

## 7. Phase 2. Trackman-only와 incremental signal 분해

기존 63개 피처를 제거하고 Trackman 계열만으로 예측해 Trackman 단독 신호를 확인한다.

비교 모델:

- Trackman raw/summary only
- Trackman reliability only
- 기존 63개 only
- 기존 63개 + Trackman

목표는 높은 점수 자체가 아니라 다음을 분리하는 것이다.

- Trackman 자체에 target signal이 있는가
- 기존 모델과 중복되는가
- 특정 연도에만 signal이 나타나는가
- Trackman이 probability보다 residual 또는 uncertainty를 설명하는가

## 8. Phase 3. Residual 방향 안정성 분석

모델을 복잡하게 만들기 전에 각 fold에서 Trackman 구간별 residual을 계산한다.

```text
r = control_success - p_base
```

확인할 구간:

- Trackman 표본 수
- 구종군 coverage
- anomaly 수준
- Trackman available/missing
- `abs(Full - F1)` baseline uncertainty
- `asof_pitcher_n`과 Trackman 표본의 불일치

각 구간에서 다음을 기록한다.

- 평균 residual
- residual 표준편차
- 행 수·coverage
- 2022·2023·2024 residual 방향

동일 구간의 residual 방향이 fold마다 바뀌면 해당 구간에는 고정 보정을 적용하지 않는다.

## 9. Phase 4. Trackman 기반 reliability·shrinkage

Trackman을 제구 확률의 직접 피처가 아니라 공식 `asof_*` 통계의 신뢰도 조정값으로 사용한다.

### 후보 reliability

- Trackman 투구 수
- Trackman 경기 수
- 구종군별 표본 coverage
- 마지막 관측 시즌 경과
- 매칭 신뢰도
- 기존 `asof_pitcher_n`

### 후보 shrinkage

```text
effective_pitcher_rate
  = reliability × asof_pitcher_rate
  + (1 - reliability) × league_or_hand_prior
```

대상은 우선 `asof_pitcher_n`이 작은 cold-start 행으로 한정한다. 전체 행에 직접 적용하지 않고, reliability가 낮은 행은 원래 `asof_*`와 baseline을 보존한다.

검증할 값:

- 원래 `asof_pitcher_success_rate`
- shrinkage rate
- shrinkage rate와 기존 rate의 차이
- cold-start gate별 Brier

## 10. Phase 5. Trackman–asof 불일치 피처

Trackman 물리값 자체보다 공식 pre-pitch 이력과 Trackman profile이 서로 얼마나 다른지를 사용한다.

### 후보

- 공식 구종 비율 − Trackman 구종 비율
- 공식 투수 표본 수 − Trackman 표본 수의 상대 차이
- 기존 이력은 안정적이나 Trackman 반복성이 낮은 정도
- 기존 구종 분포와 Trackman 구종 분포의 divergence
- Trackman profile의 최근 변화와 공식 최근 제구율 변화의 방향 불일치

처음에는 5~10개로 압축한다. 원시 Trackman 평균과 불일치 피처를 동시에 대량 투입하지 않는다.

적용 범위:

- cold-start
- baseline uncertainty 상위 구간
- Trackman·asof coverage가 모두 충분한 행

## 11. Phase 6. Trackman archetype·저차원 표현

상관이 높은 raw 피처를 모두 모델에 넣는 대신, 과거 Trackman profile을 안정적인 유형 또는 저차원 표현으로 압축한다.

### 절차

1. 목표 시즌보다 이전 Trackman profile 생성
2. season·game_type·pitcher_hand 기준 표준화
3. PCA 또는 소수 cluster로 압축
4. 투수별 archetype 확률·저차원 점수 생성
5. cold-start·불확실성 gate 안에서만 적용

archetype이 단순히 pitcher_id 또는 표본 수를 재현하는지 반드시 permutation과 비교한다.

## 12. Phase 7. 상태 전이와 최근성 재설계

기존 시즌 평균−커리어 평균 대신 시즌 내부 상태 전이를 검토한다.

### 후보 상태

- 시즌 초반 → 중반 → 후반
- 안정 → 불안정
- 불안정 → 회복
- 구속 하락 + 릴리스 유지
- 구속 유지 + 릴리스 악화
- 구종 비율 변화 + mechanical profile 변화

각 상태는 목표 행의 시즌보다 이전 Trackman만 사용한다. 상태 전이와 이후 제구율의 관계가 2022·2023·2024에서 같은 방향인지 먼저 확인한 뒤 모델 피처로 승격한다.

## 13. Phase 8. 제한적 residual 보정

Phase 3~7에서 방향이 안정적인 단일 신호가 확인된 경우에만 작은 residual 모델을 만든다.

```text
p_final = clip(p_base + lambda × residual_trackman, 0, 1)
```

원칙:

- Ridge 또는 얕은 CatBoost부터 시작
- lambda·gate·최소 표본 수는 2022·2023에서만 선택
- 2024는 설정을 고정한 최종 검증
- gate 밖과 결측 행은 정확히 baseline 사용
- baseline offset을 Trackman 실험 중 재최적화하지 않음

## 14. 추천 실행 순서

1. Phase 0: coverage·selection·분포 감사
2. Phase 1: permutation·negative control
3. Phase 2: Trackman-only incremental signal 분해
4. Phase 3: fold별 residual 방향 확인
5. Phase 4: reliability·shrinkage
6. Phase 5: Trackman–asof 불일치
7. Phase 6: archetype·저차원 표현
8. Phase 7: 상태 전이·최근성
9. Phase 8: 방향이 안정적인 단일 신호만 residual 보정

Phase 1~3에서 독립적인 signal이 확인되지 않으면 Phase 4 이후의 복잡한 피처 개발을 중단하고 Trackman은 제출에 사용하지 않는다.

## 15. 실험 산출물과 저장 위치

각 Phase는 다음을 남긴다.

- 실행 코드
- `config.json`
- 사용 피처와 cutoff 규칙
- fold별 Brier/BSS
- residual 방향과 segment metrics
- coverage·결측·표본 수 분포
- permutation/negative control 비교
- 채택·보류·기각 결론
- `report.md`

원본 산출물은 `open/experiments/trackman_strategy_v3/` 아래에 저장한다. 사람이 읽는 요약은 OneDrive `phase2/EDA/실험_요약/`에 별도로 기록한다.

`0822(std21)` ZIP, `open/champion/`, `submit/`, `reference`는 실험 중 직접 변경하지 않는다.

## 16. 최종 성공 기준

다음 조건을 모두 만족해야 제출 후보로 검토한다.

- Trackman 정상 매핑이 permutation 대조군보다 의미 있게 우수
- 2022·2023 residual 방향이 일관
- 2024 고정 holdout에서도 같은 방향
- 전체 Brier 악화 없이 충분한 coverage
- cold-start 또는 불확실성 gate 밖의 baseline 성능 보존
- Trackman·train 공식 데이터만 사용
- test 행 간 의존성 없음
- 독립 실행과 feature cutoff 감사 통과

## 17. 진행 상태

- [x] Phase 0: 데이터·coverage·selection 감사
  - Train `1,475,092`행, Trackman history `1,793,078`행, Trackman pitcher `906`명 확인
  - confirmed player mapping은 전체 Train의 `1,464,612`행(`99.29%`)으로 높았지만, repeatability 사용 가능 행은 `1,010,991`행(`68.54%`)으로 차이가 큼
  - 2020~2024 repeatability coverage는 Regular `79.30%~85.38%`, Futures `59.45%~88.05%`로 시즌·game_type별 편차가 존재
  - 2022/2023/2024 Trackman available 행 비율은 각각 `84.27%/85.66%/80.78%`
  - baseline residual은 game_type별로 방향이 바뀜: Futures 평균 residual `+0.1124→-0.0771→-0.0079` (2022→2023→2024), Regular는 `+0.0088→+0.0095→+0.0003`
  - Trackman available 행의 평균 residual도 `+0.0213→-0.0001→-0.0001`로 안정적인 보정 방향을 보이지 않음
  - 결론: 매핑률 자체는 높지만 실제 물리 피처 coverage·selection과 residual 방향이 시간·game_type에 따라 변하므로, 새 raw 피처보다 Phase 1 permutation/negative control을 먼저 수행
  - 상세 결과: `open/experiments/trackman_strategy_v3/phase0_audit_20260822_rerun/`
- [x] Phase 1: permutation·negative control
  - 정상 Trackman과 row-shuffled Trackman 모두 Ridge residual 보정에서 baseline보다 악화
  - 2023 lambda=0.10 기준 정상 delta `+0.000199`, shuffled delta `+0.000192`
  - 2024 lambda=0.10 기준 정상 delta `+0.000017`, shuffled delta `+0.000011`
  - 정상 Trackman이 shuffled 대조군보다 일관되게 우수하지 않아 독립적인 incremental signal을 확인하지 못함
  - 결론: 현재 std/MAD/repeatability 계열은 residual 보정 신호로 사용하지 않으며, Phase 2는 Trackman-only와 기존 모델의 중복 여부를 확인하는 진단 목적으로만 진행
  - 상세 결과: `open/experiments/trackman_strategy_v3/phase1_permutation_20260822/`
- [x] Phase 2: Trackman-only incremental signal 분해
  - reliability-only, physical std/MAD-only, all Trackman-only와 각 변형의 baseline probability 결합을 Ridge로 screening
  - 2023 Trackman-only delta는 `+0.00214~+0.00296`, baseline 결합 변형도 `+0.00115~+0.00177`로 악화
  - 2024 Trackman-only delta는 `+0.00282~+0.00299`, baseline 결합 변형도 `+0.000084~+0.000307`으로 악화
  - 결론: Trackman raw/reliability 피처만으로 기존 모델의 독립 신호를 확인하지 못함
  - 상세 결과: `open/experiments/trackman_strategy_v3/phase2_incremental_20260822/`
- [x] Phase 3: residual 방향 안정성
  - 동일 세그먼트의 평균 baseline residual이 2022·2023·2024에서 같은 부호인지 확인
  - 전체 18개 세그먼트 중 방향이 안정적인 세그먼트는 `2개(11.1%)`뿐
  - `game_type=F`는 `+0.1124→-0.0771→-0.0079`로 방향이 반전되어 고정 보정 근거가 없음
  - `tm_available=True`도 `+0.0213→-0.0001→-0.0001`로 안정적이지 않음
  - 안정적으로 보인 `game_type=R`은 평균 coverage `88.5%`지만 residual이 `+0.00034~+0.00952`로 작고, Trackman 고유 신호가 아니라 baseline의 연도·세그먼트 calibration 차이일 수 있음
  - `trackman_n=1000+`도 coverage `59.3%`에서만 안정적이었으나, Trackman 자체의 물리 신호보다 표본이 많은 투수군 선택 효과를 분리하지 못함
  - 결론: 고정 residual 방향을 적용할 수 있는 Trackman 세그먼트가 충분하지 않으며 Phase 1·2 결과와 함께 독립 signal gate를 통과하지 못함
  - 상세 결과: `open/experiments/trackman_strategy_v3/phase3_residual_direction_20260822/`
- [ ] Phase 4: reliability·shrinkage
- [ ] Phase 5: Trackman–asof 불일치
- [ ] Phase 6: archetype·저차원 표현
- [ ] Phase 7: 상태 전이·최근성 재설계
- [ ] Phase 8: 제한적 residual 보정

## 18. Phase 0~3 중간 결론

Phase 0~3의 신호 감사 결과, Trackman 매핑률은 높지만 현재 사용 가능한 Trackman 요약값에서 제출 후보로 승격할 만큼의 독립적이고 시간 안정적인 신호는 확인되지 않았다.

- Phase 0: 매핑률과 물리 피처 coverage가 다르고, game_type·연도별 baseline residual 방향이 불안정했다.
- Phase 1: 정상 Trackman과 shuffled 대조군이 모두 baseline을 악화시켰고 정상 매핑이 shuffled보다 일관되게 우수하지 않았다.
- Phase 2: reliability-only·physical-only·all Trackman 계열이 2023·2024에서 모두 악화했고 baseline 결합으로도 독립 개선이 확인되지 않았다.
- Phase 3: 18개 세그먼트 중 방향 안정 세그먼트가 2개뿐이며, 안정 세그먼트도 Trackman 고유 효과라고 보기 어렵다.

따라서 플랜의 중단 조건에 따라 Phase 4~8의 복잡한 Trackman 피처·모델링은 실행하지 않는다. `0822(std21)`은 변경하지 않고, Trackman은 현재 제출 후보에 추가하지 않는다. 새로운 시도는 Trackman 내부 피처를 더 만드는 방식이 아니라, 별도의 강한 가설과 독립적인 검증 설계가 생길 때 재개한다.
