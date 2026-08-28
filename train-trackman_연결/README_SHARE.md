# Train–Trackman 최종 결과 공유본

이 폴더는 Train과 Trackman 투구 데이터를 연결한 최종 분석 결과를 공유하기 위한 요약 패키지입니다.

## 전체 구성

README를 제외한 최종 결과물은 총 11개입니다.

| 구분 | 파일 수 | 용도 |
|---|---:|---|
| 분석 보고서 | 3개 | 전체 매칭 과정과 검증 결과 확인 |
| 간단 매핑표 | 4개 | 실제 조회에 바로 사용할 핵심 대응표 |
| 상세 매핑표 | 4개 | 후보·근거·충돌 정보까지 확인하는 검증용 대응표 |

## 1. 분석 보고서 3개

| 파일 | 주요 내용 |
|---|---|
| `train_trackman_matching_complete_report.md` | Trackman 정렬, Train 경기 분할, 경기 단위 매칭, gap 정렬, 최종 행 상태 분류까지 전체 과정의 통합 보고서 |
| `team_mapping_validation_report.md` | `season + game_type + train_team_id`와 Trackman 팀명의 대응 검증 결과 |
| `player_id_mapping_validation_report.md` | 시즌·게임타입·팀 단위의 엄격 선수 ID 매핑과 전역 선수 ID 매핑의 검증 결과 |

권장 읽는 순서는 다음과 같습니다.

1. `train_trackman_matching_complete_report.md`
2. `team_mapping_validation_report.md`
3. `player_id_mapping_validation_report.md`

## 2. 간단 매핑표 4개

간단 매핑표는 불필요한 후보·중간 검증 컬럼을 제외하고, 실제 조회에 필요한 키·결과·상태만 남긴 파일입니다.

| 파일 | 대응 기준 | 주요 결과 |
|---|---|---|
| `team_id_mapping_simple.csv` | `season + game_type + train_team_id` | Trackman 팀명 |
| `player_id_mapping_simple.csv` | `role + season + game_type + train_team_id + train_player_id` | Trackman 선수 ID |
| `player_id_global_mapping_simple.csv` | `role + train_player_id` | 시즌·팀을 통합한 Trackman 선수 ID |
| `team_player_id_mapping_simple.xlsx` | 위 간단 팀·선수 대응표를 한 엑셀에 통합 | `README`, `Team_Mapping`, `Player_Mapping` 시트 |

`mapping_status` 컬럼은 확정 매핑, 전역 보조 매핑, 충돌, 근거 부족을 구분하기 위해 유지했습니다.

## 3. 상세 매핑표 4개

상세 매핑표는 최종 ID뿐 아니라 매핑 근거, 후보 수, 일치율, 충돌 여부 등 검증용 컬럼을 포함합니다.

| 파일 | 내용 |
|---|---|
| `team_id_mapping_final.csv` | 팀 매핑 126개 조합의 상세 근거와 상태 |
| `player_id_mapping_final.csv` | `역할 + 시즌 + 게임타입 + 팀 ID + 선수 ID` 기준 엄격 매핑 6,566개 |
| `player_id_global_mapping_final.csv` | `역할 + 선수 ID` 기준 전역 매핑 1,622개 |
| `team_player_id_mapping_final.xlsx` | 상세 팀 매핑·엄격 선수 매핑·전역 선수 매핑을 통합한 엑셀 |

상세 파일은 검증과 재검토가 필요할 때 사용하고, 일반적인 ID 조회에는 간단 매핑표를 사용하면 됩니다.

## 매핑표 선택 기준

| 필요한 작업 | 사용할 파일 |
|---|---|
| 팀명을 빠르게 찾기 | `team_id_mapping_simple.csv` |
| 특정 시즌·팀의 선수 ID 찾기 | `player_id_mapping_simple.csv` |
| 시즌·팀 구분 없이 선수 ID 후보 확인 | `player_id_global_mapping_simple.csv` |
| 충돌·후보·일치율까지 검토 | 해당 `*_final.csv` 또는 상세 통합 엑셀 |
| 전체 분석 방법과 최종 수치 확인 | 3개 분석 보고서 |

보고서와 최종 매핑표에는 개인 PC 절대 경로, 실행 코드, 원자료 링크, 중간 산출물 링크를 포함하지 않았습니다. 매칭 방법·검증 기준·주요 수치·최종 판정은 보고서 본문에 정리되어 있습니다.
