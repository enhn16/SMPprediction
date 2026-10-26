# 운영 안내 — 매일 자동 예측 (2026-10-22 갱신: 일주일 잔차 보정 반영, 예약 작업 등록)

## 모델
| 예측 | 모델 | 학습 |
|---|---|---|
| 일주일 (1~7일 뒤, 시간별) | **Chronos-2 small** 중앙값 (휴무일 공변량) **+ 0.5 × 잔차 보정**(외부 정보 LightGBM, H별), 범위 = Chronos-2 분위수 (D+1은 90일 보정) | Chronos는 학습 없음. 잔차 보정은 매주 일요일 재학습 (약 20초, 모델 없으면 매일 실행 때 학습) |
| 한 달 (7~34일 뒤, 일별 4지표) | 기준(같은 유형일 4일 평균) + λ × (LSTM·GRU 평균 − 기준), 범위 = 주차×휴무일 오차 분위수 | 매주 일요일 LSTM·GRU 재학습 (약 35분) |
| 내일 급락 경고 | 과제1 최종 모델(Ridge+EBM+랜덤포레스트) → 확률 보정 | 매일 (판단일까지 자료) |
- 첫 운영 실행 2026-10-22: 학습 34분 + 매일 예측 약 3분. 실행 기록 `ops/run_log.txt`
- 예약 작업 등록 2026-10-22 23시 (첫 실행: 10/23 15:30 daily). 실행 방식: `wscript.exe //B ops\run_hidden.vbs ops_{daily|actuals|train}.bat` — 창 없이 실행하고 종료코드를 돌려줌 (2026-10-25 변경. 이전 `powershell -WindowStyle Hidden`은 로그온·절전 해제 때 콘솔 창이 보였음)
- **bat 파일은 cp949(ANSI)·CRLF로 저장한다** (`.gitattributes`에서 줄바꿈 변환 제외). UTF-8·LF로 저장하면 cmd가 한글 주석 일부를 명령으로 실행해 "'ily).'은(는) 내부 또는 외부 명령…이 아닙니다" 오류가 뜬다 (예측 자체는 실행됨)

## 한눈에
| 작업 (작업 스케줄러) | 시각 | 하는 일 | 실행 파일 |
|---|---|---|---|
| `SMP_forecast_daily` | 매일 15:30 (+23:30까지 매시, 로그온·절전 해제 3분 뒤) | 수집 → 밀린 기준일까지 예측(육지: 일주일·한 달·급락 확률 → 제주: 같은 셋) → 예측 기록 → 웹 데이터 → GitHub 푸시. 이미 산출됐으면 바로 끝남 | `smp_forecast_app/ops/ops_daily.bat` |
| `SMP_forecast_actuals` | 매일 21:30 · 23:50 | 내일 공표 SMP 수집, 매일 쌓는 자료(단기예보 11·14시 발표분, 육지 5분 수급, 지역별 태양광) → 웹 데이터 → 푸시 | `ops_actuals.bat` |
| `SMP_forecast_train` | 매주 일요일 03:00 | 일주일 잔차 보정 + 한 달 예측용 LSTM·GRU 재학습·저장, 제주 잔차 보정·급락 고정 모델 저장 | `ops_train.bat` |

- 세 작업 모두 "예약 시간을 놓치면 가능한 빨리 실행" + "작업 실행 위해 절전 모드 해제" + "실패 시 15분 간격 3회 재시도" 설정.
- daily는 추가로 **로그온·절전 해제(Power-Troubleshooter 이벤트 1) 3분 뒤**와 **15:30 이후 23:30까지 매시**에도 실행된다 (2026-10-25). PC가 꺼져 있거나 절전이었으면 켜질 때 바로 따라잡고, 실행 도중 끊겨도 다음 시각에 다시 돈다. `ops daily`는 15:30 전이면 어제 기준일까지만 산출하고(오늘 판단 자료가 덜 모임), 육지·제주 모두 산출돼 있으면 수집 없이 바로 끝난다.
- 2026-10-23 사례: PC 절전(10/23 09:38 ~ 10/24 16:13) 중에는 절전 해제 설정에도 깨어나지 않았다 → 켜진 뒤 따라잡기에 의존.
- **늦게 실행돼도 예측 내용은 같다**: 피처가 대상일 기준 '판단일 15시까지 알 수 있는 값'으로 날짜에 고정 (내일 수요예측·SMP 미사용, 날씨는 발표 시각으로 선택). 산출 시각(run_at)을 기록해 웹에 "지연 산출" 표시.
- 며칠 꺼져 있었으면 밀린 기준일(최대 14일)을 차례로 산출.

## 매일 수집하는 자료
| 자료 | 수집기 | 비고 |
|---|---|---|
| 기상청 단기예보(격자) 11·14시 발표분, 육지 7·제주 4지점 | `common.collect.kma_vilage_fcst` | **API가 최근 3일만 제공 → 매일 쌓음** (2026-10-26~). 아직 피처 미사용 |
| 육지 5분 수급 (수요·공급능력·예비력) | `common.collect.kpx_sukub` | **오늘분만 제공** → 15:30·21:30·23:50에 받아 하루 전체를 남김 (2026-10-26~) |
| 지역별 시간별 태양광 발전량 | `common.collect.kpx_pv_region` | 약 7주 늦게 갱신, 새 날짜만. 하루 100회 한도로 2026-01 초반은 이어 받는 중 |
| SMP·하루전 수요예측 | `common.collect.smp_forecast --recent 5` | 공공데이터 API |
| 단기 육상예보 (11시 발표) | `common.collect.kma_regional_fcst` | 기상청 API허브 키 |
| 중기 육상예보 (06시 발표) | `common.collect.mid_fcst` | **당일분만 제공 → 매일 쌓아야 함** |
| Open-Meteo 일사·운량 | `common.collect.openmeteo_prevruns` | 키 불필요 |
| 주간 전력수요 전망 | `common.collect.kpx_weekly_outlook` | 게시판 이미지 OCR(Windows 내장, 한국어) + 전력정보앱 텍스트(평일). 과거 55건 검증: 일치 88.9%, 못 읽음 10.5%(빈 값), 오독 0.6%. 평일은 앱 값 우선 |
| 연료비 (월) | `common.collect.epsis_fuel_cost` | 매월 1~10일 |
| 시장참여 설비용량 (월) | `common.collect.epsis_capacity` | 매월 1~10일 |
| 휴일 | `common.collect.holidays` | 12월에 다음 해 |
| 전력수급예보 (내일 최대 예측수요) | `common.collect.kpx_demand_fcst` | 승인됨(2026-10-22). 내일 1건만 제공·과거 조회 불가 → 대상일·수집 시각으로 축적. 피처 미사용 |

## 제주 (2026-10-25 연결, 모델·근거는 [JEJU.md](JEJU.md))
| 단계 | 내용 |
|---|---|
| 수집 (daily) | `kpx_jeju_sukub {3일 전}`(5분 계통 실적, 하루 100회 한도) · `kma_regional_fcst … jeju` · `openmeteo_prevruns … jeju`(오늘+3일) · `… jeju_ext`(오늘+8일) |
| 예측 (daily, 육지 다음) | `jz_ops.weekly`(육지 운영 기록 `hourly_log`의 같은 기준일 pred를 읽음) → `drops` → `monthly`(육지 `daily_log`를 읽음). 기준일 = 제주 백테스트·운영 기록 다음 날부터 육지가 산출된 날까지 |
| 기록 | `data/processed/smp_forecast_app/jeju/ops/jwz_log.csv`(일주일, jz_ops가 씀) · `jdrop_log.csv`(급락) · `jdaily_log.csv`(한 달) |
| 학습 (일요일) | `jz_ops.train()` — 잔차 보정 H별 + 급락 고정 모델(2024~2025 자료) |

## 파일
- 모델: `data/processed/smp_forecast_app/models/mz_seq.pt` (한 달 LSTM·GRU), `wz_resid.pkl` (일주일 잔차 보정), Chronos는 Hugging Face 캐시
- 잔차 보정 학습 이력: Chronos 과거 예측 `wzoo_hist/c2s.csv.gz`(2023~2024) + 백테스트 `wzoo/c2s.csv.gz` + 운영 기록의 `c2s` 열
- 예측 기록: `data/processed/smp_forecast_app/ops/hourly_log.csv`(H=1~7; pred = 보정 후, c2s = Chronos 원래 중앙값, rhat = 잔차 예측), `daily_log.csv`(7~34일), `drop_log.csv`(급락 확률)
- 실행 기록: `data/processed/smp_forecast_app/ops/run_log.txt`, `console_*.log`
- 웹 데이터: `smp_forecast_app/web/data/forecast_data.js`(육지·공통) + `forecast_data_jeju.js`(제주) (백테스트 결과 + 운영 기록, 같은 대상은 운영 기록 우선). 푸시는 두 파일만 커밋

## 범위 계산
- 오차 이력 = 백테스트 표본 밖 예측(wzoo c2s·c2, MZ_predictions_daily) + 운영 기록. 방법은 2025로 고른 것 그대로 (WZ_choice·MZ_choice).

## 수동 실행
```
python -m smp_forecast_app.src.ops train     # 재학습
python -m smp_forecast_app.src.ops daily     # 오늘 예측
python -m smp_forecast_app.src.ops actuals   # 공표 SMP 반영
```

## 작업 해제
```
schtasks /delete /tn "SMP_forecast_daily" /f
schtasks /delete /tn "SMP_forecast_actuals" /f
schtasks /delete /tn "SMP_forecast_train" /f
```
