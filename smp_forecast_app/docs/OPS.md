# 운영 안내 — 매일 자동 예측 (2026-10-22 갱신: 웹앱 모델 교체)

## 모델
| 예측 | 모델 | 학습 |
|---|---|---|
| 일주일 (1~7일 뒤, 시간별) | **Chronos-2 small** 중앙값 (휴무일 공변량), 범위 = Chronos-2 분위수 (D+1은 90일 보정) | 없음 (사전학습 모델, 매일 계산) |
| 한 달 (7~34일 뒤, 일별 4지표) | 기준(같은 유형일 4일 평균) + λ × (LSTM·GRU 평균 − 기준), 범위 = 주차×휴무일 오차 분위수 | 매주 일요일 LSTM·GRU 재학습 (약 35분) |
| 내일 급락 경고 | 과제1 최종 모델(Ridge+EBM+랜덤포레스트) → 확률 보정 | 매일 (판단일까지 자료) |
- 첫 운영 실행 2026-10-22: 학습 34분 + 매일 예측 약 3분. 실행 기록 `ops/run_log.txt`

## 한눈에
| 작업 (작업 스케줄러) | 시각 | 하는 일 | 실행 파일 |
|---|---|---|---|
| `SMP_forecast_daily` | 매일 15:30 | 수집 → 밀린 기준일까지 예측(내일 시간별·주간·한 달·급락 확률) → 예측 기록 → 웹 데이터 → GitHub 푸시 | `smp_forecast_app/ops/ops_daily.bat` |
| `SMP_forecast_actuals` | 매일 21:30 | 내일 공표 SMP 수집 → 웹 데이터 → 푸시 | `ops_actuals.bat` |
| `SMP_forecast_train` | 매주 일요일 03:00 | 한 달 예측용 LSTM·GRU 재학습·저장 | `ops_train.bat` |

- 세 작업 모두 "예약 시간을 놓치면 가능한 빨리 실행" + "작업 실행 위해 절전 모드 해제" 설정. PC가 꺼져 있었으면 켜질 때 실행.
- **늦게 실행돼도 예측 내용은 같다**: 피처가 대상일 기준 '판단일 15시까지 알 수 있는 값'으로 날짜에 고정 (내일 수요예측·SMP 미사용, 날씨는 발표 시각으로 선택). 산출 시각(run_at)을 기록해 웹에 "지연 산출" 표시.
- 며칠 꺼져 있었으면 밀린 기준일(최대 14일)을 차례로 산출.

## 매일 수집하는 자료
| 자료 | 수집기 | 비고 |
|---|---|---|
| SMP·하루전 수요예측 | `common.collect.smp_forecast --recent 5` | 공공데이터 API |
| 단기 육상예보 (11시 발표) | `common.collect.kma_regional_fcst` | 기상청 API허브 키 |
| 중기 육상예보 (06시 발표) | `common.collect.mid_fcst` | **당일분만 제공 → 매일 쌓아야 함** |
| Open-Meteo 일사·운량 | `common.collect.openmeteo_prevruns` | 키 불필요 |
| 주간 전력수요 전망 | `common.collect.kpx_weekly_outlook` | 게시판 이미지 OCR(Windows 내장, 한국어) + 전력정보앱 텍스트(평일). 과거 55건 검증: 일치 88.9%, 못 읽음 10.5%(빈 값), 오독 0.6%. 평일은 앱 값 우선 |
| 연료비 (월) | `common.collect.epsis_fuel_cost` | 매월 1~10일 |
| 시장참여 설비용량 (월) | `common.collect.epsis_capacity` | 매월 1~10일 |
| 휴일 | `common.collect.holidays` | 12월에 다음 해 |
| 전력수급예보 (일 최대 예측수요) | `common.collect.kpx_demand_fcst` | **활용신청 필요**(15158707). 승인 후 주간전망 대체·보완 검토 |

## 파일
- 모델: `data/processed/smp_forecast_app/models/mz_seq.pt` (한 달 LSTM·GRU), Chronos는 Hugging Face 캐시
- 예측 기록: `data/processed/smp_forecast_app/ops/hourly_log.csv`(H=1~6), `daily_log.csv`(7~34일), `drop_log.csv`(급락 확률)
- 실행 기록: `data/processed/smp_forecast_app/ops/run_log.txt`, `console_*.log`
- 웹 데이터: `smp_forecast_app/web/data/forecast_data.js` (백테스트 결과 + 운영 기록, 같은 대상은 운영 기록 우선)

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
