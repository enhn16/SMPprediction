# 제주 담당 인수 메모 (2026-10-23, 과제2 담당 채팅 작성)

> 이 문서를 받은 채팅은 **제주 SMP 예측(가격·범위)과 제주 다음 날 급락 판별**을 맡는다.
> 먼저 읽을 것: [루트 CLAUDE.md](../../CLAUDE.md) (공통 규칙) → [과제2 CLAUDE.md](../CLAUDE.md) → 이 문서 → [과제1 제주 레이블링](../../drop_classifier/docs/07_제주_레이블링.md) → [과제2 결과](01_결과.md) 10~12절.
> 공통 규칙(판단 시각 D-1 15시, 2025로만 선택, 2026은 한 번 채점, 시간 순서 분할, 한국어 보고, 단계마다 사용자 확인)은 그대로 적용한다.

## 1. 역할 분담 (두 채팅이 같은 파일을 동시에 고치지 않기)

| | 제주 채팅 (이 문서를 받은 쪽) | 과제2 채팅 (육지·웹·운영) |
|---|---|---|
| 맡는 일 | 제주 자료 수집, 제주 가격 모델·범위, 제주 급락 판별, 결과 파일 저장, 제주 결과 문서 | 제주 결과를 운영(`ops.py`)·웹(`export_web.py`, `web_src/`)에 연결, 육지 유지보수, 발표 정리 |
| 새로 만들 곳 | `smp_forecast_app/src/jz_*.py`, `smp_forecast_app/docs/JEJU.md`(결과·결정 기록), `data/processed/smp_forecast_app/jeju/`, `outputs/tables/J*` · 급락은 `drop_classifier`에 `region="제주"`로 | — |
| **고치지 말 것** | `smp_forecast_app/src/ops.py`, `export_web.py`, `final_metrics.py`, `web_src/`, `web/`, `docs/PLAN.md`·`OPS.md`·`01_결과.md`, 육지 결과 파일(`WZ*`, `MZ*`, `F*`, `wzoo*/`, `mzoo/`, `ops/`) | — |
| 공용 코드를 고쳐야 하면 | `common/`, `smp_forecast_app/src/features.py`·`wz_*.py`·`intervals.py` 등은 **인자 추가(기본값 = 육지)**로만 바꾸고, **육지 출력이 전과 똑같은지 확인**한 뒤 사용자에게 알린다 (매일 15:30 운영이 이 코드를 씀) | |

- 결정 기록·진행 상황은 `smp_forecast_app/docs/JEJU.md`에 남긴다 (과제2 PLAN.md 대신). 과제1 쪽 결정은 `drop_classifier/docs/PLAN.md`의 제주 칸·결정 기록에.
- 운영(작업 스케줄러 `SMP_forecast_daily` 15:30 · `actuals` 21:30 · `train` 일요일 03:00)이 돌고 있다. 시험 실행할 때 `SMP_NO_PUSH=1`을 쓰고, 운영 기록 파일(`data/processed/smp_forecast_app/ops/`)은 건드리지 않는다.

## 2. 지금까지 확인된 사실

### 가격
- 제주 SMP 시간별 2022-01 ~ 최신 (`common.loaders.load_smp_actual("제주")`, 엑셀 공식값 + 이후 API). 단위 원/kWh, 1~24시.
- **음수 가격 있음** (첫 음수 2024-06-01, 2026 최저 −48원). 음수는 2024Q4~2025Q2 집중, 2025Q3 이후 거의 없음 → **범위 하한을 0원에서 자르지 말 것** (육지는 자름).
- **육지와 같은 가격인 시간 비율: 2022년 31% → 2026년 91% → 2026Q3 99%.** 평소엔 육지 가격 그대로라는 뜻.
- 엑셀 vs API 제주 SMP 14일 169시간 불일치 (2024-08-19, 2024-11-30~12-05, 2026-02-24~25, 04-05, 05-27~06-01) → 실적은 엑셀(공식 확정값) 우선 (`load_smp_actual`이 이미 그렇게 함).

### 급락 (과제1 ① 완료, ②~④ 미완)
- 정의(확정): 11~15시 제주 최저 < 30원, 학습 시작 2024-01-01. 레이블 `data/processed/drop_classifier/daily_label_제주.csv`.
- 급락일: 2024년 20 · 2025년 60 · 2026년(~10/18) 53. 비율 13.6% (육지 4.6%).
- **2024 이후 육지 급락 47일은 모두 제주도 급락. 제주만 급락 86일** — 평일 69%, 한낮 제주 수요예측 낮고(558MW), 맑고(운량 3.4), 바람 강함(5.2m/s).
- 3월 최다, 6~8월 거의 없음.

### 자료 — 있는 것
| 자료 | 위치·로더 | 비고 |
|---|---|---|
| 제주 SMP | `load_smp_actual("제주")` | 위 참고 |
| 제주 하루전 수요예측 | `load_smp_forecast_api()`의 `region == "제주"` → `demand_fcst` (= API `jlfd`) | D일분은 D-1 저녁 공표 → **D-1까지의 실적만** 사용 가능 |
| 중기 육상예보 | `load_mid_forecast()` — `region == "제주도"` 행이 있음 (육지 피처는 제외 중) | D+4~D+7 |
| ASOS 제주 4지점 (제주·고산·성산·서귀포) | `data/raw/api/asos/` | **실측 = 상한선 비교용만**. 성산 일사 없음, 서귀포 일사 33% 결측 |
| 제주 출력제어 (태양광·풍력) | `data/raw/` 사용자 제공 CSV, 2021 ~ 2024-06 | 보관용, 최근 없음 |
| 공휴일·연료비·설비용량 | 육지와 공용 (`common/`) | 설비용량 파일에 제주 구분이 있는지 확인 필요 |

### 자료 — 받아야 할 것 (확인 완료)
- **기상청 단기 구역예보, 제주(11G00201)·서귀포(11G00401)**: API허브 `fct_afs_dl`로 2024-01, 2025-11, 2026-09 과거분 조회 성공 (발표별 72행/3일). 과제1에서 비어 있던 2025-11 이후도 채울 수 있다.
  - 응답 열: 구역, 발표시각, 발효시각, …, **풍향1 풍속등급 풍향2** (예: `NE 1 E`), 기온(-99 = 결측), 강수확률, 하늘(DB01~04), 강수유무, 예보문.
  - `common.loaders.load_kma_regional_fcst()`는 풍향·풍속을 읽지 않는다 → 제주용 파서를 새로 쓰거나 인자 추가.
  - 저장은 같은 폴더(`data/raw/api/kma_afs/afs_{구역}_{YYYYMM}.txt`)도 안전하다 (육지 피처는 `REGIONS` 7개 지점으로 거름). **`common/collect/kma_regional_fcst.REGIONS`에 제주를 추가하지 말 것** — 육지 피처가 바뀐다. 제주용 목록(`JEJU_REGIONS`)을 따로 두고 수집 함수에 인자로.
- **Open-Meteo Previous Runs (일사·운량 + 풍속 100m), 제주 지점**: 키 불필요. ⚠ `load_openmeteo_prev2()`는 `data/raw/api/openmeteo/prev2_*.json`을 **지점 구분 없이 모두 평균**한다 → 제주 자료를 이 폴더에 넣으면 **운영 중인 육지 피처가 오염**된다. 반드시 **별도 폴더**(예: `data/raw/api/openmeteo_jeju/`)와 별도 로더로.
- 매일 수집에 넣는 일은 과제2 채팅이 `ops.py`에 연결한다 → 제주 채팅은 **수집 명령(모듈·인자)만 정해서 JEJU.md에 적어 둘 것**.

## 3. 해야 할 일 (사용자에게 단계마다 보고·확인)

### ① 자료 (반나절)
- 제주·서귀포 구역예보 2023-12 ~ 최신 수집, Open-Meteo 제주 지점 수집 (위 주의사항).
- 판단 시각 확보 여부 확인: 구역예보 11시 발표분, Open-Meteo는 대상일 2일 전 실행분(H=1만 안전) — 육지와 같은 규칙.

### ② 제주 일주일 가격 (D+1~D+7, 시간별) + 80% 범위 (1일)
- **비교의 기준선 = "육지 최종 예측 그대로"** (`data/processed/smp_forecast_app/WZ_predictions_hourly.csv`의 `pred`, 백테스트 2025-07 ~; 2025 상반기분이 필요하면 `wzoo/c2s.csv.gz` + `wzoo_resid/외부만.csv.gz`로 `c2s + 0.5 × rhat` 재구성).
- 후보 예: 육지 예측 그대로 / Chronos-2 small(제주 SMP) / 육지 예측 + 제주 잔차 보정(제주 수요·풍속·하늘·연휴, `wz_resid.py` 방식) / 표 모델(LightGBM 등) 몇 종 / 앙상블.
- 선택 절차는 육지와 같게: **2025 상반기로 맞춤 → 2025 하반기 MAE로 비교(0.05원 이내면 단순한 쪽) → 2025 전체로 재적합 → 2026 한 번 채점**. 제주 2026 가격 채점은 아직 한 번도 안 했다 → "1차 시험".
- 범위: `intervals.select` 규칙(2025 하반기 포함률 75~85% 중 구간점수 최소). **하한 0원 자르기 제거.** 급락일이 많아 이탈이 클 수 있음 — 휴무일·한낮 구간 성적을 함께 보고.
- 성적 지표는 `smp_forecast_app/src/final_metrics.py`와 **같은 정의** (MAE·RMSE·MAPE(공표 20원 이상)·rMAE·편향·포함률·하한 아래/상한 위·평균 폭·구간점수, 구간: 전체·평일·휴무일·휴무일 한낮·급락 시간·D+H). rMAE 단순 기준은 지난주 같은 시간.

### ③ 제주 다음 날 급락 판별 (1일)
- 과제1 코드(`drop_classifier/src/`, `region` 인자)를 재사용. 육지 최종 = Ridge + EBM + 랜덤포레스트, 문턱 = 2025 급락 90%를 잡는 값, 확률 = 로지스틱 보정(`smp_forecast_app/src/drop_badge.py`·`ops_drop.py` 참고).
- 제주 후보 피처: **육지 급락 점수(육지 모델의 예상 한낮 최저)**, 제주 수요예측(D-1까지), 풍속·하늘 예보(제주), 최근 제주 SMP·급락 횟수, 달력.
- 급락 비율이 높아(13.6%) 분류·트리 모델이 유리할 수 있으니 후보를 다시 비교. 순위 규칙은 육지와 같게(급락 90% 잡을 때 헛경보 수 → 100% → PR-AUC), **2026 채점 전에 고정**.

### (참고) 한 달 예측
- 육지 한 달도 기준을 크게 못 이겼다(참고 전망). 제주는 **"육지 한 달 예측 + 최근 제주−육지 차이"** 정도의 참고값만 제안하고, 시간이 남을 때만.

## 4. 결과 파일 형식 (과제2 채팅이 웹·운영에 붙일 때 읽음)

| 파일 | 열 | 비고 |
|---|---|---|
| `data/processed/smp_forecast_app/jeju/JWZ_predictions_hourly.csv` | `base_date, H, date, hour, y, pred, lo, hi, offday` | 육지 `WZ_predictions_hourly.csv`와 같은 형식, 2025-07-01 ~ 최신(앞으로 7일 행 포함, y 빈칸) |
| `data/processed/smp_forecast_app/jeju/drop_badge_제주.csv` | `date, grade, pred_min, prob, label` | 육지 `drop_badge.csv`와 같은 형식, grade ∈ {없음, 주의, 강한 경보} |
| `outputs/tables/JWZ_choice.json`, `JD_choice.json` | 선택 모델·범위 방법·문턱·확률식 계수(`a`, `b`)·등급 문턱 확률 | 육지 `WZ_choice.json`, `D_choice.json` 참고 |
| `outputs/tables/JF1_final_metrics.csv` | `final_metrics.py`의 F1과 같은 열 | 팀 비교용 |
| `outputs/tables/JWZ1_models.csv`, `JWZ2_ensembles_2025H2.csv`, `JD*_…` | 비교한 후보와 성적 | 웹 참고 문서용 |
| **운영용 예측 함수** | `jz_ops.py`(가칭)에 `weekly(bases) -> DataFrame`(위 시간별 형식), `drops(bases) -> DataFrame`(위 급락 형식) | 과제2 채팅이 `ops.py`에서 호출. 늦게 실행해도 같은 값(피처가 날짜에 고정)이어야 함 |
| 수집 명령 목록 | JEJU.md에 모듈·인자 | 과제2 채팅이 매일 수집에 추가 |

## 5. 육지 쪽 참고 (재사용할 것)
- 일주일 최종: Chronos-2 small + 0.5 × 잔차 보정(외부 정보 LightGBM) — `wz_chronos.py`(`--hist`로 2023~ 과거 예측), `wz_resid.py`, `wz_ens.py`(후보·선택 규칙), `intervals.py`(범위 후보·선택), `model.cqr`(보정 분위수 회귀).
- 피처 표: `features.build(lead=H)` (육지 전용). 제주용은 새 함수로 (`jz_features.py`) 만들고 육지 것을 고치지 않는 편이 안전.
- 실행 형식: 루트에서 `python -m smp_forecast_app.src.<모듈>`, 시드 `common.config.SEED`, 그래프 한글 폰트 `common/`.
- 2026-10 기준 2026 시험 성적(육지): 일주일 MAE 9.62원·포함률 76.3%, 급락 판별 재현율 90%·헛경보 72일.

## 6. 일정
- 10/28 줌 미팅 → **11/4 대면(성공 기준 확정)** → 11/11 발표(팀별 5분).
- 공통 CLAUDE.md상 과제1 제주는 "선택" 항목 → 발표는 육지 중심, 제주는 확장 결과. **11/3까지 ①~③과 결과 파일**이 있으면 과제2 채팅이 웹(육지/제주 전환)에 연결할 수 있다.
