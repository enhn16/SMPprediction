# 확장 재채점: 새 판정 기준(시간 구간), 2020-01 ~ 2026-10

> 공통 규칙은 루트 [CLAUDE.md](../CLAUDE.md)를 따른다. 매 세션 이 파일과 [docs/PLAN.md](docs/PLAN.md)를 먼저 읽는다.

## 1. 목적
- 11/11 발표(팀별 5분)와 웹 '사례 분석' 탭용. 2026-11-04 수업의 새 판정 기준으로 급락·급등 모델을 다시 채점한다.
- **웹 공식 성적(최종 모델 성적표, 2026-01-01~10-18, `team_scorecard`)과 분리된 작업**이다. 결과를 섞지 않는다.
- 기준 문서: `read/SMP예측_1104수업_판정기준_노션게시안_v1.1.docx`, 피드백: `read/SMP예측_1104수업_피드백_4-1조_v1.docx`.

## 2. 판정 기준 (기준 문서 v1.1)
- 육지 급등: 실적 SMP ≥ 같은 시각 직전 28일(그날 제외) 실적 중앙값 + 20원
- 육지 급락: 11~15시 실적 SMP < 30원
- 제주 급락: 제주 SMP **≤ 0원** (음수 포함. 부서 5번 표가 이 기준 — 2026-11-05 확인)
- 하루 안에서 연속된 시간 = 1구간. 휴일 = 주말·공휴일(대체·임시·선거일). 거래시간 1~24.
- 판정 기간 2020-01-29 ~ 2026-10-31 (2,468일).
- 확률 모델: 성공 = 그날 경보, 부분 = 주의만, 실패 = 없음. 헛경보 = 주의·경보를 냈는데 그날 사건 없는 날 수(나눠 표시).

## 3. 규칙
- 운영 코드(`drop_classifier/src`, `surge_classifier/src`, `smp_forecast_app/src`)는 **import만** 하고 수정하지 않는다. 운영 산출물도 건드리지 않는다.
- 2020~2022 보강 자료(`data/raw/backfill_2020/`)는 운영 로더가 읽지 않으므로 `src/data.py`에서 읽는다.
- 2026년 구간은 「2차 시험」으로 표시한다 (이미 여러 번 본 자료).
- 판단 시각 = 전날 15:00. 누수 금지 규칙은 루트와 같다.

## 4. 폴더
```
extended_backtest/
├─ CLAUDE.md, docs/PLAN.md
├─ src/data.py      2019~ SMP·특일 로더 (보강분 + 운영 로더)
├─ src/events.py    새 기준 사건 구간 집계 → outputs/tables/E1_*
├─ src/drop_features.py  육지 급락 피처 2020~ (운영 정의 그대로) → data/processed/extended_backtest/
├─ src/drop_walk.py      육지 급락 매달 학습·문턱 → outputs/tables/D1_*
├─ src/surge_features.py, surge_walk.py  육지 급등 피처 2020~·매달 학습(옛/새 레이블) → S1_*
├─ src/score_tables.py   표 1·2·3 → T1_*, T2_*, T3_*
├─ src/ablation.py, miss_analysis.py  사후 분석(참고) → A*, M*
├─ docs/RESULTS.md (발표용 결과), ANALYSIS_NOTES.md (해 보니 이랬다·피드백 확인)
└─ outputs/tables/
```
실행(루트에서): `python -m extended_backtest.src.events` → `.drop_features` → `.drop_walk` / `.surge_features` → `.surge_walk` → `.score_tables` → `.ablation`, `.miss_analysis`

## 5. 학습·시험 (2026-11-06 확정, PLAN ②)
- 시험 2025-01-01 ~ 2026-10-31. 매달 그 전달까지 자료로 학습, 문턱은 앞 12개월 앞으로만 예측으로 매달. 급락·급등 모두 전체 날.
- 제주 급락은 별도 채팅(`docs/HANDOFF_jeju.md`).
