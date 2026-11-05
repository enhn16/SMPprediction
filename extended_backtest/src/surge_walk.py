"""육지 급등: 매달 다시 학습 + 문턱도 매달 다시 정하기 (급락 drop_walk.py와 같은 틀, 2026-11-06 확정).

모델: 운영 급등 v3 구성 그대로 (40개 입력, 부호 고정 점수 + 스플라인 로지스틱 + LightGBM 그루터기 확률 평균).
  다만 '휴무일 0' 규칙은 쓰지 않는다 (사용자: 급락·급등 모두 전체 날).
두 판을 나란히:
  A 지금 모델  = 옛 레이블 y (일 최고 ≥ 28일 일 최고 중앙값 × 1.25)로 학습  ← 피드백 4번 "지금 모델을 새 기준으로 채점"
  B 새 레이블  = y_new (그날 '같은 시각 28일 중앙값 + 20원' 이상 시간이 있음)로 학습
문턱(두 판 모두): 그달 바로 앞 12개월의 앞으로만 예측 + 새 기준 사건일(y_new)로
  주의 = 재현율 80%를 지키는 가장 높은 선 (운영 규칙 그대로, 0.01 격자).
  경보 = 주의보다 높은 선 중 정밀도 75%·적중 2일 이상인 가장 낮은 선, 없으면 그중 정밀도 최고 (2026-11-06 사용자 확정).
    운영 규칙(정밀도 50%)은 새 기준 급등일이 34~48%라 주의보다 낮은 선이 돼서 바꿈.
학습 = 2020-01부터 그 전달 말까지, 예측 2024-01 ~ 2026-10, 시험 2025-01-01 ~ 2026-10-31 (2024 = 문턱용 연습).

실행: python -m extended_backtest.src.surge_walk [regrade] → outputs/tables/S1_surge_walk_{daily,thresholds,graded}.csv
"""
import time

import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from surge_classifier.src import team_model2 as T2
from surge_classifier.src import team_model2_fa as FA
from extended_backtest.src.surge_features import CFG

FEAT = PROCESSED_DIR / "extended_backtest" / "surge_features_육지.pkl"
OUT = ROOT / "extended_backtest" / "outputs" / "tables"
FIRST_MONTH, LAST_MONTH = "2024-01", "2026-10"
TEST_START = pd.Timestamp("2025-01-01")
VERSIONS = {"A": "y", "B": "y_new"}
NAMES = {"A": "A 지금 모델 (옛 레이블 ×1.25)", "B": "B 새 레이블 (+20원)"}


def fit_predict(tr, te, label):
    y = tr[label].astype(int)
    models = {k: T2.MAKE[k]().fit(tr[T2.FEATS], y) for k in T2.KEYS}
    p = pd.DataFrame({k: m.predict(te[T2.FEATS]) for k, m in models.items()}, index=te.index)
    return p[T2.KEYS].mean(axis=1).values


def run() -> pd.DataFrame:
    d = pd.read_pickle(FEAT)
    parts = []
    FA.configure(CFG)
    try:
        for month in pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M"):
            t0 = time.time()
            te = d[(d["date"].dt.to_period("M") == month) & d["y_new"].notna()]
            row = te[["date", "is_off", "y", "y_new"]].copy()
            for v, label in VERSIONS.items():
                tr = d[(d["date"] < month.start_time) & d[label].notna()]
                row[f"score_{v}"] = fit_predict(tr, te, label)
            parts.append(row)
            print(f"{month}: {len(te)}일, 새 기준 급등일 {int(te['y_new'].sum())}, {time.time() - t0:.0f}초", flush=True)
    finally:
        FA.configure({})
    return pd.concat(parts, ignore_index=True)


ALERT_PREC = 0.75


def alert_line(y, p, caution):
    best = None
    for t in T2.GRID[T2.GRID > caution]:
        tp, fp = T2.counts(y, p, t)
        if tp >= 2:
            prec = tp / (tp + fp)
            if prec >= ALERT_PREC:
                return float(t)
            if best is None or prec > best[0]:
                best = (prec, float(t))
    return best[1] if best else float(T2.GRID[-1])


def thresholds(daily) -> pd.DataFrame:
    rows = []
    for month in pd.period_range(TEST_START.to_period("M"), LAST_MONTH, freq="M"):
        lo = (month - 12).start_time
        pool = daily[(daily["date"] >= lo) & (daily["date"] < month.start_time)]
        y = pool["y_new"].values.astype(int)
        r = {"month": str(month), "pool_days": len(pool), "pool_events": int(y.sum())}
        for v in VERSIONS:
            s = pool[f"score_{v}"].values
            r[f"주의_{v}"] = T2.caution_line(y, s)
            r[f"경보_{v}"] = alert_line(y, s, r[f"주의_{v}"])
        rows.append(r)
    return pd.DataFrame(rows)


def grade(daily, th) -> pd.DataFrame:
    d = daily[daily["date"] >= TEST_START].copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    d = d.merge(th, on="month", how="left")
    for v in VERSIONS:
        s = d[f"score_{v}"]
        d[f"grade_{v}"] = np.where(s >= d[f"경보_{v}"], "경보", np.where(s >= d[f"주의_{v}"], "주의", "없음"))
    return d


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "regrade":      # 저장된 예측으로 문턱·판정만 다시
        daily = pd.read_csv(OUT / "S1_surge_walk_daily.csv", parse_dates=["date"])
    else:
        daily = run()
        daily.to_csv(OUT / "S1_surge_walk_daily.csv", index=False, encoding="utf-8-sig")
    th = thresholds(daily)
    th.to_csv(OUT / "S1_surge_walk_thresholds.csv", index=False, encoding="utf-8-sig")
    grade(daily, th).to_csv(OUT / "S1_surge_walk_graded.csv", index=False, encoding="utf-8-sig")
    print(th.to_string())
