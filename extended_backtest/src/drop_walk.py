"""육지 급락: 매달 다시 학습 + 문턱도 매달 다시 정하기 (2026-11-06 사용자 확정).

규칙
1. 매달, 그 전달 말까지의 자료(2020-01부터 전부)로 학습해서 그달을 예측한다. (2024-01 ~ 2026-10, 34번)
2. 그달의 주의·경보 문턱은 바로 앞 12개월의 '앞으로만 예측' 결과로 정한다.
   주의 = 급락일의 80%를 잡는 점수, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨 (없으면 적중 3일 이상 중 정밀도 최고)
   — 운영 문턱 규칙(drop_classifier.src.final.threshold_for, team_final.alarm_threshold) 그대로.
3. 시험 = 2025-01-01 ~ 2026-10-31 (부서 참고안과 같음, 2026은 2차 시험). 2024년은 첫 문턱용 연습 구간(채점 안 함).
모델: 운영 팀 합의 구성 그대로 (T2 회귀 한낮 최저 · LSTM 168시간 · LightGBM 확장 연료비+수준 · 두 예측 평균, 시드 5개).
위험 점수 = −(두 예상 한낮 최저의 평균). 전체 날 학습·판정.

실행: python -m extended_backtest.src.drop_walk           → outputs/tables/D1_drop_walk_daily.csv (월별로 이어서 계산)
      python -m extended_backtest.src.drop_walk time      → 한 달 학습 시간만 재기
"""
import sys
import time

import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from drop_classifier.src.final import threshold_for
from drop_classifier.src.team_final import alarm_threshold
from drop_classifier.src.team_lstm import lgbm, lstm

FEAT = PROCESSED_DIR / "extended_backtest" / "drop_features_육지.pkl"
OUT = ROOT / "extended_backtest" / "outputs" / "tables"
DAILY = OUT / "D1_drop_walk_daily.csv"
FIRST_MONTH, LAST_MONTH = "2024-01", "2026-10"
TEST_START = pd.Timestamp("2025-01-01")
TRAIN_START = pd.Timestamp("2020-01-01")


def predict_month(df: pd.DataFrame, month: pd.Period) -> pd.DataFrame:
    tr = df[(df["date"] >= TRAIN_START) & (df["date"] < month.start_time)].reset_index(drop=True)
    te = df[df["date"].dt.to_period("M") == month].reset_index(drop=True)
    p_lstm = lstm(tr, te, "T2", 168)
    p_lgbm = lgbm(tr, te, "T2", "연료비+수준")
    return te[["date", "label", "min_smp_11_15", "offday"]].assign(
        pred_lstm=p_lstm, pred_lgbm=p_lgbm, pred=(p_lstm + p_lgbm) / 2,
        train_end=tr["date"].max(), n_train_drops=int(tr["label"].sum()))


def run():
    df = pd.read_pickle(FEAT)
    done = pd.read_csv(DAILY, parse_dates=["date", "train_end"]) if DAILY.exists() else pd.DataFrame()
    have = set(done["date"].dt.to_period("M")) if len(done) else set()
    for month in pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M"):
        if month in have:
            continue
        t0 = time.time()
        part = predict_month(df, month)
        done = pd.concat([done, part], ignore_index=True)
        done.to_csv(DAILY, index=False, encoding="utf-8-sig")
        print(f"{month}: {len(part)}일, 급락 {int(part['label'].sum())}, 학습 급락 {part['n_train_drops'].iloc[0]}일, "
              f"{time.time() - t0:.0f}초", flush=True)
    return done


def thresholds(daily: pd.DataFrame) -> pd.DataFrame:
    """시험 달마다 바로 앞 12개월 예측으로 주의·경보 문턱 (점수 = −예상 한낮 최저, 클수록 위험)."""
    rows = []
    for month in pd.period_range(TEST_START.to_period("M"), LAST_MONTH, freq="M"):
        lo = (month - 12).start_time
        pool = daily[(daily["date"] >= lo) & (daily["date"] < month.start_time)]
        y, s = pool["label"].values, -pool["pred"].values
        t_warn = threshold_for(y, s, 0.8)
        t_alarm, how = alarm_threshold(y, s)
        rows.append({"month": str(month), "pool_from": lo.date(), "pool_days": len(pool), "pool_drops": int(y.sum()),
                     "주의_예상최저이하": -t_warn, "경보_예상최저이하": None if t_alarm is None else -t_alarm, "경보규칙": how})
    return pd.DataFrame(rows)


def grade(daily: pd.DataFrame, th: pd.DataFrame) -> pd.DataFrame:
    d = daily[daily["date"] >= TEST_START].copy()
    d["month"] = d["date"].dt.to_period("M").astype(str)
    d = d.merge(th[["month", "주의_예상최저이하", "경보_예상최저이하"]], on="month", how="left")
    d["grade"] = "없음"
    d.loc[d["pred"] <= d["주의_예상최저이하"], "grade"] = "주의"
    d.loc[d["pred"] <= d["경보_예상최저이하"], "grade"] = "경보"
    return d


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        df = pd.read_pickle(FEAT)
        t0 = time.time()
        part = predict_month(df, pd.Period("2025-07", "M"))
        print(f"2025-07 한 달: {time.time() - t0:.0f}초, 예측 범위 {part['pred'].min():.1f} ~ {part['pred'].max():.1f}")
    else:
        daily = run()
        th = thresholds(daily)
        th.to_csv(OUT / "D1_drop_walk_thresholds.csv", index=False, encoding="utf-8-sig")
        g = grade(daily, th)
        g.to_csv(OUT / "D1_drop_walk_graded.csv", index=False, encoding="utf-8-sig")
        print(th.to_string())
