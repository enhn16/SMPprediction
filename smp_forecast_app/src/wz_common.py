"""일주일 예측 모델 비교(wz_*) 공용 — 기준일 I 15시 판단, 대상일 D = I+H (H=1~7), 시간별.

예측 저장 형식: data/processed/smp_forecast_app/wzoo/{모델}.csv.gz [date, hour, H, pred(, qlo, qhi)]
  (기준일 base_date = date − H)
평가 구간: 2025-01-01 ~ (2025 = 모델·앙상블·범위 선택, 2026 = 시험)
재학습: 빠른 모델 매월 1일, 느린 모델 분기 첫날. 실적은 (재학습일 − H)일까지만 사용 (판단일 규칙).
"""
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR
from smp_forecast_app.src.features import FEATURE_SETS, load_features

OUT = PROCESSED_DIR / "smp_forecast_app" / "wzoo"
HS = range(1, 8)
START_EVAL = "2025-01-01"
TRAIN_START = "2022-01-01"
COLS = FEATURE_SETS["기본+전망·날씨"]


def periods(df, freq):
    """재학습 시점 목록 (MS=매월, QS=분기)."""
    return pd.date_range(START_EVAL, df["date"].max(), freq=freq).union(pd.DatetimeIndex([pd.Timestamp(START_EVAL)]))


def rolling_fit(df, H, freq, fit_predict):
    """재학습 시점마다 학습(실적은 시점−H일까지) → 다음 시점 전까지 예측. fit_predict(train, test) → 예측 배열 또는 (pred, qlo, qhi)."""
    out = []
    ps = list(periods(df, freq)) + [pd.Timestamp("2100-01-01")]
    for a, b in zip(ps[:-1], ps[1:]):
        tr = df[(df["date"] >= TRAIN_START) & (df["date"] < a - pd.Timedelta(days=H - 1)) & df["y"].notna()]
        te = df[(df["date"] >= a) & (df["date"] < b)]
        if te.empty:
            continue
        r = fit_predict(tr, te)
        o = te[["date", "hour"]].copy()
        if isinstance(r, tuple):
            o["pred"], o["qlo"], o["qhi"] = r
        else:
            o["pred"] = r
        out.append(o)
    return pd.concat(out).assign(H=H)


def save(name, df):
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(OUT / f"{name}.csv.gz", index=False, encoding="utf-8")


def load_pred(name):
    return pd.read_csv(OUT / f"{name}.csv.gz", parse_dates=["date"])


def actual_table():
    """평가용 실적: H별 피처 표의 y (모두 같은 공표 SMP)."""
    df = load_features(1)[["date", "hour", "y", "offday", "s_d1"]]
    return df[df["date"] >= START_EVAL]


def feats(H):
    return load_features(H)
