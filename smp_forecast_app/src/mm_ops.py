"""한 달 예측 mm 운영 — 월말 판단 → 다음 달 모든 날 일평균 + 80% 범위 (MONTH.md 0-5절).

from smp_forecast_app.src import mm_ops
w = mm_ops.monthly_next("2026-10-31")      # [base_date, date, day_type, offday, pred_mean, lo_mean, hi_mean]
  · 월말이 아닌 날을 넣으면 직전 월말 판단 결과를 돌려준다 (base_date 열 = 실제 판단일). 매일 불러도 된다.
  · 판단일 SMP가 아직 없으면(월말 당일 자료 갱신 전) 오류.
  · 결과는 mm/live/MM_{판단일}.csv에 저장하고, 다음부터는 그 파일을 읽는다 (refresh=True면 다시 계산).
  · 재학습은 따로 없다: 계산할 때마다 판단일까지 자료로 LightGBM을 새로 학습한다 (약 10초).
  · 범위 보정 표본 = 백테스트(MM_predictions_calib2024 + MM_predictions_daily) + 이전 운영 결과, 대상일 2024-01-01 ~ 대상 해 전년 말.
실행: python -m smp_forecast_app.src.mm_ops [판단일]     (기본: 자료가 있는 마지막 월말)
"""
import sys
import time

import numpy as np
import pandas as pd

from smp_forecast_app.src.mm_model import COLS, MM, Data, band, calib_of

LIVE = MM / "live"


def history(D, exclude=None):
    """표본 밖 기록 [base_date, date, h, p, y, vol, lng_chg] — 백테스트 + 운영 결과, y는 지금 자료로 채움."""
    parts = [pd.read_csv(MM / f, parse_dates=["base_date", "date"]) for f in ("MM_backtest_full.csv",)]
    parts += [pd.read_csv(f, parse_dates=["base_date", "date"]) for f in sorted(LIVE.glob("MM_*_full.csv"))]
    h = pd.concat(parts, ignore_index=True).drop_duplicates(["base_date", "date"], keep="last")
    if exclude is not None:
        h = h[h["base_date"] != exclude]
    i = h["date"].dt.strftime("%Y-%m-%d").map(D.idx)
    h["y"] = np.where(D.full[i], D.DM[i], np.nan)
    return h


def monthly_next(base_date=None, refresh=False, data=None):
    D = data or Data("gas")
    b = D.month_end(base_date or D.d[D.LAST])
    t = D.idx[b.strftime("%Y-%m-%d")]
    if not D.full[t]:
        raise ValueError(f"판단일 {b.date()} SMP가 아직 없습니다 (전날 저녁 공표분이 들어온 뒤 실행).")
    LIVE.mkdir(parents=True, exist_ok=True)
    out, full = LIVE / f"MM_{b:%Y%m%d}.csv", LIVE / f"MM_{b:%Y%m%d}_full.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, parse_dates=["base_date", "date"])
    ev = D.predict(t, D.d[t + 1])
    ev["lo"], ev["hi"] = band(calib_of(history(D, exclude=b), ev["date"].dt.year.iat[0]), ev)
    ev.to_csv(full, index=False, encoding="utf-8-sig")
    w = ev.rename(columns={"p": "pred_mean", "lo": "lo_mean", "hi": "hi_mean"})[COLS].round(2)
    w.to_csv(out, index=False, encoding="utf-8-sig")
    return w


if __name__ == "__main__":
    t0 = time.time()
    w = monthly_next(sys.argv[1] if len(sys.argv) > 1 else None, refresh=True)
    print(w.to_string(index=False))
    print(f"판단일 {w['base_date'].iat[0].date()} · 월평균 예측 {w['pred_mean'].mean():.1f}원 · {time.time()-t0:.0f}초")
