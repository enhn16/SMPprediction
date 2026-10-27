"""한 달 예측 개선(mx) — Chronos 제로샷 예측 (기준일 2023-01-01 ~, 7~34일 뒤 일별 4개 지표).

  hc2s : Chronos-2 small, 시간별 SMP 최근 8,192시간 + 달력 공변량(휴무일·요일유형·명절) → 816시간 → 일별 4지표로 집계
  hc2  : Chronos-2,       시간별 최근 4,096시간 + 같은 공변량 → 816시간 → 집계
  dc2  : Chronos-2,       일별 4지표 다변량(일평균·한낮·저녁·한낮 최저) 최근 2,048일 + 달력 공변량 → 34일, 분위수 0.1·0.5·0.9
입력은 기준일 I의 24시까지 SMP (I일 SMP는 I-1 저녁 공표). 2023 기준일은 잔차 보정 학습 이력용.
출력: data/processed/smp_forecast_app/mx/chr_{tag}.csv.gz [base_date, H, date, pred_{지표}(, qlo_{지표}, qhi_{지표})]
실행: python -m smp_forecast_app.src.mx_chronos [hc2s hc2 dc2] [--until YYYY-MM-DD]  (이미 있으면 건너뜀)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.loaders import load_smp_actual
from smp_forecast_app.src.features_daily import STATS
from smp_forecast_app.src.mx_features import MX

os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODELS = {"hc2s": ("autogluon/chronos-2-small", 8192), "hc2": ("amazon/chronos-2", 4096), "dc2": ("amazon/chronos-2", 2048)}
LEADS = range(7, 35)
MID, EVE = list(range(10, 15)), list(range(17, 22))   # 0-기준 시간 위치 (11~15시, 18~22시)


def calendar_cov(days_from, days_to):
    cal = build_calendar(days_from, days_to).set_index("date")
    return pd.DataFrame({"offday": cal["is_offday"].astype(float),
                         "daytype": cal["day_type"].map({"평일": 0, "주말": 1, "공휴일": 2, "연휴": 3}).astype(float),
                         "myeongjeol": cal["is_myeongjeol"].astype(float)}, index=cal.index).astype("float32")


def agg_hourly(path):
    """816시간 경로 → {지표: 34일 배열}."""
    p = path.reshape(34, 24)
    return {"mean": p.mean(1), "mid": p[:, MID].mean(1), "eve": p[:, EVE].mean(1), "midmin": p[:, MID].min(1)}


def bases_all(s, until=None):
    days = pd.DatetimeIndex(sorted(s["date"].unique()))
    b = days[days >= "2023-01-01"]
    return b[b <= pd.Timestamp(until)] if until else b


def run_hourly(tag, bases, s):
    import torch
    from chronos import BaseChronosPipeline
    name, ctx = MODELS[tag]
    s = s.sort_values(["date", "hour"]).reset_index(drop=True)
    vals = s["smp"].values.astype("float32")
    days = pd.DatetimeIndex(s["date"].values[::24])
    cov = calendar_cov(days[0], days[-1] + pd.Timedelta(days=40))
    hourly = {k: np.repeat(cov[k].values, 24) for k in cov}
    pos = {d: i for i, d in enumerate(days)}
    pipe = BaseChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)
    rows, t0 = [], time.time()
    for k in range(0, len(bases), 32):
        batch = bases[k:k + 32]
        inputs = []
        for I in batch:
            end = (pos[I] + 1) * 24
            c = vals[max(0, end - ctx):end]
            c0 = (I - cov.index[0]).days * 24 + 24 - len(c)
            inputs.append({"target": c, "past_covariates": {n: v[c0:c0 + len(c)] for n, v in hourly.items()},
                           "future_covariates": {n: v[c0 + len(c):c0 + len(c) + 816] for n, v in hourly.items()}})
        q, _ = pipe.predict_quantiles(inputs, prediction_length=816, quantile_levels=[0.5])
        for I, qq in zip(batch, q):
            a = agg_hourly((qq[0] if qq.dim() == 3 else qq).numpy()[:, 0])
            for H in LEADS:
                rows.append({"base_date": I, "H": H, **{f"pred_{st}": float(a[st][H - 1]) for st in STATS}})
        print(f"{tag} {k + len(batch)}/{len(bases)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def run_daily(tag, bases, s):
    import torch
    from chronos import BaseChronosPipeline
    from smp_forecast_app.src.features_daily import daily_stats
    name, ctx = MODELS[tag]
    st = daily_stats()
    st = st.reindex(pd.date_range(st.index.min(), st.index.max()))
    cov = calendar_cov(st.index[0], st.index[-1] + pd.Timedelta(days=40))
    pipe = BaseChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)
    rows, t0 = [], time.time()
    for k in range(0, len(bases), 64):
        batch = bases[k:k + 64]
        inputs = []
        for I in batch:
            past = st.loc[:I].iloc[-ctx:]
            fut = pd.date_range(I + pd.Timedelta(days=1), periods=34)
            inputs.append({"target": past[STATS].values.T.astype("float32"),
                           "past_covariates": {n: cov[n].reindex(past.index).values for n in cov},
                           "future_covariates": {n: cov[n].reindex(fut).values for n in cov}})
        q, _ = pipe.predict_quantiles(inputs, prediction_length=34, quantile_levels=[0.1, 0.5, 0.9])
        for I, qq in zip(batch, q):
            qq = qq.numpy()                       # [지표 4, 34일, 분위 3]
            for H in LEADS:
                r = {"base_date": I, "H": H}
                for j, stt in enumerate(STATS):
                    r[f"pred_{stt}"], r[f"qlo_{stt}"], r[f"qhi_{stt}"] = qq[j, H - 1, 1], qq[j, H - 1, 0], qq[j, H - 1, 2]
                rows.append(r)
        print(f"{tag} {k + len(batch)}/{len(bases)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def predict(tag, bases, s=None):
    s = load_smp_actual("육지")[["date", "hour", "smp"]] if s is None else s
    out = run_daily(tag, bases, s) if tag.startswith("d") else run_hourly(tag, bases, s)
    out["date"] = out["base_date"] + pd.to_timedelta(out["H"], unit="D")
    return out


def main(tags, until=None):
    MX.mkdir(parents=True, exist_ok=True)
    s = load_smp_actual("육지")[["date", "hour", "smp"]]
    for tag in tags:
        p = MX / f"chr_{tag}.csv.gz"
        if p.exists():
            print(f"{tag}: 이미 있음 → 건너뜀")
            continue
        predict(tag, list(bases_all(s, until)), s).round(3).to_csv(p, index=False, compression="gzip")
        print(f"{tag} 저장 → {p}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    until = a[a.index("--until") + 1] if "--until" in a else None
    tags = [x for x in a if x in MODELS]
    main(tags or list(MODELS), until)
