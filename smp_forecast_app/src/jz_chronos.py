"""제주 일주일 예측 — Chronos 시계열 기초모델 (제로샷). 육지 wz_chronos와 같은 방식, 대상 = 제주 SMP.

입력: 기준일 I의 24시까지 제주 시간별 SMP, 최근 2,048시간
  jc2 / jc2s   : Chronos-2 / Chronos-2 small + 달력 공변량(휴무일·요일유형·명절, 과거·미래)
  jbolt        : Chronos-Bolt small, 제주 SMP만
  jc2x / jc2sx : 위 + 육지 공변량 — 육지 SMP(과거 = 실적, 미래 = 같은 기준일의 육지 일주일 예측 L_base),
                 제주 하루전 수요예측(과거만, I일까지)
출력: I+1 ~ I+7 시간별 분위수 0.1·0.5·0.9 → pred, qlo, qhi
실행: python -m smp_forecast_app.src.jz_chronos jc2s jc2sx ...
      python -m smp_forecast_app.src.jz_chronos --hist jc2s jc2sx  → 대상일 2023~2024 (잔차 보정 학습용, jzoo_hist/)
운영: forecast(tag, bases, land_future) — jz_ops에서 호출
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from common.calendar import build_calendar
from common.loaders import load_smp_actual, load_smp_forecast_api
from smp_forecast_app.src.jz_common import HIST, OUT, land_pred, save

CTX = 2048
MODELS = {"jc2": "amazon/chronos-2", "jc2s": "autogluon/chronos-2-small", "jbolt": "amazon/chronos-bolt-small",
          "jc2x": "amazon/chronos-2", "jc2sx": "autogluon/chronos-2-small"}


def _series():
    s = load_smp_actual("제주").sort_values(["date", "hour"]).reset_index(drop=True)
    assert (s.groupby("date").size() == 24).all()
    idx = pd.MultiIndex.from_frame(s[["date", "hour"]])
    land = load_smp_actual("육지").set_index(["date", "hour"])["smp"].reindex(idx).values.astype("float32")
    dem = load_smp_forecast_api()
    dem = dem[dem["region"] == "제주"].set_index(["date", "hour"])["demand_fcst"].reindex(idx).values.astype("float32")
    return s, s["smp"].values.astype("float32"), land, dem


def land_future_table(lp: pd.DataFrame) -> dict:
    """육지 일주일 예측 [date, hour, H, L_base] → {기준일: 168시간 배열}."""
    lp = lp.assign(base=lp["date"] - pd.to_timedelta(lp["H"], unit="D"))
    return {b: g.sort_values(["H", "hour"])["L_base"].values.astype("float32") for b, g in lp.groupby("base") if len(g) == 168}


def forecast(tag, bases, land_future=None, pipe=None, log=print):
    """기준일 목록 → [date, hour, H, pred, qlo, qhi]. land_future: {기준일: 육지 예측 168시간} (x 모델만 필요)."""
    from chronos import BaseChronosPipeline
    s, vals, land, dem = _series()
    days = pd.DatetimeIndex(s["date"].values[::24])
    cal = build_calendar(days[0], days[-1] + pd.Timedelta(days=10)).set_index("date")
    cov_day = {"offday": cal["is_offday"].astype(float), "daytype": cal["day_type"].map({"평일": 0, "주말": 1, "공휴일": 2, "연휴": 3}).astype(float),
               "myeongjeol": cal["is_myeongjeol"].astype(float)}
    hourly = {k: np.repeat(v.values, 24).astype("float32") for k, v in cov_day.items()}
    cal_start = cal.index[0]
    pos = {d: i for i, d in enumerate(days)}
    use_cov, use_land = tag != "jbolt", tag.endswith("x")
    bs = [b for b in bases if b in pos and ((not use_land) or b in land_future)]
    if len(bs) < len(bases):
        log(f"{tag}: 실적 또는 육지 예측이 없는 기준일 {len(bases) - len(bs)}개 제외")
    if pipe is None:
        pipe = BaseChronosPipeline.from_pretrained(MODELS[tag], device_map="cpu", torch_dtype=torch.float32)
    rows, t0 = [], time.time()
    for k in range(0, len(bs), 32):
        batch = bs[k:k + 32]
        inputs = []
        for I in batch:
            end = (pos[I] + 1) * 24
            st = max(0, end - CTX)
            ctx = vals[st:end]
            if use_cov:
                c0 = (I - cal_start).days * 24 + 24 - len(ctx)
                past = {n: v[c0:c0 + len(ctx)] for n, v in hourly.items()}
                futr = {n: v[c0 + len(ctx):c0 + len(ctx) + 168] for n, v in hourly.items()}
                if use_land:
                    past["land"] = land[st:end]
                    futr["land"] = land_future[I]
                    past["jdem"] = dem[st:end]
                inputs.append({"target": ctx, "past_covariates": past, "future_covariates": futr})
            else:
                inputs.append(torch.tensor(ctx))
        q, m = pipe.predict_quantiles(inputs, prediction_length=168, quantile_levels=[0.1, 0.5, 0.9])
        for I, qq in zip(batch, q):
            qq = (qq[0] if qq.dim() == 3 else qq).numpy()
            for t in range(168):
                H, h = t // 24 + 1, t % 24 + 1
                rows.append((I + pd.Timedelta(days=H), h, H, qq[t, 1], qq[t, 0], qq[t, 2]))
        if len(bs) > 32:
            log(f"{tag} {k + len(batch)}/{len(bs)} ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows, columns=["date", "hour", "H", "pred", "qlo", "qhi"])


def main(tags, hist=False):
    s = load_smp_actual("제주")
    days = pd.DatetimeIndex(sorted(s["date"].unique()))
    lfut = land_future_table(land_pred())
    lo, hi = ("2022-12-25", "2024-12-31") if hist else ("2024-12-25", "2100-01-01")
    keep = ("2023-01-01", "2025-01-01") if hist else ("2025-01-01", "2100-01-01")
    bases = [d for d in days if pd.Timestamp(lo) <= d < pd.Timestamp(hi)]
    for tag in tags:
        if ((HIST if hist else OUT) / f"{tag}.csv.gz").exists():
            print(f"{tag}: 이미 있음 → 건너뜀", flush=True)
            continue
        t0 = time.time()
        out = forecast(tag, bases, lfut, log=lambda m: print(m, flush=True))
        out = out[(out["date"] >= keep[0]) & (out["date"] < keep[1])]
        save(tag, out, hist=hist)
        print(f"{tag} 완료 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    hist = "--hist" in args
    args = [a for a in args if a != "--hist"]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main(args or ["jc2s", "jc2sx", "jbolt", "jc2", "jc2x"], hist=hist)
