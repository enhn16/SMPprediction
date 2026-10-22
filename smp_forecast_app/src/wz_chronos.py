"""일주일 예측 — Chronos 시계열 기초모델 (제로샷, 우리 자료로 추가 학습 없음).

입력: 기준일 I의 24시까지 시간별 SMP (I일 SMP는 I-1일 저녁 공표 → I일 15시에 확보), 최근 2,048시간(약 85일)
  Chronos-2 / Chronos-2 small: 휴무일·요일유형·명절(사전 확정 달력)을 과거·미래 공변량으로 함께 넣음
  Chronos-Bolt small: SMP만
출력: I+1 ~ I+7 시간별 분위수 0.1·0.5·0.9 → pred(중앙값), qlo, qhi (범위 후보로도 사용)
모델 파일: 이 PC의 Hugging Face 캐시에 이미 있는 amazon/chronos-2, autogluon/chronos-2-small, amazon/chronos-bolt-small
실행: python -m smp_forecast_app.src.wz_chronos c2 c2s bolt
      python -m smp_forecast_app.src.wz_chronos --hist c2s   → 대상일 2023~2024 예측을 wzoo_hist/에 (잔차 보정 학습용, wz_resid)
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from common.calendar import build_calendar
from common.loaders import load_smp_actual
from common.config import PROCESSED_DIR
from smp_forecast_app.src.wz_common import OUT, save

HIST = PROCESSED_DIR / "smp_forecast_app" / "wzoo_hist"

CTX = 2048
MODELS = {"c2": "amazon/chronos-2", "c2s": "autogluon/chronos-2-small", "bolt": "amazon/chronos-bolt-small"}


def main(tags, hist=False):
    from chronos import BaseChronosPipeline
    s = load_smp_actual("육지").sort_values(["date", "hour"]).reset_index(drop=True)
    vals = s["smp"].values.astype("float32")
    days = pd.DatetimeIndex(s["date"].values[::24])
    cal = build_calendar(days[0], days[-1] + pd.Timedelta(days=10)).set_index("date")
    cov_day = {"offday": cal["is_offday"].astype(float), "daytype": cal["day_type"].map({"평일": 0, "주말": 1, "공휴일": 2, "연휴": 3}).astype(float),
               "myeongjeol": cal["is_myeongjeol"].astype(float)}
    hourly = {k: np.repeat(v.values, 24).astype("float32") for k, v in cov_day.items()}
    cal_start = cal.index[0]
    lo, hi = ("2022-12-25", "2024-12-31") if hist else ("2024-12-25", "2100-01-01")
    keep = ("2023-01-01", "2025-01-01") if hist else ("2025-01-01", "2100-01-01")
    out_dir = HIST if hist else OUT
    bases = [d for d in days if pd.Timestamp(lo) <= d < pd.Timestamp(hi)]
    pos = {d: i for i, d in enumerate(days)}
    for tag in tags:
        if (out_dir / f"{tag}.csv.gz").exists():
            print(f"{tag}: 이미 있음 → 건너뜀")
            continue
        t0 = time.time()
        pipe = BaseChronosPipeline.from_pretrained(MODELS[tag], device_map="cpu", torch_dtype=torch.float32)
        use_cov = tag in ("c2", "c2s")
        rows = []
        for k in range(0, len(bases), 32):
            batch = bases[k:k + 32]
            inputs = []
            for I in batch:
                end = (pos[I] + 1) * 24
                ctx = vals[max(0, end - CTX):end]
                if use_cov:
                    c0 = (I - cal_start).days * 24 + 24 - len(ctx)
                    past = {n: v[c0:c0 + len(ctx)] for n, v in hourly.items()}
                    futr = {n: v[c0 + len(ctx):c0 + len(ctx) + 168] for n, v in hourly.items()}
                    inputs.append({"target": ctx, "past_covariates": past, "future_covariates": futr})
                else:
                    inputs.append(torch.tensor(ctx))
            q, m = pipe.predict_quantiles(inputs, prediction_length=168, quantile_levels=[0.1, 0.5, 0.9])
            for I, qq in zip(batch, q):
                qq = qq[0] if qq.dim() == 3 else qq       # Chronos-2: [변수, 시간, 분위]
                qq = qq.numpy()
                for t in range(168):
                    H, h = t // 24 + 1, t % 24 + 1
                    rows.append((I + pd.Timedelta(days=H), h, H, qq[t, 1], qq[t, 0], qq[t, 2]))
            print(f"{tag} {k + len(batch)}/{len(bases)} ({time.time()-t0:.0f}s)", flush=True)
        out = pd.DataFrame(rows, columns=["date", "hour", "H", "pred", "qlo", "qhi"])
        out = out[(out["date"] >= keep[0]) & (out["date"] < keep[1])]
        if hist:
            HIST.mkdir(parents=True, exist_ok=True)
            out.round(3).to_csv(HIST / f"{tag}.csv.gz", index=False, encoding="utf-8")
        else:
            save(tag, out)
        print(f"{tag} 완료 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    hist = "--hist" in args
    args = [a for a in args if a != "--hist"]
    main(args or (["c2s"] if hist else ["c2", "c2s", "bolt"]), hist=hist)
