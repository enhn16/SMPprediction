"""한 달 예측 mx3 — Chronos 시간별 경로 저장 (기준일 2023-01-01 ~, 7~34일 뒤 672시간, 분위수 0.1·0.5·0.9).

mx_chronos의 hc2s·hc2와 같은 입력(시간별 SMP 문맥 + 달력 공변량)이지만, 4지표로 묶지 않고 시간별 값을 그대로 저장한다.
입력은 기준일 I의 24시까지 SMP (I일 SMP는 I-1 저녁 공표).
출력: data/processed/smp_forecast_app/mx/mx3_chr_{tag}.csv.gz [base_date, H(일), hour(1~24), q10, q50, q90]
실행: python -m smp_forecast_app.src.mx3_chronos hc2s   (hc2는 다른 프로세스로 동시에; 이미 있으면 건너뜀)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from common.loaders import load_smp_actual
from smp_forecast_app.src.mx_chronos import LEADS, MODELS, bases_all, calendar_cov
from smp_forecast_app.src.mx_features import MX

os.environ.setdefault("HF_HUB_OFFLINE", "1")
QL = [0.1, 0.5, 0.9]
H0 = (LEADS[0] - 1) * 24          # 816시간 경로에서 7일 뒤 1시의 위치
CHUNK = 160                       # 기준일 160개마다 조각 저장 (mx/mx3_chr_parts/)


def run(tag, bases, s, save=True):
    """save=False (운영): 조각 저장 없이 바로 반환."""
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
    nh = len(LEADS) * 24
    Hs, hrs = np.repeat(list(LEADS), 24), np.tile(np.arange(1, 25), len(LEADS))
    bs = 32 if tag == "hc2s" else 16
    t0 = time.time()
    if not save:
        return pd.concat(sum((_batch(pipe, bases[k:k + bs], vals, cov, hourly, pos, ctx, nh, Hs, hrs) for k in range(0, len(bases), bs)), []),
                         ignore_index=True)
    for c0 in range(0, len(bases), CHUNK):          # 조각마다 저장 → 중단되면 이어서
        part = MX / "mx3_chr_parts" / f"{tag}_{c0:05d}.csv.gz"
        if part.exists():
            continue
        parts = []
        for k in range(c0, min(c0 + CHUNK, len(bases)), bs):
            parts += _batch(pipe, bases[k:k + bs], vals, cov, hourly, pos, ctx, nh, Hs, hrs)
            print(f"{tag} {min(k + bs, len(bases))}/{len(bases)} ({time.time()-t0:.0f}s)", flush=True)
        part.parent.mkdir(exist_ok=True)
        pd.concat(parts, ignore_index=True).round(2).to_csv(part, index=False, compression="gzip")
    return pd.concat([pd.read_csv(f, parse_dates=["base_date"]) for f in sorted((MX / "mx3_chr_parts").glob(f"{tag}_*.csv.gz"))],
                     ignore_index=True)


def _batch(pipe, batch, vals, cov, hourly, pos, ctx, nh, Hs, hrs):
    inputs = []
    for I in batch:
        end = (pos[I] + 1) * 24
        c = vals[max(0, end - ctx):end]
        c0 = (I - cov.index[0]).days * 24 + 24 - len(c)
        inputs.append({"target": c, "past_covariates": {n: v[c0:c0 + len(c)] for n, v in hourly.items()},
                       "future_covariates": {n: v[c0 + len(c):c0 + len(c) + 816] for n, v in hourly.items()}})
    q, _ = pipe.predict_quantiles(inputs, prediction_length=816, quantile_levels=QL)
    out = []
    for I, qq in zip(batch, q):
        a = (qq[0] if qq.dim() == 3 else qq).numpy()[H0:H0 + nh]          # [672, 3]
        out.append(pd.DataFrame({"base_date": I, "H": Hs, "hour": hrs, "q10": a[:, 0], "q50": a[:, 1], "q90": a[:, 2]}))
    return out


def main(tags, until=None):
    s = load_smp_actual("육지")[["date", "hour", "smp"]]
    for tag in tags:
        p = MX / f"mx3_chr_{tag}.csv.gz"
        if p.exists():
            print(f"{tag}: 이미 있음 → 건너뜀")
            continue
        run(tag, list(bases_all(s, until)), s).round(2).to_csv(p, index=False, compression="gzip")
        print(f"{tag} 저장 → {p}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    until = a[a.index("--until") + 1] if "--until" in a else None
    main([x for x in a if x in ("hc2s", "hc2")] or ["hc2s", "hc2"], until)
