"""일주일 v3 (R3) — Chronos 추론 (제로샷). wz_chronos와 같은 입력·방식에 공변량 묶음만 고를 수 있게 함.

입력: 기준일 I의 24시까지 시간별 SMP 최근 2,048시간. 출력: I+1 ~ I+7 시간별 분위수 0.1·0.5·0.9 → pred, qlo, qhi
공변량 묶음
  cal  : 휴무일·요일유형·명절 (wz_chronos와 같음, 사전 확정 달력)
  dem  : cal + 과거 전용 공변량 육지 하루전 수요예측(기준일 I분까지 = I-1 저녁 공표분, 미래 값 없음)  [C3]
  vc   : cal + 직도입·일반 LNG 변동비 중앙값 계단(S1 가정, 미래는 기준일 달 값까지, wz3_vc)  [⑤ 나]
  ext  : cal + 주간 수요전망 최대부하(kpx_peak) + 하늘상태·강수확률 예보(rf_sky_pm·rf_pop_pm) + 연료비단가(LNG·유연탄)  [B4]
         과거 구간 값 = 그날의 D-1 15시 판단 피처(H=1 표), 미래 I+H = 기준일 I 15시 판단 피처(H 표, wz3/features).
         빈 값은 기준일까지 알려진 마지막 값으로 채움(앞으로 채우기) — 판단 시각 이후 정보는 쓰지 않음.
실행: python -m smp_forecast_app.src.wz3_chronos c2:cal:2023-01-01:2025-01-01  c2s:ext:2023-01-01:2100-01-01
      (모델:공변량:대상일 시작:대상일 끝[:과거 길이][:미세조정판]) → wz3/chronos/{모델}_{공변량}_{시작}[_ctx길이][_미세조정판].csv.gz
      과거 길이 기본 2048 (C2 = 8192), 미세조정판 = wz3/ft/{이름} (wz3_finetune, C1)
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from common.calendar import build_calendar
from common.config import PROCESSED_DIR
from common.loaders import load_smp_actual

W3 = PROCESSED_DIR / "smp_forecast_app" / "wz3"
CH = W3 / "chronos"
CTX = 2048
MODELS = {"c2": "amazon/chronos-2", "c2s": "autogluon/chronos-2-small"}
EXT = ["kpx_peak", "rf_sky_pm", "rf_pop_pm", "fc_lng", "fc_coal"]


def feat_daily(H) -> pd.DataFrame:
    """wz3 피처 표 H의 일별 공변량 [date × EXT]."""
    f = pd.read_csv(W3 / "features" / f"features_h{H}.csv", usecols=["date", "hour"] + EXT, parse_dates=["date"])
    return f[f["hour"] == 12].set_index("date")[EXT]


def run(tag, cov, start, end, bases_only=None, ctx_len=CTX, ckpt=None, fdaily=None):
    """fdaily: ext 공변량용 {H: 일별 표} (운영 wz3_ops가 그날 만든 피처). 없으면 wz3/features 표."""
    from chronos import BaseChronosPipeline
    s = load_smp_actual("육지").sort_values(["date", "hour"]).reset_index(drop=True)
    vals = s["smp"].values.astype("float32")
    days = pd.DatetimeIndex(s["date"].values[::24])
    cal = build_calendar(days[0], days[-1] + pd.Timedelta(days=10)).set_index("date")
    cov_day = {"offday": cal["is_offday"].astype(float),
               "daytype": cal["day_type"].map({"평일": 0, "주말": 1, "공휴일": 2, "연휴": 3}).astype(float),
               "myeongjeol": cal["is_myeongjeol"].astype(float)}
    hourly = {k: np.repeat(v.values, 24).astype("float32") for k, v in cov_day.items()}
    cal_start = cal.index[0]
    if cov == "ext":
        fut_d = fdaily if fdaily is not None else {H: feat_daily(H) for H in range(1, 8)}
        past_d = fut_d[1].reindex(cal.index)                # 과거 d일 값 = d-1 15시 판단분
    lo, hi = pd.Timestamp(start) - pd.Timedelta(days=7), pd.Timestamp(end)
    bases = [d for d in days if lo <= d < hi] if bases_only is None else [d for d in days if d in set(bases_only)]
    pos = {d: i for i, d in enumerate(days)}
    t0 = time.time()
    src = str(W3 / "ft" / ckpt / "finetuned-ckpt") if ckpt else MODELS[tag]
    if ckpt and (W3 / "ft" / ckpt / "finetuned-ckpt" / "adapter_config.json").exists():   # LoRA 어댑터는 Chronos2 로더만 읽음
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained(src, device_map="cpu", torch_dtype=torch.float32)
    else:
        pipe = BaseChronosPipeline.from_pretrained(src, device_map="cpu", torch_dtype=torch.float32)
    if cov == "dem":
        from common.loaders import load_smp_forecast_api
        dm = load_smp_forecast_api()
        dm = dm[dm["region"] == "육지"][["date", "hour", "demand_fcst"]]
        dem = s[["date", "hour"]].merge(dm, on=["date", "hour"], how="left")["demand_fcst"].ffill().values.astype("float32")
    if cov == "vc":
        from smp_forecast_app.src.wz3_vc import monthly
        vcm = monthly()
        vcm.index = vcm.index.map(lambda p: p.ordinal)
    rows = []
    for k in range(0, len(bases), 32):
        batch = bases[k:k + 32]
        inputs = []
        for I in batch:
            end_i = (pos[I] + 1) * 24
            ctx = vals[max(0, end_i - ctx_len):end_i]
            c0 = (I - cal_start).days * 24 + 24 - len(ctx)
            past = {n: v[c0:c0 + len(ctx)] for n, v in hourly.items()}
            futr = {n: v[c0 + len(ctx):c0 + len(ctx) + 168] for n, v in hourly.items()}
            if cov == "dem":
                past["demand"] = dem[max(0, end_i - ctx_len):end_i]   # 과거 전용 (future_covariates에 없음)
            if cov == "vc":       # ⑤(나) 직도입·일반 LNG 변동비 계단(S1: 그달 1일 공개) — 미래 날은 기준일 달 값까지만
                npd = -(-len(ctx) // 24)
                dd = pd.date_range(end=I + pd.Timedelta(days=7), periods=npd + 7, freq="D")
                mo = np.minimum(dd.to_period("M").map(lambda p: p.ordinal).values, I.to_period("M").ordinal)
                for n, col in (("vc_dl", "dl"), ("vc_gl", "gl")):
                    x = np.repeat(vcm[col].reindex(mo).values.astype("float32"), 24)
                    past[n], futr[n] = x[:npd * 24][-len(ctx):], x[npd * 24:]
            if cov == "ext":
                npd = -(-len(ctx) // 24)
                pv = past_d.reindex(pd.date_range(end=I, periods=npd, freq="D"))
                fv = pd.DataFrame([fut_d[H].reindex([I + pd.Timedelta(days=H)]).iloc[0] for H in range(1, 8)])
                both = pd.concat([pv, fv], ignore_index=True).ffill().bfill().fillna(0)
                for n in EXT:
                    x = np.repeat(both[n].values.astype("float32"), 24)
                    past[n], futr[n] = x[:npd * 24][-len(ctx):], x[npd * 24:]
            inputs.append({"target": ctx, "past_covariates": past, "future_covariates": futr})
        q, _ = pipe.predict_quantiles(inputs, prediction_length=168, quantile_levels=[0.1, 0.5, 0.9])
        for I, qq in zip(batch, q):
            qq = (qq[0] if qq.dim() == 3 else qq).numpy()
            for t in range(168):
                H, h = t // 24 + 1, t % 24 + 1
                rows.append((I + pd.Timedelta(days=H), h, H, qq[t, 1], qq[t, 0], qq[t, 2]))
        print(f"{tag}:{cov} {k + len(batch)}/{len(bases)} ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows, columns=["date", "hour", "H", "pred", "qlo", "qhi"])
    return out[(out["date"] >= start) & (out["date"] < end)]


def main(specs):
    CH.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        tag, cov, start, end, *opt = spec.split(":")
        ctx_len = int(opt[0]) if opt and opt[0] else CTX
        ckpt = opt[1] if len(opt) > 1 else None
        suf = (f"_ctx{ctx_len}" if ctx_len != CTX else "") + (f"_{ckpt}" if ckpt else "")
        p = CH / f"{tag}_{cov}_{start}{suf}.csv.gz"
        if p.exists():
            print(f"{p.name}: 이미 있음 → 건너뜀")
            continue
        out = run(tag, cov, start, end, ctx_len=ctx_len, ckpt=ckpt)
        out.round(3).to_csv(p, index=False, encoding="utf-8")
        print(f"저장 {p.name} ({len(out):,}행)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
