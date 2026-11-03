"""일주일 v3 (R3) ⑥ M1·M2 — 다른 시계열 기초모델 제로샷 (TiRex, TimesFM 2.5). Chronos와 같은 기준일·입력(과거 2,048시간, 공변량 없음).

출력: wz3/chronos/{tirex|tfm}_2023-01-01.csv.gz [date, hour, H, pred, qlo, qhi] (분위수 0.5·0.1·0.9)
실행: python -m smp_forecast_app.src.wz3_fm tirex tfm
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from common.loaders import load_smp_actual
from smp_forecast_app.src.wz3_chronos import CH, CTX

START = "2023-01-01"


def model(name):
    if name == "tirex":
        from tirex import load_model
        m = load_model("NX-AI/TiRex", device="cpu", backend="torch")

        def f(x):
            q, _ = m.forecast(context=torch.tensor(np.stack(x)), prediction_length=168)
            q = q.numpy()
            return q[..., 4], q[..., 0], q[..., 8]
        return f
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=168, normalize_inputs=True,
                                     use_continuous_quantile_head=True, force_flip_invariance=True,
                                     infer_is_positive=False, fix_quantile_crossing=True))

    def f(x):
        _, q = m.forecast(horizon=168, inputs=x)     # q[..., 0] = 평균, 1~9 = 분위수 0.1~0.9
        return q[..., 5], q[..., 1], q[..., 9]
    return f


def run(name):
    p = CH / f"{name}_{START}.csv.gz"
    if p.exists():
        print(f"{p.name}: 이미 있음 → 건너뜀")
        return
    s = load_smp_actual("육지").sort_values(["date", "hour"]).reset_index(drop=True)
    vals = s["smp"].values.astype("float32")
    days = pd.DatetimeIndex(s["date"].values[::24])
    bases = [d for d in days if d >= pd.Timestamp(START) - pd.Timedelta(days=7)]
    pos = {d: i for i, d in enumerate(days)}
    f, rows, t0 = model(name), [], time.time()
    for k in range(0, len(bases), 32):
        batch = bases[k:k + 32]
        x = [vals[(pos[I] + 1) * 24 - CTX:(pos[I] + 1) * 24] for I in batch]
        med, lo, hi = f(x)
        for j, I in enumerate(batch):
            for t in range(168):
                H, h = t // 24 + 1, t % 24 + 1
                rows.append((I + pd.Timedelta(days=H), h, H, med[j, t], lo[j, t], hi[j, t]))
        print(f"{name} {k + len(batch)}/{len(bases)} ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows, columns=["date", "hour", "H", "pred", "qlo", "qhi"])
    out[out["date"] >= START].round(3).to_csv(p, index=False, encoding="utf-8")
    print(f"저장 {p.name}", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
