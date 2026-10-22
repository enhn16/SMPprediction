"""Chronos 시계열 기초모델(제로샷)로 D일 한낮 최저 SMP 예측 → 캐시 (육지).

입력: D-1 24시까지 시간별 SMP (D-1 SMP는 D-2 저녁 공표 → D-1 15시에 확보). 우리 자료로 학습하지 않음(제로샷).
출력: data/processed/drop_classifier/chronos_{모델}.csv
  date, chr_mid_med (11~15시 예측 중앙값의 최저), chr_mid_q10 (11~15시 10% 분위 최저)
모델: amazon/chronos-bolt-small, amazon/chronos-2 (불러오기 실패 시 건너뜀)
실행: python -m drop_classifier.src.chronos_feat
"""
import warnings

import numpy as np
import pandas as pd
import torch

from common.config import PROCESSED_DIR
from common.loaders import load_smp_excel

warnings.filterwarnings("ignore")
OUT_DIR = PROCESSED_DIR / "drop_classifier"
CTX = 512
MODELS = {"bolt": "amazon/chronos-bolt-small", "c2": "amazon/chronos-2"}
SKIP_DONE = True  # 이미 저장된 모델은 건너뜀


def run(tag, name, s, dates):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)
    vals = s["smp"].values.astype("float32")
    idx = {pd.Timestamp(d): i for i, d in enumerate(s["date"].values[::24])}   # 날짜 → 시작 위치/24
    rows = []
    for k in range(0, len(dates), 64):
        batch = dates[k:k + 64]
        ctxs = []
        for d in batch:
            end = idx[pd.Timestamp(d)] * 24      # D일 0시 위치 = D-1 24시까지 사용
            ctxs.append(torch.tensor(vals[max(0, end - CTX):end]))
        q, m = pipe.predict_quantiles(ctxs, prediction_length=24, quantile_levels=[0.1, 0.5])
        for d, qq, mm in zip(batch, q, m):
            if qq.dim() == 3:   # Chronos-2는 계열마다 [변수 수, 시간, 분위] 형태
                qq, mm = qq[0], mm[0]
            rows.append({"date": d, f"{tag}_mid_med": float(mm[10:15].min()), f"{tag}_mid_q10": float(qq[10:15, 0].min())})
        print(tag, k + len(batch), "/", len(dates), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / f"chronos_{tag}.csv", index=False, encoding="utf-8-sig")
    return out


def main():
    s = load_smp_excel("육지").sort_values(["date", "hour"]).reset_index(drop=True)
    assert (s.groupby("date").size() == 24).all()
    dates = pd.date_range("2024-01-01", s["date"].max())
    for tag, name in MODELS.items():
        if SKIP_DONE and (OUT_DIR / f"chronos_{tag}.csv").exists():
            print(tag, "이미 있음 → 건너뜀")
            continue
        try:
            run(tag, name, s, dates)
        except Exception as e:
            print(tag, "실패:", repr(e)[:200], flush=True)


if __name__ == "__main__":
    main()
