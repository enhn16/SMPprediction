"""한 달 예측 개선(mx) v2 — LSTM·GRU 걷기 전진 백테스트 (기준일 2024-01-01 ~, 분기 첫날 재학습).

모델·입력은 지금 운영 한 달 모델과 같음 (mz.seq_data·seq_fit: 최근 56일 4지표 ÷ 28일 수준 + 7~34일 뒤 달력 → 28일 × 4지표, 시드 3개 평균).
다른 점 (팀 규칙): 학습 표본은 대상일이 2023-01-01 이후인 기준일만 (B + 7일 ≥ 2023-01-01), 그리고 34일 목표가 재학습일 전에 모두 확정된 것만.
2024 예측은 v2 선택에서 가중치·λ를 맞추는 데 쓴다 (mx_ens --v2).
출력: data/processed/smp_forecast_app/mx/seq_{lstm,gru}.csv.gz [base_date, H, date, pred_{지표}]
실행: python -m smp_forecast_app.src.mx_seq lstm   (gru는 다른 프로세스로 동시에; 이미 있으면 건너뜀)
"""
import sys
import time

import numpy as np
import pandas as pd

from common.config import SEED
from smp_forecast_app.src.mx_features import MX, STATS
from smp_forecast_app.src.mz import seq_data, seq_fit, seq_predict

START, TRAIN_FROM = "2024-01-01", pd.Timestamp("2023-01-01")


def run(kind):
    Xp, Xf, Y, L, B = seq_data()
    ps = list(pd.date_range(START, B.max(), freq="QS")) + [pd.Timestamp("2100-01-01")]
    rows, t0 = [], time.time()
    for a, b in zip(ps[:-1], ps[1:]):
        tr = np.where((B + pd.Timedelta(days=34) < a) & (B + pd.Timedelta(days=7) >= TRAIN_FROM))[0]
        te = np.where((B >= a) & (B < b))[0]
        if len(te) == 0:
            continue
        ms = [seq_fit(kind, Xp, Xf, Y, tr, SEED + sd) for sd in range(3)]
        p = seq_predict(ms, Xp, Xf, te) * L[te, None, None]
        for j, I in enumerate(B[te]):
            for h in range(28):
                rows.append({"base_date": I, "H": h + 7, **{f"pred_{s}": float(p[j, h, k]) for k, s in enumerate(STATS)}})
        print(f"{kind} {a.date()} 학습 {len(tr)}건 → 예측 {len(te)}기준일 ({time.time()-t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    out["date"] = out["base_date"] + pd.to_timedelta(out["H"], unit="D")
    out.round(3).to_csv(MX / f"seq_{kind}.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    for k in sys.argv[1:] or ["lstm", "gru"]:
        if (MX / f"seq_{k}.csv.gz").exists():
            print(f"{k}: 이미 있음 → 건너뜀")
            continue
        run(k)
