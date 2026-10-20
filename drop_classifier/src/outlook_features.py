"""전력거래소 주간 전력수요전망 → 대상일 피처 (판단 시점에 공개된 전망만 사용).

공개 시점 = 게시판 등록일(posted). 등록 시각을 모르므로 보수적으로 '등록일 다음 날부터' 사용 가능하다고 본다.
- 하루 전 모델(D-1 17시 판단): posted ≤ D-2
- 일주일 모델(판단일 I 18시): posted ≤ I-1
같은 대상일을 다룬 전망이 여러 개면 조건을 만족하는 가장 최근 등록분을 쓴다.

피처
- kpx_peak: 대상일 예측 최대부하(GW)
- kpx_peak_rel: 같은 전망 안 최댓값 대비 비율 (연휴·주말에 수요가 얼마나 꺼지는지)
"""
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR

PATH = PROCESSED_DIR / "drop_classifier" / "kpx_weekly_outlook.csv"
OUTLOOK = ["kpx_peak", "kpx_peak_rel"]


def load() -> pd.DataFrame:
    df = pd.read_csv(PATH, parse_dates=["written", "target_date", "posted"])
    df = df.dropna(subset=["peak_gw"])
    df["kpx_peak_rel"] = df["peak_gw"] / df.groupby("list_no")["peak_gw"].transform("max")
    return df.rename(columns={"peak_gw": "kpx_peak"})


def attach(targets: pd.Series, decision_dates: pd.Series, out: pd.DataFrame | None = None) -> pd.DataFrame:
    """대상일·판단일 쌍마다, 판단일 전날까지 등록된 전망 중 가장 최근 것의 피처를 붙인다."""
    ol = load().sort_values("posted")
    by_target = {d: g for d, g in ol.groupby("target_date")}
    rows = []
    for t, dec in zip(targets, decision_dates):
        g = by_target.get(t)
        r = {c: np.nan for c in OUTLOOK}
        if g is not None:
            ok = g[g["posted"] <= dec - pd.Timedelta(days=1)]
            if len(ok):
                last = ok.iloc[-1]
                r = {c: last[c] for c in OUTLOOK}
        rows.append(r)
    return pd.DataFrame(rows, index=targets.index)
