"""일주일 v3 (R3) ⑧ K3 중심 + 팀원 범위 방식(오차 분위수, 지평×시간×평일/휴무일) — 사후 후보. 규칙은 WEEK_V3.md ⑧ (결과 보기 전 고정).

  select : 2024로 맞춘 K3 중심, 2025 범위 비교 (A1 vs B1·B2·B3) → WZ3_X7_rangeB_2025.csv
  test   : (B가 뽑혔을 때만) 2024~2025로 다시 맞춘 K3, 2026-01-01 ~ 10-18 한 번 채점 → WZ3_X8_rangeB_2026.csv
실행: python -m smp_forecast_app.src.wz3_rangeB select | test
"""
import sys

import numpy as np
import pandas as pd

from common.loaders import load_smp_actual
from smp_forecast_app.src import range_adjust
from smp_forecast_app.src.final_metrics import hourly_segments, metrics
from smp_forecast_app.src.wz3_select import (DATA, FIT, SEL, TAB, TEST, TIE_RANGE, W3, between, build_point,
                                             c2_quantiles, make_range, range_segments)

K3 = ("mix|ftf|ext", "one", False)
A1 = dict(kind="sym", src="Chronos-2", window=365)
REFIT = ("2024-01-01", "2025-12-31")
OUT1, OUT2 = TAB / "WZ3_X7_rangeB_2025.csv", TAB / "WZ3_X8_rangeB_2026.csv"
KEYS = ["H", "hour", "offday"]


def _q(err_rows):
    g = err_rows.groupby(KEYS)["err"]
    return g.quantile([0.1, 0.9]).unstack(), g.size()


def b_ranges(d, pred, ev_per):
    """d(전 지평)·pred → {B1, B2, B3: (lo, hi)}, ev_per 대상일 행만 값."""
    d = d.assign(err=d["y"].values - pred, pred=pred).reset_index(drop=True)
    ev = np.asarray(between(d, ev_per))
    out = {k: (np.full(len(d), np.nan), np.full(len(d), np.nan)) for k in ("B1", "B2", "B3")}
    hist = d[d["y"].notna()]
    for yr in sorted(d.loc[ev, "date"].dt.year.unique()):       # B1: 전년도 오차로 1년 고정
        q, _ = _q(hist[hist["date"].dt.year == yr - 1])
        rows = ev & (d["date"].dt.year == yr).values
        j = d.loc[rows, KEYS].merge(q, left_on=KEYS, right_index=True, how="left")
        out["B1"][0][rows], out["B1"][1][rows] = d.loc[rows, "pred"].values + j[0.1].values, d.loc[rows, "pred"].values + j[0.9].values
    hd, hb = hist["date"].values, None
    for I in np.unique(d.loc[ev, "base_date"].values):        # B2·B3: 기준일 I까지 확정된 오차(대상일 ≤ I)
        rows = np.where(ev & (d["base_date"].values == I))[0]
        tg = d.iloc[rows]
        for name, win, fb in (("B2", 365, False), ("B3", 90, True)):
            w = hist[(hd <= I) & (hd > I - np.timedelta64(win, "D"))]
            q, n = _q(w)
            if fb:
                q = q[n >= 20]
            j = tg[KEYS].merge(q, left_on=KEYS, right_index=True, how="left")
            lo, hi = tg["pred"].values + j[0.1].values, tg["pred"].values + j[0.9].values
            if fb:
                miss = np.isnan(lo)
                lo[miss], hi[miss] = out["B1"][0][rows][miss], out["B1"][1][rows][miss]
            out[name][0][rows], out[name][1][rows] = lo, hi
    p = d["pred"].values
    return {k: (np.clip(np.minimum(lo, p), 0, None), np.maximum(hi, p)) for k, (lo, hi) in out.items()}


def candidates(d, pred, ev_per):
    lo, hi = make_range(d, pred, A1, ev_per, c2_quantiles())
    c = {"A1 (K3 지금 범위)": (lo, hi)}
    names = {"B1": "B1 팀원 B (전년도 고정)", "B2": "B2 직전 365일", "B3": "B3 팀원 C (직전 90일)"}
    c |= {names[k]: v for k, v in b_ranges(d, pred, ev_per).items()}
    return c


def select():
    d, pred, _ = build_point(K3, FIT)
    m = between(d, SEL)
    c = candidates(d, pred, SEL)
    ok = m.copy()
    for lo, hi in c.values():
        ok &= ~np.isnan(lo) & ~np.isnan(hi)
    print(f"공통 채점 시간 {ok.sum():,} / 2025 {m.sum():,}")
    t = pd.DataFrame([{"후보": k, "시간수": int(ok.sum()), **range_segments(d, lo, hi, ok)} for k, (lo, hi) in c.items()])
    inr = t["포함률"].between(0.75, 0.85)
    best = t.loc[inr, "구간점수"].min()
    pick = t[inr & (t["구간점수"] <= best * (1 + TIE_RANGE))].iloc[0]["후보"]      # 표 순서 = A1 먼저(단순·현행)
    t["선택"] = np.where(t["후보"] == pick, "✓", "")
    t.round(4).to_csv(OUT1, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(t[["후보", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수", "휴무일 포함률", "휴무일 구간점수",
             "휴무일 한낮 포함률", "휴무일 한낮 구간점수", "D+1 포함률", "D+1 구간점수", "선택"]].round(3).to_string(index=False))
    print("선택:", pick)


def test():
    t = pd.read_csv(OUT1)
    pick = t.loc[t["선택"] == "✓", "후보"].iloc[0]
    if pick.startswith("A1"):
        print("A1 유지 → 2026은 보지 않음")
        return
    d, pred, _ = build_point(K3, REFIT)
    c = candidates(d, pred, ("2025-01-01", "2100-01-01"))
    w = d.assign(pred_k3=pred, lo_a1=c["A1 (K3 지금 범위)"][0], hi_a1=c["A1 (K3 지금 범위)"][1], lo_b=c[pick][0], hi_b=c[pick][1])
    cur = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["base_date", "date"])
    cur = range_adjust.apply(cur[cur["y"].notna()].reset_index(drop=True), "육지")
    m = cur.merge(w[["date", "hour", "H", "pred_k3", "lo_a1", "hi_a1", "lo_b", "hi_b"]], on=["date", "hour", "H"])
    m = m[m["date"].between(*TEST) & m["y"].notna()]
    act = load_smp_actual("육지")[["date", "hour", "smp"]]
    m = m.merge(act.assign(date=act["date"] + pd.Timedelta(days=7)).rename(columns={"smp": "lag7"}), on=["date", "hour"], how="left")
    m["offday"] = m["offday"].astype(bool)
    segs = hourly_segments(m) | {"D+1": (m["H"] == 1).values, "D+2~7": (m["H"] >= 2).values}
    rows = []
    for seg, s in segs.items():
        g = m[s]
        for lab, p, lo, hi in (("지금 모델", "pred", "lo", "hi"), ("K3 + A1", "pred_k3", "lo_a1", "hi_a1"),
                               (f"K3 + {pick}", "pred_k3", "lo_b", "hi_b")):
            rows.append({"구분": seg, "모델": lab, **metrics(g["y"], g[p], g[lo], g[hi], g["lag7"])})
    o = pd.DataFrame(rows)
    o.round(4).to_csv(OUT2, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(o[["구분", "모델", "표본수", "MAE", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수"]].round(3).to_string(index=False))


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
