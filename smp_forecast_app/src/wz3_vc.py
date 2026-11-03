"""일주일 v3 (R3) ⑤ — 변동비 순위 피처 (현업 공유 2026-10-30, 팀 13종 밖 실험용).

공개 시점 가정: S1 = 그달 1일 공개, S2 = 전달 25일 공개. 기준일 I 15시에 공개 전인 달은 마지막 공개 월 값.
  대상일 D(= I+H)에 쓰는 달 = min(D의 달, I에 알려진 마지막 달)
피처 (WEEK_V3.md ⑤)
  vs 계단  : vc_dl·vc_gl·vc_coal (직도입 LNG·일반 LNG·석탄 변동비 중앙값), vc_dl_chg·vc_gl_chg (기준일 7일 전 달 대비)
  vk 위치  : vc_stack (순부하가 닿는 발전기 변동비), vc_stack_up (+3GW 위치까지 상승폭), vc_c_vs_stack (Chronos − vc_stack, wz3_resid에서)
             순부하 = d_d1_h(기준일분 하루전 수요예측, 같은 시간) − 시장 태양광 추정(1년 전 ±14일 같은 시간 평균)
             변동비 순위는 변동비 0(수력·양수 등) 제외, 낮은 순 누적
"""
import numpy as np
import pandas as pd

from common.config import RAW_DIR
from common.loaders import load_renewables

VS = ["vc_dl", "vc_gl", "vc_coal", "vc_dl_chg", "vc_gl_chg"]
VK = ["vc_stack", "vc_stack_up", "vc_c_vs_stack"]
_cache = {}


def monthly() -> pd.DataFrame:
    """월별 중앙값 [월 ordinal → dl, gl, coal], 빠진 달은 직전 달 값."""
    if "m" not in _cache:
        s = pd.read_csv(RAW_DIR / "변동비_연료별_월별요약_202203-202611.csv", encoding="utf-8-sig")
        p = s.pivot(index="연월", columns="연료구분", values="변동비_중앙값")
        p.index = pd.PeriodIndex(p.index, freq="M")
        p = p.reindex(pd.period_range(p.index.min(), p.index.max(), freq="M")).ffill()
        _cache["m"] = pd.DataFrame({"dl": p["LNG(직도입)"], "gl": p["LNG"], "coal": p["석탄"]})
    return _cache["m"]


def stacks() -> dict:
    """월별 변동비 순위 (변동비 0 제외) → {월 ordinal: (누적출력 MW, 변동비)}, 빠진 달은 직전 달."""
    if "s" not in _cache:
        g = pd.read_csv(RAW_DIR / "변동비순위_발전기별_202203-202611.csv", encoding="utf-8-sig")
        g = g[g["변동비_원kWh"] > 0]
        out = {}
        for m, x in g.groupby("연월"):
            x = x.sort_values("변동비_원kWh")
            out[pd.Period(m, freq="M").ordinal] = (x["최대출력_MW"].cumsum().values, x["변동비_원kWh"].values)
        ks = sorted(out)
        for k in range(ks[0], ks[-1] + 1):
            if k not in out:
                out[k] = out[k - 1]
        _cache["s"] = out
    return _cache["s"]


def solar_est(dates: pd.Series, hours: pd.Series) -> np.ndarray:
    """대상일·시간의 육지 시장 태양광 추정 = 1년 전 ±14일 같은 시간 평균 (기준일보다 1년 가까이 앞선 실적만)."""
    if "pv" not in _cache:
        r = load_renewables()
        r = r[r["region"] == "육지"].pivot(index="date", columns="hour", values="solar").sort_index()
        r = r.reindex(pd.date_range(r.index.min(), r.index.max()))
        _cache["pv"] = r.rolling(29, center=True, min_periods=10).mean()
    pv = _cache["pv"]
    ago = pd.DatetimeIndex(dates) - pd.Timedelta(days=365)
    ri = pv.index.get_indexer(ago)
    ci = pv.columns.get_indexer(hours)
    v = np.where(ri >= 0, pv.values[np.clip(ri, 0, None), ci], np.nan)
    return np.nan_to_num(v, nan=0.0)


def features(d: pd.DataFrame, H: int, scen: str) -> pd.DataFrame:
    """피처 표 d(대상일 date, hour, d_d1_h 포함, 지평 H)에 vc_* 열을 붙여 돌려줌. scen = S1 | S2."""
    base = d["date"] - pd.Timedelta(days=H)
    known = base.dt.to_period("M").map(lambda p: p.ordinal) + ((base.dt.day >= 25).astype(int) if scen == "S2" else 0)
    use = np.minimum(d["date"].dt.to_period("M").map(lambda p: p.ordinal).values, known.values)
    ref = (base - pd.Timedelta(days=7)).dt.to_period("M").map(lambda p: p.ordinal).values
    m = monthly()
    mi = {p.ordinal: i for i, p in enumerate(m.index)}
    ui, rfi = np.array([mi[u] for u in use]), np.array([mi[r] for r in ref])
    d = d.copy()
    d["vc_dl"], d["vc_gl"], d["vc_coal"] = m["dl"].values[ui], m["gl"].values[ui], m["coal"].values[ui]
    d["vc_dl_chg"] = d["vc_dl"] - m["dl"].values[rfi]
    d["vc_gl_chg"] = d["vc_gl"] - m["gl"].values[rfi]
    net = d["d_d1_h"].values - solar_est(d["date"], d["hour"])
    st = stacks()
    stack, up = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for u in np.unique(use):
        k = use == u
        cum, cost = st[u]
        at = lambda x: cost[np.clip(np.searchsorted(cum, x), 0, len(cost) - 1)]
        stack[k] = at(net[k])
        up[k] = at(net[k] + 3000) - stack[k]
    d["vc_stack"], d["vc_stack_up"] = np.where(np.isnan(net), np.nan, stack), np.where(np.isnan(net), np.nan, up)
    return d
