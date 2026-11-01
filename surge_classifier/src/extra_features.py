"""급등 추가 피처 후보 (13종 밖 포함, 2026-10-28 수집분) — 대상일 D, 판단 = D-1 15시.

- 공급능력 (EPSIS 일별 최대전력기준 실적, collect_extra.supply): D-1 최대는 저녁일 수 있어 **D-2일분**만 사용
    sup_rr_d2      공급예비율(%)
    sup_avail_d2   공급능력 / 설비용량 (정비·고장으로 빠진 정도)
    sup_rel_d2     공급능력 / 직전 28일 중앙값
- 발전원별 5분 실적 (2023: surge_extra, 2024~: 공용 폴더): **D-1 00:00 ~ 14:55**
    nuc_rel        원자력 평균 / 직전 28일(D-2까지) 일평균 중앙값 (정비·불시정지)
    nuc_gw         원자력 평균(GW)
    gas_rel        가스 평균 / 직전 28일 중앙값
- Open-Meteo JMA 한낮 운량·일사 (2023 과거 실행분 포함, 해마다 같은 모델)
    cloud_mid_jma, rad_mid_jma, wk_cloud_mid_jma (평일만)
- 기상특보 이력 (API허브 wrn_met_data, collect_extra.warn): **D-1 15시까지 입력(TM_IN)된 특보**로 그 시각 육지 구역 상태
    cold_n / cold_warn_n   한파 주의보 이상 / 경보 구역 수 (발효 예정 포함)
    heat_n / heat_warn_n   폭염 주의보 이상 / 경보 이상 구역 수

실행: python -m surge_classifier.src.extra_features → data/processed/surge_classifier/extra_features_육지.csv
"""
import json
import re

import numpy as np
import pandas as pd

from common.config import API_RAW_DIR, RAW_DIR
from surge_classifier.src.collect_extra import OUT as EXTRA_RAW
from surge_classifier.src.label_days import OUT_DATA

PS_COLS = ["date", "time", "solar_btm", "solar_ppa", "solar_mkt", "pump", "hydro", "gas", "wind", "renew", "ess",
           "renew_x", "new_etc", "oil", "coal_dom", "coal_bit", "coal", "nuclear"]


def load_supply() -> pd.DataFrame:
    recs = []
    for p in sorted((EXTRA_RAW / "epsis_supply").glob("epsis_supply_*.txt")):
        t = p.read_text(encoding="utf-8")
        # 각 날: c1·c3·c4·c5·c6 값 다음에 gridData.push({"year":..,"month":..,"day":..})
        for m in re.finditer(r'c1 = textFormmat\("([\d.]*)".*?c3 = textFormmat\("([\d.]*)".*?c4 = textFormmat\("([\d.]*)"'
                             r'.*?c5 = textFormmat\("([\d.\-]*)".*?c6 = textFormmat\("([\d.\-]*)".*?c7 = textFormmat\("([^"]*)"'
                             r'.*?"year":"(\d+)",\s*"month":"(\d+)",\s*"day":"(\d+)"', t, re.S):
            c1, c3, c4, c5, c6, c7, y, mo, d = m.groups()
            recs.append({"date": pd.Timestamp(int(y), int(mo), int(d)), "cap": float(c1), "supply": float(c3),
                         "peak": float(c4), "reserve": float(c5), "rr": float(c6), "peak_time": c7})
    return pd.DataFrame(recs).drop_duplicates("date").set_index("date").sort_index()


def load_power_source() -> pd.DataFrame:
    frames = []
    files = sorted((EXTRA_RAW / "kpx_power_source").glob("ps_*.xlsx")) + sorted((API_RAW_DIR / "kpx_power_source").glob("ps_*.xlsx"))
    for p in files:
        frames.append(pd.read_excel(p, header=None, skiprows=2, names=PS_COLS))
    d = pd.concat(frames, ignore_index=True)
    d["ts"] = pd.to_datetime(d["date"].astype(str) + " " + d["time"].astype(str), errors="coerce")
    d = d.dropna(subset=["ts"]).drop_duplicates("ts")
    for c in ("gas", "nuclear"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = d["ts"].dt.normalize()
    return d[["ts", "date", "gas", "nuclear"]].sort_values("ts").reset_index(drop=True)


def load_openmeteo_jma() -> pd.DataFrame:
    """JMA 2일 전 실행분 → [date, rad_mid_jma, cloud_mid_jma] (loaders.load_openmeteo_prev2와 같은 집계, JMA만)."""
    rows = []
    paths = sorted((API_RAW_DIR / "openmeteo_2023_backfill").glob("prev2_jma*.json")) + \
        sorted((API_RAW_DIR / "openmeteo").glob("prev2_jma*.json"))
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        for loc in (data if isinstance(data, list) else [data]):
            h = pd.DataFrame(loc["hourly"])
            h["time"] = pd.to_datetime(h["time"])
            h = h[h["time"].dt.hour.between(11, 14)]
            g = h.groupby(h["time"].dt.normalize()).agg(rad=("shortwave_radiation_previous_day2", "sum"),
                                                       cloud=("cloud_cover_previous_day2", "mean"),
                                                       n=("shortwave_radiation_previous_day2", "count"))
            g.loc[g["n"] < 4, ["rad", "cloud"]] = np.nan
            rows.append(g.assign(src=p.name))
    d = pd.concat(rows).rename_axis("date").reset_index()
    # 같은 날이 여러 파일에 있으면 뒤 파일(최근 수집) 값을 도시별로 쓰지 않고 날짜·파일별 7개 도시 평균 후 마지막 파일
    d = d.groupby(["date", "src"]).agg(rad_mid_jma=("rad", "mean"), cloud_mid_jma=("cloud", "mean")).reset_index()
    return d.sort_values(["date", "src"]).groupby("date").last()[["rad_mid_jma", "cloud_mid_jma"]]


def load_warnings() -> pd.DataFrame:
    """특보 이력 → [tm_in, reg, wrn, lvl, cmd] (육지 구역 = L1로 시작, 제주 L109 제외; 한파 C·폭염 H만)."""
    rows = []
    for p in sorted((EXTRA_RAW / "kma_warning").glob("wrn_2*.txt")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line[:1].isdigit():
                rows.append([x.strip() for x in line.split(",")[:8]])
    d = pd.DataFrame(rows, columns=["tm_fc", "tm_ef", "tm_in", "stn", "reg", "wrn", "lvl", "cmd"]).drop_duplicates()
    d = d[d["reg"].str.startswith("L1") & ~d["reg"].str.startswith("L109") & d["wrn"].isin(["C", "H"])]
    d["tm_in"] = pd.to_datetime(d["tm_in"], format="%Y%m%d%H%M")
    d["lvl"], d["cmd"] = d["lvl"].astype(int), d["cmd"].astype(int)
    return d[["tm_in", "reg", "wrn", "lvl", "cmd"]].sort_values("tm_in").reset_index(drop=True)


def warning_state(days: pd.DatetimeIndex) -> pd.DataFrame:
    """대상일 D마다 D-1 15:00 시점 구역별 마지막 특보 명령 → 발표·대치·연장·변경(1·2·5·6)이면 걸린 상태."""
    w = load_warnings()
    keys = w[["reg", "wrn"]].drop_duplicates()
    cut = pd.DataFrame({"date": days, "tm_in": days - pd.Timedelta(hours=9)})        # D-1 15:00
    grid = cut.merge(keys, how="cross").sort_values("tm_in")
    st = pd.merge_asof(grid, w, on="tm_in", by=["reg", "wrn"], direction="backward")
    st["on"] = st["cmd"].isin([1, 2, 5, 6])
    out = pd.DataFrame(index=days)
    for code, name in (("C", "cold"), ("H", "heat")):
        g = st[st["wrn"] == code]
        out[f"{name}_n"] = g[g["on"]].groupby("date").size().reindex(days, fill_value=0)
        out[f"{name}_warn_n"] = g[g["on"] & (g["lvl"] >= 3)].groupby("date").size().reindex(days, fill_value=0)
    return out


def build(days: pd.DatetimeIndex, is_offday: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=days)
    s = load_supply().reindex(pd.date_range(days[0] - pd.Timedelta(days=40), days[-1]))
    s2 = s.shift(2)
    out["sup_rr_d2"] = s2["rr"].reindex(days)
    out["sup_avail_d2"] = (s2["supply"] / s2["cap"]).reindex(days)
    out["sup_rel_d2"] = (s2["supply"] / s2["supply"].rolling(28, min_periods=20).median()).reindex(days)

    ps = load_power_source()
    am = ps[ps["ts"].dt.hour < 15].groupby("date")[["nuclear", "gas"]].mean()      # 00:00 ~ 14:55
    full = ps.groupby("date")[["nuclear", "gas"]].mean()
    idx = pd.date_range(days[0] - pd.Timedelta(days=40), days[-1])
    am, full = am.reindex(idx), full.reindex(idx)
    ref = full.shift(2).rolling(28, min_periods=20).median()                        # D-2까지의 일평균
    am1 = am.shift(1)                                                                # D-1 오전~14:55
    out["nuc_gw"] = (am1["nuclear"] / 1000).reindex(days)
    out["nuc_rel"] = (am1["nuclear"] / ref["nuclear"]).reindex(days)
    out["gas_rel"] = (am1["gas"] / ref["gas"]).reindex(days)

    om = load_openmeteo_jma().reindex(days)
    out["cloud_mid_jma"], out["rad_mid_jma"] = om["cloud_mid_jma"], om["rad_mid_jma"]
    out["wk_cloud_mid_jma"] = (1 - is_offday.reindex(days).values) * out["cloud_mid_jma"]
    out = out.join(warning_state(days))
    return out


def main():
    f = pd.read_csv(OUT_DATA / "features_육지.csv", parse_dates=["date"]).set_index("date")
    ex = build(f.index, f["is_offday"])
    out = OUT_DATA / "extra_features_육지.csv"
    ex.rename_axis("date").reset_index().to_csv(out, index=False, encoding="utf-8-sig")
    cov = ex.groupby(ex.index.year).apply(lambda g: g.notna().mean()).T.round(2)
    print(f"저장: {out} ({len(ex)}일)\n[연도별 값 있는 비율]\n{cov.to_string()}")


if __name__ == "__main__":
    main()
