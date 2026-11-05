"""육지 급등 피처 표 2020-01 ~ (운영 급등 v3와 같은 40개 입력 + 옛 레이블 y + 새 기준 레이블).

운영 코드(surge_classifier.src.team_model2.build_daily, team_model3.data)와 같은 정의. 차이는 자료 범위뿐:
- team_model2 모듈이 쓰는 로더 이름(load_smp_actual, load_smp_forecast_api, build_calendar, weather_daily)만
  보강분 + 운영분을 합친 함수로 잠시 바꿔 끼워 build_daily를 그대로 부른다 (운영 코드는 고치지 않음).
- wk_sky_pm(평일 × 오후 하늘), kpx_peak(주간전망), wk_cloud_mid_jma(평일 × 한낮 JMA 운량)는 같은 식으로 여기서 만든다.
  JMA 운량: 보강분(2020~2023)은 일사가 2022-07부터라, '4시간 이상 있음' 판정을 운량 개수로 한다 (운영 파일은 그대로).
새 기준 레이블 y_new = 그날 '같은 시각 직전 28일 중앙값 + 20원' 이상인 시간이 있음 (events.py와 같음).

검증: python -m extended_backtest.src.surge_features check → 2023-01-01 이후 운영 v3 입력 표와 대조
저장: python -m extended_backtest.src.surge_features       → data/processed/extended_backtest/surge_features_육지.pkl
"""
import json
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

import common.calendar as cal_mod
from common.collect.kma_regional_fcst import REGIONS as SURGE_REGIONS
from common.collect.openmeteo_prevruns import CITIES
from common.config import API_RAW_DIR, PROCESSED_DIR, RAW_DIR
from surge_classifier.src import team_model2 as T2
from surge_classifier.src import team_model2_fa as FA
from extended_backtest.src.data import BF, load_holidays_all, load_smp_all
from extended_backtest.src.drop_features import kma_regional_all, outlook_attach, smp_forecast_all
from extended_backtest.src.events import hourly_flags

OUT = PROCESSED_DIR / "extended_backtest"
FIRST = "2020-01-01"
CFG = FA.CANDS["C A+B"]   # v3 입력 구성 (40개). 휴무일 0 규칙은 쓰지 않는다 (2026-11-06 사용자: 전체 날).


def _smp_actual(region):
    return load_smp_all(region)


def _calendar(start, end):
    orig = cal_mod.load_holidays
    cal_mod.load_holidays = load_holidays_all
    try:
        return cal_mod.build_calendar(start, end)
    finally:
        cal_mod.load_holidays = orig


def _read_prev2(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        for site, loc in zip(CITIES, json.loads(p.read_text(encoding="utf-8"))):
            w = pd.DataFrame(loc["hourly"])
            w.columns = [c.replace("_previous_day2", "") for c in w.columns]
            rows.append(w.assign(site=site))
    return pd.concat(rows)


def _weather_daily() -> pd.DataFrame:
    """T2.weather_daily와 같은 집계.
    보강 구간: best(2020~2023)가 빈 칸은 JMA 2일 전 실행분으로 채움 (한국 best_match = JMA, 2023년 차이 0 — backfill 인계).
    그 위에 운영 파일(수정 시각 순)이 같은 시각을 덮어씀 (운영과 같음)."""
    best = _read_prev2(sorted((BF / "openmeteo_best").glob("prev2_best_*.json"))).drop_duplicates(["time", "site"], keep="last")
    jma = _read_prev2(sorted((BF / "openmeteo_jma").glob("prev2_jma_*.json"))).drop_duplicates(["time", "site"], keep="last")
    bf = best.set_index(["time", "site"]).combine_first(jma.set_index(["time", "site"])[best.columns.drop(["time", "site"])])
    bf = bf.reset_index()
    ops = _read_prev2(sorted((T2.EXTRA_RAW / "openmeteo_best").glob("prev2_best_*.json"), key=lambda p: p.stat().st_mtime))
    w = pd.concat([bf, ops]).drop_duplicates(["time", "site"], keep="last")
    w["date"] = w["time"].str[:10]
    site = w.groupby(["date", "site"]).agg(tmax=("temperature_2m", "max"), tmin=("temperature_2m", "min"),
                                           tmean=("temperature_2m", "mean"), cloud=("cloud_cover", "mean"),
                                           precip=("precipitation", "sum"), n=("temperature_2m", "count"))
    site = site[site["n"] >= 20]
    g = site.groupby("date")
    out = pd.DataFrame({"wx_tmax": g["tmax"].mean(), "wx_tmin": g["tmin"].mean(), "wx_tmean": g["tmean"].mean(),
                        "wx_cloud": g["cloud"].mean(), "wx_precip": g["precip"].mean()})
    out["wx_hdd"] = (18 - out["wx_tmean"]).clip(lower=0)
    out["wx_cdd"] = (out["wx_tmean"] - 24).clip(lower=0)
    return out


def jma_cloud_mid() -> pd.Series:
    """extra_features.load_openmeteo_jma의 cloud_mid_jma (11~14시 운량 평균, 7개 도시 평균, 뒤 파일 우선) + 보강분."""
    rows = []
    paths = [(p, "cloud") for p in sorted((BF / "openmeteo_jma").glob("prev2_jma_*.json"))] + \
        [(p, "rad") for p in sorted((API_RAW_DIR / "openmeteo_2023_backfill").glob("prev2_jma*.json"))] + \
        [(p, "rad") for p in sorted((API_RAW_DIR / "openmeteo").glob("prev2_jma*.json"))]
    for i, (p, count_on) in enumerate(paths):
        data = json.loads(p.read_text(encoding="utf-8"))
        for loc in (data if isinstance(data, list) else [data]):
            h = pd.DataFrame(loc["hourly"])
            h["time"] = pd.to_datetime(h["time"])
            h = h[h["time"].dt.hour.between(11, 14)]
            col = "shortwave_radiation_previous_day2" if count_on == "rad" else "cloud_cover_previous_day2"
            g = h.groupby(h["time"].dt.normalize()).agg(cloud=("cloud_cover_previous_day2", "mean"), n=(col, "count"))
            g.loc[g["n"] < 4, "cloud"] = np.nan
            rows.append(g.assign(order=i))
    d = pd.concat(rows).rename_axis("date").reset_index()
    d = d.groupby(["date", "order"])["cloud"].mean().reset_index()
    return d.sort_values(["date", "order"]).groupby("date")["cloud"].last()


def sky_pm(days: pd.DatetimeIndex) -> pd.Series:
    """surge features._weather의 sky_pm (D-1 11시, 없으면 05시 발표 오후 하늘, 7개 구역 평균)."""
    f = kma_regional_all()
    f = f[f["reg"].isin(SURGE_REGIONS)].copy()
    f["date"] = f["tm_ef"].dt.normalize()
    f["seg"] = np.where(f["tm_ef"].dt.hour == 12, "pm", "am")
    f["issue_day"] = (f["date"] - f["tm_fc"].dt.normalize()).dt.days
    f = f[(f["issue_day"] == 1) & f["tm_fc"].dt.hour.isin([5, 11])]
    f = f.sort_values("tm_fc").groupby(["reg", "date", "seg"]).tail(1)
    return f[f["seg"] == "pm"].groupby("date")["sky"].mean().reindex(days)


@contextmanager
def patched_t2():
    names = {"load_smp_actual": _smp_actual, "load_smp_forecast_api": smp_forecast_all,
             "build_calendar": _calendar, "weather_daily": _weather_daily}
    old = {k: getattr(T2, k) for k in names}
    for k, v in names.items():
        setattr(T2, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(T2, k, v)


def build() -> pd.DataFrame:
    with patched_t2():
        d = T2.build_daily(extend_days=0)
    days = pd.DatetimeIndex(d["date"])
    wk = 1 - d["is_off"].values
    d["wk_sky_pm"] = wk * sky_pm(days).values
    d["kpx_peak"] = outlook_attach(pd.Series(days), pd.Series(days) - pd.Timedelta(days=1))["kpx_peak"].values
    d["wk_cloud_mid_jma"] = wk * d["date"].map(jma_cloud_mid()).values
    # 새 기준 레이블 (날짜 단위: 급등 구간이 하루에 하나라도 있으면 1)
    h = hourly_flags("land_surge")
    y_new = h.groupby("date")["flag"].any().astype(float)
    d["y_new"] = d["date"].map(y_new)
    return d[d["date"] >= FIRST].reset_index(drop=True)


def feats() -> list[str]:
    FA.configure(CFG)
    try:
        return list(T2.FEATS)
    finally:
        FA.configure({})


def check(d: pd.DataFrame):
    from surge_classifier.src.team_model3 import data as ops_data
    ops = ops_data(extend_days=0)
    cols = ["y"] + feats()
    a, b = d.set_index("date"), ops.set_index("date")
    idx = a.index.intersection(b.index)
    idx = idx[(idx >= "2023-03-01")]          # 운영 표는 2022-01부터라 90·365일 창이 2023 초에 덜 참
    print(f"대조 {len(idx)}일 ({idx.min().date()} ~ {idx.max().date()})")
    bad = []
    for c in cols:
        x, y = a.loc[idx, c].astype(float), b.loc[idx, c].astype(float)
        same = np.isclose(x, y, rtol=1e-6, atol=1e-6, equal_nan=True)
        if not same.all():
            bad.append(c)
            dd = idx[~same]
            print(f"  다름 {c}: {len(dd)}일 ({dd.min().date()} ~ {dd.max().date()}), 예 {dd[0].date()} 우리 {x[dd[0]]:.4g} / 운영 {y[dd[0]]:.4g}")
    print("모두 같음" if not bad else f"다른 열 {len(bad)}개")
    print("\n연도별 빈칸 비율(%), 빈칸 있는 열만:")
    t = (d[cols].isna().groupby(d["date"].dt.year).mean() * 100).round(0).T
    print(t[t.max(axis=1) > 0].to_string())
    print("\n연도별 레이블: 옛(×1.25) 일수 / 새(+20원) 일수")
    print(d.groupby(d["date"].dt.year)[["y", "y_new"]].sum().to_string())


if __name__ == "__main__":
    d = build()
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check(d)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        d.to_pickle(OUT / "surge_features_육지.pkl")
        print(f"저장: {len(d)}일")
