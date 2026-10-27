"""② 급등 피처 테이블 (육지) — 팀 데이터 목록 12종만 사용.

대상일 D, 판단일 I = D-1. 각 피처가 언제 확보되는지:
- 달력 (특일정보): 미리 확정
- SMP 실적: I일까지 (I일 SMP는 I-1일 저녁 공표 → I일 15시에 있음). 기준값(직전 28일 일 최고 중앙값)도 I일까지
- 하루전 수요예측(mlfd): 1차(15시)는 I일분까지. 2차(21시)는 D일분 추가 (D일 SMP와 같이 I일 저녁 공표, ROLES.md 4.4)
- 주간 전력수요전망 최대부하: I-1일까지 게시된 것 (과제1 outlook_features.attach와 같은 규칙), 2023-12 ~
- 구역별 육상예보: I일 11시 발표분(없으면 05시)의 D일 오전·오후, 육지 7개 구역 평균, 2023-12 ~
- Open-Meteo 2일 전 실행 예보: D일 한낮 일사·운량 (I-1일 이전 실행), 2024-01 ~
- 연료비단가(LNG·유연탄)·시장참여 설비용량(태양광): M월 값을 M+2월 1일부터
2023년 행은 날씨·주간전망이 없다(빈칸) → 모델 쪽에서 결측 처리.

실행: python -m surge_classifier.src.features   → data/processed/surge_classifier/features_육지.csv
"""
import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.collect.kma_regional_fcst import REGIONS
from common.loaders import (load_fuel_cost, load_kma_regional_fcst, load_market_capacity, load_openmeteo_prev2,
                            load_smp_forecast_api)
from drop_classifier.src import outlook_features
from surge_classifier.src.label_days import OUT_DATA, RATIO, SENS_RATIOS, base_28d, daily_max

CALENDAR = ["is_offday", "is_weekend", "is_holiday_any", "is_after_offday", "is_before_offday", "is_monday",
            "month_sin", "month_cos", "is_summer", "is_winter", "is_fuel_change_window"]
SMP_HIST = ["log_base", "prev_ratio", "prev_mean_ratio", "prev_spike", "max7_ratio", "same_dow_ratio",
            "surges_last14", "ratio_std14", "base_trend"]
DEMAND_PREV = ["dem_max_prev", "dem_max_prev_rel", "dem_am_prev_rel", "dem_eve_prev_rel"]
OUTLOOK = ["kpx_peak", "kpx_peak_rel", "kpx_peak_vs_prev"]
WEATHER = ["sky_pm", "sky_am", "pop_pm", "rain_pm", "ta_pm", "ta_am", "cdd", "hdd", "wk_sky_pm", "wk_cdd"]
OPENMETEO = ["rad_mid", "cloud_mid", "wk_cloud_mid"]
SUPPLY = ["cap_solar_gw", "fc_lng", "lng_coal_ratio"]
DEMAND_D = ["dem_max_d", "dem_max_d_rel", "dem_am_d_rel", "dem_eve_d_rel", "dem_max_d_vs_prev"]   # 2차(21시) 전용

AM, EVE = [8, 9, 10, 11], [17, 18, 19, 20, 21]


def _calendar(days: pd.DatetimeIndex) -> pd.DataFrame:
    cal = build_calendar(days[0] - pd.Timedelta(days=3), days[-1] + pd.Timedelta(days=3)).set_index("date")
    off = cal["is_offday"].astype(int)
    c = cal.reindex(days)
    df = pd.DataFrame(index=days)
    df["is_offday"] = c["is_offday"].astype(int)
    df["is_weekend"] = c["is_weekend"].astype(int)
    df["is_holiday_any"] = c["is_holiday"].astype(int)
    df["is_after_offday"] = ((off == 0) & (off.shift(1) == 1)).astype(int).reindex(days)
    df["is_before_offday"] = ((off == 0) & (off.shift(-1) == 1)).astype(int).reindex(days)
    df["is_monday"] = (days.dayofweek == 0).astype(int)
    m = days.month
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    df["is_summer"] = m.isin([7, 8, 9]).astype(int)
    df["is_winter"] = m.isin([12, 1, 2]).astype(int)
    df["is_fuel_change_window"] = (days.day <= 3).astype(int)
    df["day_type"] = c["day_type"]
    return df


def _smp_history(dm: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """D일 행에 I = D-1일까지의 SMP 실적 요약."""
    mx = dm["max_smp"].reindex(days)
    base = base_28d(mx)                       # D일 기준값 (D-1일까지)
    ratio = mx / base                          # D일 실제 비율 (레이블용, 피처 아님)
    lab = (ratio >= RATIO).astype(float).where(ratio.notna())
    prev_mx, prev_mean = mx.shift(1), dm["mean_smp"].reindex(days).shift(1)
    # 지난 날들의 비율은 그날 자신의 기준값으로 계산 (모두 D-1일까지의 정보)
    df = pd.DataFrame(index=days)
    df["base_28d"] = base
    df["log_base"] = np.log(base)
    df["prev_ratio"] = prev_mx / base
    df["prev_mean_ratio"] = prev_mean / base
    df["prev_spike"] = prev_mx / prev_mean
    df["max7_ratio"] = mx.shift(1).rolling(7, min_periods=5).max() / base
    df["same_dow_ratio"] = mx.shift(7) / base
    df["surges_last14"] = lab.shift(1).rolling(14, min_periods=7).sum()
    df["ratio_std14"] = ratio.shift(1).rolling(14, min_periods=7).std()
    df["base_trend"] = base / base.shift(28)
    df["max_smp"], df["ratio"] = mx, ratio
    return df


def _demand(days: pd.DatetimeIndex) -> pd.DataFrame:
    """하루전 수요예측(mlfd): 일 최대·오전·저녁. 상대값 = 그날 이전 28일 일 최대 중앙값 대비."""
    fc = load_smp_forecast_api()
    fc = fc[fc["region"] == "육지"].pivot_table(index="date", columns="hour", values="demand_fcst")
    fc = fc[fc.notna().sum(axis=1) == 24].reindex(pd.date_range(days[0] - pd.Timedelta(days=40), days[-1]))
    mx, am, eve = fc.max(axis=1), fc[AM].max(axis=1), fc[EVE].max(axis=1)
    ref = mx.shift(1).rolling(28, min_periods=20).median()   # d일 값의 기준 (d-1일까지)
    df = pd.DataFrame(index=fc.index)
    df["dem_max"], df["dem_max_rel"] = mx, mx / ref
    df["dem_am_rel"], df["dem_eve_rel"] = am / ref, eve / ref
    df["dem_ref"] = ref
    out = pd.DataFrame(index=days)
    p = df.shift(1).reindex(days)    # I = D-1일분 (1차)
    out["dem_max_prev"], out["dem_max_prev_rel"] = p["dem_max"], p["dem_max_rel"]
    out["dem_am_prev_rel"], out["dem_eve_prev_rel"] = p["dem_am_rel"], p["dem_eve_rel"]
    d = df.reindex(days)             # D일분 (2차 21시 전용)
    out["dem_max_d"] = d["dem_max"]
    out["dem_max_d_rel"], out["dem_am_d_rel"], out["dem_eve_d_rel"] = d["dem_max_rel"], d["dem_am_rel"], d["dem_eve_rel"]
    out["dem_max_d_vs_prev"] = d["dem_max"] / p["dem_max"]
    return out


def _weather(days: pd.DatetimeIndex) -> pd.DataFrame:
    """구역별 육상예보: I일 11시(없으면 05시) 발표분의 D일 오전·오후 → 7개 구역 평균 (과제1 fcst15와 같은 선택 규칙)."""
    f = load_kma_regional_fcst()
    f = f[f["reg"].isin(REGIONS)].copy()
    f["date"] = f["tm_ef"].dt.normalize()
    f["seg"] = np.where(f["tm_ef"].dt.hour == 12, "pm", "am")
    f["issue_day"] = (f["date"] - f["tm_fc"].dt.normalize()).dt.days
    f = f[(f["issue_day"] == 1) & f["tm_fc"].dt.hour.isin([5, 11])]
    f = f.sort_values("tm_fc").groupby(["reg", "date", "seg"]).tail(1)   # 11시분 우선
    g = f.groupby(["date", "seg"])[["sky", "pop", "rain", "ta"]].mean().unstack("seg")
    out = pd.DataFrame(index=days)
    out["sky_pm"], out["sky_am"] = g[("sky", "pm")].reindex(days), g[("sky", "am")].reindex(days)
    out["pop_pm"], out["rain_pm"] = g[("pop", "pm")].reindex(days), g[("rain", "pm")].reindex(days)
    out["ta_pm"], out["ta_am"] = g[("ta", "pm")].reindex(days), g[("ta", "am")].reindex(days)
    out["cdd"] = (out["ta_pm"] - 28).clip(lower=0)    # 폭염 (오후 기온)
    out["hdd"] = (0 - out["ta_am"]).clip(lower=0)     # 한파 (오전 기온)
    return out


def _openmeteo(days: pd.DatetimeIndex) -> pd.DataFrame:
    o = load_openmeteo_prev2().groupby("date")[["rad_mid", "cloud_mid"]].mean()
    return o.reindex(days)


def _supply(days: pd.DatetimeIndex) -> pd.DataFrame:
    t = pd.DataFrame({"date": days})
    cap = load_market_capacity().sort_values("usable_from")
    fuel = load_fuel_cost().sort_values("usable_from")
    t = pd.merge_asof(t, cap[["usable_from", "cap_solar"]], left_on="date", right_on="usable_from").drop(columns="usable_from")
    t = pd.merge_asof(t, fuel[["usable_from", "fc_lng", "fc_coal"]], left_on="date", right_on="usable_from")
    out = pd.DataFrame(index=days)
    out["cap_solar_gw"] = t["cap_solar"].values / 1000
    out["fc_lng"] = t["fc_lng"].values
    out["lng_coal_ratio"] = (t["fc_lng"] / t["fc_coal"]).values
    return out


def build(region: str = "육지", extend_days: int = 1) -> pd.DataFrame:
    """대상일별 피처. 마지막 SMP 실적일 + extend_days 까지 행을 만든다 (운영: 내일 행)."""
    assert region == "육지", "제주는 아직 범위 밖"
    dm = daily_max(region)
    days = pd.date_range("2022-01-01", dm.index.max() + pd.Timedelta(days=extend_days))
    df = pd.concat([_calendar(days), _smp_history(dm, days), _demand(days), _weather(days),
                    _openmeteo(days), _supply(days)], axis=1)
    ol = outlook_features.attach(pd.Series(days), pd.Series(days) - pd.Timedelta(days=1))
    df["kpx_peak"], df["kpx_peak_rel"] = ol["kpx_peak"].values, ol["kpx_peak_rel"].values
    df["kpx_peak_vs_prev"] = df["kpx_peak"] * 1000 / df["dem_max_prev"]
    wk = 1 - df["is_offday"]
    df["wk_sky_pm"], df["wk_cdd"], df["wk_cloud_mid"] = wk * df["sky_pm"], wk * df["cdd"], wk * df["cloud_mid"]
    for r in SENS_RATIOS:
        df[f"label_{round(r * 100) - 100}"] = (df["ratio"] >= r).astype(float).where(df["ratio"].notna())
    df["label"] = df["label_25"]
    df = df.rename_axis("date").reset_index()
    return df[df["date"] >= "2023-01-01"].reset_index(drop=True)


def main():
    df = build()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    out = OUT_DATA / "features_육지.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out} ({len(df)}일, {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d})")
    cols = CALENDAR + SMP_HIST + DEMAND_PREV + OUTLOOK + WEATHER + OPENMETEO + SUPPLY + DEMAND_D
    cov = df.groupby(df["date"].dt.year)[cols].apply(lambda g: g.notna().mean()).T.round(2)
    print("\n[연도별 값 있는 비율]")
    print(cov[(cov < 0.99).any(axis=1)].to_string())


if __name__ == "__main__":
    main()
