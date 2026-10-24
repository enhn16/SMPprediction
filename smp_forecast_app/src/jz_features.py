"""제주 일주일 예측용 피처 표 (판단일 I = D-H의 15시). 육지 features.py와 같은 규칙, 제주 자료로 새로 만듦.

| 묶음 | 피처 | 확보 시점 |
|---|---|---|
| 달력 | 육지와 같음 (common/calendar) | 사전 확정 |
| 제주 SMP | D-H·D-H-1·D-7 같은 시간, D-H 일평균/최저/최고/한낮최저, 7일 같은 시간 평균, 28일 평균, 직전 같은 유형일 | D-H분은 D-H-1 저녁 공표 |
| 제주 수요 | 제주 하루전 수요예측(jlfd) D-H·D-7 같은 시간, D-H 일평균·한낮 평균, 직전 같은 유형일 | D-H분은 D-H-1 저녁 공표 |
| 제주−육지 | 최근 7·28일 같은 가격 시간 비율, 최근 7일 같은 시간 차이 평균, 최근 28일 제주만 급락 일수·제주 급락 일수 | D-H까지 실적 |
| 육지 | 육지 피처 표(features.build(H), 새로 계산)의 SMP 지연값·연료비·설비·주간전망·날씨 (L_ 접두) | 육지 표와 같음 |
| 제주 날씨 | H=1~3: I일 11시(없으면 05시) 제주·서귀포 구역예보 하늘·강수확률·기온·강수, 풍향(북풍 성분) / H≥4: 06시 중기예보 '제주도' | 15시 이전 발표 |
| 제주 Open-Meteo | 4지점·모델 평균 한낮 일사·운량, 하루·한낮·새벽 10m·100m 풍속. H일 뒤에는 (H+1)일 전 실행분 (H=1~6, H=7 없음) | 판단일 이전 실행분 |
| 제주 계통 실적 | 제주계통운영정보 5분: 판단일 I 0~14:25 풍력·태양광·수요·순부하, I-1일 하루·한낮, 최근 7일 (js_) | 실시간 (I일 14:25까지, 공개 지연 여유) |
y = 제주 SMP. 저장: data/processed/smp_forecast_app/jeju/features_hourly_제주_h{H}.csv
실행: python -m smp_forecast_app.src.jz_features [H ...]   (기본 1~7)
"""
import sys

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import API_RAW_DIR, PROCESSED_DIR
from common.loaders import (load_kma_afs_jeju, load_kpx_jeju_sukub, load_mid_forecast, load_openmeteo_prev2_jeju, load_openmeteo_prevx_jeju,
                            load_smp_actual,
                            load_smp_forecast_api)
from smp_forecast_app.src.features import build as land_build

JDIR = PROCESSED_DIR / "smp_forecast_app" / "jeju"
API_SUKUB = API_RAW_DIR / "kpx_jeju_sukub"
DAYTYPE_CODE = {"평일": 0, "주말": 1, "공휴일": 2, "연휴": 3}
MID = [11, 12, 13, 14, 15]

CAL = ["hour", "dow", "month", "doy_sin", "doy_cos", "offday", "daytype", "off_block_len", "off_block_day",
       "is_myeongjeol", "is_labor_day", "prev_offday", "next_offday", "type_change"]
SMP = ["s_d1", "s_d2", "s_d7", "s_d1_mean", "s_d1_min", "s_d1_max", "s_d1_midmin", "s_ma7_h", "s_ma28",
       "s_same_h", "s_same_mean", "s_same_midmin", "s_d1_shape"]
DEM = ["d_d1_h", "d_d7_h", "d_d1_mean", "d_d1_mid", "d_same_h", "d_d1_shape"]
SPREAD = ["j_eq7", "j_eq28", "j_diff7_h", "j_drop28", "j_only28", "j_min7_mid"]
LAND = ["L_s_d1", "L_s_d1_midmin", "L_s_same_h", "L_s_ma28", "L_d_d1_h", "fc_lng", "fc_coal", "cap_solar",
        "kpx_peak", "kpx_peak_rel", "L_rf_sky_pm", "L_om_rad_mid"]
WEATHER = ["jf_sky_pm", "jf_sky_am", "jf_pop_pm", "jf_ta_pm", "jf_rain_pm", "jf_north_pm"]
OM = ["jo_rad_mid", "jo_cloud_mid", "jo_ws10_day", "jo_ws100_day", "jo_ws100_mid", "jo_ws100_night"]
SUKUB = ["js_wind_i_am", "js_solar_i_mid", "js_dem_i_mid", "js_net_i_mid", "js_share_i_mid", "js_wind_i1_day",
         "js_solar_i1_mid", "js_net_i1_mid", "js_share_i1_mid", "js_solar_max7_mid", "js_wind7", "js_net_min7_mid"]
# SUKUB(5분 계통 실적)은 과거 자료 수집이 끝나야 쓸 수 있음 (API 개발계정 하루 100회 한도, 2026-10-23 2개월분만) → 지금은 제외
FEATURES = CAL + SMP + DEM + SPREAD + LAND + WEATHER + OM

# 16방위 → 북풍 성분(바람이 불어오는 방향의 cos): 북풍 1, 남풍 −1
_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
NORTH = {d: np.cos(np.deg2rad(22.5 * i)) for i, d in enumerate(_DIRS)}


def _wide(df, col):
    return df.pivot_table(index="date", columns="hour", values=col)


def jeju_weather(H: int) -> pd.DataFrame:
    """판단일 I = D-H의 15시 이전 발표 제주 예보 → D일 [jf_*] (제주·서귀포 평균)."""
    if H <= 3:
        f = load_kma_afs_jeju()
        f["date"] = f["tm_ef"].dt.normalize()
        f["seg"] = np.where(f["tm_ef"].dt.hour == 12, "pm", "am")
        f = f[((f["date"] - f["tm_fc"].dt.normalize()).dt.days == H) & f["tm_fc"].dt.hour.isin([5, 11])]
        f = f.sort_values("tm_fc").groupby(["reg", "date", "seg"]).tail(1)   # 11시분 우선
        f["north"] = (f["wd1"].map(NORTH) + f["wd2"].map(NORTH)) / 2
        pm, am = f[f["seg"] == "pm"].groupby("date"), f[f["seg"] == "am"].groupby("date")
        return pd.DataFrame({"jf_sky_pm": pm["sky"].mean(), "jf_sky_am": am["sky"].mean(), "jf_pop_pm": pm["pop"].mean(),
                             "jf_ta_pm": pm["ta"].mean(), "jf_rain_pm": pm["rain"].mean(),
                             "jf_north_pm": pm["north"].mean()}).rename_axis("date").reset_index()
    m = load_mid_forecast()
    m = m[(m["issue_hour"] == 6) & (m["lead"] == H) & (m["region"] == "제주도")]
    pm, am = m[m["ampm"] == "오후"].groupby("target_date"), m[m["ampm"] == "오전"].groupby("target_date")
    out = pd.DataFrame({"jf_sky_pm": pm["sky"].mean(), "jf_sky_am": am["sky"].mean(), "jf_pop_pm": pm["pop"].mean(),
                        "jf_rain_pm": pm["rain"].mean()}).rename_axis("date").reset_index()
    out["jf_ta_pm"], out["jf_north_pm"] = np.nan, np.nan
    return out


def jeju_openmeteo(H: int = 1) -> pd.DataFrame:
    """제주 Open-Meteo (H+1)일 전 실행 예보 → 일별 [jo_*] (지점·모델 평균). 일사·운량은 시작 시각 11~14시, 풍속 km/h."""
    if H == 1:
        o = load_openmeteo_prev2_jeju()
    else:
        o = load_openmeteo_prevx_jeju()
        o = o[o["k"] == H + 1].drop(columns="k")
    o["date"] = o["time"].dt.normalize()
    hr = o["time"].dt.hour
    g = o.groupby(["date", "model", "point"])
    d = pd.DataFrame({
        "jo_rad_mid": o[hr.between(11, 14)].groupby(["date", "model", "point"])["rad"].sum(min_count=4),
        "jo_cloud_mid": o[hr.between(11, 14)].groupby(["date", "model", "point"])["cloud"].mean(),
        "jo_ws10_day": g["ws10"].mean(), "jo_ws100_day": g["ws100"].mean(),
        "jo_ws100_mid": o[hr.between(10, 15)].groupby(["date", "model", "point"])["ws100"].mean(),
        "jo_ws100_night": o[hr.between(0, 6)].groupby(["date", "model", "point"])["ws100"].mean(),
    })
    return d.groupby("date").mean().reset_index()


def jeju_sukub(H: int) -> pd.DataFrame:
    """제주 5분 계통 실적 → 대상일 D 행에 판단일 I = D-H의 값 [js_*]. I일은 15시 판단이라 14:25까지만 쓴다.
    순부하 = 발전단 수요 − 신재생(태양광·풍력 등), 비중 = 신재생 ÷ 수요."""
    k = load_kpx_jeju_sukub()
    k["day"] = k["time"].dt.normalize()
    k["net"] = k["demand"] - k["renew"]
    mod = k["time"].dt.hour * 60 + k["time"].dt.minute
    upto15 = k[mod <= 865]               # I일에 쓸 수 있는 부분 (0:00 ~ 14:25, 공개 지연 여유 30분)
    mid = k[mod.between(660, 865)]       # 한낮 11:00 ~ 14:25 (I일·이전 날 모두 같은 정의)
    g15, gm, gd = upto15.groupby("day"), mid.groupby("day"), k.groupby("day")
    n15 = g15.size()
    day = pd.DataFrame({
        "wind_am": g15["wind"].mean().where(n15 >= 140), "solar_mid": gm["solar"].mean(), "dem_mid": gm["demand"].mean(),
        "net_mid": gm["net"].mean(), "share_mid": gm["renew"].sum() / gm["demand"].sum(),
        "wind_day": gd["wind"].mean().where(gd.size() >= 250)})
    day.loc[gm.size().reindex(day.index).fillna(0) < 30, ["solar_mid", "dem_mid", "net_mid", "share_mid"]] = np.nan
    full = pd.date_range(day.index.min(), day.index.max())
    day = day.reindex(full)
    sh = lambda c, k_: day[c].shift(k_)
    out = pd.DataFrame({
        "js_wind_i_am": sh("wind_am", H), "js_solar_i_mid": sh("solar_mid", H), "js_dem_i_mid": sh("dem_mid", H),
        "js_net_i_mid": sh("net_mid", H), "js_share_i_mid": sh("share_mid", H),
        "js_wind_i1_day": sh("wind_day", H + 1), "js_solar_i1_mid": sh("solar_mid", H + 1),
        "js_net_i1_mid": sh("net_mid", H + 1), "js_share_i1_mid": sh("share_mid", H + 1),
        "js_solar_max7_mid": day["solar_mid"].rolling(7, min_periods=4).max().shift(H),
        "js_wind7": day["wind_day"].rolling(7, min_periods=4).mean().shift(H + 1),
        "js_net_min7_mid": day["net_mid"].rolling(7, min_periods=4).min().shift(H)}, index=full)
    return out.rename_axis("date").reset_index()


def build(H: int = 1, land_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """land_features: 육지 피처 표(features.build(H) 결과). 없으면 새로 계산한다
    (저장된 features_hourly_육지_h*.csv는 매일 갱신되지 않아 최근 며칠이 비므로 쓰지 않음, 2026-10-23)."""
    smp = load_smp_actual("제주")[["date", "hour", "smp"]]
    land = load_smp_actual("육지")[["date", "hour", "smp"]]
    dem = load_smp_forecast_api()
    dem = dem[dem["region"] == "제주"]
    S, D, L = _wide(smp, "smp"), _wide(dem, "demand_fcst"), _wide(land, "smp")
    last = S.index.max()
    days = pd.date_range("2022-01-08", last + pd.Timedelta(days=H))
    full = pd.date_range("2021-12-01", days[-1])
    S, D, L = S.reindex(full), D.reindex(full), L.reindex(full)
    cal = build_calendar(days[0] - pd.Timedelta(days=60), days[-1] + pd.Timedelta(days=1)).set_index("date")

    def shift(W, k):
        return W.shift(k).reindex(days)

    feats = {"s_d1": shift(S, H), "s_d2": shift(S, H + 1), "s_d7": shift(S, 7 if H <= 7 else 14),
             "s_ma7_h": sum(shift(S, k) for k in range(H, H + 7)) / 7,
             "d_d1_h": shift(D, H), "d_d7_h": shift(D, 7 if H <= 7 else 14)}
    S1, D1 = shift(S, H), shift(D, H)
    daily = {"s_d1_mean": S1.mean(axis=1), "s_d1_min": S1.min(axis=1), "s_d1_max": S1.max(axis=1),
             "s_d1_midmin": S1[MID].min(axis=1), "s_ma28": S.mean(axis=1).rolling(28).mean().shift(H).reindex(days),
             "d_d1_mean": D1.mean(axis=1), "d_d1_mid": D1[MID].mean(axis=1)}

    off = cal["is_offday"]
    same_idx = []
    for d in days:
        k = H
        while off.get(d - pd.Timedelta(days=k), False) != off[d]:
            k += 1
        same_idx.append(d - pd.Timedelta(days=k))
    same_idx = pd.DatetimeIndex(same_idx)
    Ssame, Dsame = S.reindex(same_idx).set_axis(days), D.reindex(same_idx).set_axis(days)
    feats["s_same_h"], feats["d_same_h"] = Ssame, Dsame
    daily["s_same_mean"] = Ssame.mean(axis=1)
    daily["s_same_midmin"] = Ssame[MID].min(axis=1)
    feats["s_d1_shape"] = S1.div(S1.mean(axis=1).clip(lower=20), axis=0)
    feats["d_d1_shape"] = D1.div(D1.mean(axis=1), axis=0)

    # 제주−육지 관계 (D-H까지 실적)
    eq = ((S - L).abs() < 0.01).mean(axis=1)
    jmid, lmid = S[MID].min(axis=1), L[MID].min(axis=1)
    jdrop, only = (jmid < 30).astype(float), ((jmid < 30) & (lmid >= 30)).astype(float)
    jdrop[jmid.isna()] = np.nan
    only[jmid.isna()] = np.nan
    daily["j_eq7"] = eq.rolling(7, min_periods=5).mean().shift(H).reindex(days)
    daily["j_eq28"] = eq.rolling(28, min_periods=20).mean().shift(H).reindex(days)
    daily["j_drop28"] = jdrop.rolling(28, min_periods=20).sum().shift(H).reindex(days)
    daily["j_only28"] = only.rolling(28, min_periods=20).sum().shift(H).reindex(days)
    daily["j_min7_mid"] = jmid.rolling(7, min_periods=5).min().shift(H).reindex(days)
    feats["j_diff7_h"] = sum(shift(S - L, k) for k in range(H, H + 7)) / 7

    long = pd.concat({k: v.stack(dropna=False) for k, v in feats.items()}, axis=1)
    long.index.names = ["date", "hour"]
    long = long.reset_index()
    long = long.merge(pd.DataFrame(daily).rename_axis("date").reset_index(), on="date", how="left")
    long["hour"] = long["hour"].astype(int)

    c = cal.reindex(days)
    cd = pd.DataFrame({
        "date": days, "dow": c["weekday"].values, "month": days.month,
        "doy_sin": np.sin(2 * np.pi * days.dayofyear / 365.25), "doy_cos": np.cos(2 * np.pi * days.dayofyear / 365.25),
        "offday": c["is_offday"].astype(int).values, "daytype": c["day_type"].map(DAYTYPE_CODE).values,
        "off_block_len": c["off_block_len"].values, "off_block_day": c["off_block_day"].values,
        "is_myeongjeol": c["is_myeongjeol"].astype(int).values, "is_labor_day": c["is_labor_day"].astype(int).values,
        "prev_offday": cal["is_offday"].shift(1).reindex(days).astype(float).values,
        "next_offday": cal["is_offday"].shift(-1).reindex(days).astype(float).values,
    })
    cd["type_change"] = (cd["offday"] != cd["prev_offday"]).astype(int)
    cd = cd.merge(jeju_weather(H), on="date", how="left")
    if (load_kpx_jeju_sukub is not None) and any((API_SUKUB).glob("jeju5m_*.csv")):
        cd = cd.merge(jeju_sukub(H), on="date", how="left")
    if H <= 6:   # (H+1)일 전 실행분 = 판단일 D-H 이전 실행 (H=1은 육지와 같은 D-2 실행분)
        cd = cd.merge(jeju_openmeteo(H), on="date", how="left")
    else:
        for k in OM:
            cd[k] = np.nan

    # 육지 피처 표 (같은 H, 같은 판단 규칙) — 읽기만 함
    lf = (land_build(H) if land_features is None else land_features)[["date", "hour", "s_d1", "s_d1_midmin", "s_same_h", "s_ma28", "d_d1_h", "fc_lng", "fc_coal",
                                "cap_solar", "kpx_peak", "kpx_peak_rel", "rf_sky_pm", "om_rad_mid"]]
    lf = lf.rename(columns={k: f"L_{k}" for k in ["s_d1", "s_d1_midmin", "s_same_h", "s_ma28", "d_d1_h", "rf_sky_pm", "om_rad_mid"]})

    out = long.merge(cd, on="date", how="left").merge(lf, on=["date", "hour"], how="left")
    out = out.merge(smp.rename(columns={"smp": "y"}), on=["date", "hour"], how="left")
    out = out.merge(land.rename(columns={"smp": "y_land"}), on=["date", "hour"], how="left")
    return out.sort_values(["date", "hour"]).reset_index(drop=True)


def path(H: int):
    return JDIR / f"features_hourly_제주_h{H}.csv"


def load(H: int = 1) -> pd.DataFrame:
    return pd.read_csv(path(H), parse_dates=["date"])


def main(hs):
    JDIR.mkdir(parents=True, exist_ok=True)
    for H in hs:
        df = build(H)
        df.to_csv(path(H), index=False, encoding="utf-8-sig")
        cov = df[df["date"] >= "2024-01-01"]
        miss = cov[FEATURES].isna().mean().round(3)
        print(f"[H={H}] {len(df):,}행 {df.date.min().date()} ~ {df.date.max().date()} · 2024~ 결측률:",
              miss[miss > 0].to_dict(), flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or range(1, 8))
