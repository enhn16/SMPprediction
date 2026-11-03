"""한 달 예측 mx3 — 시간별 표 (행 = 기준일 I × H(7~34일) × 시간 1~24, 대상일 D = I + H). 판단 시각 I일 15시.

시간별 피처 (모두 기준일 I까지 확정된 값만; I일 SMP는 I-1 저녁 공표):
| 피처 | 내용 |
|---|---|
| h_same4·h_same8 | 대상일과 같은 유형(평일/휴무일) 최근 4·8일의 같은 시간 평균 |
| h_l7·h_l28 | 최근 7·28일 같은 시간 평균 |
| h_shape4·h_shape28 | 위 값 ÷ 그 기간 일평균 (하루 모양) |
| h_ly | 작년 대상일(364일 전) ±3일 같은 시간 평균 |
| h_seasN | 같은 시간의 연료 정규화 여러 해 계절비 ('SMP ÷ 그 달 LNG'의 대상일 ±7일 ÷ 기준일 직전 28일, 1~3년 전 평균) |
| h_pv_exp | 작년 대상일 ±3일 같은 시간 태양광 × 최근 성장률 (태양광 실적은 I−56일까지만) |
| hol_cls | 달력 분류: 0 평일, 1 주말, 2 단독 공휴일, 3 명절 아닌 연휴, 4 명절 연휴 (특일정보로 미리 앎) |
+ mx 일별 피처(mx_features, 일평균 기준 수준·계절비·수요·연료·태양광 설비 등)를 (base_date, H)로 붙임.
저장: data/processed/smp_forecast_app/mx/mx3_hourly.pkl  (기준일 2022-12-01 ~, 대상 실적 y는 있는 곳만)
실행: python -m smp_forecast_app.src.mx3_data
"""
import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.loaders import load_fuel_cost, load_renewables, load_smp_actual
from smp_forecast_app.src.mx_features import MX, PV_LAG, load, multi_year
from common.config import API_RAW_DIR

OUT = MX / "mx3_hourly.pkl"
LEADS = range(7, 35)
HOURS = np.arange(1, 25)
DAILY = ["mean_same4", "mean_same8", "mean_l7", "mean_l28", "mean_l91", "mean_trend", "mean_ly", "mean_seasN", "mean_seasM",
         "mean_fuelimp", "mean_r_now", "mid_same4", "eve_same4", "midmin_same4", "seas_dem_m", "dem_l28", "pv_exp", "pv_share",
         "pvmid_exp", "cap_solar", "fc_lng", "fc_lng_chg1", "mon_cross", "dom", "dow", "month", "doy_sin", "doy_cos", "offday",
         "day_type", "off_block_len", "off_block_day", "is_myeongjeol", "prev_offday", "next_offday"]


def pv_hourly():
    """육지 시간별 태양광 (date × hour)."""
    r = load_renewables()
    r = r[r["region"] == "육지"][["date", "hour", "solar"]]
    fs = sorted((API_RAW_DIR / "kpx_pv_region").glob("pv_region_*.csv"))
    if fs:
        a = pd.concat([pd.read_csv(f, dtype={"tradeYmd": str}) for f in fs])
        a = a[~a["regionNm"].astype(str).str.contains("제주")]
        a = a.assign(date=pd.to_datetime(a["tradeYmd"]), hour=a["tradeNo"].astype(int)).groupby(["date", "hour"])["amgo"].sum()
        a = a.rename("solar").reset_index()
        r = pd.concat([r[r["date"] < a["date"].min()], a])
    return r.pivot_table(index="date", columns="hour", values="solar")


def build(d=None):
    """d: 일별 mx 피처 (없으면 저장 파일 = 기준일 2022-12-01 ~). 운영에서는 그날 기준일 행만 넘김."""
    d = load() if d is None else d
    s = load_smp_actual("육지")
    W = s.pivot_table(index="date", columns="hour", values="smp")
    full = pd.date_range("2021-01-01", d["date"].max() + pd.Timedelta(days=1))
    W = W.reindex(full)
    cal = build_calendar(full[0], full[-1]).set_index("date").reindex(full)
    off = cal["is_offday"].astype(bool)
    dm = W.mean(axis=1)
    fc = load_fuel_cost().set_index("month")
    lng = pd.Series(fc["fc_lng"].reindex(full.to_period("M").to_timestamp()).values, index=full)
    pv = pv_hourly().reindex(full)
    p28 = pv.sum(axis=1, min_count=20).rolling(28, min_periods=20).mean()

    B = pd.DatetimeIndex(d["base_date"])
    T = pd.DatetimeIndex(d["date"])
    offt = d["offday"].values == 1
    at = lambda ser, idx: ser.reindex(idx).values
    known = B - pd.Timedelta(days=PV_LAG)
    growth = at(p28, known) / at(p28, known - pd.Timedelta(days=364))

    # 같은 유형 최근 n일 (일평균·시간별)
    def same(x, n):
        o = {k: x[off == bool(k)].rolling(n, min_periods=n - 1).mean().reindex(full).ffill() for k in (0, 1)}
        return o
    dm_s4, dm_l28 = same(dm, 4), dm.rolling(28, min_periods=20).mean()
    parts = []
    for h in HOURS:
        x = W[h]
        s4, s8 = same(x, 4), same(x, 8)
        f = pd.DataFrame({"base_date": d["base_date"].values, "H": d["H"].values, "hour": h})
        f["h_same4"] = np.where(offt, at(s4[1], B), at(s4[0], B))
        f["h_same8"] = np.where(offt, at(s8[1], B), at(s8[0], B))
        f["h_l7"] = at(x.rolling(7, min_periods=5).mean(), B)
        f["h_l28"] = at(x.rolling(28, min_periods=20).mean(), B)
        f["h_shape4"] = f["h_same4"] / np.where(offt, at(dm_s4[1], B), at(dm_s4[0], B))
        f["h_shape28"] = f["h_l28"] / at(dm_l28, B)
        f["h_ly"] = at(x.rolling(7, center=True, min_periods=5).mean(), T - pd.Timedelta(days=364))
        r = x / lng
        f["h_seasN"] = multi_year(r.rolling(15, center=True, min_periods=10).mean(), r.rolling(28, min_periods=20).mean(), T, B)
        f["h_pv_exp"] = at(pv[h].rolling(7, center=True, min_periods=5).mean(), T - pd.Timedelta(days=364)) * growth
        f["y"] = at(x, T)
        parts.append(f)
    h = pd.concat(parts, ignore_index=True)
    dd = d[["base_date", "H", "date"] + DAILY].copy()
    dt = dd["day_type"]
    dd["hol_cls"] = np.select([dd["is_myeongjeol"] == 1, dt == "연휴", dt == "공휴일", dt == "주말"], [4, 3, 2, 1], 0)
    h = h.merge(dd, on=["base_date", "H"], how="left")
    num = h.select_dtypes("float64").columns
    h[num] = h[num].astype("float32")
    return h.sort_values(["base_date", "H", "hour"]).reset_index(drop=True)


OUT_X = MX / "mx3_hourly_x.pkl"


def build_extra():
    """추가 피처 (mx3_hourly_x.pkl, 같은 행 순서). 모두 기준일 I까지 확정된 값 (수요예측은 I일분까지, 태양광은 I−56일까지).
    | x_q10·x_q90·x_min28 | 최근 28일 같은 시간 10%·90% 분위수·최솟값 |
    | x_low28 | 최근 28일 같은 시간 30원 미만 비율 (급락 빈도) |
    | x_sd8 | 같은 유형 최근 8일 같은 시간 표준편차 |
    | x_ly_shape | 작년 대상일 ±3일 같은 시간 ÷ 그 기간 일평균 (작년 하루 모양) |
    | x_seasM | 같은 시간 SMP 여러 해 계절비 (연료 정규화 없이) |
    | x_dem_l28·x_dem_seas | 최근 28일 같은 시간 수요예측, 같은 시간 수요 여러 해 계절비 |
    | x_netload | 예상 수요(x_dem_l28 × x_dem_seas) − 예상 태양광(h_pv_exp) |
    | x_pv_ratio | 예상 태양광 ÷ 예상 수요 |
    | x_vol28 | 최근 28일 일평균 하루 변화 절대값 평균 |
    | kpx_peak_rel·rf_sky_pm·rf_pop_pm | 주간 최대부하 전망·중기예보 (H≤10만, 나머지 빈칸) |
    | fc_coal·fc_lng_chg3 | 판단일에 쓸 수 있는 유연탄 단가·LNG 3개월 변화 (M+2) |"""
    from common.loaders import load_smp_forecast_api
    h = load_hourly()[["base_date", "H", "hour", "date", "h_pv_exp"]]
    d = load()
    s = load_smp_actual("육지")
    W = s.pivot_table(index="date", columns="hour", values="smp")
    full = pd.date_range("2021-01-01", d["date"].max() + pd.Timedelta(days=1))
    W = W.reindex(full)
    cal = build_calendar(full[0], full[-1]).set_index("date").reindex(full)
    off = cal["is_offday"].astype(bool)
    dem = load_smp_forecast_api()
    dem = dem[dem["region"] == "육지"].pivot_table(index="date", columns="hour", values="demand_fcst").reindex(full)
    dm = W.mean(axis=1)
    vol = dm.diff().abs().rolling(28, min_periods=20).mean()
    B, T = pd.DatetimeIndex(d["base_date"]), pd.DatetimeIndex(d["date"])
    offt = d["offday"].values == 1
    at = lambda ser, idx: ser.reindex(idx).values
    dm_ly = dm.rolling(7, center=True, min_periods=5).mean()
    parts = []
    for hr in HOURS:
        x = W[hr]
        f = pd.DataFrame({"base_date": d["base_date"].values, "H": d["H"].values, "hour": hr})
        r28 = x.rolling(28, min_periods=20)
        f["x_q10"], f["x_q90"], f["x_min28"] = at(r28.quantile(0.1), B), at(r28.quantile(0.9), B), at(r28.min(), B)
        f["x_low28"] = at((x < 30).astype(float).where(x.notna()).rolling(28, min_periods=20).mean(), B)
        sd = {k: x[off == bool(k)].rolling(8, min_periods=6).std().reindex(full).ffill() for k in (0, 1)}
        f["x_sd8"] = np.where(offt, at(sd[1], B), at(sd[0], B))
        f["x_ly_shape"] = at(x.rolling(7, center=True, min_periods=5).mean() / dm_ly, T - pd.Timedelta(days=364))
        f["x_seasM"] = multi_year(x.rolling(15, center=True, min_periods=10).mean(), x.rolling(28, min_periods=20).mean(), T, B)
        dh = dem[hr]
        d28 = dh.rolling(28, min_periods=20).mean()
        f["x_dem_l28"] = at(d28, B)
        f["x_dem_seas"] = multi_year(dh.rolling(15, center=True, min_periods=10).mean(), d28, T, B)
        f["x_vol28"] = at(vol, B)
        parts.append(f)
    x = pd.concat(parts, ignore_index=True)
    x = h[["base_date", "H", "hour", "h_pv_exp"]].merge(x, on=["base_date", "H", "hour"], how="left")
    x["x_netload"] = x["x_dem_l28"] * x["x_dem_seas"] - x["h_pv_exp"]
    x["x_pv_ratio"] = x["h_pv_exp"] / (x["x_dem_l28"] * x["x_dem_seas"])
    x = x.merge(d[["base_date", "H", "kpx_peak_rel", "rf_sky_pm", "rf_pop_pm", "fc_coal", "fc_lng_chg3"]], on=["base_date", "H"],
                how="left").drop(columns=["h_pv_exp"])
    num = x.select_dtypes("float64").columns
    x[num] = x[num].astype("float32")
    return x


def load_hourly(extra=False) -> pd.DataFrame:
    h = pd.read_pickle(OUT)
    if extra:
        x = pd.read_pickle(OUT_X)
        assert (x["base_date"].values == h["base_date"].values).all() and (x["hour"].values == h["hour"].values).all()
        h = pd.concat([h, x.drop(columns=["base_date", "H", "hour"])], axis=1)
    return h


def main():
    if "--extra" in __import__("sys").argv[1:]:
        x = build_extra()
        x.to_pickle(OUT_X)
        print(f"저장: {OUT_X} ({len(x):,}행)", x.isna().mean().round(3)[lambda v: v > 0].to_dict())
        return
    h = build()
    h.to_pickle(OUT)
    print(f"저장: {OUT} ({len(h):,}행, 기준일 {h.base_date.min().date()} ~ {h.base_date.max().date()})")
    c = h[h["date"] >= "2023-01-01"]
    print("2023~ 결측률:", c.isna().mean().round(3)[lambda x: x > 0].to_dict())


if __name__ == "__main__":
    main()
