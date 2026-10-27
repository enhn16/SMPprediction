"""한 달 예측 개선(mx) — 일별 피처 표 (육지). 행 = 기준일 I × H(7~34일), 대상일 D = I + H. 판단 시각 I일 15시.

기존 features_daily(달력·최근 수준·작년 계절비·수요·연료·설비)에 팀 데이터 13종에서 더한 것:
| 묶음 | 피처 | 확보 시점 |
|---|---|---|
| 최근 수준 | 지표별 14·56·91일 평균, 같은 유형일 8일 평균, 7일/28일 추세 | I일 SMP (I-1 저녁 공표) |
| 작년 값 | 지표별 작년 대상일(364일 전) ±3일 평균 | 확정 |
| 여러 해 계절비 | 지표별 '작년·재작년·3년 전 (대상일 ±7일 ÷ 기준일 직전 28일)' 평균 (SMP) | 확정 |
| 연료 정규화 계절비 | 같은 계산을 'SMP ÷ 그 달 LNG 연료비단가'로 (연료 변동을 뺀 계절 모양) | 과거 해 연료비는 확정 |
| 연료 함의 수준 | 과거 해 (SMP ÷ LNG) 대상일 계절값 × 판단일에 쓸 수 있는 최신 LNG 단가(M+2), 현재 SMP÷LNG | M+2 규칙 |
| 연료 변화 | LNG·유연탄 최신 단가의 1개월 변화, 대상 월 − 기준 월(월 경계 수), 대상일 '며칠' | M+2 규칙 |
| 수요 | 여러 해 수요예측 계절비, 최근 28일 수요예측 | I일분까지 (I-1 저녁 공표) |
| 태양광 발전 | 지역별 시간별 태양광(육지 합) — 작년 대상일 ±3일 × 최근 성장률(공개 지연 56일 가정) → 예상 발전량, 수요 대비 비율 | 대상일 실적은 약 7주 뒤 공개 → I−56일까지만 |
| 주간 전망·중기예보 | 대상일 주간 최대부하 전망(전날까지 게시분), 중기 육상예보(06시, 4~10일 뒤) | 가까운 지평(H≤10)만, 나머지 빈칸 |
학습은 대상일 2023-01-01 이후(팀 규칙), 지연 입력(작년 값 등)은 2022년 SMP도 참조.
저장: data/processed/smp_forecast_app/mx/features_mx.csv.gz
실행: python -m smp_forecast_app.src.mx_features
"""
import numpy as np
import pandas as pd

from common.config import API_RAW_DIR, PROCESSED_DIR
from common.loaders import load_fuel_cost, load_renewables, load_smp_forecast_api
from smp_forecast_app.src import features_daily as fd
from smp_forecast_app.src.features import _monthly_usable, weather_for_lead

MX = PROCESSED_DIR / "smp_forecast_app" / "mx"
OUT = MX / "features_mx.csv.gz"
STATS = fd.STATS
PV_LAG = 56
NEW = ([f"{s}_{k}" for s in STATS for k in ("l14", "l56", "l91", "same8", "trend", "ly", "seasM", "seasN", "fuelimp", "r_now")]
       + ["seas_dem_m", "dem_l28", "pv_growth", "pv_ly", "pv_exp", "pv_share", "pvmid_exp",
          "mon_cross", "dom", "fc_lng_chg1", "fc_coal_chg1", "kpx_peak_rel", "rf_sky_pm", "rf_pop_pm"])
FEATURES = fd.FEATURES + NEW


def pv_daily():
    """육지 태양광 일 발전량·한낮(11~15시) 발전량 (raw 파일 ~2025-12 + data.go.kr API 2026~)."""
    r = load_renewables()
    r = r[r["region"] == "육지"][["date", "hour", "solar"]]
    fs = sorted((API_RAW_DIR / "kpx_pv_region").glob("pv_region_*.csv"))
    if fs:
        a = pd.concat([pd.read_csv(f, dtype={"tradeYmd": str}) for f in fs])
        a = a[~a["regionNm"].astype(str).str.contains("제주")]
        a = a.assign(date=pd.to_datetime(a["tradeYmd"]), hour=a["tradeNo"].astype(int)).groupby(["date", "hour"])["amgo"].sum()
        a = a.rename("solar").reset_index()
        r = pd.concat([r[r["date"] < a["date"].min()], a])
    day = r.groupby("date")["solar"].sum()
    mid = r[r["hour"].between(11, 15)].groupby("date")["solar"].sum()
    return day, mid


def _at(ser, dates):
    return ser.reindex(pd.DatetimeIndex(dates)).values


def multi_year(xc, xb, tgt, base, years=(1, 2, 3)):
    """여러 해 계절비: 평균_k [xc(대상 − 364k) / xb(기준 − 364k)] (둘 다 있는 해만)."""
    vals = [(_at(xc, tgt - pd.Timedelta(days=364 * k)) / _at(xb, base - pd.Timedelta(days=364 * k))) for k in years]
    return np.nanmean(np.vstack(vals), axis=0)


def build() -> pd.DataFrame:
    df = fd.build()
    df = df[df["base_date"] >= "2022-12-01"].reset_index(drop=True)
    st = fd.daily_stats()
    full = pd.date_range("2021-01-01", df["date"].max() + pd.Timedelta(days=1))
    st = st.reindex(full)
    cal = fd.build_calendar(full[0], full[-1]).set_index("date").reindex(full)
    B, T = pd.DatetimeIndex(df["base_date"]), pd.DatetimeIndex(df["date"])
    off_t = df["offday"].values == 1

    fc = load_fuel_cost().set_index("month")
    lng_d = pd.Series(fc["fc_lng"].reindex(full.to_period("M").to_timestamp()).values, index=full)
    dec = df["base_date"] + pd.Timedelta(days=1)    # 판단일 I에 쓸 수 있는 월값 (features_daily와 같은 호출)
    fct = load_fuel_cost().sort_values("month")
    fct["fc_lng_chg1"] = fct["fc_lng"] / fct["fc_lng"].shift(1) - 1
    fct["fc_coal_chg1"] = fct["fc_coal"] / fct["fc_coal"].shift(1) - 1
    mu = _monthly_usable(dec, fct, ["fc_lng", "fc_lng_chg1", "fc_coal_chg1"])
    lng_now = mu["fc_lng"].values
    df["fc_lng_chg1"], df["fc_coal_chg1"] = mu["fc_lng_chg1"].values, mu["fc_coal_chg1"].values

    for s in STATS:
        x = st[s]
        for w in (14, 56, 91):
            df[f"{s}_l{w}"] = _at(x.rolling(w, min_periods=int(w * 0.7)).mean(), B)
        same8 = {o: x[cal["is_offday"] == bool(o)].rolling(8, min_periods=6).mean().reindex(full).ffill() for o in (0, 1)}
        df[f"{s}_same8"] = np.where(off_t, _at(same8[1], B), _at(same8[0], B))
        df[f"{s}_trend"] = df[f"{s}_l7"] / df[f"{s}_l28"].clip(lower=10)
        df[f"{s}_ly"] = _at(x.rolling(7, center=True, min_periods=5).mean(), T - pd.Timedelta(days=364))
        xc15 = x.rolling(15, center=True, min_periods=10).mean()
        x28 = x.rolling(28, min_periods=20).mean()
        df[f"{s}_seasM"] = multi_year(xc15, x28, T, B)
        r = x / lng_d
        rc15, r28 = r.rolling(15, center=True, min_periods=10).mean(), r.rolling(28, min_periods=20).mean()
        df[f"{s}_seasN"] = multi_year(rc15, r28, T, B)
        prof = np.nanmean(np.vstack([_at(rc15, T - pd.Timedelta(days=364 * k)) for k in (1, 2, 3)]), axis=0)
        df[f"{s}_fuelimp"] = prof * lng_now
        df[f"{s}_r_now"] = df[f"{s}_l28"] / lng_now

    dem = load_smp_forecast_api()
    dem = dem[dem["region"] == "육지"].groupby("date")["demand_fcst"].mean().reindex(full)
    d28 = dem.rolling(28, min_periods=20).mean()
    df["dem_l28"] = _at(d28, B)
    df["seas_dem_m"] = multi_year(dem.rolling(15, center=True, min_periods=10).mean(), d28, T, B)

    pvd, pvm = pv_daily()
    pvd, pvm = pvd.reindex(full), pvm.reindex(full)
    known = B - pd.Timedelta(days=PV_LAG)
    p28 = pvd.rolling(28, min_periods=20).mean()
    df["pv_growth"] = _at(p28, known) / _at(p28, known - pd.Timedelta(days=364))
    df["pv_ly"] = _at(pvd.rolling(7, center=True, min_periods=5).mean(), T - pd.Timedelta(days=364))
    df["pv_exp"] = df["pv_ly"] * df["pv_growth"]
    pm_ly = _at(pvm.rolling(7, center=True, min_periods=5).mean(), T - pd.Timedelta(days=364))
    df["pvmid_exp"] = pm_ly * df["pv_growth"]
    df["pv_share"] = df["pv_exp"] / 24 / (df["dem_l28"] * df["seas_dem_m"])

    df["mon_cross"] = (T.year * 12 + T.month) - (B.year * 12 + B.month)
    df["dom"] = T.day
    # 주간 전망·중기예보: 가까운 지평만
    from drop_classifier.src import outlook_features
    df["kpx_peak_rel"], df["rf_sky_pm"], df["rf_pop_pm"] = np.nan, np.nan, np.nan
    near = df["H"] <= 10
    ol = outlook_features.attach(df.loc[near, "date"], df.loc[near, "base_date"] + pd.Timedelta(days=1))
    df.loc[near, "kpx_peak_rel"] = ol["kpx_peak_rel"].values
    for H in range(7, 11):
        w = weather_for_lead(H)[["date", "rf_sky_pm", "rf_pop_pm"]].set_index("date")
        m = df["H"] == H
        df.loc[m, "rf_sky_pm"] = _at(w["rf_sky_pm"], df.loc[m, "date"])
        df.loc[m, "rf_pop_pm"] = _at(w["rf_pop_pm"], df.loc[m, "date"])
    return df


def load() -> pd.DataFrame:
    return pd.read_csv(OUT, parse_dates=["base_date", "date"])


def build_csv() -> pd.DataFrame:
    """build()를 CSV로 한 번 거친 값 — 백테스트(load)와 비트 단위로 같은 입력 (마지막 자리 차이로 트리 분할이 달라지는 것 방지)."""
    import io
    buf = io.StringIO()
    build().to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf, parse_dates=["base_date", "date"])


def main():
    MX.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(OUT, index=False, encoding="utf-8", compression="gzip")
    cov = df[df["date"] >= "2023-01-01"]
    print(f"저장: {OUT} ({len(df):,}행, 기준일 {df.base_date.min().date()} ~ {df.base_date.max().date()})")
    na = cov[FEATURES].isna().mean().round(3)
    print("2023~ 결측률:", na[na > 0].to_dict())
    for y in (2023, 2024, 2025, 2026):
        c = df[df["date"].dt.year == y][NEW].isna().mean()
        print(y, "결측 많은 피처:", c[c > 0.3].round(2).to_dict())


if __name__ == "__main__":
    main()
