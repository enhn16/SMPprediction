"""한 달 전망용 일별 피처 테이블 (육지). 행 = 기준일 I × 지평 H(7~34일), 대상일 D = I + H.

판단 시각: I일 15시 (I일 SMP는 I-1일 저녁 공표, I일 하루전 수요예측도 같이 공표 → 사용 가능).
| 묶음 | 피처 |
|---|---|
| 대상일 달력 | 요일·월·연중일, 휴무일·요일유형·연휴 길이/며칠째·명절·근로자의날, 앞뒤 휴무 (사전 확정) |
| 최근 수준 (I까지) | 최근 7·28일 일평균, 최근 같은 유형일(휴무/평일) 4일 평균 — 일평균·한낮·저녁·한낮최저 각각 |
| 계절 비율 (작년) | 작년 대상일 ±3일 평균 ÷ 작년 기준일 직전 28일 평균 (SMP·수요예측) — 작년 실적이라 확정 |
| 수요 | 최근 7일 하루전 수요예측 일평균 |
| 연료·설비 | EPSIS LNG·유연탄 연료비단가와 3개월 전 대비 변화(M+2 규칙), 태양광 설비(M+2) |
목표: 대상일 일평균(y_mean), 한낮 11~15시 평균(y_mid), 저녁 18~22시 평균(y_eve), 한낮 최저(y_midmin)
저장: data/processed/smp_forecast_app/features_daily_육지.csv
실행: python -m smp_forecast_app.src.features_daily
"""
import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR
from common.loaders import load_fuel_cost, load_market_capacity, load_smp_actual, load_smp_forecast_api
from smp_forecast_app.src.features import DAYTYPE_CODE, _monthly_usable

OUT = PROCESSED_DIR / "smp_forecast_app" / "features_daily_육지.csv"
LEADS = range(7, 35)
STATS = ["mean", "mid", "eve", "midmin"]
TARGETS = [f"y_{s}" for s in STATS]

CAL = ["H", "dow", "month", "doy_sin", "doy_cos", "offday", "daytype", "off_block_len", "off_block_day",
       "is_myeongjeol", "is_labor_day", "prev_offday", "next_offday"]
LEVEL = [f"{s}_{w}" for s in STATS for w in ("l7", "l28", "same4")] + ["seas_smp", "seas_dem", "dem_l7", "lastyear_mean"]
FUEL = ["fc_lng", "fc_coal", "fc_lng_chg3", "cap_solar"]
FEATURES = CAL + LEVEL + FUEL


def daily_stats() -> pd.DataFrame:
    s = load_smp_actual("육지")
    w = s.pivot_table(index="date", columns="hour", values="smp")
    return pd.DataFrame({"mean": w.mean(axis=1), "mid": w[[11, 12, 13, 14, 15]].mean(axis=1),
                         "eve": w[[18, 19, 20, 21, 22]].mean(axis=1), "midmin": w[[11, 12, 13, 14, 15]].min(axis=1)})


def build() -> pd.DataFrame:
    st = daily_stats()
    last = st.index.max()
    full = pd.date_range("2021-01-01", last + pd.Timedelta(days=max(LEADS) + 1))
    st = st.reindex(full)
    cal = build_calendar(full[0], full[-1]).set_index("date").reindex(full)
    dem = load_smp_forecast_api()
    dem = dem[dem["region"] == "육지"].groupby("date")["demand_fcst"].mean().reindex(full)

    # 기준일 I 기준 값 (I 포함 이전)
    base = pd.DataFrame(index=full)
    for s in STATS:
        base[f"{s}_l7"] = st[s].rolling(7, min_periods=5).mean()
        base[f"{s}_l28"] = st[s].rolling(28, min_periods=20).mean()
        for off in (0, 1):  # 같은 유형일 최근 4일
            ser = st[s][cal["is_offday"] == bool(off)].rolling(4, min_periods=3).mean()
            base[f"{s}_same4_{off}"] = ser.reindex(full).ffill()
    base["dem_l7"] = dem.rolling(7, min_periods=5).mean()
    base["smp_l28_all"] = st["mean"].rolling(28, min_periods=20).mean()
    base["dem_l28_all"] = dem.rolling(28, min_periods=20).mean()
    st_c7 = st["mean"].rolling(7, center=True, min_periods=5).mean()
    dem_c7 = dem.rolling(7, center=True, min_periods=5).mean()

    I = pd.date_range("2022-02-01", last)
    rows = pd.DataFrame([(i, h) for i in I for h in LEADS], columns=["base_date", "H"])
    rows["date"] = rows["base_date"] + pd.to_timedelta(rows["H"], unit="D")
    b = base.reindex(rows["base_date"]).reset_index(drop=True)
    c = cal.reindex(rows["date"]).reset_index(drop=True)
    off = c["is_offday"].astype(int)
    out = rows.copy()
    out["dow"] = c["weekday"].values
    out["month"] = rows["date"].dt.month
    doy = rows["date"].dt.dayofyear
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    out["offday"] = off.values
    out["day_type"] = c["day_type"].values
    out["daytype"] = c["day_type"].map(DAYTYPE_CODE).values
    for k in ("off_block_len", "off_block_day"):
        out[k] = c[k].values
    out["is_myeongjeol"] = c["is_myeongjeol"].astype(int).values
    out["is_labor_day"] = c["is_labor_day"].astype(int).values
    out["prev_offday"] = cal["is_offday"].shift(1).reindex(rows["date"]).astype(float).values
    out["next_offday"] = cal["is_offday"].shift(-1).reindex(rows["date"]).astype(float).values
    for s in STATS:
        out[f"{s}_l7"], out[f"{s}_l28"] = b[f"{s}_l7"].values, b[f"{s}_l28"].values
        out[f"{s}_same4"] = np.where(off.values == 1, b[f"{s}_same4_1"].values, b[f"{s}_same4_0"].values)
    out["dem_l7"] = b["dem_l7"].values
    # 작년 계절 비율: 작년 대상일(364일 전, 같은 요일) ±3일 평균 ÷ 작년 기준일 직전 28일 평균
    ly_d, ly_i = rows["date"] - pd.Timedelta(days=364), rows["base_date"] - pd.Timedelta(days=364)
    out["seas_smp"] = st_c7.reindex(ly_d).values / base["smp_l28_all"].reindex(ly_i).values
    out["seas_dem"] = dem_c7.reindex(ly_d).values / base["dem_l28_all"].reindex(ly_i).values
    out["lastyear_mean"] = st["mean"].reindex(ly_d).values

    fc = load_fuel_cost()
    fc["fc_lng_chg3"] = fc["fc_lng"] / fc["fc_lng"].shift(3) - 1
    # 판단일 I에 쓸 수 있는 월값 → _monthly_usable(대상, lead)는 (대상 - lead)를 판단일로 보므로 I+1, lead=1로 호출
    dec = rows["base_date"] + pd.Timedelta(days=1)
    out[["fc_lng", "fc_coal", "fc_lng_chg3"]] = _monthly_usable(dec, fc, ["fc_lng", "fc_coal", "fc_lng_chg3"]).values
    out[["cap_solar"]] = _monthly_usable(dec, load_market_capacity(), ["cap_solar"]).values

    tg = st.reindex(rows["date"]).reset_index(drop=True)
    for s in STATS:
        out[f"y_{s}"] = tg[s].values
    return out


def load_daily() -> pd.DataFrame:
    return pd.read_csv(OUT, parse_dates=["base_date", "date"])


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} ({len(df):,}행, 기준일 {df.base_date.min().date()} ~ {df.base_date.max().date()})")
    print("2024~ 결측률:", df[df["base_date"] >= "2024-01-01"][FEATURES].isna().mean().round(3).loc[lambda s: s > 0].to_dict())


if __name__ == "__main__":
    main()
