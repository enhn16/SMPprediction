"""제주 급락(SMP ≤ 0원) 새 기준용 날짜별 피처 — 판단 시각 = 전날(D-1) 15:00.

두 종류의 '하루 계통 모양' 피처를 같은 함수(day_shape)로 만든다.
  - 실제값(act_*): 5분 실적의 시간별 중앙값. 1단계(잉여일) 학습에만 쓴다 (그날이 지나야 아는 값).
  - 전망값(fc_*):  D-1 15시에 알 수 있는 값으로 만든 내일 시간별 수요·태양광·풍력 전망. 예측에 쓴다.
      태양광 = Open-Meteo 일사(2일 전 실행, 3개 모델·4지점 평균) × 최근 14일 '실제 태양광 ÷ 일사' 비율 (설비 증가 자동 반영)
      풍력   = Open-Meteo 100m 풍속(ECMWF·GFS 평균) → 최근 60일 '풍속 → 풍력' 3차 회귀 (0 ~ 최근 최대로 자름)
      수요   = 최근 21일 안의 같은 유형일(휴일/평일) 3일 시간별 중앙값
    보정·수요에 쓰는 실적은 D-2까지(완전한 날)만 쓴다. Open-Meteo가 2024-01부터라 전망값은 2024-01 ~.
  - 그 밖(D-1 15시 기준): 최근 제주 0원 일수, 전날 최저 SMP, 육지 전날 한낮 최저, D-1 오전(11~14시) 실제 순부하, 달력, LNG 연료비.
레이블: J1(0원일), 출력제어(2021-01 ~ 2024-06-03, 사후 정보 → 레이블 전용).
저장: data/processed/extended_backtest/jeju_features.csv
실행(루트에서): python -m extended_backtest.src.jeju_features
"""
import numpy as np
import pandas as pd

from common import loaders
from common.config import PROCESSED_DIR, RAW_DIR
from extended_backtest.src.data import load_jeju_sukub_all, load_smp_all, offday_flags

OUT = PROCESSED_DIR / "extended_backtest" / "jeju_features.csv"
LAB = __import__("pathlib").Path(__file__).resolve().parents[1] / "outputs" / "tables" / "J1_labels_jeju.csv"
START, END = pd.Timestamp("2020-01-29"), pd.Timestamp("2026-10-31")
MID, NIGHT = range(11, 16), range(1, 8)     # 거래시간: 한낮 11~15시, 새벽 1~7시


# ---------- 시간별 실제값 ----------
def hourly_actual() -> pd.DataFrame:
    """[date, hour(1~24), dem, sol, wind] 5분 실적 시간별 중앙값 (한 칸 튐 제거). 2023-4분기 같은 이상값은 결측."""
    k = load_jeju_sukub_all()
    k.loc[(k["demand"] > 1500) | (k["wind"] > 450) | (k["solar"] > 800), ["demand", "solar", "wind"]] = np.nan
    k["date"] = k["time"].dt.normalize()
    k["hour"] = k["time"].dt.hour + 1
    h = k.groupby(["date", "hour"])[["demand", "solar", "wind"]].median().reset_index()
    return h.rename(columns={"demand": "dem", "solar": "sol"})


def day_shape(h: pd.DataFrame, p: str) -> pd.DataFrame:
    """시간별 [date, hour, dem, sol, wind] → 날짜별 계통 모양 피처 (접두어 p)."""
    h = h.copy()
    h["net"] = h["dem"] - h["sol"] - h["wind"]
    h["re"] = (h["sol"] + h["wind"]) / h["dem"]
    g = h.groupby("date")
    mid, night = h[h["hour"].isin(MID)].groupby("date"), h[h["hour"].isin(NIGHT)].groupby("date")
    out = pd.DataFrame({
        f"{p}net_min": g["net"].min(), f"{p}net_min_mid": mid["net"].min(), f"{p}net_min_night": night["net"].min(),
        f"{p}sol_max": g["sol"].max(), f"{p}wind_mean": g["wind"].mean(), f"{p}wind_night": night["wind"].mean(),
        f"{p}dem_min_mid": mid["dem"].min(), f"{p}dem_mean": g["dem"].mean(), f"{p}re_max": g["re"].max(),
        f"{p}n_hours": g["net"].count()})
    return out.reset_index()


# ---------- 전망값 (D-1 15시) ----------
def openmeteo_hourly() -> pd.DataFrame:
    o = loaders.load_openmeteo_prev2_jeju()
    o["date"] = o["time"].dt.normalize()
    o["hour"] = o["time"].dt.hour + 1                       # HH시 시작 값 = 거래시간 HH+1
    rad = o.groupby(["date", "hour"])["rad"].mean()
    ws = o[o["model"].isin(["ecmwf", "gfs"])].groupby(["date", "hour"])["ws100"].mean()
    cloud = o.groupby(["date", "hour"])["cloud"].mean()
    return pd.concat({"rad": rad, "ws": ws, "cloud": cloud}, axis=1).reset_index()


def hourly_forecast(act: pd.DataFrame, om: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """날짜 D마다 내일 시간별 [dem, sol, wind] 전망. 보정에는 D-2까지 실적만 쓴다."""
    a = act.merge(om, on=["date", "hour"], how="left")
    a["off"] = offday_flags(a["date"]).values
    om_i = om.set_index("date")
    off_all = dict(zip(dates, offday_flags(pd.Series(dates)).values))
    rows = []
    for D in dates:
        if D not in om_i.index:
            continue
        past = a[(a["date"] <= D - pd.Timedelta(days=2))]
        # 태양광: 최근 14일, 8~18시
        s = past[(past["date"] >= D - pd.Timedelta(days=15)) & past["hour"].between(8, 18)].dropna(subset=["sol", "rad"])
        k_sol = s["sol"].sum() / s["rad"].sum() if s["date"].nunique() >= 5 and s["rad"].sum() > 0 else np.nan
        # 풍력: 최근 60일, 3차 회귀
        w = past[past["date"] >= D - pd.Timedelta(days=61)].dropna(subset=["wind", "ws"])
        # 수요: 최근 21일 안 같은 유형일 3일 중앙값
        dd = past[(past["date"] >= D - pd.Timedelta(days=22)) & (past["off"] == off_all[D])]
        last3 = sorted(dd["date"].unique())[-3:]
        dem = dd[dd["date"].isin(last3)].groupby("hour")["dem"].median()
        f = om_i.loc[[D]].set_index("hour")
        out = pd.DataFrame({"date": D, "hour": f.index, "sol": k_sol * f["rad"].values})
        if len(w) >= 200:
            X = np.vander(w["ws"].values, 4)
            coef = np.linalg.lstsq(X, w["wind"].values, rcond=None)[0]
            out["wind"] = np.clip(np.vander(f["ws"].fillna(w["ws"].median()).values, 4) @ coef, 0, w["wind"].quantile(0.99))
        else:
            out["wind"] = np.nan
        out["dem"] = dem.reindex(f.index).values
        out["cloud"] = f["cloud"].values
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


# ---------- 출력제어 (레이블 전용) ----------
def curtail_days() -> pd.DataFrame:
    """[date, curt_hours] 제주 태양광·풍력 출력제어 시간 수 (2021-01-13 ~ 2024-06-03)."""
    parts = []
    for name, enc in [("제주 태양광 출력제어횟수.csv", "utf-8-sig"), ("제주 풍력 출력제어횟수 및 제어량.csv", "cp949")]:
        d = pd.read_csv(RAW_DIR / name, encoding=enc, dtype=str)
        d = d.rename(columns={d.columns[0]: "date"})
        hours = [c for c in d.columns if c.endswith("시")]
        v = d[hours].fillna("0").apply(lambda s: ~s.str.strip().isin(["0", "", "0.0"]))
        parts.append(pd.DataFrame({"date": pd.to_datetime(d["date"]), **{c: v[c] for c in hours}}))
    c = pd.concat(parts).groupby("date").max()
    return pd.DataFrame({"curt_hours": c.sum(axis=1)}).reset_index()


CURT_START, CURT_END = pd.Timestamp("2021-01-01"), pd.Timestamp("2024-05-31")


def build() -> pd.DataFrame:
    days = pd.DataFrame({"date": pd.date_range(START, END)})
    days["offday"] = offday_flags(days["date"]).values.astype(int)
    days["month"] = days["date"].dt.month
    days["m_sin"], days["m_cos"] = np.sin(2 * np.pi * days["month"] / 12), np.cos(2 * np.pi * days["month"] / 12)

    lab = pd.read_csv(LAB, parse_dates=["date"])[["date", "event", "n_intervals"]]
    cur = curtail_days()
    df = days.merge(lab, on="date", how="left").merge(cur, on="date", how="left")
    df["curt_hours"] = df["curt_hours"].fillna(0)
    in_c = df["date"].between(CURT_START, CURT_END)
    df["surplus"] = np.where(in_c, ((df["curt_hours"] > 0) | (df["event"] == 1)).astype(float), np.nan)

    act = hourly_actual()
    df = df.merge(day_shape(act, "act_"), on="date", how="left")
    om = openmeteo_hourly()
    fc = hourly_forecast(act, om, pd.date_range("2024-01-01", END))
    df = df.merge(day_shape(fc, "fc_"), on="date", how="left")
    df = df.merge(fc.groupby("date")["cloud"].mean().rename("fc_cloud").reset_index(), on="date", how="left")

    # D-1 오전 실제 순부하 (11~14시 = 10:00~14:00, 14:25까지 공개분 안)
    a = act.copy()
    a["net"] = a["dem"] - a["sol"] - a["wind"]
    y = a[a["hour"].between(11, 14)].groupby("date").agg(yd_net_min_mid=("net", "min"), yd_sol_max=("sol", "max"),
                                                         yd_wind_mean=("wind", "mean"))
    y.index = y.index + pd.Timedelta(days=1)
    df = df.merge(y.rename_axis("date").reset_index(), on="date", how="left")

    # 최근 제주 SMP (D-1 실적은 D-2 저녁 공표 → D-1 15시에 앎)
    sj = load_smp_all("제주")
    dj = sj.groupby("date")["smp"].agg(["min"]).rename(columns={"min": "smp_min"})
    dj["zero"] = (dj["smp_min"] <= 0).astype(int)
    dj["mid_min"] = sj[sj["hour"].isin(MID)].groupby("date")["smp"].min()
    sl = load_smp_all("육지")
    dj["land_mid_min"] = sl[sl["hour"].isin(MID)].groupby("date")["smp"].min()
    dj = dj.reindex(pd.date_range(dj.index.min(), END + pd.Timedelta(days=1)))
    sh = dj.shift(1)                                             # D 행 = D-1 값
    rec = pd.DataFrame({"yd_smp_min": sh["smp_min"], "yd_mid_min": sh["mid_min"], "yd_zero": sh["zero"],
                        "zero_7": sh["zero"].rolling(7, min_periods=1).sum(),
                        "zero_28": sh["zero"].rolling(28, min_periods=1).sum(),
                        "yd_land_mid_min": sh["land_mid_min"],
                        "mid_min_28med": sh["mid_min"].rolling(28, min_periods=7).median()}).rename_axis("date").reset_index()
    df = df.merge(rec, on="date", how="left")
    df["yd_mid_rel"] = df["yd_mid_min"] / df["mid_min_28med"]

    fuel = loaders.load_fuel_cost()
    m = df["date"].dt.to_period("M").dt.to_timestamp()
    lng = fuel.set_index("usable_from")["fc_lng"]
    df["fuel_lng"] = m.map(lng)
    return df


def main():
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{len(df)}일, 열 {df.shape[1]} → {OUT}")
    print("잉여일(2021-01~2024-05):", int(df["surplus"].sum()), "/", int(df["surplus"].notna().sum()))
    fc = df[df["date"] >= "2024-01-01"]
    print("전망 피처 결측률(2024~):", fc.filter(like="fc_").isna().mean().round(3).to_dict())


if __name__ == "__main__":
    main()
