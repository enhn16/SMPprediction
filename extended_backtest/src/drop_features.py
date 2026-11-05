"""육지 급락 피처 표 2020-01 ~ (운영 팀 모델과 같은 33개 + 연료비·가격 수준 + 168시간 SMP).

운영 코드(drop_classifier.src.features.build, team_select.data23, ens8.data, team_lstm.data)와 같은 정의로 다시 만든다.
차이는 자료 범위뿐: 2020~2022 보강분(data/raw/backfill_2020)을 운영 자료 앞에 붙인다.
운영 코드는 import만 하고 고치지 않는다. 운영 로더는 읽는 폴더만 잠시 바꿔(보강 폴더) 한 번 더 부른다.

검증: python -m extended_backtest.src.drop_features check  → 2023-01-01 이후 운영 피처 표와 값 대조
저장: python -m extended_backtest.src.drop_features        → data/processed/extended_backtest/drop_features_육지.parquet
"""
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

import common.calendar as cal_mod
import common.loaders as L
from common.config import PROCESSED_DIR
from drop_classifier.src.ens8 import COLS
from drop_classifier.src.features import START as TREND_START
from drop_classifier.src.fuel_level import FUEL, LEVEL
from drop_classifier.src.outlook_features import load as outlook_load_ops, OUTLOOK
from drop_classifier.src.supply_level import supply_features
from drop_classifier.src.team_lstm import add_seqs
from extended_backtest.src.data import BF, load_backfill_smp_forecast, load_holidays_all, load_smp_all

MID = (11, 15)
REGIONS7 = None  # fair_select.REGIONS (아래에서 import)
OUT = PROCESSED_DIR / "extended_backtest"
FIRST = "2020-01-01"


@contextmanager
def raw_dir(path):
    """운영 로더가 읽는 원본 폴더(API_RAW_DIR)를 잠시 바꾼다."""
    old = L.API_RAW_DIR
    L.API_RAW_DIR = path
    try:
        yield
    finally:
        L.API_RAW_DIR = old


def kma_regional_all() -> pd.DataFrame:
    with raw_dir(BF):
        bf = L.load_kma_regional_fcst()
    ops = L.load_kma_regional_fcst()
    out = pd.concat([bf, ops], ignore_index=True)
    return out.drop_duplicates(["reg", "tm_fc", "tm_ef"], keep="last").reset_index(drop=True)


def fuel_cost_all() -> pd.DataFrame:
    with raw_dir(BF):
        bf = L.load_fuel_cost()
    ops = L.load_fuel_cost()
    out = pd.concat([bf[~bf["month"].isin(ops["month"])], ops], ignore_index=True)
    return out.sort_values("month").reset_index(drop=True)


def outlook_all() -> pd.DataFrame:
    """주간전망: 운영 표 + 2020~2022 보강 표 (게시물 번호가 겹치지 않는 것만)."""
    ops = outlook_load_ops()
    bf = pd.read_csv(BF / "weekly_outlook_2020_2022.csv", parse_dates=["posted", "target_date"])
    bf = bf[~bf["list_no"].isin(ops["list_no"])].dropna(subset=["peak_gw"])
    bf["kpx_peak_rel"] = bf["peak_gw"] / bf.groupby("list_no")["peak_gw"].transform("max")
    bf = bf.rename(columns={"peak_gw": "kpx_peak"})
    return pd.concat([bf[["list_no", "posted", "target_date", "kpx_peak", "kpx_peak_rel"]],
                      ops[["list_no", "posted", "target_date", "kpx_peak", "kpx_peak_rel"]]], ignore_index=True)


def outlook_attach(targets: pd.Series, decision: pd.Series) -> pd.DataFrame:
    """outlook_features.attach와 같은 규칙 (판단일 전날까지 등록된 가장 최근 전망)."""
    ol = outlook_all().sort_values("posted")
    by_target = {d: g for d, g in ol.groupby("target_date")}
    rows = []
    for t, dec in zip(targets, decision):
        g = by_target.get(t)
        r = {c: np.nan for c in OUTLOOK}
        if g is not None:
            ok = g[g["posted"] <= dec - pd.Timedelta(days=1)]
            if len(ok):
                r = {c: ok.iloc[-1][c] for c in OUTLOOK}
        rows.append(r)
    return pd.DataFrame(rows, index=targets.index)


def fcst15_all() -> pd.DataFrame:
    """fair_select.fcst15와 같은 규칙 (D-1 11시, 없으면 05시 발표분, 7개 구역 평균)."""
    from drop_classifier.src.fair_select import REGIONS
    f = kma_regional_all()
    f = f[f["reg"].isin(REGIONS)].copy()
    f["date"] = f["tm_ef"].dt.normalize()
    f["seg"] = np.where(f["tm_ef"].dt.hour == 12, "pm", "am")
    f["issue_day"] = (f["date"] - f["tm_fc"].dt.normalize()).dt.days
    f = f[(f["issue_day"] == 1) & f["tm_fc"].dt.hour.isin([5, 11])]
    f = f.sort_values("tm_fc").groupby(["reg", "date", "seg"]).tail(1)
    pm, am = f[f["seg"] == "pm"], f[f["seg"] == "am"]
    out = pd.DataFrame({
        "rf_sky_pm": pm.groupby("date")["sky"].mean(),
        "rf_clear_pm": pm.groupby("date")["sky"].apply(lambda s: (s <= 2).mean()),
        "rf_pop_pm": pm.groupby("date")["pop"].mean(),
        "rf_rain_pm": pm.groupby("date")["rain"].mean(),
        "rf_ta_pm": pm.groupby("date")["ta"].mean(),
        "rf_sky_am": am.groupby("date")["sky"].mean(),
    })
    return out.rename_axis("date").reset_index()


def smp_forecast_all() -> pd.DataFrame:
    bf = load_backfill_smp_forecast()
    ops = L.load_smp_forecast_api()
    out = pd.concat([bf[bf["date"] < ops["date"].min()], ops], ignore_index=True)
    return out.sort_values(["region", "date", "hour"]).reset_index(drop=True)


def build() -> pd.DataFrame:
    smp = load_smp_all("육지")
    # 레이블: 11~15시 최저 < 30원 (= 새 기준 육지 급락 구간이 있는 날)
    mid = smp[smp["hour"].between(*MID)].groupby("date")["smp"].min().rename("min_smp_11_15")
    full = smp.groupby("date")["hour"].transform("size") == 24
    days = smp[full]["date"].unique()
    df = mid.reindex(days).reset_index().rename(columns={"index": "date"})
    df["label"] = (df["min_smp_11_15"] < 30).astype(int)

    # 달력 (특일 = 보강 2019~2021 + 운영 2022~)
    orig = cal_mod.load_holidays
    cal_mod.load_holidays = load_holidays_all
    try:
        cal = cal_mod.build_calendar(df["date"].min(), df["date"].max() + pd.Timedelta(days=2))
    finally:
        cal_mod.load_holidays = orig
    df = df.merge(cal, on="date", how="left")

    # 수요예측 (features.build와 같음)
    fc = smp_forecast_all()
    f = fc[(fc["region"] == "육지") & fc["hour"].between(*MID)]
    dem = f.groupby("date")["demand_fcst"].mean().asfreq("D")
    base28 = dem.shift(1).rolling(28, min_periods=20).median()
    demand = pd.DataFrame({"demand_fcst_prev": dem.shift(1), "demand_prev_rel28": (dem / base28).shift(1)})
    df = df.set_index("date").join(demand, how="left").reset_index()

    # 달력 피처
    df["is_offday"] = (df["day_type"] != "평일").astype(int)
    df["is_weekend_only"] = (df["day_type"] == "주말").astype(int)
    df["is_holiday_single"] = (df["day_type"] == "공휴일").astype(int)
    df["is_long_holiday"] = (df["day_type"] == "연휴").astype(int)
    df["is_myeongjeol"] = df["is_myeongjeol"].astype(int)
    df["is_labor_day"] = df["is_labor_day"].astype(int)
    m = df["date"].dt.month
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)

    # 시차 SMP (D-1 이전만)
    df = df.sort_values("date").reset_index(drop=True)
    df["prev_day_min_smp"] = df["min_smp_11_15"].shift(1)
    prev_same, last = [], {}
    for off, v in zip(df["is_offday"], df["min_smp_11_15"]):
        prev_same.append(last.get(off, np.nan))
        last[off] = v
    df["prev_same_type_min_smp"] = prev_same
    df["drops_last14"] = df["label"].shift(1).rolling(14, min_periods=1).sum()
    df["min_smp_last7"] = df["min_smp_11_15"].shift(1).rolling(7, min_periods=1).min()
    off = df["is_offday"]
    df["is_bridge"] = ((off == 0) & (off.shift(1) == 1) & (off.shift(-1) == 1)).astype(int)
    df["is_after_offday"] = ((off == 0) & (off.shift(1) == 1)).astype(int)
    df["trend_years"] = (df["date"] - pd.Timestamp(TREND_START)).dt.days / 365.25

    # 주간전망 (등록 다음 날부터: posted ≤ D-2), 15시 예보, 태양광 설비, 연료비
    df = pd.concat([df, outlook_attach(df["date"], df["date"] - pd.Timedelta(days=1))], axis=1)
    df = df.merge(fcst15_all(), on="date", how="left")
    df["cap_solar_gw"] = supply_features(df["date"])["cap_solar_gw"].values
    fu = fuel_cost_all().sort_values("usable_from")
    mm = pd.merge_asof(df[["date"]].sort_values("date"), fu[["usable_from", "fc_lng", "fc_coal"]],
                       left_on="date", right_on="usable_from").set_index("date")
    df["fc_lng"] = df["date"].map(mm["fc_lng"]).values
    df["fc_coal"] = df["date"].map(mm["fc_coal"]).values
    df["fc_lng_coal_gap"] = df["fc_lng"] - df["fc_coal"]

    # 가격 수준: D-7 ~ D-1 일 최고·평균의 평균 (min_periods 5)
    daily = smp.groupby("date")["smp"].agg(["max", "mean"])
    roll = daily.rolling(7, min_periods=5).mean().shift(1)
    df["smp_max7"] = df["date"].map(roll["max"]).values
    df["smp_mean7"] = df["date"].map(roll["mean"]).values

    df["offday"] = df["day_type"] != "평일"
    df = df[df["date"] >= FIRST].reset_index(drop=True)
    df = add_seqs(df, smp=smp)
    return df[["date", "label", "min_smp_11_15", "day_type", "offday"] + COLS + FUEL + LEVEL + ["seq168"]]


def check(df: pd.DataFrame):
    """운영 피처 표(team_lstm.data)와 2023-01-01 이후 값 대조."""
    from drop_classifier.src.team_lstm import data as ops_data
    ops = ops_data()
    cols = ["label", "min_smp_11_15"] + COLS + FUEL + LEVEL
    a = df.set_index("date")
    b = ops.set_index("date")
    common_d = a.index.intersection(b.index)
    common_d = common_d[common_d >= "2023-01-01"]
    print(f"대조 날짜 {len(common_d)}일 ({common_d.min().date()} ~ {common_d.max().date()})")
    bad = []
    for c in cols:
        x, y = a.loc[common_d, c].astype(float), b.loc[common_d, c].astype(float)
        same = np.isclose(x, y, rtol=1e-6, atol=1e-6, equal_nan=True)
        if not same.all():
            bad.append(c)
            d = common_d[~same]
            print(f"  다름 {c}: {len(d)}일, 예 {d[0].date()} 우리 {x[d[0]]} / 운영 {y[d[0]]}")
    sx = np.stack(a.loc[common_d, "seq168"]); sy = np.stack(b.loc[common_d, "seq168"])
    print("  seq168 같음" if np.allclose(sx, sy, equal_nan=True) else f"  seq168 다름 {(~np.isclose(sx, sy, equal_nan=True)).any(axis=1).sum()}일")
    print("모두 같음" if not bad else f"다른 열 {len(bad)}개")
    print("\n연도별 빈칸 비율(%):")
    print((df[COLS + FUEL + LEVEL].isna().groupby(df["date"].dt.year).mean() * 100).round(0).T
          .loc[lambda t: t.max(axis=1) > 0].to_string())


if __name__ == "__main__":
    df = build()
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check(df)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_pickle(OUT / "drop_features_육지.pkl")
        print(f"저장: {len(df)}일, 급락 {int(df['label'].sum())}일")
