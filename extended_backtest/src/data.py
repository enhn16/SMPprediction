"""2020~ 확장 채점용 자료 로더.

- 2020-01 ~ 2021-12: data/raw/backfill_2020/ (운영 로더는 읽지 않는 보강분)
- 2022-01 ~       : 운영 로더(common.loaders) 그대로
- 2019 육지 SMP  : data/raw/smpInLand_2019.xlsx (2020년 초 28일 기준값용)
시간(hour)은 거래시간 1~24 (11 = 10:00~11:00).
"""
import pandas as pd

from common.config import RAW_DIR
from common import loaders

BF = RAW_DIR / "backfill_2020"


def load_backfill_smp_forecast() -> pd.DataFrame:
    """backfill_2020/smp_forecast → [date, hour, region, smp, demand_fcst] (2020-01 ~ 2021-12)."""
    files = sorted((BF / "smp_forecast").glob("smp_forecast_*.csv"))
    df = pd.concat([pd.read_csv(p, dtype={"date": str, "hour": str}) for p in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["hour"] = df["hour"].astype(int)
    df["region"] = df["areaName"]
    df["demand_fcst"] = df["mlfd"].where(df["region"] == "육지", df["jlfd"])
    out = df[["date", "hour", "region", "smp", "demand_fcst"]]
    return out.drop_duplicates(["date", "hour", "region"]).sort_values(["region", "date", "hour"]).reset_index(drop=True)


def load_smp_2019_land() -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / "smpInLand_2019.xlsx", header=1)
    df.columns = ["date"] + list(range(1, 25)) + ["max", "min", "mean"]
    df = df.drop(columns=["max", "min", "mean"])
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    out = df.melt(id_vars="date", var_name="hour", value_name="smp")
    out["hour"] = out["hour"].astype(int)
    return out


def load_smp_all(region: str) -> pd.DataFrame:
    """[date, hour, smp, source] 2019(육지만)·2020 ~ 최신."""
    parts = []
    if region == "육지":
        parts.append(load_smp_2019_land().assign(source="엑셀2019"))
    bf = load_backfill_smp_forecast()
    bf = bf[bf["region"] == region][["date", "hour", "smp"]].assign(source="API보강")
    parts.append(bf)
    ops = loaders.load_smp_actual(region)
    parts.append(ops[ops["date"] > bf["date"].max()])
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["date", "hour"], keep="last")
    return out.sort_values(["date", "hour"]).reset_index(drop=True)


def load_holidays_all() -> pd.DataFrame:
    """[date, name] 2019 ~ 2027: 보강 특일(2019~2021) + 운영 특일(2022~)."""
    df = pd.read_csv(BF / "holidays" / "holidays_2019_2021.csv")
    df = df[df["isHoliday"] == "Y"]
    df["date"] = pd.to_datetime(df["locdate"].astype(str), format="%Y%m%d")
    bf = df.groupby("date")["dateName"].agg(lambda s: "·".join(dict.fromkeys(s))).reset_index()
    bf = bf.rename(columns={"dateName": "name"})
    ops = loaders.load_holidays()
    out = pd.concat([bf[bf["date"] < ops["date"].min()], ops], ignore_index=True)
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def offday_flags(dates: pd.Series) -> pd.Series:
    """휴일 = 주말·공휴일(대체·임시·선거일 포함). 판정 기준 문서 5번과 같은 정의."""
    hol = set(load_holidays_all()["date"])
    d = pd.to_datetime(dates)
    return (d.dt.weekday >= 5) | d.isin(hol)


def load_jeju_sukub_all() -> pd.DataFrame:
    """제주 5분 실적 2019-12 ~ 최신 → [time, supply, demand, renew, solar, wind] (MW). 운영 로더와 같은 열·처리.

    - 2019-12·2020-01, 빈 날(2023-07-02, 2026-10-29~31): backfill_2020/kpx_jeju_sukub (API, jeju_collect_sukub)
    - 2020-02 ~ 2022-12: backfill_2020/kpx_jeju_sukub_dl/chejusukub_YYYY.csv (사용자 내려받기, cp949, 2026-11-06)
    - 2023-01 ~      : 운영 로더 common.loaders.load_kpx_jeju_sukub (겹치면 운영 값 우선)
    공급능력·수요 0 이하는 결측 처리 (운영 로더와 같음).
    """
    import numpy as np
    api = pd.concat([pd.read_csv(p, dtype={"baseDatetime": str})
                     for p in sorted((BF / "kpx_jeju_sukub").glob("jeju5m_*.csv"))], ignore_index=True)
    api = pd.DataFrame({"time": pd.to_datetime(api["baseDatetime"], format="%Y%m%d%H%M%S"),
                        "supply": api["suppAbility"], "demand": api["currPwrTot"], "renew": api["renewPwrTot"],
                        "solar": api["renewPwrSolar"], "wind": api["renewPwrWind"]})
    dl = pd.concat([pd.read_csv(p, encoding="cp949", dtype=str)
                    for p in sorted((BF / "kpx_jeju_sukub_dl").glob("chejusukub_*.csv"))], ignore_index=True)
    dl.columns = ["time", "supply", "demand", "renew", "solar", "wind"]
    dl["time"] = pd.to_datetime(dl["time"], format="%Y%m%d%H%M%S")
    ops = loaders.load_kpx_jeju_sukub()
    out = pd.concat([ops, dl, api], ignore_index=True)   # 앞쪽 우선
    num = ["supply", "demand", "renew", "solar", "wind"]
    out[num] = out[num].apply(pd.to_numeric, errors="coerce")
    out = out.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    out.loc[out["supply"] <= 0, "supply"] = np.nan
    out.loc[out["demand"] <= 0, ["demand", "renew", "solar", "wind"]] = np.nan
    return out
