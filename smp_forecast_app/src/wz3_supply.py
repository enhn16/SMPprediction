"""일주일 v3 (R3) ⑥ N1~N3 — 공급 쪽 피처 (팀 13종 밖, 실험용). 모두 기준일 I(= 대상일 − H) 15시에 알 수 있는 값.

  nuc  : 원전 계획예방정비 — 대상일 가용·정비 중 용량, 기준일 대비·기준일 7일 전 대비 가용 변화
         (2025 이전은 실제 종료일 기준이라 연장분을 약간 앞서 앎 → 약한 누수, WEEK_V3.md ⑥)
  sup  : 발전원별 5분 실적(실시간 공개)의 기준일 전날(I−1) 하루 — 원전·LNG·석탄 평균, 시장 태양광 한낮(11~15시) 평균, 양수 평균
         2024-01부터 (그 전 빈 값). 자가용·PPA 추정은 2024-11부터라 쓰지 않음
  curt : 출력제어 시행일 — 게시 지연 감안 I−63 ~ I−35일 시행일 수, 1년 전 대상일 ±15일 시행일 수 (자료 끝 이후는 빈 값)
"""
import numpy as np
import pandas as pd

from common.loaders import load_kpx_power_source, load_land_curtailment, load_nuclear_available

NUC = ["nuc_avail", "nuc_off", "nuc_chg_base", "nuc_chg_7"]
SUP = ["sup_nuc", "sup_gas", "sup_coal", "sup_solar_mid", "sup_pump"]
CURT = ["curt_recent", "curt_ly"]
_cache = {}


def _nuc():
    if "nuc" not in _cache:
        n = load_nuclear_available("2022-01-01", "2027-12-31").set_index("date")
        _cache["nuc"] = n
    return _cache["nuc"]


def _sup():
    if "sup" not in _cache:
        p = load_kpx_power_source()
        p["coal"] = p["coal_dom"] + p["coal_bit"]
        p["hour"] = p["ts"].dt.hour
        day = p.groupby("date")[["nuclear", "gas", "coal", "pump"]].mean()
        mid = p[p["hour"].between(11, 14)].groupby("date")["solar_mkt"].mean()
        _cache["sup"] = day.assign(solar_mid=mid)
    return _cache["sup"]


def _curt():
    if "curt" not in _cache:
        c = load_land_curtailment()
        s = pd.Series(1.0, index=pd.DatetimeIndex(c["date"]).normalize().unique())
        days = pd.date_range("2021-01-01", s.index.max())
        _cache["curt"] = (s.reindex(days, fill_value=0.0), s.index.max())
    return _cache["curt"]


def features(d: pd.DataFrame, H: int, parts) -> tuple[pd.DataFrame, list]:
    d = d.copy()
    base = d["date"] - pd.Timedelta(days=H)
    cols = []
    if "nuc" in parts:
        n = _nuc()["nuc_avail_mw"]
        d["nuc_avail"] = d["date"].map(n)
        d["nuc_off"] = d["date"].map(_nuc()["nuc_off_mw"])
        d["nuc_chg_base"] = d["nuc_avail"] - base.map(n)
        d["nuc_chg_7"] = d["nuc_avail"] - (base - pd.Timedelta(days=7)).map(n)
        cols += NUC
    if "sup" in parts:
        s = _sup()
        prev = base - pd.Timedelta(days=1)
        for c, k in zip(SUP, ["nuclear", "gas", "coal", "solar_mid", "pump"]):
            d[c] = prev.map(s[k])
        cols += SUP
    if "curt" in parts:
        cs, last = _curt()
        cum = cs.cumsum()
        at = lambda t: t.map(cum)
        hi, lo = base - pd.Timedelta(days=35), base - pd.Timedelta(days=64)
        d["curt_recent"] = np.where(hi <= last, at(hi) - at(lo), np.nan)
        ly = d["date"] - pd.Timedelta(days=365)
        d["curt_ly"] = np.where(ly + pd.Timedelta(days=15) <= last,
                                at(ly + pd.Timedelta(days=15)) - at(ly - pd.Timedelta(days=16)), np.nan)
        cols += CURT
    return d, cols
