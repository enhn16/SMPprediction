"""한 달 예측 고도화 ① 진단 — v2 (최종 운영 모델)의 약한 구간 찾기. 2025 검증 구간만 (2026은 보지 않음).

입력: mx/MX_predictions_daily_v2.csv 의 대상일 2025 행 (= 2024로 맞춘 선택 예측 + 2025로 고른 범위 → 범위 포함률은 선택 구간 안의 값),
      mx 피처의 same4(단순 기준), 연료비단가(사후 진단용 표시에만 씀, 피처 아님).
나누는 축: 지표 × 주차(1~4) × 대상 월(연료비 변경 달 표시) × 평일·휴무일, + 월 경계 수, + 실제 수준 변화 크기.
연료비 변경 달: 대상 월 LNG 단가가 전월 대비 ±4% 이상 바뀐 달 (사후 표시. 판단 시각에는 M+2 규칙으로 알 수 없음).
실제 수준 변화: 대상 월 실제 평균 − 기준일 직전 28일 실제 평균 (같은 지표). 사후 값.
이탈 원인 분해 (오차 e = 실제 − 예측 = 월 편향 + 주 편향 + 날짜 흔들림):
  월 편향 = 같은 지표·같은 대상 월·같은 주차 전체 오차 평균 (그 달 수준을 다 같이 놓친 정도: 연료비·수급 국면)
  주 편향 = 같은 기준일·같은 주차 7일 오차 평균 − 월 편향 (그 예측 한 번이 그 주 수준을 놓친 정도)
  날짜 흔들림 = 나머지
  범위 밖 행 → 월 편향을 빼면 범위 안: '월 수준', 그래도 밖인데 주 편향까지 빼면 안: '주 수준', 그래도 밖: '날짜 흔들림'
출력: outputs/tables/MX3_D1_by_week.csv, MX3_D2_by_month.csv, MX3_D3_by_offday.csv, MX3_D4_full.csv(지표×주차×월×평일·휴무일),
      MX3_D5_by_cross.csv, MX3_D6_by_shift.csv, MX3_D7_miss_cause.csv
실행: python -m smp_forecast_app.src.mx3_diag
"""
import numpy as np
import pandas as pd

from common.loaders import load_fuel_cost
from smp_forecast_app.src.final_metrics import metrics
from smp_forecast_app.src.model_monthly import WEEK
from smp_forecast_app.src.mx_ens import SN, TAB
from smp_forecast_app.src.mx_features import MX, STATS, load

FUEL_CHG = 0.04
KEEP = ["표본수", "MAE", "단순 기준 MAE", "rMAE", "편향(공표−예측)", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수"]


def fuel_months():
    fc = load_fuel_cost().sort_values("month")
    fc["lng_chg"] = fc["fc_lng"] / fc["fc_lng"].shift(1) - 1
    fc["m"] = pd.to_datetime(fc["month"]).dt.to_period("M")
    return fc.set_index("m")["lng_chg"]


def table(s):
    """지표 s의 2025 대상일 행 + 진단용 열."""
    X = pd.read_csv(MX / "MX_predictions_daily_v2.csv", parse_dates=["base_date", "date"])
    f = load()[["base_date", "H", f"{s}_same4", "mon_cross"]]
    d = X[["base_date", "H", "date", "day_type", "offday", f"y_{s}", f"pred_{s}", f"lo_{s}", f"hi_{s}"]].rename(
        columns={f"y_{s}": "y", f"pred_{s}": "p", f"lo_{s}": "lo", f"hi_{s}": "hi"})
    # 실제 일별 값 (모든 기준일 행에서 모음) → 수준 변화 계산
    act = d.dropna(subset=["y"]).drop_duplicates("date").set_index("date")["y"].sort_index()
    act = act.reindex(pd.date_range(act.index.min(), act.index.max()))
    d = d.merge(f.rename(columns={f"{s}_same4": "same4"}), on=["base_date", "H"], how="left")
    d = d[(d["date"].dt.year == 2025) & d["y"].notna() & d["lo"].notna()].reset_index(drop=True)
    d["W"] = WEEK(d["H"])
    d["month"] = d["date"].dt.month
    lng = fuel_months()
    d["lng_chg"] = d["date"].dt.to_period("M").map(lng).astype(float)
    d["fuel_mon"] = d["lng_chg"].abs() >= FUEL_CHG
    mon_mean = act.groupby(act.index.to_period("M")).mean()
    l28 = act.rolling(28, min_periods=20).mean()       # 기준일까지 28일 (기준일 I의 실적은 I일 15시에 전부는 아님 — 사후 진단용이라 무시)
    d["shift"] = d["date"].dt.to_period("M").map(mon_mean).astype(float) - l28.reindex(d["base_date"]).values
    d["cross"] = d["mon_cross"].clip(upper=2)
    # 오차 분해
    d["e"] = d["y"] - d["p"]
    d["eM"] = d.groupby(["month", "W"])["e"].transform("mean")
    d["eW"] = d.groupby(["base_date", "W"])["e"].transform("mean") - d["eM"]
    d["eD"] = d["e"] - d["eM"] - d["eW"]
    return d


def mrow(x):
    m = metrics(x["y"], x["p"], x["lo"], x["hi"], x["same4"])
    return {k: m[k] for k in KEEP}


def group(d, s, keys):
    rows = []
    for k, x in d.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({"지표": SN[s], **dict(zip(keys, k)), **mrow(x)})
    return rows


def cause(d, s, keys):
    """이탈 원인 분해 (keys별)."""
    out = []
    inside = lambda y: (y >= d["lo"]) & (y <= d["hi"])
    miss = ~inside(d["y"])
    in1 = inside(d["y"] - d["eM"])
    in2 = inside(d["y"] - d["eM"] - d["eW"])
    d = d.assign(miss=miss, c_mon=miss & in1, c_wk=miss & ~in1 & in2, c_day=miss & ~in1 & ~in2,
                 up=d["y"] > d["hi"], e2=d["e"] ** 2, m2=d["eM"] ** 2, w2=d["eW"] ** 2, d2=d["eD"] ** 2)
    for k, x in (d.groupby(keys) if keys else [(("전체",), d)]):
        k = k if isinstance(k, tuple) else (k,)
        n = x["miss"].sum()
        out.append({"지표": SN[s], **(dict(zip(keys, k)) if keys else {"구간": "전체"}), "표본수": len(x), "이탈률": x["miss"].mean(),
                    "이탈 중 위쪽": x.loc[x["miss"], "up"].mean() if n else np.nan,
                    "이탈 중 월 수준": x["c_mon"].sum() / n if n else np.nan, "이탈 중 주 수준": x["c_wk"].sum() / n if n else np.nan,
                    "이탈 중 날짜 흔들림": x["c_day"].sum() / n if n else np.nan,
                    "제곱오차 중 월 편향": x["m2"].sum() / x["e2"].sum(), "제곱오차 중 주 편향": x["w2"].sum() / x["e2"].sum(),
                    "제곱오차 중 날짜": x["d2"].sum() / x["e2"].sum()})
    return out


def main():
    D = {k: [] for k in ("D1", "D2", "D3", "D4", "D5", "D6", "D7")}
    for s in STATS:
        d = table(s)
        D["D1"] += group(d, s, ["W"])
        mon = group(d, s, ["month"])
        lng = d.groupby("month")["lng_chg"].first()
        sh = d.groupby("month")["shift"].mean()
        for r in mon:
            r["LNG 전월 대비"], r["연료비 변경 달"] = lng[r["month"]], "●" if abs(lng[r["month"]]) >= FUEL_CHG else ""
            r["평균 실제 수준 변화"] = sh[r["month"]]
        D["D2"] += mon
        D["D3"] += group(d, s, ["offday"]) + group(d, s, ["day_type"])
        D["D4"] += group(d, s, ["W", "month", "fuel_mon", "offday"])
        D["D5"] += group(d, s, ["W", "cross"])
        d["shift_bin"] = pd.cut(d["shift"], [-np.inf, -15, -5, 5, 15, np.inf], labels=["≤−15", "−15~−5", "−5~5", "5~15", "≥15"])
        D["D6"] += group(d, s, ["shift_bin"])
        D["D7"] += cause(d, s, []) + cause(d, s, ["W"]) + cause(d, s, ["month"]) + cause(d, s, ["fuel_mon"]) + cause(d, s, ["offday"])
    names = {"D1": "by_week", "D2": "by_month", "D3": "by_offday", "D4": "full", "D5": "by_cross", "D6": "by_shift", "D7": "miss_cause"}
    ren = {"W": "주차", "month": "대상 월", "offday": "휴무일", "fuel_mon": "연료비 변경 달", "cross": "월 경계 수", "shift_bin": "실제 수준 변화(원)", "day_type": "요일유형"}
    for k, rows in D.items():
        t = pd.DataFrame(rows).rename(columns=ren)
        t.round(4).to_csv(TAB / f"MX3_{k}_{names[k]}.csv", index=False, encoding="utf-8-sig")
    print("저장:", ", ".join(f"MX3_{k}_{v}.csv" for k, v in names.items()))


if __name__ == "__main__":
    main()
