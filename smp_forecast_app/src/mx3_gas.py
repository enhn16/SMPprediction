"""한 달 예측 mx3 — 가스공사 발전용 천연가스 요금(13종 밖, 실험) 정보 가치 확인. 운영 모델은 그대로.

자료: 한국가스공사 홈페이지 '발전용 천연가스 요금' 연도별 엑셀 (일반발전 합계, 원/GJ).
  게시 시점(파일 수정 시각): 다음 달 요금이 전달 26~27일쯤 게시 (2022-11-29, 2024-11-27, 2025-11-26, 2026-10-27 17:57)
  → 판단 시각 기준 가정(보수): 이번 달 요금은 항상 앎, 다음 달 요금은 기준일이 28일 이상일 때만 앎. 그보다 먼 달은 마지막으로 아는 값.
방법 (결과 보기 전 고정):
  1) 일평균 LightGBM(mx_tab lgb_r와 같은 설정·목표 = same4 비율, 매월 재학습, 학습 대상일 2023~)을 가스 피처 없이/있게 두 번 학습
     가스 피처: 대상월에 쓸 수 있는 요금(천원/GJ), 그 요금 ÷ 기준일 7일 전 달 요금 − 1, 그 요금 ÷ 최근 28일 일평균 SMP, 대상월 요금을 아는지(0/1)
  2) mx3 시간별 예측 × (가스 있음 ÷ 가스 없음, 같은 기준일·대상일) = 가스 피처만의 효과
  3) 2025(검증)·2026(시험, 이미 본 구간 — 정보 가치 확인용) 시간별·일평균 MAE, 수준 변화 달 따로
입력: KOGAS 엑셀 폴더(인자), mx 피처, MX3_predictions_hourly
출력: outputs/tables/MX3_G1_gas_test.csv, MX3_G2_gas_month.csv
실행: python -m smp_forecast_app.src.mx3_gas <엑셀 폴더>
"""
import sys

import numpy as np
import pandas as pd

from smp_forecast_app.src.mx_ens import TAB
from smp_forecast_app.src.mx_features import FEATURES, MX, load
from smp_forecast_app.src.mx_tab import LGB, START, TRAIN_FROM

KNOW_DAY = 28


def tariff(folder):
    """연도별 엑셀 → 월별 일반발전 합계(원/GJ)."""
    rows = []
    for y in range(2021, 2027):
        x = pd.read_excel(f"{folder}/{y}.xlsx", header=None)
        mon, kind = x.iloc[2].ffill(), x.iloc[3]
        gj = x.index[x[0].astype(str).str.contains("원/GJ")][0]
        for j in range(2, x.shape[1]):
            m = str(mon[j]).replace("월", "")
            v = pd.to_numeric(x.iloc[gj + 2, j], errors="coerce")
            if m.isdigit() and kind[j] == "일반발전" and v > 0:
                rows.append((pd.Period(f"{y}-{int(m):02d}", "M").ordinal, v))
    return pd.Series(dict(rows))


def gas_features(d, kg):
    tm = d["date"].dt.to_period("M").map(lambda p: p.ordinal)
    bm = d["base_date"].dt.to_period("M").map(lambda p: p.ordinal)
    known = bm + (d["base_date"].dt.day >= KNOW_DAY).astype(int)
    use = np.minimum(tm, known)
    ref = (d["base_date"] - pd.Timedelta(days=7)).dt.to_period("M").map(lambda p: p.ordinal)
    g = pd.DataFrame(index=d.index)
    g["gas_kg"] = kg.reindex(use).values / 1000
    g["gas_chg"] = kg.reindex(use).values / kg.reindex(ref).values - 1
    g["gas_ratio"] = kg.reindex(use).values / 1000 / d["mean_l28"].clip(lower=10) * 100
    g["gas_known"] = (tm <= known).astype(int).values
    return g


def walk(d, cols, y, base):
    from lightgbm import LGBMRegressor
    yt = (y / base).clip(0, 3)
    pred = pd.Series(np.nan, index=d.index)
    ps = list(pd.date_range(START, d["base_date"].max(), freq="MS")) + [pd.Timestamp("2100-01-01")]
    for a, b in zip(ps[:-1], ps[1:]):
        tr = (d["date"] >= TRAIN_FROM) & (d["date"] < a) & yt.notna() & base.notna()
        te = (d["base_date"] >= a) & (d["base_date"] < b)
        if te.sum():
            pred[te] = LGBMRegressor(**LGB).fit(d.loc[tr, cols], yt[tr]).predict(d.loc[te, cols])
    return pred * base


def main(folder):
    kg = tariff(folder)
    d = load()
    d = pd.concat([d, gas_features(d, kg)], axis=1)
    base = d["mean_same4"].clip(lower=10)
    G = ["gas_kg", "gas_chg", "gas_ratio", "gas_known"]
    d["p0"] = walk(d, list(FEATURES), d["y_mean"], base)
    d["p1"] = walk(d, list(FEATURES) + G, d["y_mean"], base)
    d["ratio"] = (d["p1"] / d["p0"]).clip(0.7, 1.4)
    h = pd.read_csv(MX / "MX3_predictions_hourly.csv.gz", parse_dates=["base_date", "date"])
    h = h.merge(d[["base_date", "H", "ratio", "p0", "p1", "y_mean"]], on=["base_date", "H"], how="left")
    h["pred_gas"] = h["pred"] * h["ratio"]
    h = h[h["y"].notna() & h["ratio"].notna()]
    dl = h.groupby(["base_date", "H", "date"]).agg(y=("y", "mean"), mx3=("pred", "mean"), mx3_gas=("pred_gas", "mean"),
                                                   lgb=("p0", "first"), lgb_gas=("p1", "first")).reset_index()
    per = {"2025 (검증)": ("2025-01-01", "2025-12-31"), "2026 (시험·이미 본 구간)": ("2026-01-01", "2026-10-18")}
    rows = []
    for pn, (a, b) in per.items():
        hh, dd = h[(h["date"] >= a) & (h["date"] <= b)], dl[(dl["date"] >= a) & (dl["date"] <= b)]
        for seg, mh, md in (("전체", slice(None), slice(None)),
                            ("대상월 요금을 앎", h.loc[hh.index, "date"].dt.to_period("M") <= h.loc[hh.index, "base_date"].dt.to_period("M") + (h.loc[hh.index, "base_date"].dt.day >= KNOW_DAY).astype(int), None)):
            x = hh if seg == "전체" else hh[mh]
            rows.append({"기간": pn, "구간": seg, "시간별 행": len(x), "mx3 시간별 MAE": (x["y"] - x["pred"]).abs().mean(),
                         "mx3+가스 시간별 MAE": (x["y"] - x["pred_gas"]).abs().mean()})
        rows[-2].update({"mx3 일평균 MAE": (dd["y"] - dd["mx3"]).abs().mean(), "mx3+가스 일평균 MAE": (dd["y"] - dd["mx3_gas"]).abs().mean(),
                         "LightGBM 일평균 MAE": (dd["y"] - dd["lgb"]).abs().mean(), "LightGBM+가스 일평균 MAE": (dd["y"] - dd["lgb_gas"]).abs().mean()})
    t = pd.DataFrame(rows)
    t.round(3).to_csv(TAB / "MX3_G1_gas_test.csv", index=False, encoding="utf-8-sig")
    m = dl[dl["date"] >= "2025-01-01"].copy()
    m["월"] = m["date"].dt.strftime("%Y-%m")
    mt = m.groupby("월").apply(lambda g: pd.Series({"mx3": (g.y - g.mx3).abs().mean(), "mx3+가스": (g.y - g.mx3_gas).abs().mean(),
                                                    "LightGBM": (g.y - g.lgb).abs().mean(), "LightGBM+가스": (g.y - g.lgb_gas).abs().mean()}))
    mt.round(2).to_csv(TAB / "MX3_G2_gas_month.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print(mt.round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
