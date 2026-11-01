"""④-9 실험: 변동비 순위 자료가 급등 경보 v3를 낫게 하는가 (PLAN.md ④-9에 사전 등록한 후보·규칙). v3 산출물은 건드리지 않는다.

자료: data/raw/변동비순위_발전기별_202203-202611.csv, 변동비_연료별_월별요약_202203-202611.csv (현업 공유 2026-10-30, 13종 밖)
공개 시점: S1 = 그달 1일 공개 (대상일 3일 이후 그달 표, 1~2일은 전달 표) / S2 = 전달 25일 공개 (대상일 그달 표). 없는 달은 직전 달 표.
피처 (p = ref_max, 직전 28일 일 최고 중앙값): vc_next_ratio, vc_gap_ratio, vc_slope3, vc_band_gw, vc_headroom_gw, vc_chg
실행: python -m surge_classifier.src.varcost_exp        → 검증만, outputs/tables/08_varcost_select_육지.csv
      python -m surge_classifier.src.varcost_exp test   → 채택 후보만 2026 (참고), 08_varcost_test_육지.csv
"""
import sys

import numpy as np
import pandas as pd

from common.config import RAW_DIR
from surge_classifier.src import team_model2 as T2
from surge_classifier.src import team_model2_fa as FA
from surge_classifier.src import team_model3 as V3
from surge_classifier.src.label_days import OUT_TAB

RANK = RAW_DIR / "변동비순위_발전기별_202203-202611.csv"
SUMMARY = RAW_DIR / "변동비_연료별_월별요약_202203-202611.csv"
VC_SIGNS = {"vc_next_ratio": +1, "vc_gap_ratio": +1, "vc_slope3": +1, "vc_band_gw": -1, "vc_headroom_gw": -1, "vc_chg": +1}
CANDS = {"v3 (기준)": None, "S1 · 입력+부호": ("S1", True), "S2 · 입력+부호": ("S2", True),
         "S1 · 입력만": ("S1", False), "S2 · 입력만": ("S2", False)}


def ladder_tables():
    r = pd.read_csv(RANK, encoding="utf-8-sig").sort_values(["연월", "변동비순위"])
    curves = {m: (g["변동비_원kWh"].values, g["누적출력_MW"].values) for m, g in r.groupby("연월")}
    s = pd.read_csv(SUMMARY, encoding="utf-8-sig").pivot_table(index="연월", columns="연료구분", values="변동비_출력가중평균")
    return curves, s[["LNG", "LNG(직도입)"]]


def table_month(date: pd.Timestamp, scenario: str) -> pd.Period:
    m = date.to_period("M")
    return m - 1 if scenario == "S1" and date.day <= 2 else m


def vc_features(d: pd.DataFrame, scenario: str) -> pd.DataFrame:
    curves, summ = ladder_tables()
    months = sorted(curves)
    out = []
    for date, p, dem in zip(d["date"], d["ref_max"], d["prev_dem_max"]):
        key = str(table_month(date, scenario))
        avail = [m for m in months if m <= key]
        if not avail or p != p:
            out.append([np.nan] * 6)
            continue
        m = avail[-1]
        prev_m = [x for x in summ.index if x < m]
        cost, cum = curves[m]
        cum_at = lambda price: float(cum[cost <= price].max()) if (cost <= price).any() else 0.0
        c_ref = cum_at(p)
        i3 = np.searchsorted(cum, c_ref + 3000)
        cost3 = cost[min(i3, len(cost) - 1)]
        lng, dlng = summ.loc[m, "LNG"], summ.loc[m, "LNG(직도입)"]
        chg = lng / summ.loc[prev_m[-1], "LNG"] if prev_m else np.nan
        out.append([lng / p, (lng - dlng) / p, cost3 / p - 1, (cum_at(1.25 * p) - c_ref) / 1000,
                    (cum_at(1.25 * p) - dem) / 1000 if dem == dem else np.nan, chg])
    return pd.DataFrame(out, index=d.index, columns=list(VC_SIGNS))


def configure(spec):
    FA.configure(V3.CFG)
    if spec:
        T2.FEATS.extend(VC_SIGNS)
        if spec[1]:
            T2.SIGNS.update(VC_SIGNS)


def run(test=False):
    base = V3.data()
    vc = {sc: vc_features(base, sc) for sc in ("S1", "S2")}
    rows = []
    for name, spec in CANDS.items():
        if test and spec is not None and name not in TEST_PICK:
            continue
        d = base.join(vc[spec[0]]) if spec else base
        configure(spec)
        try:
            val = T2.walk_forward(d)
            s = FA.combine(val, d, V3.CFG)
            y = d.loc[s.index, "y"].values.astype(int)
            th = {"주의": T2.caution_line(y, s.values), "경보": T2.alert_line(y, s.values)}
            r = {"후보": name, "구간": "검증 2024~2025", "급등일": int(y.sum()), **FA.summary(y, s, th)}
            for yr in ("2024", "2025"):
                m = (d.loc[s.index, "date_s"].str[:4] == yr).values
                r[f"{yr} 주의 적중/헛경보"] = "%d / %d" % T2.counts(y[m], s.values[m], th["주의"])
            rows.append(r)
            if test:
                te = d[(d["date_s"] >= V3.TEST[0]) & (d["date_s"] <= V3.TEST[1]) & d["y"].notna()]
                st = FA.combine(T2.predict(T2.final_models(d), te), d, V3.CFG)
                rows.append({"후보": name, "구간": "2026 공식 ~10-18 (참고)", "급등일": int(te["y"].sum()),
                             **FA.summary(te["y"].values, st.values, th)})
        finally:
            FA.configure({})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / ("08_varcost_test_육지.csv" if test else "08_varcost_select_육지.csv"), index=False, encoding="utf-8-sig")
    print(out.round(3).to_string(index=False))
    return vc


TEST_PICK = sys.argv[2:] if len(sys.argv) > 2 else []   # test 모드에서 채점할 채택 후보 이름

if __name__ == "__main__":
    run(test=len(sys.argv) > 1 and sys.argv[1] == "test")
