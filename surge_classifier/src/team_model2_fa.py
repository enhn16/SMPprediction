"""④-6 헛경보 줄이기 후보 (v2 레시피 위에 하나씩) — PLAN.md ④-6에 사전 등록한 후보·규칙.

A 휴무일 제외 / B 흐린 평일·주간전망 입력 추가 / C A+B / D 평균 가중 (0.5, 0.25, 0.25)
고르는 지표 = 2024~2025 검증 주의 문턱의 헛경보 수 (재현율은 규칙으로 80% 이상), v2보다 10% 넘게 줄 때만 채택.
실행: python -m surge_classifier.src.team_model2_fa          → 검증만, outputs/tables/07_fa_select_육지.csv
      python -m surge_classifier.src.team_model2_fa test     → 선택 후보의 2026 (이미 본 구간, 참고)
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from surge_classifier.src import team_model2 as T2
from surge_classifier.src.label_days import OUT_DATA, OUT_TAB

BASE_FEATS, BASE_SIGNS = list(T2.FEATS), dict(T2.SIGNS)
ADD = {"wk_sky_pm": +1, "wk_cloud_mid_jma": +1, "kpx_peak": +1}
CANDS = {"v2 (기준)": dict(), "A 휴무일 제외": dict(offday=True), "B 흐린 평일·주간전망 추가": dict(add=True),
         "C A+B": dict(offday=True, add=True), "D 가중 (0.5, 0.25, 0.25)": dict(w=(.5, .25, .25))}


def data():
    d = T2.build_daily()
    f = pd.read_csv(OUT_DATA / "features_육지.csv", parse_dates=["date"]).set_index("date")
    e = pd.read_csv(OUT_DATA / "extra_features_육지.csv", parse_dates=["date"]).set_index("date")
    x = f[["wk_sky_pm", "kpx_peak"]].join(e[["wk_cloud_mid_jma"]])
    return d.join(x, on="date")


def configure(c):
    T2.FEATS[:] = BASE_FEATS + (list(ADD) if c.get("add") else [])
    T2.SIGNS.clear()
    T2.SIGNS.update(BASE_SIGNS | (ADD if c.get("add") else {}))


def combine(p: pd.DataFrame, d, c) -> pd.Series:
    w = c.get("w", (1 / 3, 1 / 3, 1 / 3))
    s = sum(wi * p[k] for wi, k in zip(w, T2.KEYS))
    if c.get("offday"):
        s = s.where(d.loc[p.index, "is_off"] == 0, 0.0)
    return s


def summary(y, s, th) -> dict:
    y, s = np.asarray(y, int), np.asarray(s, float)
    (wt, wf), (at, af) = T2.counts(y, s, th["주의"]), T2.counts(y, s, th["경보"])
    return {"PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s), "주의 문턱": th["주의"],
            "주의 적중": wt, "주의 헛경보": wf, "경보 문턱": th["경보"], "경보 적중": at, "경보 헛경보": af}


def run(test=False):
    d = data()
    rows = []
    for name, c in CANDS.items():
        configure(c)
        val = T2.walk_forward(d)
        s = combine(val, d, c)
        y = d.loc[s.index, "y"].values
        th = {"주의": T2.caution_line(y, s.values), "경보": T2.alert_line(y, s.values)}
        r = {"후보": name, "구간": "검증 2024~2025", "급등일": int(y.sum()), **summary(y, s, th)}
        for yr in ("2024", "2025"):
            m = (d.loc[s.index, "date_s"].str[:4] == yr).values
            r[f"{yr} 주의 적중/헛경보"] = "%d / %d" % T2.counts(y[m], s.values[m], th["주의"])
        rows.append(r)
        if test:
            te = d[(d["date_s"] >= T2.TEST[0]) & (d["date_s"] <= T2.TEST[1]) & d["y"].notna()]
            st = combine(T2.predict(T2.final_models(d), te), d, c)
            rows.append({"후보": name, "구간": "2026 공식 ~10-18 (참고: 이미 본 구간)", "급등일": int(te["y"].sum()),
                         **summary(te["y"].values, st.values, th)})
    configure({})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / ("07_fa_test_육지.csv" if test else "07_fa_select_육지.csv"), index=False, encoding="utf-8-sig")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    run(test=len(sys.argv) > 1 and sys.argv[1] == "test")
