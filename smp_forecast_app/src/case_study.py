"""11/11 발표 과제 — 급락·급등 네 가지 사례의 원인 분석용 자료 (2026-11-04, 총괄).

현업 1028 피드백 '분석 순서'대로 날짜마다:
  1) 그래프: 시간별 SMP(그날·전날·지난주 같은 요일), 하루전 수요예측, 태양광 실적(5분 → 시간 평균)
  2) 비교 숫자: 시간대별 SMP·수요예측·태양광·원전·LNG, 직전 28일 일 최고 중앙값(급등 기준)
  3) 그날 자료: 실측 날씨(ASOS 7개 지점), 기상특보, 공급능력·예비율(EPSIS), 원전 가용, 출력제어 공지, 가격결정 연료(시간 수)
  4) 가격과 같은 단가의 발전기 (그달 변동비 순위, 현업 공유 자료 — 단순 비교)
  5) 우리 모델의 판단 (급락·급등 등급·확률, 근거 문장)
시간 표기: 전력시장 거래시간 h = 실제 (h−1)시 ~ h시.
출력: outputs/case_study/{날짜}.png, case_facts.json, (요약) case_summary_events.csv
실행: python -m smp_forecast_app.src.case_study
"""
import json

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import RAW_DIR, ROOT
from common.loaders import (load_asos, load_kpx_power_source, load_land_curtailment, load_nuclear_available,
                            load_smp_actual, load_smp_forecast_api)
from common.plot import plt, setup_korean_font
from surge_classifier.src.extra_features import load_supply, load_warnings

OUT = ROOT / "smp_forecast_app" / "outputs" / "case_study"
CASES = {"2026-06-13": "휴일 급락", "2026-09-06": "휴일 급락", "2026-03-26": "평일 급락",
         "2024-05-04": "휴일 급등", "2026-01-22": "평일 급등"}
STN = [129, 146, 156, 165, 136, 143, 192]   # 서산·전주·광주·목포·안동·대구·진주 (구역예보와 같은 7곳)
WRN = {"C": "한파", "H": "폭염", "W": "강풍", "R": "호우", "S": "대설", "D": "건조", "V": "풍랑", "T": "태풍", "Y": "황사"}


def load_all():
    smp = load_smp_actual("육지")[["date", "hour", "smp"]]
    fc = load_smp_forecast_api()
    fc = fc[fc["region"] == "육지"][["date", "hour", "demand_fcst"]]
    ps = load_kpx_power_source()
    ps["solar"] = ps[["solar_btm", "solar_ppa", "solar_mkt"]].sum(axis=1)
    ps["hour"] = ps["ts"].dt.hour + 1                     # 5분 시각 → 거래시간 (00:00~00:55 = 1시)
    ph = ps.groupby(["date", "hour"])[["solar", "nuclear", "gas", "coal_bit", "pump"]].mean().reset_index()
    h = smp.merge(fc, on=["date", "hour"], how="left").merge(ph, on=["date", "hour"], how="left")
    day = h.groupby("date").agg(mx=("smp", "max"), mean=("smp", "mean"))
    day["mid_min"] = h[h["hour"].between(11, 15)].groupby("date")["smp"].min()
    day["ref"] = day["mx"].shift(1).rolling(28, min_periods=14).median()
    cal = build_calendar(day.index.min(), day.index.max() + pd.Timedelta(days=1)).set_index("date")
    day["off"] = cal["is_offday"].reindex(day.index).astype(bool)
    return h, day


def fuel_counts(d):
    m = pd.Timestamp(d).strftime("%Y%m")
    p = RAW_DIR / "api" / "smp_fuel" / f"smp_fuel_{m}.csv"
    if not p.exists():
        return {}
    f = pd.read_csv(p)
    f = f[(f["areaNm"] == "육지") & (f["tradeDay"].astype(str) == pd.Timestamp(d).strftime("%Y%m%d"))]
    return {r.fuelType: int(r.cnt) for r in f.itertuples()}


def units_near(d, price, tol=1.0):
    v = pd.read_csv(RAW_DIR / "변동비순위_발전기별_202203-202611.csv", encoding="utf-8-sig")
    v = v[v["연월"] == pd.Timestamp(d).strftime("%Y-%m")]
    if v.empty:
        return []
    near = v[(v["변동비_원kWh"] - price).abs() <= tol].sort_values("변동비_원kWh")
    return [f"{r.발전기명}({r.연료구분}, {r.변동비_원kWh:.2f}원)" for r in near.head(5).itertuples()]


def model_view(d):
    t = ROOT / "drop_classifier" / "outputs" / "tables" / "drop_육지_15h.csv"
    s = ROOT / "surge_classifier" / "outputs" / "tables" / "surge_육지_15h.csv"
    dr, su = pd.read_csv(t), pd.read_csv(s)
    r = {}
    x = dr[dr["date"] == d]
    if len(x):
        r["급락"] = {"등급": x["grade"].iloc[0], "확률": round(float(x["prob"].iloc[0]), 3), "예상 한낮 최저(원)": round(float(x["score"].iloc[0]), 1)}
    y = su[su["date"] == d]
    if len(y):
        r["급등"] = {"등급": y["grade"].iloc[0], "확률": round(float(y["prob"].iloc[0]), 3)}
    why = ROOT / "smp_forecast_app" / "outputs" / "tables" / "R5_why_육지.json"
    if why.exists():
        w = json.loads(why.read_text(encoding="utf-8"))
        for k, nm in (("drop", "급락 근거"), ("surge", "급등 근거")):
            if d in w.get(k, {}):
                r[nm] = [f"{'▲' if z['dir'] == 'up' else '▼' if z['dir'] == 'down' else '·'} {z['t']}" for z in w[k][d]]
    return r


def case(d, kind, h, day, asos, sup, nuc, curt, wrn):
    D = pd.Timestamp(d)
    g = lambda x: h[h["date"] == x].set_index("hour")
    t, p, wk, n = g(D), g(D - pd.Timedelta(days=1)), g(D - pd.Timedelta(days=7)), g(D + pd.Timedelta(days=1))
    # 그래프
    setup_korean_font()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    hrs = range(1, 25)
    ax.plot(hrs, t["smp"].reindex(hrs), color="#d62728", lw=2.4, marker="o", ms=3.5, label=f"SMP {D:%m/%d}")
    ax.plot(hrs, p["smp"].reindex(hrs), color="#7f7f7f", lw=1.3, ls="--", label="SMP 전날")
    ax.plot(hrs, wk["smp"].reindex(hrs), color="#1f77b4", lw=1.3, ls=":", label="SMP 지난주 같은 요일")
    ax.set_xlabel("거래시간 (h = 실제 h−1 ~ h시)")
    ax.set_ylabel("SMP (원/kWh)")
    ax.set_xticks(range(1, 25, 2))
    ax.axvspan(10.5, 15.5, color="#eeeeee", zorder=0)
    ax2 = ax.twinx()
    ax2.plot(hrs, t["demand_fcst"].reindex(hrs) / 1000, color="#2ca02c", lw=1.6, label="하루전 수요예측 (GW)")
    ax2.plot(hrs, t["solar"].reindex(hrs) / 1000, color="#ff9f1c", lw=1.6, label="태양광 실적 (GW)")
    ax2.set_ylabel("GW")
    ls = ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()
    ax.legend(ls[0][0] + ls[1][0], ls[0][1] + ls[1][1], fontsize=8, loc="upper left", ncol=2)
    ax.set_title(f"{D:%Y-%m-%d} ({'월화수목금토일'[D.weekday()]}) {kind} — 시간별 SMP·수요예측·태양광 (회색 = 한낮 11~15시)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{d}.png", dpi=150)
    plt.close(fig)

    # 시간대 표 (거래시간 기준)
    bands = {"새벽 1~7시": range(1, 8), "아침 8~10시": range(8, 11), "한낮 11~15시": range(11, 16), "오후 16~18시": range(16, 19), "저녁 19~22시": range(19, 23), "밤 23~24시": range(23, 25)}
    rows = []
    for nm, r in bands.items():
        rows.append({"시간대(거래시간)": nm, "SMP": [round(float(t['smp'].reindex(r).min()), 1), round(float(t['smp'].reindex(r).max()), 1)],
                     "전날 SMP": round(float(p["smp"].reindex(r).mean()), 1), "지난주 SMP": round(float(wk["smp"].reindex(r).mean()), 1),
                     "수요예측(GW)": round(float(t["demand_fcst"].reindex(r).mean()) / 1000, 1),
                     "전날 수요예측(GW)": round(float(p["demand_fcst"].reindex(r).mean()) / 1000, 1),
                     "태양광(GW)": round(float(t["solar"].reindex(r).mean()) / 1000, 1), "전날 태양광(GW)": round(float(p["solar"].reindex(r).mean()) / 1000, 1),
                     "원전(GW)": round(float(t["nuclear"].reindex(r).mean()) / 1000, 1), "LNG(GW)": round(float(t["gas"].reindex(r).mean()) / 1000, 1),
                     "수요예측−태양광(GW)": round(float((t["demand_fcst"] - t["solar"]).reindex(r).mean()) / 1000, 1)})
    # 날씨 (7개 지점 평균)
    a = asos[(asos["date"] == D) & asos["stn_id"].isin(STN)]
    ap = asos[(asos["date"] == D - pd.Timedelta(days=1)) & asos["stn_id"].isin(STN)]
    am = a[a["hour"].between(11, 15)]
    wx = {"기온 평균·최저·최고(℃)": [round(a["ta"].mean(), 1), round(a.groupby("stn_id")["ta"].min().mean(), 1), round(a.groupby("stn_id")["ta"].max().mean(), 1)],
          "전날 기온 평균(℃)": round(ap["ta"].mean(), 1), "한낮 운량(0~10)": round(am["cloud"].mean(), 1), "한낮 일조(시간/시간)": round(am["ss"].mean(), 2),
          "한낮 일사(MJ/m²)": round(am["icsr"].mean(), 2), "강수 시간 수": int((a["rn"].fillna(0) > 0).sum())}
    ww = wrn[(wrn["tm_in"].dt.normalize() <= D) & (wrn["tm_in"].dt.normalize() >= D - pd.Timedelta(days=2))]
    wx["특보(발표, 전후 2일)"] = sorted({WRN.get(c, c) for c in ww["wrn"]})
    s = sup.reindex([D - pd.Timedelta(days=1), D])
    supply = {"공급능력(GW) 전날→그날": [round(v / 1000, 1) for v in s["supply"]], "최대전력(GW)": [round(v / 1000, 1) for v in s["peak"]],
              "공급예비율(%)": list(s["rr"]), "최대전력 시각": str(s["peak_time"].iloc[-1]),
              "원전 가용(GW)": round(float(nuc.set_index("date")["nuc_avail_mw"].get(D, np.nan)) / 1000, 1),
              "출력제어 공지(태양광·풍력)": curt.set_index("date").reindex([D])[["curt_solar", "curt_wind"]].fillna(0).astype(int).iloc[0].tolist()}
    dd = day.loc[D]
    fx = {"날짜": d, "유형": kind, "요일": "월화수목금토일"[D.weekday()], "휴무일": bool(dd["off"]),
          "일 최고": round(float(dd["mx"]), 2), "일 최고 시간": [int(x) for x in t.index[t["smp"] >= dd["mx"] - 0.01]],
          "한낮 최저": round(float(dd["mid_min"]), 2), "한낮 최저 시간": [int(x) for x in t.index[(t["smp"] <= dd["mid_min"] + 0.01) & t.index.isin(range(11, 16))]],
          "직전 28일 일 최고 중앙값": round(float(dd["ref"]), 2), "배율": round(float(dd["mx"] / dd["ref"]), 3),
          "가격결정 연료(시간 수)": fuel_counts(D), "시간대": rows, "날씨": wx, "공급": supply,
          "같은 단가 발전기(일 최고 ±1원)": units_near(D, float(dd["mx"])), "같은 단가 발전기(평소 가격 ±1원)": units_near(D, float(p["smp"].median())),
          "우리 모델": model_view(d), "SMP 24시간": [round(float(x), 2) for x in t["smp"].reindex(hrs)]}
    return fx


def events_summary(h, day, asos, sup):
    """급락·급등 사유 요약용: 2024-01 ~ 2026-10 사건일을 평일·휴무일로 나눠 평소 날과 비교."""
    d = day[(day.index >= "2024-01-01") & (day.index <= "2026-10-31")].copy()
    d["drop"] = d["mid_min"] < 30
    d["surge"] = d["mx"] >= d["ref"] * 1.25
    hm = h[h["hour"].between(11, 15)].groupby("date").agg(dem_mid=("demand_fcst", "mean"), sol_mid=("solar", "mean"), nuc=("nuclear", "mean"))
    hp = h.groupby("date").agg(dem_max=("demand_fcst", "max"))
    a = asos[asos["stn_id"].isin(STN)].groupby("date").agg(ta=("ta", "mean"))
    am = asos[asos["stn_id"].isin(STN) & asos["hour"].between(11, 15)].groupby("date").agg(cloud=("cloud", "mean"), ss=("ss", "mean"))
    d = d.join(hm).join(hp).join(a).join(am).join(sup[["rr"]])
    d["net_mid"] = d["dem_mid"] - d["sol_mid"]
    d["month"] = d.index.month
    d["dom"] = d.index.day
    d.to_csv(OUT / "case_summary_events.csv", encoding="utf-8-sig")
    out = {}
    for ev in ("drop", "surge"):
        for off in (True, False):
            base = d[(d["off"] == off) & ~d[ev]]
            x = d[(d["off"] == off) & d[ev]]
            if not len(x):
                continue
            k = f"{'급락' if ev == 'drop' else '급등'} · {'휴무일' if off else '평일'}"
            out[k] = {"일수": len(x), "평소 일수": len(base),
                      "월 분포": x["month"].value_counts().sort_index().to_dict(),
                      **{nm: [round(float(x[c].mean()), 2), round(float(base[c].mean()), 2)] for nm, c in
                         (("한낮 수요예측(MW) 사건·평소", "dem_mid"), ("한낮 태양광(MW)", "sol_mid"), ("한낮 수요−태양광(MW)", "net_mid"),
                          ("원전(MW)", "nuc"), ("하루 최대 수요예측(MW)", "dem_max"), ("기온 평균(℃)", "ta"), ("한낮 운량", "cloud"),
                          ("한낮 일조", "ss"), ("공급예비율(%)", "rr"), ("일 최고 ÷ 28일 기준", "mx"))},
                      "월 1~3일 비율": round(float((x["dom"] <= 3).mean()), 2)}
            if ev == "surge":
                out[k]["일 최고 ÷ 28일 기준"] = [round(float((x["mx"] / x["ref"]).mean()), 2), round(float((base["mx"] / base["ref"]).mean()), 2)]
    return out


def main():
    h, day = load_all()
    asos = load_asos()
    sup = load_supply()
    sup.index = pd.to_datetime(sup.index)
    nuc, curt, wrn = load_nuclear_available(), load_land_curtailment(), load_warnings()
    facts = {d: case(d, k, h, day, asos, sup, nuc, curt, wrn) for d, k in CASES.items()}
    facts["_요약"] = events_summary(h, day, asos, sup)
    (OUT / "case_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
