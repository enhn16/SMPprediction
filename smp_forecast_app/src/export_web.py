"""웹앱 데이터 내보내기 → smp_forecast_app/web/data/forecast_data.js (window.SMP_DATA = {...}). 자체 모델만.

화면 구성 (2026-10-23 개편)
  일주일 예측   : 최신 기준일의 앞으로 7일(시간별 168시간) 예측 + 80% 범위, 내일(D+1)에 과제1 급락 배지
  한 달 예측    : 최신 기준일의 7~34일 뒤 일별(일평균·한낮·저녁·한낮 최저) 예측 + 범위, 주·월 평균
  이전 흐름 확인 : 달력에서 기간(최대 14일)을 골라 지나간 예측 vs 공표 SMP (다음 날·일주일 D+1~7·한 달 7·14·21·28일 전)
  참고 문서     : 개요·다음 날·일주일·한 달 — 모델, 비교 모델과 성능, 선택 이유, 사용 데이터, 표준 성적표 (docs)
  설명
입력: WZ_predictions_hourly.csv(일주일, wz_ens) · MZ_predictions_daily.csv(한 달, mz_ens — 없으면 M_predictions_daily.csv)
      ops 운영 기록(있으면 같은 대상 덮어씀) · drop_badge.csv / ops drop_log.csv (급락 확률)
실행: python -m smp_forecast_app.src.export_web
"""
import json

import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from common.loaders import load_smp_actual
from smp_forecast_app.src.features_daily import daily_stats

APP = ROOT / "smp_forecast_app"
TAB = APP / "outputs" / "tables"
DATA = PROCESSED_DIR / "smp_forecast_app"
OPS = DATA / "ops"
OUT = APP / "web" / "data" / "forecast_data.js"
STATS = ["mean", "mid", "eve", "midmin"]
PAST_SINCE = "2025-07-01"
DAILY_H = [7, 14, 21, 28]


def _n(v):
    return None if pd.isna(v) else round(float(v), 1)


def _h24(g, col):
    return [_n(v) for v in g.set_index("hour").reindex(range(1, 25))[col]]


def _table(name, cols=None):
    p = TAB / name
    if not p.exists():
        return []
    t = pd.read_csv(p)
    if cols:
        t = t[[c for c in cols if c in t.columns]]
    return json.loads(t.round(4).to_json(orient="records", force_ascii=False))


def _json(name):
    p = TAB / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _ops(name, **kw):
    p = OPS / name
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def hourly_frame():
    w = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["base_date", "date"])[["base_date", "H", "date", "hour", "pred", "lo", "hi"]]
    oh = _ops("hourly_log.csv", parse_dates=["base_date", "date"])
    if len(oh):
        key = lambda x: x["base_date"].astype(str) + "_" + x["H"].astype(str)
        o = oh[["base_date", "H", "date", "hour", "pred", "lo", "hi"]]
        w = pd.concat([w[~key(w).isin(key(o))], o])
    return w


def daily_frame():
    p = DATA / "MZ_predictions_daily.csv"
    m = pd.read_csv(p if p.exists() else DATA / "M_predictions_daily.csv", parse_dates=["base_date", "date"])
    om = _ops("daily_log.csv", parse_dates=["base_date", "date"])
    if len(om):
        m = pd.concat([m[~m["base_date"].isin(om["base_date"])], om])
    return m


def main():
    act = load_smp_actual("육지")[["date", "hour", "smp"]].rename(columns={"smp": "y"})
    last_pub = act["date"].max()
    w = hourly_frame()
    cal = build_calendar(w["date"].min() - pd.Timedelta(days=40), w["date"].max() + pd.Timedelta(days=40)).set_index("date")
    actual = {d.strftime("%Y-%m-%d"): _h24(g, "y") for d, g in act[act["date"] >= PAST_SINCE].groupby("date")}
    dayinfo = {d.strftime("%Y-%m-%d"): {"dt": r.day_type, "off": bool(r.is_offday)} for d, r in cal.iterrows()}

    drop = pd.read_csv(DATA / "drop_badge.csv") if (DATA / "drop_badge.csv").exists() else pd.DataFrame(columns=["date", "grade", "pred_min", "prob"])
    od = _ops("drop_log.csv")
    if len(od):
        drop = pd.concat([drop[~drop["date"].isin(od["date"])], od])
    drops = {r.date: {"grade": r.grade, "min": _n(r.pred_min), "prob": round(float(r.prob), 3)} for r in drop.itertuples()}

    # 일주일: 7개 지평이 모두 있는 가장 최근 기준일
    full = w.groupby("base_date")["H"].nunique()
    base = full[full == 7].index.max()
    wk = w[w["base_date"] == base].sort_values(["H", "hour"])
    week = {"base": base.strftime("%Y-%m-%d"), "days": []}
    for H, g in wk.groupby("H"):
        d = g["date"].iloc[0].strftime("%Y-%m-%d")
        week["days"].append({"date": d, "H": int(H), **dayinfo.get(d, {}), "p": _h24(g, "pred"), "lo": _h24(g, "lo"),
                             "hi": _h24(g, "hi"), "y": actual.get(d, [None] * 24)})
    week["drop"] = drops.get(week["days"][0]["date"])

    # 지난 예측 (시간별): 지평별 대상일 → [p, lo, hi]
    past_h = {}
    for H, g in w[w["date"] >= PAST_SINCE].groupby("H"):
        past_h[int(H)] = {d.strftime("%Y-%m-%d"): [_h24(x, "pred"), _h24(x, "lo"), _h24(x, "hi")] for d, x in g.groupby("date")}

    # 한 달
    m = daily_frame()
    fullm = m.groupby("base_date")["H"].nunique()
    mbase = fullm[fullm >= 28].index.max()
    mm = m[m["base_date"] == mbase].sort_values("H")
    ds = daily_stats()
    month = {"base": mbase.strftime("%Y-%m-%d"), "d": mm["date"].dt.strftime("%Y-%m-%d").tolist(),
             "dt": [dayinfo.get(x, {}).get("dt") for x in mm["date"].dt.strftime("%Y-%m-%d")]}
    for s in STATS:
        for k, c in (("p", "pred"), ("lo", "lo"), ("hi", "hi")):
            month[f"{k}_{s}"] = [_n(v) for v in mm[f"{c}_{s}"]]
        month[f"y_{s}"] = [_n(ds[s].get(x)) for x in mm["date"]]
    past_d = {}
    for H in DAILY_H:
        g = m[(m["H"] == H) & (m["date"] >= PAST_SINCE)]
        past_d[H] = {r.date.strftime("%Y-%m-%d"): {s: [_n(getattr(r, f"pred_{s}")), _n(getattr(r, f"lo_{s}")), _n(getattr(r, f"hi_{s}"))]
                                                   for s in STATS} for r in g.itertuples()}
    daily_act = {d.strftime("%Y-%m-%d"): {s: _n(r[s]) for s in STATS} for d, r in ds[ds.index >= PAST_SINCE].iterrows()}

    perf = {
        "week": {"models": _table("WZ1_models.csv"), "ens": _table("WZ2_ensembles_2025H2.csv"), "test": _table("WZ3_test2026.csv"),
                 "choice": _json("WZ_choice.json")},
        "month": {"models": _table("MZ1_models.csv"), "ens": _table("MZ2_ensembles_2025H2.csv"), "test": _table("MZ3_test2026.csv"),
                  "choice": _json("MZ_choice.json")},
        "dropCal": _table("D1_drop_calibration.csv"), "dropChoice": _json("D_choice.json"),
    }
    t1 = ROOT / "drop_classifier" / "outputs" / "tables"
    docs = {"final": _json("F_final.json"), "metrics": _table("F1_final_metrics.csv"), "nextday": _table("F2_nextday_compare.csv"),
            "dropTest": json.loads(pd.read_csv(t1 / "13_final_test2026_육지.csv").round(4).to_json(orient="records", force_ascii=False)),
            "dropZoo": json.loads(pd.read_csv(t1 / "17_zoo_cv2025_육지.csv").drop(columns=["초"], errors="ignore").head(15)
                                  .round(4).to_json(orient="records", force_ascii=False))}
    ol = OPS / "run_log.txt"
    runs = [l for l in ol.read_text(encoding="utf-8").splitlines() if "매일 예측 완료" in l] if ol.exists() else []
    data = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "lastPublished": last_pub.strftime("%Y-%m-%d"),
            "lastRun": runs[-1][1:17] if runs else None, "week": week, "month": month,
            "past": {"hourly": past_h, "actual": actual, "daily": past_d, "dailyActual": daily_act, "drop": drops},
            "dayinfo": {k: v for k, v in dayinfo.items() if k >= PAST_SINCE}, "perf": perf, "docs": docs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.SMP_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=lambda o: None) + ";\n",
                   encoding="utf-8")
    print(f"저장: {OUT} ({OUT.stat().st_size/1024:.0f} KB) — 일주일 기준일 {week['base']}, 한 달 기준일 {month['base']}, 최신 공표 {data['lastPublished']}")


if __name__ == "__main__":
    main()
