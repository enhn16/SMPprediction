"""웹앱 데이터 내보내기 → smp_forecast_app/web/data/forecast_data.js (window.SMP_DATA = {...}). 자체 모델만.

화면 구성 (2026-10-23 개편)
  일주일 예측   : 최신 기준일의 앞으로 7일(시간별 168시간) 예측 + 80% 범위, 내일(D+1)에 과제1 급락 배지
  한 달 예측    : 최신 기준일의 7~34일 뒤 일별(일평균·한낮·저녁·한낮 최저) 예측 + 범위, 주·월 평균
  이전 흐름 확인 : 달력에서 기간(최대 14일)을 골라 지나간 예측 vs 공표 SMP (다음 날·일주일 D+1~7·한 달 7·14·21·28일 전)
  참고 문서     : 개요·다음 날·일주일·한 달 — 모델, 비교 모델과 성능, 선택 이유, 사용 데이터, 표준 성적표 (docs)
  설명
계통: 육지 = forecast_data.js(window.SMP_DATA), 제주 = forecast_data_jeju.js(window.SMP_JEJU, 같은 구조 + 제주 참고 문서 jdocs).
      웹은 제주를 고를 때만 제주 파일을 읽음 (왼쪽 메뉴 전환)
입력: WZ_predictions_hourly.csv(일주일, wz_ens) · mx/MX_predictions_daily_v2.csv(한 달, R3 mx v2)
      ops 운영 기록(있으면 같은 대상 덮어씀) · 급락 = drop_classifier/outputs/tables/drop_육지_15h.csv(R1) / ops drop_log.csv
      제주: jeju/JWZ_predictions_hourly.csv · JMZ_predictions_daily.csv · drop_badge_제주.csv + jeju/ops/{jwz,jdaily,jdrop}_log.csv
실행: python -m smp_forecast_app.src.export_web
"""
import json

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from common.loaders import load_smp_actual
from smp_forecast_app.src.features_daily import daily_stats
from smp_forecast_app.src.jz_month import daily_stats as jz_daily_stats

APP = ROOT / "smp_forecast_app"
TAB = APP / "outputs" / "tables"
DATA = PROCESSED_DIR / "smp_forecast_app"
OPS = DATA / "ops"
JDIR = DATA / "jeju"
OUT = APP / "web" / "data" / "forecast_data.js"
OUT_J = APP / "web" / "data" / "forecast_data_jeju.js"   # 제주는 따로 (웹이 제주를 고를 때만 읽음)
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


def _table_at(p):
    return json.loads(pd.read_csv(p).round(4).to_json(orient="records", force_ascii=False)) if p.exists() else []


LAND_DROP_CHOICE = ROOT / "drop_classifier" / "outputs" / "tables" / "drop_choice_육지.json"


def _json_at(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _json(name):
    p = TAB / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# 계통별 입력 (제주는 제주 채팅 결과 파일 + 제주 운영 기록, docs/JEJU.md 1절)
SRC = {
    "육지": {"hourly": DATA / "WZ_predictions_hourly.csv", "daily": DATA / "mx" / "MX_predictions_daily_v2.csv", "drop": ROOT / "drop_classifier" / "outputs" / "tables" / "drop_육지_15h.csv",   # R1 산출 (2026-10-26 교체)
             "ops": OPS, "logs": ("hourly_log.csv", "daily_log.csv", "drop_log.csv")},
    "제주": {"hourly": JDIR / "JWZ_predictions_hourly.csv", "daily": JDIR / "JMZ_predictions_daily.csv", "drop": JDIR / "drop_badge_제주.csv",
             "ops": JDIR / "ops", "logs": ("jwz_log.csv", "jdaily_log.csv", "jdrop_log.csv")},
}


def _log(region, i, **kw):
    p = SRC[region]["ops"] / SRC[region]["logs"][i]
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def hourly_frame(region="육지"):
    w = pd.read_csv(SRC[region]["hourly"], parse_dates=["base_date", "date"])[["base_date", "H", "date", "hour", "pred", "lo", "hi"]]
    oh = _log(region, 0, parse_dates=["base_date", "date"])
    if len(oh):
        key = lambda x: x["base_date"].astype(str) + "_" + x["H"].astype(str)
        o = oh[["base_date", "H", "date", "hour", "pred", "lo", "hi"]]
        w = pd.concat([w[~key(w).isin(key(o))], o])
    return w


def daily_frame(region="육지"):
    p = SRC[region]["daily"]
    m = pd.read_csv(p if p.exists() else DATA / "M_predictions_daily.csv", parse_dates=["base_date", "date"])
    om = _log(region, 1, parse_dates=["base_date", "date"])
    if len(om):
        m = pd.concat([m[~m["base_date"].isin(om["base_date"])], om])
    return m


def drop_meta(region, drop):
    """급락 등급 문턱 확률·모델 이름·2026년 등급별 실제 급락 비율 (웹 눈금 막대용)."""
    d = drop[drop["date"] >= "2026-01-01"].dropna(subset=["label"]) if "label" in drop else drop.iloc[:0]
    real = {g: {"rate": round(float(x["label"].mean()), 4), "k": int(x["label"].sum()), "n": len(x)} for g, x in d.groupby("grade")}
    if region == "육지":   # 문턱은 예상 한낮 최저(원) → 확률식으로 눈금 막대 위치
        ch = _json_at(LAND_DROP_CHOICE)
        a, b = ch["확률식"]["a"], ch["확률식"]["b"]
        pr = lambda x: round(float(1 / (1 + np.exp(-(a * -x + b)))), 4)
        return {"warn": pr(ch["문턱"]["주의"]), "strong": pr(ch["문턱"]["경보"]), "real": real, "lines": ch.get("문턱 표시", {}),
                "model": "과제1 모델(Ridge+EBM+랜덤포레스트)"}
    thr = _json("JD_choice.json").get("등급 문턱 확률", {})
    return {"warn": thr.get("주의"), "strong": thr.get("강한 경보"), "real": real, "model": "과제1 제주 모델(KNN 분류: 과거 비슷한 날 25일)"}


def drop_cal(region):
    """급락 확률 점검표: 연도 × 등급·확률 구간 (label이 있는 날만 — 육지는 휴일·주말), 2025 = 검증, 2026 = 시험."""
    d = _drop_file(region).dropna(subset=["label"])
    d["연도"] = d["date"].str[:4].astype(int)
    rows = []
    for y, g in d.groupby("연도"):
        parts = [(f"등급 {k}", g[g["grade"] == k]) for k in ("경보", "강한 경보", "주의", "없음")]
        bins = [(0, .05, "0%~5%"), (.05, .2, "5%~20%"), (.2, .5, "20%~50%"), (.5, 1.01, "50%~100%")]
        parts += [(f"확률 {t}", g[(g["prob"] >= a) & (g["prob"] < b)]) for a, b, t in bins]
        for k, x in parts:
            if len(x):
                rows.append({"연도": int(y), "구분": k, "일수": len(x), "평균 확률": round(float(x["prob"].mean()), 4),
                             "실제 급락 비율": round(float(x["label"].mean()), 4), "급락일": int(x["label"].sum())})
    return rows


def _drop_file(region):
    """경보 백테스트 파일 (열 이름을 웹 공통으로: score → pred_min)."""
    p = SRC[region]["drop"]
    d = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["date", "grade", "pred_min", "prob"])
    return d.rename(columns={"score": "pred_min"}) if "score" in d and "pred_min" not in d else d


def region_payload(region, dayinfo):
    """한 계통의 일주일·한 달·이전 흐름·급락 데이터."""
    act = load_smp_actual(region)[["date", "hour", "smp"]].rename(columns={"smp": "y"})
    actual = {d.strftime("%Y-%m-%d"): _h24(g, "y") for d, g in act[act["date"] >= PAST_SINCE].groupby("date")}
    w = hourly_frame(region)

    drop = _drop_file(region)
    od = _log(region, 2)
    if len(od):
        drop = pd.concat([drop[~drop["date"].isin(od["date"])], od])
    drops = {r.date: {"grade": r.grade, "min": _n(r.pred_min), "prob": round(float(r.prob), 3)} for r in drop.itertuples()}

    # 일주일: 7개 지평이 모두 있는 가장 최근 기준일
    full = w[w["pred"].notna()].groupby("base_date")["H"].nunique()   # 예측값이 실제로 있는 기준일만
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
    m = daily_frame(region)
    fullm = m[m["pred_mean"].notna()].groupby("base_date")["H"].nunique()   # 빈 백테스트 행(피처만 있는 최신 기준일) 제외
    mbase = fullm[fullm >= 28].index.max()
    mm = m[m["base_date"] == mbase].sort_values("H")
    ds = daily_stats() if region == "육지" else jz_daily_stats(region)
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
    return {"lastPublished": act["date"].max().strftime("%Y-%m-%d"), "week": week, "month": month, "dropMeta": drop_meta(region, drop),
            "past": {"hourly": past_h, "actual": actual, "daily": past_d, "dailyActual": daily_act, "drop": drops}}


def main():
    w = hourly_frame()
    cal = build_calendar(w["date"].min() - pd.Timedelta(days=40), w["date"].max() + pd.Timedelta(days=40)).set_index("date")
    dayinfo = {d.strftime("%Y-%m-%d"): {"dt": r.day_type, "off": bool(r.is_offday)} for d, r in cal.iterrows()}
    land = region_payload("육지", dayinfo)

    perf = {
        "week": {"models": _table("WZ1_models.csv"), "ens": _table("WZ2_ensembles_2025H2.csv"), "test": _table("WZ3_test2026.csv"),
                 "choice": _json("WZ_choice.json")},
        # 한 달 = R3 mx v2 (2026-10-27 교체, docs/MONTH.md)
        "month": {"models": _table("MX1_models_v2.csv"), "ens": _table("MX2_ensembles_2025_v2.csv"), "test": _table("MX4_test2026_v2.csv"),
                  "choice": _json("MX_choice_v2.json")},
        "dropCal": drop_cal("육지"), "dropChoice": _json_at(LAND_DROP_CHOICE),
    }
    t1 = ROOT / "drop_classifier" / "outputs" / "tables"
    docs = {"final": _json("F_final.json"), "metrics": _table("F1_final_metrics.csv"), "nextday": _table("F2_nextday_compare.csv"),
            "dropTest": _table_at(t1 / "13_final_test2026_육지.csv"), "dropScore": _table_at(t1 / "drop_score_육지.csv"), "dropSelect": _table_at(t1 / "18_team_select_육지.csv")[:10],
            "dropZoo": json.loads(pd.read_csv(t1 / "17_zoo_cv2025_육지.csv").drop(columns=["초"], errors="ignore").head(15)
                                  .round(4).to_json(orient="records", force_ascii=False))}
    # 팀 비교표 (team_scorecard.py: 2026-01-01 ~ 10-18, 급락 = 휴일·주말) — 계통별로 나눠 참고 문서 개요에
    tp, td, ti = _table("T1_team_price.csv"), _table("T2_team_drop.csv"), _json("T_team.json")
    team = lambda reg: {"price": [r for r in tp if r["계통"] == reg], "drop": [r for r in td if r["계통"] == reg], "info": ti}
    docs["team"] = team("육지")
    try:
        jeju = region_payload("제주", dayinfo)
        jc = _json("JD_choice.json")
        jeju["jdocs"] = {"metrics": _table("JF1_final_metrics.csv"), "weekTest": _table("JWZ3_test2026.csv"),
                         "monthTest": _table("JMZ2_test2026.csv"), "dropTest": _table_at(t1 / "J3_test2026.csv"),
                         "weekChoice": _json("JWZ_choice.json"), "monthChoice": _json("JMZ_choice.json"),
                         "dropChoice": {k: v for k, v in jc.items() if k != "피처 목록"}, "dropFeatures": len(jc.get("피처 목록", [])),
                         # 비교 후보 (육지 참고 문서와 같은 꼴로 보여 주기 위함)
                         "weekModels": _table("JWZ1_models.csv", ["모델", "2025 상반기", "2025 하반기", "2025 전체"] + [f"하반기 D+{h}" for h in range(1, 8)])[:20],
                         "weekEns": _table("JWZ2_ensembles_2025H2.csv", ["앙상블", "구성 수", "2025 하반기", "구성"])[:12],
                         "monthCompare": _table("JMZ1_compare_2025H2.csv"),
                         "dropZoo": _table_at(t1 / "J8_zoo_stability_2025.csv")[:12],
                         "dropCal": drop_cal("제주"), "team": team("제주")}
    except Exception as e:  # noqa: BLE001  제주가 실패해도 육지 화면은 내보냄
        print(f"제주 내보내기 실패: {e!r}")
        jeju = None
    ol = OPS / "run_log.txt"
    runs = [l for l in ol.read_text(encoding="utf-8").splitlines() if "매일 예측 완료" in l] if ol.exists() else []
    data = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "lastRun": runs[-1][1:17] if runs else None,
            "dayinfo": {k: v for k, v in dayinfo.items() if k >= PAST_SINCE}, "perf": perf, "docs": docs, **land, "hasJeju": jeju is not None}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"), default=lambda x: None)
    OUT.write_text("window.SMP_DATA = " + dump(data) + ";\n", encoding="utf-8")
    if jeju:
        OUT_J.write_text("window.SMP_JEJU = " + dump(jeju) + ";\n", encoding="utf-8")
    msg = f"일주일 기준일 {land['week']['base']}, 한 달 기준일 {land['month']['base']}, 최신 공표 {land['lastPublished']}"
    if jeju:
        msg += f" / 제주 일주일 {jeju['week']['base']}, 한 달 {jeju['month']['base']}"
    kb = OUT.stat().st_size / 1024, (OUT_J.stat().st_size / 1024 if OUT_J.exists() else 0)
    print(f"저장: {OUT} ({kb[0]:.0f} KB) + 제주 {OUT_J.name} ({kb[1]:.0f} KB) — {msg}")


if __name__ == "__main__":
    main()
