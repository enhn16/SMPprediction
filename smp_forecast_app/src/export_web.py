"""웹앱 데이터 내보내기 → smp_forecast_app/web/data/forecast_data.js (window.SMP_DATA = {...}).

서버 없이 index.html을 파일로 열어도 되도록 JSON 대신 JS 파일로 저장한다.
포함: 날짜별 24시간 [예측, 하한, 상한, 실적] (B 자체 15시 모델, A 담당자 시간별 모델), 과제1 급락 등급,
      2026 시험 성적표(구간별·시간대별·월별), 선택 정보.
실행: python -m smp_forecast_app.src.export_web
"""
import json

import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT

APP = ROOT / "smp_forecast_app"
TAB = APP / "outputs" / "tables"
OUT = APP / "web" / "data" / "forecast_data.js"
DROP = ROOT / "drop_classifier" / "outputs" / "tables" / "drop_prob_육지.csv"


def _num(v):
    return None if pd.isna(v) else round(float(v), 1)


def _days(df, src):
    out = {}
    for date, g in df.groupby("date"):
        g = g.set_index("hour").reindex(range(1, 25))
        out[date.strftime("%Y-%m-%d")] = {k: [_num(v) for v in g[c]] for k, c in
                                          (("p", "pred"), ("lo", "lo"), ("hi", "hi"), ("y", "y"))}
    return out


def _table(path, cols=None):
    if not path.exists():
        return []
    t = pd.read_csv(path)
    if cols:
        t = t[[c for c in cols if c in t.columns]]
    return json.loads(t.round(4).to_json(orient="records", force_ascii=False))


def main():
    b = pd.read_csv(PROCESSED_DIR / "smp_forecast_app" / "B_predictions_hourly.csv", parse_dates=["date"])
    a = pd.read_csv(PROCESSED_DIR / "smp_forecast_app" / "A_intervals_hourly.csv", parse_dates=["date"])
    cal = b.groupby("date")[["day_type", "offday"]].first()
    days = {}
    for d, row in cal.iterrows():
        days[d.strftime("%Y-%m-%d")] = {"dt": row["day_type"], "off": bool(row["offday"])}
    for key, df in (("B", b), ("A", a)):
        for d, v in _days(df, key).items():
            days.setdefault(d, {})[key] = v
    if DROP.exists():
        dp = pd.read_csv(DROP)
        for r in dp.itertuples():
            if r.date in days:
                days[r.date]["drop"] = {"grade": r.등급, "min": _num(r.예측_한낮최저SMP)}

    # 주간 보기: 기준일(판단일) → 지평 1~6일 × 24시간 (2026년 대상분부터, 파일 크기 제한)
    weekly = {}
    wpath = PROCESSED_DIR / "smp_forecast_app" / "W_predictions_hourly.csv"
    if wpath.exists():
        w = pd.read_csv(wpath, parse_dates=["base_date", "date"])
        w = w[w["base_date"] >= "2025-12-31"]
        full = w.groupby("base_date")["H"].nunique()
        for bd, g in w[w["base_date"].isin(full[full == 6].index)].groupby("base_date"):
            weekly[bd.strftime("%Y-%m-%d")] = [
                {k: [_num(v) for v in x.set_index("hour").reindex(range(1, 25))[c]]
                 for k, c in (("p", "pred"), ("lo", "lo"), ("hi", "hi"), ("y", "y"))}
                for _, x in g.sort_values("H").groupby("H")]

    cols = ["방법", "구간", "시간수", "포함률", "하한아래", "상한위", "평균폭", "구간점수", "MAE", "선택"]
    data = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "days": days,
        "B": {"choice": json.loads((TAB / "B_choice.json").read_text(encoding="utf-8")),
              "test": _table(TAB / "B3_test2026.csv", cols), "byHour": _table(TAB / "B3_test2026_by_hour.csv"),
              "byMonth": _table(TAB / "B3_test2026_by_month.csv"), "weekly": _table(TAB / "B4_weekly_ref.csv")},
        "weekly": weekly,
        "W": {"test": _table(TAB / "W2_test2026.csv", ["H"] + cols), "ref": _table(TAB / "W3_operator_weekly_ref.csv")},
        "A": {"choice": json.loads((TAB / "A_choice.json").read_text(encoding="utf-8")),
              "test": _table(TAB / "A2_test2026.csv", cols), "byHour": _table(TAB / "A2_test2026_by_hour.csv"),
              "byMonth": _table(TAB / "A2_test2026_by_month.csv"), "operational": _table(TAB / "A3_operational_ref.csv", cols)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=lambda o: None)
    OUT.write_text("window.SMP_DATA = " + txt + ";\n", encoding="utf-8")
    print(f"저장: {OUT} ({OUT.stat().st_size/1024:.0f} KB, {len(days)}일)")


if __name__ == "__main__":
    main()
