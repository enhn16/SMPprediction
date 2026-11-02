"""과제3 근거 문장 → 웹·운영 형식 (R5). 총괄 export_web이 R5_why_육지.json을 읽어 D.why에 넣는다 (data.ts의 Why 타입과 같은 모양).

출력 (smp_forecast_app/outputs/tables)
  R5_why_육지.json      {"meta": {...}, "drop": {대상일: [{t, dir}]}, "surge": {대상일: [{t, dir}]}}
                        dir = up(위험을 높임) / down(낮춤) / none(확실한 근거 없음 — 한 줄 고정 문장)
  R5_reason_육지.csv    계산용 긴 표: base_date, date, kind, grade, score, rank, t, dir, group, feature, contrib, unit
대상 기간: 급락 2025-01-01 ~ 내일 (대상일 전년도 말까지 학습 = predict_team과 같음),
          급등 2026-01-01 ~ 내일 (저장된 2023~2025 모델만 있어 2025는 학습 구간 안 → 내지 않음).
운영: 15:30 경보 예측 뒤 `python -m smp_forecast_app.src.reason_web --ops [판단일, 기본 오늘]` → 판단일 다음 날 하루만 계산해 두 파일에 덧붙임 (약 40초).
전체 다시 만들기: python -m smp_forecast_app.src.reason_web
"""
import json
import sys
import time

import pandas as pd

from smp_forecast_app.src import reason_drop as RD
from smp_forecast_app.src import reason_surge as RS
from smp_forecast_app.src.team_scorecard import TAB as OUT_TAB

REGION = "육지"
WHY_JSON = OUT_TAB / f"R5_why_{REGION}.json"
LONG_CSV = OUT_TAB / f"R5_reason_{REGION}.csv"
START = {"drop": "2025-01-01", "surge": "2026-01-01"}
NONE_TEXT = {"drop": "평소와 다른 급락 위험 신호가 없습니다", "surge": "평소와 다른 급등 위험 신호가 없습니다"}
COLS = ["base_date", "date", "kind", "grade", "score", "rank", "t", "dir", "group", "feature", "contrib", "unit"]


def _long(kind: str, rs: pd.DataFrame) -> pd.DataFrame:
    r = rs.rename(columns={"won": "contrib", "p": "contrib"}).copy()
    r["kind"], r["unit"] = kind, "원 (예상 한낮 최저 변화)" if kind == "drop" else "점수 (세 확률 평균 변화)"
    r["base_date"] = (pd.to_datetime(r["date"]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    none = r["rank"] == 0
    r.loc[none, "t"], r.loc[none, "dir"] = NONE_TEXT[kind], "none"
    return r[COLS]


def build(dates: dict) -> pd.DataFrame:
    """dates = {"drop": 대상일 목록, "surge": 대상일 목록} → 긴 표."""
    parts = []
    if len(dates.get("drop", [])):
        parts.append(_long("drop", RD.explain(dates["drop"])[0]))
    if len(dates.get("surge", [])):
        parts.append(_long("surge", RS.explain(dates["surge"])[0]))
    return pd.concat(parts, ignore_index=True)


def to_json(long: pd.DataFrame) -> dict:
    out = {"meta": {"made": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "region": REGION,
                    "기준": {"drop": "평소 날 (학습 구간 평균)", "surge": "평소 평일 (학습 구간 평일 평균, 휴일은 모델이 0점)"},
                    "표시 문턱": {"drop": f"묶음 기여 {RD.MIN_WON:g}원 이상", "surge": f"묶음 기여 {RS.MIN_P:g} 이상"},
                    "dir": "up = 위험을 높임(▲), down = 낮춤(▼), none = 확실한 근거 없음(표시 없이 문장만)"}}
    for kind, g in long.sort_values(["date", "rank"]).groupby("kind"):
        out[kind] = {d: [{"t": r.t, "dir": r.dir} for r in x.itertuples()] for d, x in g.groupby("date")}
    return out


def main():
    t0 = time.time()
    if "--ops" in sys.argv:                                  # 판단일(기본 오늘)의 다음 날 하루: 기존 표에서 그 날을 바꿔 끼움
        args = [a for a in sys.argv[1:] if a != "--ops"]
        base = pd.Timestamp(args[0]) if args else pd.Timestamp.today().normalize()
        tom = (base + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        new = build({"drop": [tom], "surge": [tom]})
        old = pd.read_csv(LONG_CSV, dtype={"date": str}) if LONG_CSV.exists() else pd.DataFrame(columns=COLS)
        long = pd.concat([old[old["date"] != tom], new], ignore_index=True)
    else:
        end = pd.Timestamp(RS.V3.data()["date"].max())
        long = build({k: pd.date_range(s, end) for k, s in START.items()})
    long = long.sort_values(["kind", "date", "rank"])
    long.to_csv(LONG_CSV, index=False, encoding="utf-8-sig")
    WHY_JSON.write_text(json.dumps(to_json(long), ensure_ascii=False, indent=1), encoding="utf-8")
    n = long.groupby("kind")["date"].agg(["nunique", "min", "max"])
    print(n.to_string(), f"\n{time.time() - t0:.0f}초 → {WHY_JSON.name}, {LONG_CSV.name}")


if __name__ == "__main__":
    main()
