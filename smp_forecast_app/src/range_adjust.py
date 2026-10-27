"""일주일 80% 범위 보정 — 휴일·주말 한낮 하한 열기 (2026-10-27, 총괄).

배경: 일주일 범위는 급락 시간(30원 미만)을 거의 담지 못하고, 휴일·주말 한낮은 하한 아래 이탈이 많다.
  - D+1   : 내일 급락 경보(R1 육지 / 제주 KNN)가 '주의' 이상인 휴일·주말 11~15시의 하한을 연다
  - D+2~7 : 급락 모델 없이(정확도가 낮아 만들지 않음) 휴일·주말 11~15시 하한을 줄인다 (규칙)
점예측은 바꾸지 않는다 (예전에 급락 결합으로 점예측을 내렸을 때 MAE가 나빠졌음).

선택 규칙 (2026 채점 전 고정)
  - 후보: D+1 = A0~A3, D+2~7 = B0~B4 (아래 CAND)
  - 기간: 2025-07-01 ~ 2025-12-31 (일주일 백테스트가 있는 2025 구간), 계통별로 따로
  - 지표: 바뀌는 시간(그 지평의 휴일·주말 11~15시)의 구간점수(Winkler) 최소, 가장 좋은 값과 1% 이내면 단순한 후보(번호가 작은 쪽)
  - 2026-01-01 ~ 10-18 한 번 채점 (--score2026), 보정 전후를 같은 시간끼리 비교
적용: 운영 기록(원래 범위)은 그대로 두고, 웹 데이터(export_web)·팀 비교표(team_scorecard)가 읽을 때 apply()로 보정한다.
출력: outputs/tables/RA_choice.json, RA1_select_2025H2.csv, (--score2026) RA2_test2026.csv
실행: python -m smp_forecast_app.src.range_adjust [--score2026]
"""
import json
import sys

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from common.loaders import load_smp_actual

DATA = PROCESSED_DIR / "smp_forecast_app"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
CHOICE = TAB / "RA_choice.json"
MID = range(11, 16)
SEL = ("2025-07-01", "2025-12-31")
TEST = ("2026-01-01", "2026-10-18")
ALPHA = 0.2

# 후보 (번호가 작을수록 단순)
CAND_D1 = {"A0 안 함": None, "A1 주의 이상 → 0원": "zero", "A2 경보 → 0원, 주의 → 절반": "half", "A3 확률 비례 (경보 문턱 확률에서 0원)": "prob"}
CAND_DN = {"B0 안 함": 1.0, "B1 하한 × 0.85": 0.85, "B2 하한 × 0.7": 0.7, "B3 하한 × 0.5": 0.5, "B4 하한 0원": 0.0}

SRC = {"육지": {"week": [DATA / "WZ_predictions_hourly.csv"], "log": DATA / "ops" / "hourly_log.csv",
               "drop": [ROOT / "drop_classifier" / "outputs" / "tables" / "drop_육지_15h.csv", DATA / "ops" / "drop_log.csv"],
               "strong": ("drop_choice", ROOT / "drop_classifier" / "outputs" / "tables" / "drop_choice_육지.json")},
       "제주": {"week": [DATA / "jeju" / "JWZ_predictions_hourly.csv"], "log": DATA / "jeju" / "ops" / "jwz_log.csv",
               "drop": [DATA / "jeju" / "drop_badge_제주.csv", DATA / "jeju" / "ops" / "jdrop_log.csv"],
               "strong": ("jd_choice", TAB / "JD_choice.json")}}


def strong_prob(region) -> float:
    """'경보' 문턱에 해당하는 확률 (A3 후보의 0원 지점)."""
    kind, p = SRC[region]["strong"]
    ch = json.loads(p.read_text(encoding="utf-8"))
    if kind == "drop_choice":
        a, b = ch["확률식"]["a"], ch["확률식"]["b"]
        return float(1 / (1 + np.exp(-(a * -ch["문턱"]["경보"] + b))))
    return float(ch["등급 문턱 확률"]["강한 경보"])


def drop_alerts(region) -> pd.DataFrame:
    """대상일별 급락 경보 [date, grade, prob] — 백테스트 + 운영 기록(같은 날은 운영 기록)."""
    fs = [pd.read_csv(f) for f in SRC[region]["drop"] if f.exists()]
    d = pd.concat([x[["date", "grade", "prob"]] for x in fs]).drop_duplicates("date", keep="last")
    d["date"] = pd.to_datetime(d["date"])
    d["grade"] = d["grade"].replace({"강한 경보": "경보"})
    return d


def apply(w: pd.DataFrame, region: str, choice: dict | None = None, alerts=None, offday=None) -> pd.DataFrame:
    """시간별 일주일 예측표 w [date, hour, H, lo, ...]의 lo를 보정해 돌려줌 (다른 열은 그대로)."""
    ch = (choice or (json.loads(CHOICE.read_text(encoding="utf-8")) if CHOICE.exists() else {})).get(region)
    if not ch or w.empty:
        return w
    w = w.copy()
    dates = pd.to_datetime(w["date"])
    if offday is None:
        cal = build_calendar(dates.min(), dates.max()).set_index("date")["is_offday"]
        offday = dates.map(cal).fillna(False).astype(bool).values
    tgt = np.asarray(offday) & w["hour"].isin(MID).values
    lo = w["lo"].to_numpy(dtype=float).copy()
    # D+1: 급락 경보
    mode = CAND_D1.get(ch["D+1"])
    if mode:
        al = alerts if alerts is not None else drop_alerts(region)
        m = dates.map(al.set_index("date")["grade"]).fillna("없음").values
        p = dates.map(al.set_index("date")["prob"]).fillna(0).values
        on = tgt & (w["H"].values == 1) & np.isin(m, ["주의", "경보"])
        pos = lo > 0
        if mode == "zero":
            lo = np.where(on & pos, 0.0, lo)
        elif mode == "half":
            lo = np.where(on & pos & (m == "경보"), 0.0, np.where(on & pos & (m == "주의"), lo / 2, lo))
        elif mode == "prob":
            k = np.clip(1 - p / ch["경보 확률"], 0, 1)
            lo = np.where(on & pos, lo * k, lo)
    # D+2~7: 휴일·주말 한낮 규칙
    f = CAND_DN.get(ch["D+2~7"], 1.0)
    on = tgt & (w["H"].values >= 2) & (lo > 0)
    lo = np.where(on, lo * f, lo)
    w["lo"] = lo
    return w


def winkler(y, lo, hi):
    below, above = y < lo, y > hi
    return float(((hi - lo) + (2 / ALPHA) * ((lo - y) * below + (y - hi) * above)).mean())


def summary(x) -> dict:
    y, lo, hi = x["y"].values, x["lo"].values, x["hi"].values
    b, a = y < lo, y > hi
    return {"시간수": len(x), "포함률": 1 - b.mean() - a.mean(), "하한 아래": b.mean(), "상한 위": a.mean(),
            "평균 폭": float((hi - lo).mean()), "구간점수": winkler(y, lo, hi)}


def week_table(region) -> pd.DataFrame:
    w = pd.concat([pd.read_csv(f, parse_dates=["base_date", "date"]) for f in SRC[region]["week"]])
    log = SRC[region]["log"]
    if log.exists():   # 운영 기록 (같은 기준일·지평은 운영 쪽)
        o = pd.read_csv(log, parse_dates=["base_date", "date"])
        key = lambda x: x["base_date"].astype(str) + "_" + x["H"].astype(str)
        w = pd.concat([w[~key(w).isin(key(o))], o[[c for c in w.columns if c in o.columns]]])
    act = load_smp_actual(region)[["date", "hour", "smp"]].rename(columns={"smp": "y"})
    w = w.drop(columns=["y"], errors="ignore").merge(act, on=["date", "hour"], how="inner")
    cal = build_calendar(w["date"].min(), w["date"].max()).set_index("date")["is_offday"]
    w["offday"] = w["date"].map(cal).astype(bool)
    return w


def select():
    out, rows = {}, []
    for region in SRC:
        w = week_table(region)
        w = w[w["date"].between(*SEL)]
        al, sp = drop_alerts(region), strong_prob(region)
        pick = {"경보 확률": sp}
        for part, cands, hsel in (("D+1", CAND_D1, w["H"] == 1), ("D+2~7", CAND_DN, w["H"] >= 2)):
            seg = hsel & w["offday"] & w["hour"].isin(MID)
            res = []
            for i, name in enumerate(cands):
                ch = {"D+1": "A0 안 함", "D+2~7": "B0 안 함", "경보 확률": sp, part: name}
                adj = apply(w, region, {region: ch}, al, w["offday"].values)
                s = summary(adj[seg.values])
                res.append((i, name, s))
                rows.append({"계통": region, "구분": part, "후보": name, **{k: round(v, 4) for k, v in s.items()},
                             "전체 포함률": round(summary(adj[hsel.values])["포함률"], 4)})
            best = min(r[2]["구간점수"] for r in res)
            pick[part] = next(name for i, name, s in res if s["구간점수"] <= best * 1.01)
        out[region] = pick
    sel = pd.DataFrame(rows)
    for region, pk in out.items():
        sel.loc[(sel["계통"] == region) & sel["후보"].isin([pk["D+1"], pk["D+2~7"]]), "선택"] = "✓"
    sel.to_csv(TAB / "RA1_select_2025H2.csv", index=False, encoding="utf-8-sig")
    CHOICE.write_text(json.dumps({**out, "선택 기간": f"{SEL[0]} ~ {SEL[1]}", "규칙": "바뀌는 시간(휴일·주말 11~15시) 구간점수 최소, 1% 이내면 단순한 후보"},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(sel.to_string(index=False))
    print(json.dumps(out, ensure_ascii=False))


def score2026():
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))
    rows = []
    for region in SRC:
        w = week_table(region)
        w = w[w["date"].between(*TEST)]
        adj = apply(w, region, ch, None, w["offday"].values)
        segs = {"전체": np.ones(len(w), bool), "휴일·주말 한낮(11~15시)": (w["offday"] & w["hour"].isin(MID)).values,
                "급락 시간(30원 미만)": (w["y"] < 30).values, "D+1": (w["H"] == 1).values, "D+2~7": (w["H"] >= 2).values,
                "D+1 휴일·주말 한낮": ((w["H"] == 1) & w["offday"] & w["hour"].isin(MID)).values}
        for nm, m in segs.items():
            b, a = summary(w[m]), summary(adj[m])
            rows.append({"계통": region, "구분": nm, **{f"보정 전 {k}": round(v, 4) for k, v in b.items() if k != "시간수"},
                         **{f"보정 후 {k}": round(v, 4) for k, v in a.items()}})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "RA2_test2026.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t[["계통", "구분", "보정 후 시간수", "보정 전 포함률", "보정 후 포함률", "보정 전 하한 아래", "보정 후 하한 아래",
             "보정 전 평균 폭", "보정 후 평균 폭", "보정 전 구간점수", "보정 후 구간점수"]].to_string(index=False))


if __name__ == "__main__":
    score2026() if "--score2026" in sys.argv else select()
