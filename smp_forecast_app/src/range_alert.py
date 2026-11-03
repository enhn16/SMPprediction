"""일주일 K3 내일(D+1) 80% 범위 경보 보정 — 급락 경보로 하한, 급등 경보로 상한 (2026-11-03, 총괄).

배경: K3 범위는 보정 없이 골랐다 (R3, range_adjust 미적용). 내일 하루는 급락·급등 경보가 있으므로
      그 경보로 범위만 넓힌다 (점예측은 그대로). 2~7일 뒤는 경보가 없어 손대지 않는다. 육지만 (제주는 range_adjust 그대로).

후보 (2026 채점 전 고정, 번호가 작을수록 단순)
  급락 (D+1, 11~15시, 모든 날 — 급락 = 모든 날 규칙)
    D0 안 함 / D1 주의 이상 → 하한 0원 / D2 경보 → 0원, 주의 → 절반 / D3 확률 비례 (경보 문턱 확률에서 0원)
  급등 (D+1, 기준 가격 = 직전 28일 일 최고 SMP 중앙값 — D-1까지 실적이라 15시에 앎, 급등 = 그날 최고 ≥ 기준 × 1.25)
    S0 안 함 / S1 경보 → 그날 예측 상위 6시간 상한을 기준 × 1.25 이상으로 / S2 주의 이상 → S1과 같은 규칙 / S3 경보 → 모든 시간 상한을 기준 × 1.25 이상으로
선택: 2025-01-01 ~ 12-31 (K3 백테스트), 급락·급등 따로. 지표 = 그 경보 후보들이 바꿀 수 있는 시간 전체
      (급락: D+1 11~15시, 급등: D+1 모든 시간)의 구간점수 최소, 가장 좋은 값과 1% 이내면 단순한 후보.
      주의: 2025 경보 등급의 문턱은 R1·R4가 2025(급등은 2024~2025) 검증으로 정한 것이라 경보 자체는 그 구간에 맞춰져 있다.
채점: 2026-01-01 ~ 10-18 한 번 (--score2026), 보정 전후 같은 시간끼리.
적용: 운영 기록은 원래 범위 그대로, 웹(export_web)·팀 비교표(team_scorecard)가 읽을 때 apply()로.
출력: outputs/tables/RL_choice.json, RL1_select_2025.csv, (--score2026) RL2_test2026.csv
실행: python -m smp_forecast_app.src.range_alert [--score2026]
"""
import json
import sys

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from common.loaders import load_smp_actual
from smp_forecast_app.src.range_adjust import summary

DATA = PROCESSED_DIR / "smp_forecast_app"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
CHOICE = TAB / "RL_choice.json"
MID = range(11, 16)
SEL = ("2025-01-01", "2025-12-31")
TEST = ("2026-01-01", "2026-10-18")
SPIKE = 1.25
TOPK = 6

CAND_DROP = {"D0 안 함": None, "D1 주의 이상 → 하한 0원": "zero", "D2 경보 → 0원, 주의 → 절반": "half", "D3 확률 비례": "prob"}
CAND_SURGE = {"S0 안 함": None, "S1 경보 → 상위 6시간 상한 ≥ 기준×1.25": ("경보", "top"),
              "S2 주의 이상 → 상위 6시간 상한 ≥ 기준×1.25": ("주의", "top"), "S3 경보 → 모든 시간 상한 ≥ 기준×1.25": ("경보", "all")}

DROP = [ROOT / "drop_classifier" / "outputs" / "tables" / "drop_육지_15h.csv", DATA / "ops" / "drop_log.csv"]
SURGE = [ROOT / "surge_classifier" / "outputs" / "tables" / "surge_육지_15h.csv", DATA / "ops" / "surge_log.csv"]
DROP_CHOICE = ROOT / "drop_classifier" / "outputs" / "tables" / "drop_choice_육지.json"
WEEK = DATA / "wz3" / "WZ3_K3_backtest_hourly.csv"
LOG = DATA / "ops" / "hourly_log_k3.csv"


def _alerts(files) -> pd.DataFrame:
    fs = [pd.read_csv(f) for f in files if f.exists()]
    d = pd.concat([x[["date", "grade", "prob"]] for x in fs]).drop_duplicates("date", keep="last")
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")


def strong_prob() -> float:
    ch = json.loads(DROP_CHOICE.read_text(encoding="utf-8"))
    a, b = ch["확률식"]["a"], ch["확률식"]["b"]
    return float(1 / (1 + np.exp(-(a * -ch["문턱"]["경보"] + b))))


def ref_price() -> pd.Series:
    """대상일 → 기준 가격 (직전 28일 일 최고 SMP 중앙값, D-1까지 실적)."""
    act = load_smp_actual("육지")
    mx = act.groupby("date")["smp"].max()
    mx = mx.reindex(pd.date_range(mx.index.min(), mx.index.max() + pd.Timedelta(days=1)))
    return mx.shift(1).rolling(28, min_periods=14).median()


def apply(w: pd.DataFrame, choice: dict | None = None, ctx: dict | None = None) -> pd.DataFrame:
    """일주일 예측표 w [date, hour, H, pred, lo, hi]의 D+1 행 lo·hi를 경보로 보정 (다른 열·다른 지평은 그대로)."""
    ch = choice or (json.loads(CHOICE.read_text(encoding="utf-8")) if CHOICE.exists() else {})
    if not ch or w.empty:
        return w
    ctx = ctx or {"drop": _alerts(DROP), "surge": _alerts(SURGE), "sp": strong_prob(), "ref": ref_price()}
    w = w.copy()
    dates = pd.to_datetime(w["date"])
    d1 = (w["H"].values == 1)
    lo, hi = w["lo"].to_numpy(dtype=float).copy(), w["hi"].to_numpy(dtype=float).copy()

    mode = CAND_DROP.get(ch.get("급락"))
    if mode:
        g = dates.map(ctx["drop"]["grade"]).fillna("없음").replace({"강한 경보": "경보"}).values
        p = dates.map(ctx["drop"]["prob"]).fillna(0).values
        on = d1 & w["hour"].isin(MID).values & np.isin(g, ["주의", "경보"]) & (lo > 0)
        if mode == "zero":
            lo = np.where(on, 0.0, lo)
        elif mode == "half":
            lo = np.where(on & (g == "경보"), 0.0, np.where(on & (g == "주의"), lo / 2, lo))
        elif mode == "prob":
            lo = np.where(on, lo * np.clip(1 - p / ctx["sp"], 0, 1), lo)

    rule = CAND_SURGE.get(ch.get("급등"))
    if rule:
        need, where = rule
        g = dates.map(ctx["surge"]["grade"]).fillna("없음").values
        lv = np.isin(g, ["경보"] if need == "경보" else ["주의", "경보"])
        cap = dates.map(ctx["ref"]).values * SPIKE
        on = d1 & lv & np.isfinite(cap)
        if where == "top":   # 그날 예측이 높은 6시간 (같은 기준일·대상일 안에서)
            rk = w.groupby(["base_date", "date"])["pred"].rank(ascending=False, method="first").values
            on &= rk <= TOPK
        hi = np.where(on, np.maximum(hi, cap), hi)
    w["lo"], w["hi"] = lo, hi
    return w


def table() -> pd.DataFrame:
    w = pd.read_csv(WEEK, parse_dates=["base_date", "date"])
    if LOG.exists():
        o = pd.read_csv(LOG, parse_dates=["base_date", "date"])
        key = lambda x: x["base_date"].astype(str) + "_" + x["H"].astype(str)
        w = pd.concat([w[~key(w).isin(key(o))], o[[c for c in w.columns if c in o.columns]]])
    act = load_smp_actual("육지")[["date", "hour", "smp"]].rename(columns={"smp": "y"})
    w = w.drop(columns=["y"], errors="ignore").merge(act, on=["date", "hour"], how="inner")
    w["offday"] = w["date"].map(build_calendar(w["date"].min(), w["date"].max()).set_index("date")["is_offday"]).astype(bool)
    return w.reset_index(drop=True)


def _ctx():
    return {"drop": _alerts(DROP), "surge": _alerts(SURGE), "sp": strong_prob(), "ref": ref_price()}


def select():
    w = table()
    w = w[w["date"].between(*SEL)].reset_index(drop=True)
    ctx, rows, pick = _ctx(), [], {}
    d1 = (w["H"] == 1).values
    for part, cands, seg in (("급락", CAND_DROP, d1 & w["hour"].isin(MID).values), ("급등", CAND_SURGE, d1)):
        res = []
        for i, name in enumerate(cands):
            adj = apply(w, {part: name}, ctx)
            s = summary(adj[seg])
            res.append((name, s))
            ev = (adj["y"] < 30) if part == "급락" else adj["date"].map(ctx["ref"]).mul(SPIKE).le(adj.groupby("date")["y"].transform("max"))
            e = seg & ev.values
            rows.append({"구분": part, "후보": name, **{k: round(v, 4) for k, v in s.items()},
                         "사건 시간수": int(e.sum()), "사건 포함률": round(summary(adj[e])["포함률"], 4) if e.any() else None,
                         "D+1 전체 포함률": round(summary(adj[d1])["포함률"], 4)})
        best = min(s["구간점수"] for _, s in res)
        pick[part] = next(name for name, s in res if s["구간점수"] <= best * 1.01)
    sel = pd.DataFrame(rows)
    sel.loc[sel["후보"].isin(pick.values()), "선택"] = "✓"
    sel.to_csv(TAB / "RL1_select_2025.csv", index=False, encoding="utf-8-sig")
    CHOICE.write_text(json.dumps({**pick, "선택 기간": f"{SEL[0]} ~ {SEL[1]}", "규칙": "바뀔 수 있는 시간(급락: D+1 11~15시, 급등: D+1 전체) 구간점수 최소, 1% 이내면 단순한 후보",
                                  "적용": "육지 일주일 K3의 D+1만, 점예측 그대로"}, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(sel.to_string(index=False))
    print(pick)


def score2026():
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))
    w = table()
    w = w[w["date"].between(*TEST)].reset_index(drop=True)
    ctx = _ctx()
    adj = apply(w, ch, ctx)
    d1 = (w["H"] == 1).values
    dmax = w.groupby("date")["y"].transform("max")
    segs = {"전체 (D+1~7)": np.ones(len(w), bool), "D+1": d1, "D+1 한낮(11~15시)": d1 & w["hour"].isin(MID).values,
            "D+1 급락 시간(30원 미만)": d1 & (w["y"] < 30).values,
            "D+1 급등일": d1 & (w["date"].map(ctx["ref"]) * SPIKE <= dmax).values}
    rows = []
    for nm, m in segs.items():
        b, a = summary(w[m]), summary(adj[m])
        rows.append({"구분": nm, **{f"보정 전 {k}": round(v, 4) for k, v in b.items() if k != "시간수"},
                     **{f"보정 후 {k}": round(v, 4) for k, v in a.items()}})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "RL2_test2026.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t[["구분", "보정 후 시간수", "보정 전 포함률", "보정 후 포함률", "보정 전 평균 폭", "보정 후 평균 폭", "보정 전 구간점수", "보정 후 구간점수"]].to_string(index=False))


if __name__ == "__main__":
    score2026() if "--score2026" in sys.argv else select()
