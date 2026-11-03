"""일주일 v3 (R3) ⑩ K3 위 추가 고도화 — G1 여러 Chronos 쌓기(stk), G2 예상 태양광(pvx), G3 둘 다. 규칙은 WEEK_V3.md ⑩ (결과 보기 전 고정).

  select : 2024·2025로 K3 대비 판정 → WZ3_G1_2025.csv
  test   : 인정된 최선 하나만 2026-01-01 ~ 10-18 한 번 채점 (점예측 + A1 범위) → WZ3_G2_test2026.csv
실행: python -m smp_forecast_app.src.wz3_g select | test
"""
import json
import sys

import pandas as pd

from common.loaders import load_smp_actual
from smp_forecast_app.src import range_adjust
from smp_forecast_app.src import wz3_resid as resid
from smp_forecast_app.src.final_metrics import hourly_segments, metrics
from smp_forecast_app.src.wz3_select import CHOICE, DATA, TAB, TEST, TIE_MAE, W3, accept, eval_cfg, final_predictions

K3 = "mix|ftf|ext"
CANDS = [("G1 여러 Chronos 쌓기", "stk"), ("G2 예상 태양광", "pvx"), ("G3 G1+G2", "stk+pvx")]   # 단순한 순
OUT1, OUT2 = TAB / "WZ3_G1_2025.csv", TAB / "WZ3_G2_test2026.csv"


def cfg_of(part):
    return f"mix|ftf+{part}|ext+{part}"


def select():
    for _, part in CANDS:
        for b in ("ftf", "ext"):
            resid.run(f"{b}+{part}")
    k3 = eval_cfg("K3", K3)
    rows = [k3 | {"설정": K3}]
    for name, part in CANDS:
        r = eval_cfg(name, cfg_of(part))
        r |= {"설정": cfg_of(part), "Δ2024": r["2024 전체"] - k3["2024 전체"], "Δ2025": r["2025 전체"] - k3["2025 전체"],
              "인정": accept(r, k3)}
        rows.append(r)
    t = pd.DataFrame(rows)
    ok = t[t["인정"].fillna(False).astype(bool)]
    pick = ""
    if len(ok):
        best = ok["2025 전체"].min()
        pick = ok[ok["2025 전체"] <= best + TIE_MAE].iloc[0]["후보"]
    t["선택"] = (t["후보"] == pick).map({True: "✓", False: ""})
    t.round(3).to_csv(OUT1, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(t[["후보", "λ", "2024 전체", "2025 전체", "Δ2024", "Δ2025", "2025 평일", "2025 휴무일", "2025 D+1", "인정", "선택"]]
          .round(3).to_string(index=False))
    print("선택:", pick or "없음 → K3 유지, 2026은 보지 않음")


def test():
    t = pd.read_csv(OUT1)
    sel = t[t["선택"] == "✓"]
    if sel.empty:
        print("선택 없음 → 2026은 보지 않음")
        return
    name, cfg = sel.iloc[0]["후보"], sel.iloc[0]["설정"]
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))     # 범위 설정은 P8·K3와 같은 A1 (파일은 고치지 않음)
    ch["점예측"] = {"spec": [cfg, "one", False]}
    g, info = final_predictions(ch)
    print("재맞춤:", info)
    g.assign(stage="15h").round(2).to_csv(W3 / "WZ3_predictions_hourly_G.csv", index=False, encoding="utf-8-sig")
    k3 = pd.read_csv(W3 / "WZ3_predictions_hourly_K3.csv", parse_dates=["date"])[["date", "hour", "H", "pred", "lo", "hi"]]
    cur = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["base_date", "date"])
    cur = range_adjust.apply(cur[cur["y"].notna()].reset_index(drop=True), "육지")
    g = g[["date", "hour", "H", "pred", "lo", "hi"]]
    m = (cur.rename(columns={"pred": "pred_cur", "lo": "lo_cur", "hi": "hi_cur"})
         .merge(k3.rename(columns={"pred": "pred_k3", "lo": "lo_k3", "hi": "hi_k3"}), on=["date", "hour", "H"])
         .merge(g.rename(columns={"pred": "pred_g", "lo": "lo_g", "hi": "hi_g"}), on=["date", "hour", "H"]))
    m = m[m["date"].between(*TEST) & m["y"].notna()]
    act = load_smp_actual("육지")[["date", "hour", "smp"]]
    m = m.merge(act.assign(date=act["date"] + pd.Timedelta(days=7)).rename(columns={"smp": "lag7"}), on=["date", "hour"], how="left")
    m["offday"] = m["offday"].astype(bool)
    segs = hourly_segments(m) | {"월초 7일": (m["date"].dt.day <= 7).values, "D+1": (m["H"] == 1).values,
                                 "D+2~7": (m["H"] >= 2).values}
    rows = []
    for seg, s in segs.items():
        x = m[s]
        for who, lab in (("cur", "지금 모델"), ("k3", "K3"), ("g", name)):
            rows.append({"구분": seg, "모델": lab, **metrics(x["y"], x[f"pred_{who}"], x[f"lo_{who}"], x[f"hi_{who}"], x["lag7"])})
    o = pd.DataFrame(rows)
    o.round(4).to_csv(OUT2, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(o[["구분", "모델", "표본수", "MAE", "RMSE", "포함률", "평균 폭", "구간점수"]].round(3).to_string(index=False))


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
