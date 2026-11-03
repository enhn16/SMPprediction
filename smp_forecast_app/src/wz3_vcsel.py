"""일주일 v3 (R3) ⑤ 변동비 순위 실험 — 2025 비교표와 (채택 시) 2026 한 번 채점. 규칙은 WEEK_V3.md ⑤ (결과 보기 전 고정).

  선택: 2025 MAE가 비교 기준(C0 위 후보 = C0, P8 위 후보 = P8)보다 0.05원 넘게 좋을 때만 채택, 2024는 함께 표시
  (가) P8 위 후보 = C0 위 6개 중 S1·S2 각각 2025 최선 묶음을 mix|new+X|ext+X 로
  2026: 채택 후보만, 2024~2025로 λ 다시 맞춰 2026-01-01 ~ 10-18, 비교 기준·지금 모델과 같은 시간끼리, 월초 7일·그 밖
출력: outputs/tables/WZ3_V1_vc_2025.csv, (채택 시) WZ3_V2_vc_test2026.csv
실행: python -m smp_forecast_app.src.wz3_vcsel select | test
"""
import sys

import numpy as np
import pandas as pd

from smp_forecast_app.src import wz3_resid as resid
from smp_forecast_app.src.wz3_select import DATA, TAB, TIE_MAE, build_point, eval_cfg

T0, T1 = "2026-01-01", "2026-10-18"
REFIT = ("2024-01-01", "2025-12-31")
C0_SET = {f"{g}{s}": f"new+{g}{s}" for s in ("S1", "S2") for g in ("vs", "vk", "va")}
NAMES = {"vs": "계단", "vk": "누적출력 위치", "va": "계단+위치"}
OUT1, OUT2 = TAB / "WZ3_V1_vc_2025.csv", TAB / "WZ3_V2_vc_test2026.csv"


def row(name, cfg, base, kind):
    r = eval_cfg(name, cfg)
    return r | {"비교 기준": base, "구조": kind}


def select():
    rows = [row("C0 기준(P1)", "new", "-", "기준"), row("P8 v3", "mix", "-", "기준")]
    for k, cfg in C0_SET.items():
        rows.append(row(f"C0 + {NAMES[k[:2]]} {k[2:]}", cfg, "C0 기준(P1)", "잔차 피처"))
    t = pd.DataFrame(rows)
    for s in ("S1", "S2"):     # (가) P8 위: 공개 가정마다 C0판 최선 묶음
        c = t[t["설정"].str.endswith(s)].sort_values("2025 전체").iloc[0]["설정"].split("+")[1]
        resid.run(f"ext+{c}")      # 휴무일 행용 잔차 (없으면 계산)
        rows.append(row(f"P8 + {NAMES[c[:2]]} {s}", f"mix|new+{c}|ext+{c}", "P8 v3", "P8 위"))
    rows.append(row("Chronos 변동비 공변량(S1)", "cvc", "C0 기준(P1)", "Chronos 공변량"))
    t = pd.DataFrame(rows)
    ref = t.set_index("후보")
    for col in ("2024 전체", "2025 전체", "2025 D+1", "2025 D+2~7", "2025 휴무일"):
        t[f"Δ{col}"] = [
            np.nan if b == "-" else x - ref.loc[b, col] for b, x in zip(t["비교 기준"], t[col])]
    t["채택"] = (t["비교 기준"] != "-") & (t["Δ2025 전체"] < -TIE_MAE)
    t.round(3).to_csv(OUT1, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(t[["후보", "λ", "2024 전체", "2025 전체", "Δ2024 전체", "Δ2025 전체", "Δ2025 D+1", "Δ2025 D+2~7",
             "Δ2025 휴무일", "채택"]].round(3).to_string(index=False))
    ok = t[t["채택"]]
    if ok.empty:
        print("채택 없음 → 2026은 보지 않음")
    else:
        best = ok["2025 전체"].min()
        print("채택:", ok[ok["2025 전체"] <= best + TIE_MAE].iloc[0]["후보"])


def test():
    t = pd.read_csv(OUT1)
    ok = t[t["채택"].astype(bool)]
    if ok.empty:
        print("채택 없음 → 2026은 보지 않음")
        return
    pick = ok[ok["2025 전체"] <= ok["2025 전체"].min() + TIE_MAE].iloc[0]
    base_cfg = t.set_index("후보").loc[pick["비교 기준"], "설정"]
    preds = {}
    for who, cfg in (("v", pick["설정"]), ("b", base_cfg)):
        d, p, _ = build_point((cfg, "one", False), REFIT)
        preds[who] = d.assign(pred=p)[["date", "hour", "H", "y", "pred"]]
    m = preds["v"].merge(preds["b"].drop(columns="y"), on=["date", "hour", "H"], suffixes=("_v", "_b"))
    cur = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["date"])[["date", "hour", "H", "pred"]]
    m = m.merge(cur.rename(columns={"pred": "pred_c"}), on=["date", "hour", "H"])
    m = m[m["date"].between(T0, T1) & m["y"].notna()]
    segs = {"전체": m["date"].notna(), "월초 7일": m["date"].dt.day <= 7, "그 밖": m["date"].dt.day > 7,
            "D+1": m["H"] == 1, "D+2~7": m["H"] >= 2}
    out = []
    for k, s in segs.items():
        x = m[s]
        r = {"구분": k, "시간수": len(x)}
        for who, lab in (("c", "지금 모델"), ("b", pick["비교 기준"]), ("v", pick["후보"])):
            r[f"MAE {lab}"] = (x["y"] - x[f"pred_{who}"]).abs().mean()
        out.append(r)
    o = pd.DataFrame(out)
    o.round(3).to_csv(OUT2, index=False, encoding="utf-8-sig")
    print(o.round(3).to_string(index=False))


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
