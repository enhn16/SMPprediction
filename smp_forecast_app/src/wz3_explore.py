"""일주일 v3 (R3) ⑥ 전면 탐색 — 후보 판정(2024·2025)과 최종 조합 K3, (채택 시) 2026 한 번 채점. 규칙은 WEEK_V3.md ⑥ (결과 보기 전 고정).

  인정: 2025 MAE가 기준(C0, 구조 후보는 P8)보다 0.05원 넘게 좋고 2024에서도 좋을 것
  K3 : P8 틀에 인정된 것만 — 평일 = (인정된 최선 기초모델 판 또는 new) + 인정 N 피처 (+ vkS1),
       휴무일 = ext + 인정 N 피처 (+ vkS1) (+ offonly). P8보다 두 해 모두 좋고 2025가 0.05원 넘게 좋을 때 채택
출력: outputs/tables/WZ3_X1_explore.csv, WZ3_X2_K3.csv, (채택 시) WZ3_X3_test2026.csv
실행: python -m smp_forecast_app.src.wz3_explore select | test
"""
import sys

import pandas as pd

from smp_forecast_app.src import wz3_resid as resid
from smp_forecast_app.src.wz3_select import DATA, TAB, TIE_MAE, accept, build_point, eval_cfg

T0, T1 = "2026-01-01", "2026-10-18"
REFIT = ("2024-01-01", "2025-12-31")
P8 = "mix"
CANDS = [  # (이름, 설정, 비교 기준, 묶음)
    ("N1 원전 정비", "new+nuc", "C0", "N"), ("N2 발전원 실적(전날)", "new+sup", "C0", "N"),
    ("N3 출력제어 이력", "new+curt", "C0", "N"), ("N4 N1+N2+N3", "new+nuc+sup+curt", "C0", "N"),
    ("M1 TiRex", "tirex", "C0", "M"), ("M2 TimesFM 2.5", "tfm", "C0", "M"),
    ("M3 Chronos·TiRex·TimesFM 평균", "avg_cal_tirex_tfm", "C0", "M"),
    ("M4 LoRA 미세조정(1e-5)", "ftl", "C0", "M"), ("M5 전체 미세조정(1e-6)", "ftf", "C0", "M"),
    ("S1 휴무일 전용 잔차", "mix|new|ext+offonly", "P8", "S"), ("S2 P8 + 누적출력 위치", "mix|new+vkS1|ext+vkS1", "P8", "S"),
]
OUT1, OUT2, OUT3 = TAB / "WZ3_X1_explore.csv", TAB / "WZ3_X2_K3.csv", TAB / "WZ3_X3_test2026.csv"


def ready(cfg) -> bool:
    for part in cfg.split("|")[1:] if cfg.startswith("mix|") else [cfg]:
        if not (resid.RES3 / f"{part}.csv.gz").exists():
            return False
    return True


def select():
    base = {"C0": eval_cfg("C0 기준(P1)", "new"), "P8": eval_cfg("P8 v3", P8)}
    rows = [base["C0"] | {"묶음": "기준", "비교 기준": "-"}, base["P8"] | {"묶음": "기준", "비교 기준": "-"}]
    for name, cfg, ref, grp in CANDS:
        if not ready(cfg):
            print(f"{name}: 잔차 파일 없음 → 건너뜀")
            continue
        r = eval_cfg(name, cfg)
        r |= {"묶음": grp, "비교 기준": ref, "Δ2024": r["2024 전체"] - base[ref]["2024 전체"],
              "Δ2025": r["2025 전체"] - base[ref]["2025 전체"], "인정": accept(r, base[ref])}
        rows.append(r)
    t = pd.DataFrame(rows)
    t.round(3).to_csv(OUT1, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(t[["후보", "λ", "2024 전체", "2025 전체", "Δ2024", "Δ2025", "2025 평일", "2025 휴무일", "인정"]].round(3).to_string(index=False))
    ok = t[t["인정"].fillna(False).astype(bool)]
    if ok.empty:
        print("인정된 후보 없음 → K3 없음, 2026은 보지 않음")
        return
    m = ok[ok["묶음"] == "M"].sort_values("2025 전체")
    kind = m.iloc[0]["설정"] if len(m) else "new"
    nparts = sorted({p for c in ok[ok["묶음"] == "N"]["설정"] for p in c.split("+")[1:]})
    vk = ["vkS1"] if "S2 P8 + 누적출력 위치" in set(ok["후보"]) else []
    off = ["offonly"] if "S1 휴무일 전용 잔차" in set(ok["후보"]) else []
    wk = "+".join([kind] + nparts + vk)
    of = "+".join(["ext"] + nparts + vk + off)
    for c in (wk, of):
        resid.run(c)
    cfg = f"mix|{wk}|{of}"
    r = eval_cfg(f"K3 (평일 {wk} · 휴무일 {of})", cfg)
    r |= {"설정": cfg, "Δ2024": r["2024 전체"] - base["P8"]["2024 전체"], "Δ2025": r["2025 전체"] - base["P8"]["2025 전체"],
          "채택": accept(r, base["P8"])}
    k = pd.DataFrame([base["P8"] | {"채택": "-"}, r])
    k.round(3).to_csv(OUT2, index=False, encoding="utf-8-sig")
    print(k[["후보", "2024 전체", "2025 전체", "Δ2024", "Δ2025", "채택"]].round(3).to_string(index=False))


def test():
    k = pd.read_csv(OUT2)
    r = k.iloc[-1]
    if str(r["채택"]) != "True":
        print("K3 채택 안 됨 → 2026은 보지 않음")
        return
    preds = {}
    for who, cfg in (("k", r["설정"]), ("p8", P8)):
        d, p, _ = build_point((cfg, "one", False), REFIT)
        preds[who] = d.assign(pred=p)[["date", "hour", "H", "y", "pred", "offday"]]
    m = preds["k"].merge(preds["p8"][["date", "hour", "H", "pred"]], on=["date", "hour", "H"], suffixes=("_k", "_p8"))
    cur = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["date"])[["date", "hour", "H", "pred"]]
    m = m.merge(cur.rename(columns={"pred": "pred_c"}), on=["date", "hour", "H"])
    m = m[m["date"].between(T0, T1) & m["y"].notna()]
    segs = {"전체": m["date"].notna(), "평일": m["offday"] == 0, "휴무일": m["offday"] == 1,
            "월초 7일": m["date"].dt.day <= 7, "그 밖": m["date"].dt.day > 7, "D+1": m["H"] == 1, "D+2~7": m["H"] >= 2}
    out = []
    for name, s in segs.items():
        x = m[s]
        out.append({"구분": name, "시간수": len(x)} | {f"MAE {lab}": (x["y"] - x[f"pred_{w}"]).abs().mean()
                                                    for w, lab in (("c", "지금 모델"), ("p8", "v3 P8"), ("k", "K3"))})
    o = pd.DataFrame(out)
    o.round(3).to_csv(OUT3, index=False, encoding="utf-8-sig")
    print(o.round(3).to_string(index=False))


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
