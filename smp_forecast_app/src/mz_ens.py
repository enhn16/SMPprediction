"""한 달 예측 — 모델 비교 + 앙상블 + 80% 범위 + 2026 채점 (지표별: 일평균·한낮·저녁·한낮 최저).

절차는 일주일(wz_ens)과 같음: 2025 상반기로 순위·가중치 → 2025 하반기로 앙상블 비교(0.05원 이내면 구성 적은 쪽)
  → 2025 전체로 재적합 → 2026 한 번 채점 (한 달 예측 3차 시험 표기). 지평 묶음 = 주차(7~13, 14~20, 21~27, 28~34일 뒤).
  섞기 후보 추가: 기준(same4) + λ × (선택 앙상블 − 기준), λ ∈ {0.25, 0.5, 0.75}
범위: model_monthly.interval (주차 × 휴무일; 고정 반년·갱신 90·180일; 가산·배수형), 2025 하반기로 선택
출력: outputs/tables/MZ1_models.csv, MZ2_ensembles_2025H2.csv, MZ3_test2026.csv, MZ_choice.json,
      data/processed/smp_forecast_app/MZ_predictions_daily.csv
실행: python -m smp_forecast_app.src.mz_ens
"""
import json

import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from smp_forecast_app.src.features_daily import STATS, load_daily
from smp_forecast_app.src.intervals import score, select
from smp_forecast_app.src.model_monthly import WEEK, interval
from smp_forecast_app.src.mz import OUT
from smp_forecast_app.src.wz_ens import ens_candidates, mae

TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
NAMES = {"same4": "기준: 최근 같은 유형일 4일 평균", "seas": "기준: 작년 계절비 보정", "lgb": "LightGBM(기준비율)", "xgb": "XGBoost(기준비율)",
         "ridge": "Ridge(기준비율)", "cat": "CatBoost(기준비율)", "rf": "랜덤포레스트(기준비율)", "et": "엑스트라트리(기준비율)",
         "ebm": "EBM(기준비율)", "mlp": "MLP(기준비율)", "tabpfn": "TabPFN(기준비율)", "chr_c2": "Chronos-2", "chr_bolt": "Chronos-Bolt",
         "lstm": "LSTM", "gru": "GRU"}
FAMILY = {"부스팅": ["lgb", "xgb", "cat"], "트리": ["rf", "et"], "선형·GAM": ["ridge", "ebm"], "신경망(표)": ["mlp", "tabpfn"],
          "시계열 신경망": ["lstm", "gru"], "Chronos": ["chr_c2", "chr_bolt"]}
BASE = ["same4", "seas"]
INTV = {"고정반년·가산": (None, "add"), "고정반년·배수": (None, "mult"), "갱신90일·가산": (90, "add"), "갱신90일·배수": (90, "mult"),
        "갱신180일·가산": (180, "add"), "갱신180일·배수": (180, "mult")}


def main():
    dd = load_daily()
    dd = dd[dd["base_date"] >= "2025-01-01"][["base_date", "H", "date", "day_type", "offday"] + [f"y_{s}" for s in STATS]]
    preds = {p.name.replace(".csv.gz", ""): pd.read_csv(p, parse_dates=["base_date", "date"]) for p in sorted(OUT.glob("*.csv.gz"))}
    models_all = [m for m in NAMES if m in preds and m not in BASE]
    mrows, erows, trows, choice, outs = [], [], [], {}, dd.copy()
    for s in STATS:
        P = dd[["base_date", "H", "date", "offday", f"y_{s}"]].rename(columns={f"y_{s}": "y"}).copy()
        for m, f in preds.items():
            P = P.merge(f[["base_date", "H", f"pred_{s}"]].rename(columns={f"pred_{s}": m}), on=["base_date", "H"], how="left")
        P["W"] = WEEK(P["H"])
        models = [m for m in models_all if P[m].notna().any()]
        y25 = P["date"].dt.year == 2025
        h1, h2 = y25 & (P["date"] < "2025-07-01"), y25 & (P["date"] >= "2025-07-01")
        full = P[models + BASE].notna().all(axis=1) & P["y"].notna()
        for m in BASE + models:
            mrows.append({"지표": s, "모델": NAMES[m], "2025 상반기": mae(P.loc[h1 & full, "y"], P.loc[h1 & full, m]),
                          "2025 하반기": mae(P.loc[h2 & full, "y"], P.loc[h2 & full, m]), "2025 전체": mae(P.loc[y25 & full, "y"], P.loc[y25 & full, m])})
        PW = P.assign(H=P["W"])   # 지평 묶음 = 주차
        cands, rank = ens_candidates(PW, h1 & full, models, family=FAMILY, hs=range(1, 5))
        allc = {"단일 최고 (" + NAMES[rank[0]] + ")": (P[rank[0]], [rank[0]]), f"{NAMES['same4']} 그대로": (P["same4"], ["same4"]), **cands}
        for name, (p, ms) in list(cands.items()):
            for lam in (0.25, 0.5, 0.75):
                allc[f"기준 + {lam} × ({name} − 기준)"] = (P["same4"] + lam * (p - P["same4"]), ["same4"] + list(ms))
        er = [{"지표": s, "앙상블": n, "구성 수": len(ms), "2025 하반기": mae(P.loc[h2 & full, "y"], p[h2 & full]),
               "구성": ", ".join(NAMES.get(m, m) for m in ms)} for n, (p, ms) in allc.items()]
        er = pd.DataFrame(er).sort_values("2025 하반기")
        erows.append(er)
        top = er.iloc[0]["2025 하반기"]
        pick = er[er["2025 하반기"] <= top + 0.05].sort_values(["구성 수", "2025 하반기"]).iloc[0]["앙상블"]
        # 2025 전체로 재적합
        cands_f, rank_f = ens_candidates(PW, y25 & full, models, family=FAMILY, hs=range(1, 5))
        allf = {"단일 최고 (" + NAMES[rank[0]] + ")": (P[rank_f[0]], [rank_f[0]]), f"{NAMES['same4']} 그대로": (P["same4"], ["same4"]), **cands_f}
        for name, (p, ms) in list(cands_f.items()):
            for lam in (0.25, 0.5, 0.75):
                allf[f"기준 + {lam} × ({name} − 기준)"] = (P["same4"] + lam * (p - P["same4"]), ["same4"] + list(ms))
        P["pred"] = allf[pick][0] if pick in allf else allc[pick][0]
        print(f"[{s}] 선택: {pick} (2025 하반기 {er.iloc[0]['2025 하반기']:.2f} 1위 {er.iloc[0]['앙상블']})", flush=True)
        # 범위
        d = P.rename(columns={"y": f"y_{s}", "pred": f"pred_{s}"})
        ev = (d["date"] >= "2025-07-01").values & d[f"y_{s}"].notna().values & d[f"pred_{s}"].notna().values
        part = []
        for name, (w, form) in INTV.items():
            lo, hi = interval(d, ev, s, f"pred_{s}", w, form)
            part.append({"방법": name, "구간": "전체", **score(d.loc[ev, f"y_{s}"], lo, hi, d.loc[ev, f"pred_{s}"])})
        ich = select(pd.DataFrame(part))
        allm = (d["date"] >= "2025-07-01").values & d[f"pred_{s}"].notna().values
        w, form = INTV[ich]
        lo, hi = interval(d, allm, s, f"pred_{s}", w, form)
        d.loc[allm, f"lo_{s}"], d.loc[allm, f"hi_{s}"] = lo, hi
        outs = outs.merge(d[["base_date", "H", f"pred_{s}", f"lo_{s}", f"hi_{s}"]], on=["base_date", "H"], how="left")
        te = d[(d["date"] >= "2026-01-01") & d[f"y_{s}"].notna() & d[f"lo_{s}"].notna()]
        for wk in ["전체", 1, 2, 3, 4]:
            g = te if wk == "전체" else te[WEEK(te["H"]) == wk]
            r = {"지표": s, "주차": wk, **score(g[f"y_{s}"], g[f"lo_{s}"], g[f"hi_{s}"], g[f"pred_{s}"]),
                 "기준(same4) MAE": mae(g[f"y_{s}"], g["same4"])}
            r["기준 대비 개선"] = 1 - r["MAE"] / r["기준(same4) MAE"]
            trows.append(r)
        choice[s] = {"점예측": pick, "범위": ich}
        print(f"[{s}] 범위 {ich}, 2026 포함률 {trows[-5]['포함률']:.1%}, MAE {trows[-5]['MAE']:.2f} (기준 {trows[-5]['기준(same4) MAE']:.2f})", flush=True)
    pd.DataFrame(mrows).round(3).to_csv(TAB / "MZ1_models.csv", index=False, encoding="utf-8-sig")
    pd.concat(erows).round(3).to_csv(TAB / "MZ2_ensembles_2025H2.csv", index=False, encoding="utf-8-sig")
    test = pd.DataFrame(trows)
    test.round(4).to_csv(TAB / "MZ3_test2026.csv", index=False, encoding="utf-8-sig")
    outs = outs[outs["date"] >= "2025-07-01"]
    outs.round(2).to_csv(PROCESSED_DIR / "smp_forecast_app" / "MZ_predictions_daily.csv", index=False, encoding="utf-8-sig")
    info = {"지표별": choice, "선택 앙상블": choice["mean"]["점예측"], "시험": "2026-01-01 ~ (한 달 예측 3차 시험)",
            "선택 절차": "상반기로 맞춤 → 2025 하반기 비교 → 2025 전체로 재적합"}
    (TAB / "MZ_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(test[test["주차"] == "전체"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
