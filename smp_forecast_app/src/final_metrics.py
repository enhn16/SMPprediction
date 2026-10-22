"""최종 모델 성적표 (팀 비교용 표준 지표) — 다음 날·일주일·한 달.

최종 모델 (2025로 선택, 2026-10-23 확정)
  다음 날 급락 판별      : 과제1 최종 모델 Ridge + EBM + 랜덤포레스트 (내일 한낮 최저 SMP → 등급·확률). 성적은 과제1 표 그대로
  일주일 (D+1~D+7, 시간별): Chronos-2 small + 0.5 × 잔차 보정 (내일 시간별 가격 = 이 모델의 D+1)
  한 달 (7~34일 뒤, 일별 4지표): 기준(같은 유형일 4일 평균) + λ × (LSTM·GRU 평균 − 기준)
지표
  MAE·RMSE(원), MAPE(공표 20원 이상 시간만), rMAE = MAE / 단순 기준 MAE, 편향 = 평균(공표 − 예측)
  포함률(80% 범위), 하한 아래·상한 위, 평균 폭(원), 구간점수(Winkler, 낮을수록 좋음)
  단순 기준: 일주일 = 지난주 같은 시간 (D+1 행은 전날 같은 시간도 병기), 한 달 = 같은 유형일 4일 평균
기간: 2025 하반기 = 선택 구간(표본 밖), 2026-01-01 ~ 최신 공표일 = 시험
출력: outputs/tables/F1_final_metrics.csv (기간×구간), F2_nextday_compare.csv (내일 시간별 가격 후보 비교, 일주일 D+1), F_final.json
실행: python -m smp_forecast_app.src.final_metrics
"""
import json

import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from smp_forecast_app.src.features import load_features
from smp_forecast_app.src.features_daily import STATS, load_daily
from smp_forecast_app.src.intervals import ALPHA
from smp_forecast_app.src.wz_common import load_pred

DATA = PROCESSED_DIR / "smp_forecast_app"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
PERIODS = {"2025 하반기 (선택)": ("2025-07-01", "2026-01-01"), "2026 (시험)": ("2026-01-01", "2100-01-01")}
SN = {"mean": "일평균", "mid": "한낮 11~15시", "eve": "저녁 18~22시", "midmin": "한낮 최저"}


def metrics(y, p, lo=None, hi=None, naive=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    big = y >= 20
    out = {"표본수": len(y), "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
           "MAPE(20원 이상)": np.abs(e[big] / y[big]).mean(), "편향(공표−예측)": e.mean()}
    if naive is not None:
        nv = np.asarray(naive, float)
        ok = ~np.isnan(nv)
        out["단순 기준 MAE"] = np.abs(y[ok] - nv[ok]).mean()
        out["rMAE"] = np.abs(e[ok]).mean() / out["단순 기준 MAE"]
    if lo is not None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        below, above = y < lo, y > hi
        out.update({"포함률": 1 - below.mean() - above.mean(), "하한 아래": below.mean(), "상한 위": above.mean(),
                    "평균 폭": (hi - lo).mean(),
                    "구간점수": ((hi - lo) + (2 / ALPHA) * ((lo - y) * below + (y - hi) * above)).mean()})
    return out


def hourly_segments(d):
    mid = d["hour"].between(11, 15)
    off = d["offday"].astype(bool)
    return {"전체": np.ones(len(d), bool), "평일": ~off, "휴무일": off, "휴무일 한낮(11~15시)": off & mid,
            "급락 시간(30원 미만)": d["y"] < 30}


def weekly_table():
    """WZ_predictions_hourly + 단순 기준(지난주 같은 시간, 다음 날은 전날 같은 시간)."""
    w = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["base_date", "date"])
    w = w[w["y"].notna()]
    n7 = load_pred("naive7")[["date", "hour", "H", "pred"]].rename(columns={"pred": "naive7"})
    d1 = load_features(1)[["date", "hour", "s_d1"]]
    return w.merge(n7, on=["date", "hour", "H"], how="left").merge(d1, on=["date", "hour"], how="left")


def run_hourly(w, product, naive_col):
    rows = []
    for per, (a, b) in PERIODS.items():
        g = w[(w["date"] >= a) & (w["date"] < b)]
        for seg, m in hourly_segments(g).items():
            x = g[m]
            if len(x):
                rows.append({"예측": product, "기간": per, "구분": seg,
                             **metrics(x["y"], x["pred"], x["lo"], x["hi"], x[naive_col])})
        if product == "일주일":
            for H, x in g.groupby("H"):
                rows.append({"예측": product, "기간": per, "구분": f"D+{H}",
                             **metrics(x["y"], x["pred"], x["lo"], x["hi"], x[naive_col])})
    return rows


def run_monthly():
    m = pd.read_csv(DATA / "MZ_predictions_daily.csv", parse_dates=["base_date", "date"])
    base = load_daily()[["base_date", "H"] + [f"{s}_same4" for s in STATS]]
    m = m.merge(base, on=["base_date", "H"], how="left")
    rows = []
    for per, (a, b) in PERIODS.items():
        g = m[(m["date"] >= a) & (m["date"] < b)]
        for s in STATS:
            x = g[g[f"y_{s}"].notna()]
            rows.append({"예측": "한 달", "기간": per, "구분": SN[s],
                         **metrics(x[f"y_{s}"], x[f"pred_{s}"], x[f"lo_{s}"], x[f"hi_{s}"], x[f"{s}_same4"])})
            for wk in range(1, 5):
                xx = x[(x["H"] - 7) // 7 + 1 == wk]
                rows.append({"예측": "한 달", "기간": per, "구분": f"{SN[s]} · {wk}주차",
                             **metrics(xx[f"y_{s}"], xx[f"pred_{s}"], xx[f"lo_{s}"], xx[f"hi_{s}"], xx[f"{s}_same4"])})
    return rows


def nextday_compare():
    """내일 시간별 가격(D+1) 후보: 같은 날짜·시간 행에서 2025 상·하반기·2026 MAE (점예측만)."""
    V = pd.read_csv(DATA / "V_predictions_hourly.csv", parse_dates=["date"])
    B = pd.read_csv(DATA / "B_predictions_hourly.csv", parse_dates=["date"])
    c = load_pred("c2s").query("H == 1")[["date", "hour", "pred"]].rename(columns={"pred": "c2s"})
    r = pd.read_csv(DATA / "wzoo_resid" / "외부만.csv.gz", parse_dates=["date"]).query("H == 1")[["date", "hour", "rhat"]]
    lam = json.loads((TAB / "WZ_choice.json").read_text(encoding="utf-8")).get("잔차 보정 λ", 0.5)
    m = (V[["date", "hour", "y", "pred", "s_d1"]].rename(columns={"pred": "v2"})
         .merge(B[["date", "hour", "pred"]].rename(columns={"pred": "b1"}), on=["date", "hour"])
         .merge(c, on=["date", "hour"]).merge(r, on=["date", "hour"]))
    m["final"] = m["c2s"] + lam * m["rhat"]
    m = m[m["y"].notna()]
    names = {"final": "Chronos-2 small + 잔차 보정 (최종)", "c2s": "Chronos-2 small 단독",
             "v2": "내일 전용 2차 (LightGBM·CatBoost 앙상블 + 급락 결합)", "b1": "내일 전용 1차 (LightGBM 8구성 중 선택)",
             "s_d1": "단순 기준: 전날 같은 시간"}
    pers = {"2025 상반기": ("2025-01-01", "2025-07-01"), "2025 하반기": ("2025-07-01", "2026-01-01"),
            "2025 전체": ("2025-01-01", "2026-01-01"), "2026 (시험)": ("2026-01-01", "2100-01-01")}
    rows = []
    for k, nm in names.items():
        row = {"모델": nm}
        for per, (a, b) in pers.items():
            g = m[(m["date"] >= a) & (m["date"] < b)]
            row[per] = (g["y"] - g[k]).abs().mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("2025 전체")


def main():
    w = weekly_table()
    rows = run_hourly(w, "일주일", "naive7") + run_hourly(w[w["H"] == 1], "일주일 D+1", "s_d1") + run_monthly()
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "F1_final_metrics.csv", index=False, encoding="utf-8-sig")
    nd = nextday_compare()
    nd.round(3).to_csv(TAB / "F2_nextday_compare.csv", index=False, encoding="utf-8-sig")
    wz = json.loads((TAB / "WZ_choice.json").read_text(encoding="utf-8"))
    mz = json.loads((TAB / "MZ_choice.json").read_text(encoding="utf-8"))
    last = w["date"].max().strftime("%Y-%m-%d")
    info = {"확정일": "2026-10-23", "시험 기간": f"2026-01-01 ~ {last}",
            "다음 날 급락 판별": {"모델": "과제1 최종 모델: Ridge + EBM + 랜덤포레스트 평균 (내일 11~15시 최저 SMP 예측)",
                             "판정": "예상 한낮 최저 ≤ 90.1원 주의, ≤ 75.3원 강한 경보 (2025 급락 90%를 잡는 문턱), 급락 = 한낮 최저 30원 미만",
                             "확률": "로지스틱 보정 (2025로 적합, D_choice.json)",
                             "2026 시험": "급락 20일 · 주의 이상 18일 적중·헛경보 72 · 강한 경보 16일 적중·헛경보 27 · ROC-AUC 0.932"},
            "일주일": {"모델": f"Chronos-2 small + {wz.get('잔차 보정 λ')} × 잔차 보정({wz.get('잔차 보정 피처 묶음')})",
                     "범위": wz["범위"]},
            "한 달": {"모델": {SN[s]: v["점예측"] for s, v in mz["지표별"].items()},
                    "범위": {SN[s]: v["범위"] for s, v in mz["지표별"].items()}}}
    (TAB / "F_final.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = ["예측", "기간", "구분", "표본수", "MAE", "RMSE", "MAPE(20원 이상)", "rMAE", "편향(공표−예측)", "포함률", "평균 폭", "구간점수"]
    print(t[t["구분"].isin(["전체", "일평균", "한낮 11~15시", "저녁 18~22시", "한낮 최저"])][show].round(3).to_string(index=False))
    print(nd.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
