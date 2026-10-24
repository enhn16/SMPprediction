"""제주 최종 모델 성적표 — 육지 final_metrics.py와 같은 지표·구간 정의 (팀 비교용).

  일주일 (D+1~D+7, 시간별): JWZ_choice.json의 모델 (JWZ_predictions_hourly.csv)
  지표: MAE·RMSE·MAPE(공표 20원 이상)·rMAE(단순 기준 = 지난주 같은 시간, D+1 행은 전날 같은 시간)·편향·포함률·하한 아래·상한 위·평균 폭·구간점수
  구간: 전체·평일·휴무일·휴무일 한낮(11~15시)·급락 시간(30원 미만)·D+H
  다음 날 급락: 2025 표본 밖 (헛경보·적중) — 2026은 --score2026
기간: 2025 하반기 (선택) · 2026 (시험, --score2026 일 때만)
출력: outputs/tables/JF1_final_metrics.csv
실행: python -m smp_forecast_app.src.jz_final_metrics [--score2026]
"""
import sys

import pandas as pd

from common.config import ROOT
from smp_forecast_app.src.final_metrics import metrics, hourly_segments
from smp_forecast_app.src.jz_common import JDIR, load_pred
from smp_forecast_app.src.jz_features import load as load_feats

TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"


def main(score2026=False):
    periods = {"2025 하반기 (선택)": ("2025-07-01", "2026-01-01")}
    if score2026:
        periods["2026 (시험)"] = ("2026-01-01", "2100-01-01")
    w = pd.read_csv(JDIR / "JWZ_predictions_hourly.csv", parse_dates=["base_date", "date"])
    w = w[w["y"].notna()]
    n7 = load_pred("jnaive7")[["date", "hour", "H", "pred"]].rename(columns={"pred": "naive7"})
    d1 = load_feats(1)[["date", "hour", "s_d1"]]
    w = w.merge(n7, on=["date", "hour", "H"], how="left").merge(d1, on=["date", "hour"], how="left")
    rows = []
    for product, g0, naive in (("일주일", w, "naive7"), ("일주일 D+1", w[w["H"] == 1], "s_d1")):
        for per, (a, b) in periods.items():
            g = g0[(g0["date"] >= a) & (g0["date"] < b)]
            for seg, m in hourly_segments(g).items():
                x = g[m]
                if len(x):
                    rows.append({"계통": "제주", "예측": product, "기간": per, "구분": seg,
                                 **metrics(x["y"], x["pred"], x["lo"], x["hi"], x[naive])})
            if product == "일주일":
                for H, x in g.groupby("H"):
                    rows.append({"계통": "제주", "예측": product, "기간": per, "구분": f"D+{H}",
                                 **metrics(x["y"], x["pred"], x["lo"], x["hi"], x[naive])})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "JF1_final_metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    show = ["예측", "기간", "구분", "표본수", "MAE", "RMSE", "MAPE(20원 이상)", "rMAE", "편향(공표−예측)", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수"]
    print(t[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main("--score2026" in sys.argv[1:])
