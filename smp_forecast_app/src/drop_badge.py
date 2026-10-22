"""웹앱 급락 배지용 '내일 한낮 급락 확률' — 과제1 예상 한낮최저 SMP(원)를 확률로 보정.

과제1 최종 모델(drop_classifier/outputs/tables/drop_prob_육지.csv)은 확률이 아니라 내일 11~15시 최저 SMP 예측값과
등급(과제1 13_final_choice.json 문턱, 2026-10-21: 주의 ≤ 90.1원, 강한 경보 ≤ 75.3원)을 낸다. 이를 로지스틱 보정(Platt)으로 P(한낮 최저 < 30원)로 바꾼다.
- 보정식 적합: 2025 (2024 학습 모델의 표본 밖 예측) — 급락 정의는 과제1과 같음 (11~15시 최저 < 30원)
- 2026: 보정식 고정 후 신뢰도만 확인 (등급별·확률 구간별 실제 급락 비율, Brier) — 튜닝 없음
출력: data/processed/smp_forecast_app/drop_badge.csv [date, grade, pred_min, prob, label]
      outputs/tables/D1_drop_calibration.csv, D_choice.json
실행: python -m smp_forecast_app.src.drop_badge
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from common.config import PROCESSED_DIR, ROOT
from smp_forecast_app.src.ops_drop import STRONG, WARN

SRC = ROOT / "drop_classifier" / "outputs" / "tables" / "drop_prob_육지.csv"
LABEL = PROCESSED_DIR / "drop_classifier" / "daily_label_육지.csv"
OUT = PROCESSED_DIR / "smp_forecast_app" / "drop_badge.csv"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"


def main():
    d = pd.read_csv(SRC, parse_dates=["date"]).rename(columns={"예측_한낮최저SMP": "pred_min", "등급": "grade"})
    lab = pd.read_csv(LABEL, parse_dates=["date"])[["date", "label"]]
    d = d.merge(lab, on="date", how="left")
    fit = d[(d["date"].dt.year == 2025) & d["label"].notna()]
    lr = LogisticRegression(C=1e6).fit(fit[["pred_min"]], fit["label"].astype(int))
    d["prob"] = lr.predict_proba(d[["pred_min"]])[:, 1]

    rows = []
    for yr in (2025, 2026):
        g = d[(d["date"].dt.year == yr) & d["label"].notna()]
        for grade in ("강한 경보", "주의", "없음"):
            x = g[g["grade"] == grade]
            rows.append({"연도": yr, "구분": f"등급 {grade}", "일수": len(x), "평균 확률": x["prob"].mean(),
                         "실제 급락 비율": x["label"].mean(), "급락일": int(x["label"].sum())})
        for lo, hi in ((0, .05), (.05, .2), (.2, .5), (.5, 1.01)):
            x = g[(g["prob"] >= lo) & (g["prob"] < hi)]
            rows.append({"연도": yr, "구분": f"확률 {lo:.0%}~{min(hi, 1):.0%}", "일수": len(x), "평균 확률": x["prob"].mean(),
                         "실제 급락 비율": x["label"].mean(), "급락일": int(x["label"].sum())})
        rows.append({"연도": yr, "구분": "전체 Brier", "일수": len(g), "평균 확률": brier_score_loss(g["label"], g["prob"]),
                     "실제 급락 비율": roc_auc_score(g["label"], g["prob"]), "급락일": int(g["label"].sum())})
    cal = pd.DataFrame(rows)
    cal.round(4).to_csv(TAB / "D1_drop_calibration.csv", index=False, encoding="utf-8-sig")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d[["date", "grade", "pred_min", "prob", "label"]].round(4).to_csv(OUT, index=False, encoding="utf-8-sig")
    a, b = float(lr.intercept_[0]), float(lr.coef_[0][0])
    info = {"방법": "로지스틱 보정 P(급락) = 1 / (1 + exp(-(a + b × 예상 한낮최저)))", "a": round(a, 4), "b": round(b, 5),
            "적합 기간": "2025 (과제1 2024 학습 모델의 표본 밖 예측)",
            "등급 문턱 확률": {f"주의({WARN}원)": round(float(lr.predict_proba([[WARN]])[0, 1]), 3),
                          f"강한 경보({STRONG}원)": round(float(lr.predict_proba([[STRONG]])[0, 1]), 3)},
            "자료 마지막 날": str(d["date"].max().date())}
    (TAB / "D_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    print(cal.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
