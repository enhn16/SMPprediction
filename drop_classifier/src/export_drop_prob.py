"""과제2·3 연계용 일별 급락 위험 파일 (육지, 판단 시각 D-1 15:00, 2단계 경보).

모두 학습 구간 밖 예측이다.
- 2025년: 2024년 학습 모델 (문턱을 정한 검증 예측)
- 2026년~: 2024~2025년 학습 모델 (최종 시험 예측)
문턱은 two_tier.py가 2025년으로 정한 값(10_two_tier_choice.json)을 읽는다.
저장: outputs/tables/drop_prob_육지.csv
  date, 계통, 예측_한낮최저SMP(원), 위험점수(= -예측), 등급(없음/주의/강한 경보), 판정(주의 이상 = 1)
실행: python -m drop_classifier.src.export_drop_prob
"""
import json
import warnings

import numpy as np
import pandas as pd

from drop_classifier.src.fair_select import fcst15
from drop_classifier.src.search import day_table
from drop_classifier.src.two_tier import OUT_TAB, static_scores

warnings.filterwarnings("ignore")
REGION = "육지"


def main():
    choice = json.loads((OUT_TAB / "10_two_tier_choice.json").read_text(encoding="utf-8"))
    warn, strong = choice["주의 문턱"], choice["강한 경보 문턱"]
    df = day_table().merge(fcst15(), on="date", how="left")
    parts = [static_scores(df, ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31")),
             static_scores(df, ("2024-01-01", "2025-12-31"), ("2026-01-01", "2099-12-31"))]
    d = pd.concat(parts)[["date", "score"]]
    pred = -d["score"]
    out = pd.DataFrame({
        "date": d["date"].dt.strftime("%Y-%m-%d"), "계통": REGION,
        "예측_한낮최저SMP": pred.round(1), "위험점수": d["score"].round(3),
        "등급": np.where(pred <= strong, "강한 경보", np.where(pred <= warn, "주의", "없음")),
        "판정": (pred <= warn).astype(int)})
    out.to_csv(OUT_TAB / f"drop_prob_{REGION}.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {len(out)}일, 문턱 주의 {warn}원 / 강한 경보 {strong}원")
    print(out.groupby([out["date"].str[:4], "등급"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
