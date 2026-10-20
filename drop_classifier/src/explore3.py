"""탐색 4차: 경부하기 대책기간 (정부 사전 발표, 원전 감발·출력제어 등 계통 안정화 조치 기간).

기간 (사용자 제공, 2026-10-20; 산업부·기후부 발표 기준. 시작 전에 발표되므로 D-1 15시에 확보)
  2024 봄 03-23 ~ 06-02, 가을 09-21 ~ 11-03
  2025 봄 03-01 ~ 06-01, 가을 09-20 ~ 11-16
  2026 봄 02-28 ~ 06-14, 가을 09-19 ~ 11-15
피처: lowload(기간 여부), lowload_spring(봄 기간 여부), lowload_day(기간 시작 후 경과일 ÷ 기간 길이)
검증·채택 규칙: explore.py와 동일 (2025 확장 창 2회, D-1 평일 날 채점, 현 모델 대비 15% 이상 개선 + 전체 비악화 시 교체)
실행: python -m drop_classifier.src.explore3
"""
import warnings

import numpy as np
import pandas as pd

from drop_classifier.src.explore import CAT_R, CURRENT, EBM_R, RIDGE, avg, cv_scores, score
from drop_classifier.src.explore2 import KPX, OM, data2
from drop_classifier.src.fuel_level import BASE
from drop_classifier.src.supply_level import SOLAR
from drop_classifier.src.two_tier import OUT_TAB

warnings.filterwarnings("ignore")
PERIODS = [("2024-03-23", "2024-06-02", "봄"), ("2024-09-21", "2024-11-03", "가을"),
           ("2025-03-01", "2025-06-01", "봄"), ("2025-09-20", "2025-11-16", "가을"),
           ("2026-02-28", "2026-06-14", "봄"), ("2026-09-19", "2026-11-15", "가을")]
LL = ["lowload", "lowload_spring", "lowload_day"]


def add_lowload(df: pd.DataFrame) -> pd.DataFrame:
    df["lowload"], df["lowload_spring"], df["lowload_day"] = 0, 0, 0.0
    for a, b, season in PERIODS:
        m = df["date"].between(a, b)
        df.loc[m, "lowload"] = 1
        df.loc[m, "lowload_spring"] = int(season == "봄")
        df.loc[m, "lowload_day"] = (df.loc[m, "date"] - pd.Timestamp(a)).dt.days / (pd.Timestamp(b) - pd.Timestamp(a)).days
    return df


FEATS = {
    "현재(기본+태양광설비)": BASE + SOLAR,
    "+경부하기": BASE + SOLAR + LL,
    "+경부하기+KPX실적": BASE + SOLAR + LL + KPX,
    "+경부하기+일사예보": BASE + SOLAR + LL + OM,
}
MODELS = {"Ridge+EBM(현재)": CURRENT, "Ridge+EBM+CatBoost 회귀": avg(RIDGE, EBM_R, CAT_R)}


def main():
    df = add_lowload(data2())
    x = df[df["date"].between("2024-01-01", "2025-12-31") & (df["is_offday"] == 1)]
    print("휴무일 급락률 (2024~2025):", x.groupby("lowload")["label"].mean().round(3).to_dict(),
          "| 봄 기간:", x[x.lowload_spring == 1]["label"].mean().round(3), flush=True)
    rows = []
    for fname, cols in FEATS.items():
        for mname, fn in MODELS.items():
            p = cv_scores(df, cols, fn)
            sw, aw = score(p[p["d1_weekday"] == 1])
            sa, aa = score(p)
            rows.append({"피처": fname, "모델": mname, "D-1평일 선택점수": sw, "D-1평일 PR-AUC": aw,
                         "전체 선택점수": sa, "전체 PR-AUC": aa})
            print(f"{fname} / {mname}: D-1평일 {sw} (PR {aw:.3f}) | 전체 {sa} (PR {aa:.3f})", flush=True)
    res = pd.DataFrame(rows).sort_values(["D-1평일 선택점수", "D-1평일 PR-AUC"], ascending=[True, False])
    res.round(3).to_csv(OUT_TAB / "16_explore3_cv2025_육지.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n", res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
