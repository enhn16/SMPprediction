"""운영용 내일 급등 경보 v3 (육지, 15시 하나) — ROLES.md 4.1 운영 함수. 2026-10-28 교체 (예전 지수식·21시는 ops_surge_v1.py).

predict(bases) -> DataFrame[base_date, date, stage, grade, prob, score]
- bases: 판단일 목록 (대상일 = 판단일 + 1). 판단일 15시까지 자료만 쓴다 (모든 입력이 D-1 이전 실적·2일 전 실행 예보·D-1 11시 예보).
- team_model3가 저장한 모델(v3_models.pkl, 2023~2025 학습)과 surge_choice_육지_15h.json의 문턱·확률식을 그대로 적용. 매일 다시 학습하지 않는다.
- 21시 판단 없음 (현업 10/28): 내일 SMP가 공표되면 웹이 공표값으로 교체 (총괄 처리).
- 매일 순서 (15:30 전): 공용 수집(SMP·수요예측·구역예보·주간전망·Open-Meteo JMA) → `python -m surge_classifier.src.collect_extra openmeteo`
  (Open-Meteo best_match 기온·운량·강수, 급등 v3 전용) → predict([오늘]).
- 운영 기록 폴더는 건드리지 않는다.

검증: python -m surge_classifier.src.ops_surge  → 백테스트(surge_육지_15h.csv) 2026 행과 같은지, 소요 시간
"""
import json
import pickle
import time

import numpy as np
import pandas as pd

from surge_classifier.src import team_model2 as T2
from surge_classifier.src import team_model2_fa as FA
from surge_classifier.src import team_model3 as V3
from surge_classifier.src.label_days import OUT_TAB

REGION = "육지"


def predict(bases, d: pd.DataFrame | None = None) -> pd.DataFrame:
    bases = pd.to_datetime(pd.Series(list(bases))).dt.normalize()
    ch = json.loads((OUT_TAB / f"surge_choice_{REGION}_15h.json").read_text(encoding="utf-8"))
    models = pickle.loads(V3.MODEL_PKL.read_bytes())
    d = V3.data() if d is None else d
    rows = d.set_index("date").reindex(bases + pd.Timedelta(days=1))
    if rows["prev_max"].isna().any():
        raise ValueError("대상일 입력이 없음 — 판단일 SMP 실적이 아직 수집되지 않았을 수 있음")
    if rows["wx_tmean"].isna().any():
        raise ValueError("대상일 Open-Meteo 예보가 없음 — collect_extra openmeteo를 먼저 실행")
    FA.configure(V3.CFG)
    try:
        rows = rows.reset_index().rename(columns={"index": "date"})
        p = T2.predict(models, rows)
        s = FA.combine(p, rows, V3.CFG).values
    finally:
        FA.configure({})
    return pd.DataFrame({"base_date": bases.dt.strftime("%Y-%m-%d").values,
                         "date": (bases + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d").values, "stage": "15h",
                         "grade": V3.grade(s, ch["문턱"]), "prob": V3.prob(s, ch["확률식"]).round(4), "score": s.round(4)})


def main():
    t0 = time.time()
    d = V3.data()
    t_data = time.time() - t0
    bt = pd.read_csv(OUT_TAB / f"surge_{REGION}_15h.csv", dtype={"base_date": str, "date": str})
    bt = bt[bt["date"] >= "2026-01-01"]
    t1 = time.time()
    p = predict(pd.to_datetime(bt["base_date"]), d)
    t_pred = time.time() - t1
    m = bt.merge(p, on=["base_date", "date"], suffixes=("_bt", "_op"))
    diff = int((m["grade_bt"] != m["grade_op"]).sum() + (np.abs(m["score_bt"] - m["score_op"]) > 1e-3).sum())
    print(f"2026 백테스트와 비교 {len(m)}일, 다른 칸 {diff}")
    t2 = time.time()
    one = predict([d.loc[d["y"].notna(), "date"].max()], d)
    print(f"입력 표 만들기 {t_data:.1f}초, 판단 {len(m)}일 {t_pred:.2f}초, 하루 판단 {time.time() - t2:.2f}초")
    print("마지막 실적일 기준 내일:", one.to_dict("records"))


if __name__ == "__main__":
    main()
