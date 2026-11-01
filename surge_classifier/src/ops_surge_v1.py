"""[기록용, 2026-10-28 v3로 교체됨] 예전 운영 함수 — 지수식(15h)·공표 확정(21h). 파일 이름 *_v1_index·*_v1.

predict(bases, stage) -> DataFrame[base_date, date, stage, grade, prob, score]
- bases: 판단일 목록 (대상일 = 판단일 + 1). 판단일까지 자료만 쓴다 (features.build가 D-1 이전 값만 만듦).
- 15h: 선택된 지수(surge_choice_육지_15h.json)의 식·z 통계·문턱·확률식을 그대로 적용. 학습 없음.
- 21h: ① 내일 공표 SMP(24시간)가 있으면 확인(예측 아님): 최고 ÷ 기준 ≥ 1.25 → 경보(prob 1), 아니면 없음(prob 0), score = 실제 비율
       ② 없으면 2차 선택 결과(surge_choice_육지_21h.json) 적용 — 2026-10-26 선택은 1차와 같은 지수(내일 수요예측 미사용)
- 운영 기록 폴더는 건드리지 않는다. 호출 순서: (매일 수집 끝난 뒤) predict([오늘], "15h") 15:30, predict([오늘], "21h") 21:30·23:30.

검증: python -m surge_classifier.src.ops_surge_v1  → 백테스트 파일(surge_육지_{stage}.csv)과 2026 행이 같은지, 소요 시간
"""
import json
import time

import numpy as np
import pandas as pd

from surge_classifier.src.features import build
from surge_classifier.src.label_days import OUT_TAB, RATIO

REGION = "육지"
SUFFIX = {"15h": "_v1_index", "21h": "_v1"}


def _choice(stage: str) -> dict:
    return json.loads((OUT_TAB / f"surge_choice_{REGION}_{stage}{SUFFIX[stage]}.json").read_text(encoding="utf-8"))


def _index(d: pd.DataFrame, ch: dict) -> np.ndarray:
    """choice json의 지수식: Σ z(피처), 빈칸 z = 0, 평일만이면 휴무일 = -10."""
    z = sum(((d[c] - v["평균"]) / v["표준편차"]).fillna(0) for c, v in ch["z 통계"].items())
    if "평일만" in ch["점수"]:
        z = np.where(d["is_offday"] == 1, -10.0, z)
    return np.asarray(z, dtype=float)


def _apply(d: pd.DataFrame, ch: dict) -> pd.DataFrame:
    if ch["피처 묶음"] != "지수":
        raise NotImplementedError("학습 모델 선택 시 운영 함수 추가 필요 (2026-10-26 선택은 지수)")
    s = _index(d, ch)
    a, b = ch["확률식"]["a"], ch["확률식"]["b"]
    th = ch["문턱 (모델 내부 점수)"]
    return pd.DataFrame({"grade": np.where(s >= th["경보"], "경보", np.where(s >= th["주의"], "주의", "없음")),
                         "prob": 1 / (1 + np.exp(-(a * s + b))), "score": s}, index=d.index)


def predict(bases, stage: str = "15h", feats: pd.DataFrame | None = None) -> pd.DataFrame:
    bases = pd.to_datetime(pd.Series(list(bases))).dt.normalize()
    df = build(REGION) if feats is None else feats
    d = df.set_index("date").reindex(bases + pd.Timedelta(days=1))
    if d["sky_pm"].isna().all() and d["is_offday"].isna().any():
        raise ValueError("대상일 피처가 없음 — 판단일 SMP 실적이 아직 수집되지 않았을 수 있음")
    res = _apply(d, _choice("15h" if stage == "15h" else "21h"))
    if stage == "21h":
        pub = d["ratio"].notna().values   # 내일 공표 SMP 24시간이 있음 → 확정
        res.loc[pub, "grade"] = np.where(d["ratio"].values[pub] >= RATIO, "경보", "없음")
        res.loc[pub, "prob"] = (d["ratio"].values[pub] >= RATIO).astype(float)
        res.loc[pub, "score"] = d["ratio"].values[pub]
    out = pd.DataFrame({"base_date": bases.dt.strftime("%Y-%m-%d").values,
                        "date": (bases + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d").values, "stage": stage,
                        "grade": res["grade"].values, "prob": res["prob"].astype(float).round(4).values,
                        "score": res["score"].astype(float).round(4).values})
    return out


def main():
    t0 = time.time()
    feats = build(REGION)
    t_feat = time.time() - t0
    last = feats.loc[feats["ratio"].notna(), "date"].max()
    for stage in ("15h", "21h"):
        bt = pd.read_csv(OUT_TAB / f"surge_{REGION}_{stage}{SUFFIX[stage]}.csv", dtype={"base_date": str, "date": str})
        bt = bt[bt["date"] >= "2026-01-01"]
        t1 = time.time()
        p = predict(pd.to_datetime(bt["base_date"]), stage, feats)
        m = bt.merge(p, on=["base_date", "date"], suffixes=("_bt", "_op"))
        if stage == "21h":   # 백테스트는 화요일 판단을 미공표로 처리, 운영은 실제 공표 여부 → 화요일 판단 행은 비교에서 뺌
            m = m[pd.to_datetime(m["base_date"]).dt.dayofweek != 1]
        diff = (m["grade_bt"] != m["grade_op"]).sum() + (np.abs(m["prob_bt"] - m["prob_op"]) > 1e-3).sum()
        print(f"[{stage}] 2026 백테스트와 비교 {len(m)}일, 다른 칸 {diff}, 판단 {time.time() - t1:.2f}초")
    today = predict([last], "15h", feats)
    print(f"피처 만들기 {t_feat:.1f}초. 마지막 실적일 {last:%Y-%m-%d} 기준 내일:", today.to_dict("records"))


if __name__ == "__main__":
    main()
