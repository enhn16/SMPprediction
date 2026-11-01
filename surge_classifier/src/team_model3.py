"""④ 급등 경보 v3 (육지, 15시) — 최종 운영 모델 산출물 (ROLES.md 4.1, kind=surge). 2026-10-28 확정.

v3 = v2(팀원 spike_simple3 재구현, team_model2) + ④-6 후보 C(team_model2_fa):
  · 입력 40개 = 팀원 37개 + 평일×오후 흐림(wk_sky_pm)·평일×한낮 JMA 운량(wk_cloud_mid_jma)·주간전망 최대부하(kpx_peak),
    세 입력은 부호 고정 점수에 (+)로도 추가
  · 세 모델(부호 고정·스플라인·그루터기) 확률의 단순 평균, 휴무일은 0 (휴무일엔 주의·경보 없음)
  · 2024-01 ~ 2025-12 매달 앞으로만 검증 → 문턱(주의 = 재현율 80% 최고선, 경보 = 정밀도 50%·적중 2일 / 없으면 정밀도 최고선,
    0.01 격자 — 급락과 같은 규칙) · Platt 확률 보정 → 2023~2025 재학습 → 2026
출력 (outputs/tables): surge_육지_15h.csv (2025-01-01 ~ 내일; 2025 = 매달 앞으로만 예측, 2026~ = 최종 모델),
      surge_choice_육지_15h.json, surge_score_육지_15h.csv; 학습한 모델 data/processed/surge_classifier/v3_models.pkl (운영 함수용)
실행: python -m surge_classifier.src.team_model3
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from surge_classifier.src import team_model2 as T2
from surge_classifier.src import team_model2_fa as FA
from surge_classifier.src.extra_features import load_openmeteo_jma
from surge_classifier.src.features import build as build_surge_features
from surge_classifier.src.label_days import OUT_DATA, OUT_TAB
from surge_classifier.src.models import score_rows

REGION = "육지"
CFG = FA.CANDS["C A+B"]
MODEL_PKL = OUT_DATA / "v3_models.pkl"
TEST = ("2026-01-01", "2026-10-18")


def data(extend_days: int = 1) -> pd.DataFrame:
    """v3 입력 표: 팀원 방식 일별 표 + 13종 피처 3개 (모두 D-1 15시에 확보)."""
    d = T2.build_daily(extend_days)
    f = build_surge_features(REGION, extend_days).set_index("date")[["wk_sky_pm", "kpx_peak"]]
    jma = load_openmeteo_jma()["cloud_mid_jma"]
    d = d.join(f, on="date")
    d["wk_cloud_mid_jma"] = (1 - d["is_off"]) * d["date"].map(jma)
    return d


def grade(s, th):
    return np.where(s >= th["경보"], "경보", np.where(s >= th["주의"], "주의", "없음"))


def prob(s, cal):
    return 1 / (1 + np.exp(-(cal["a"] * np.asarray(s, float) + cal["b"])))


def main():
    FA.configure(CFG)
    try:
        d = data()
        val = T2.walk_forward(d)
        sv = FA.combine(val, d, CFG)
        yv = d.loc[sv.index, "y"].values.astype(int)
        th = {"주의": T2.caution_line(yv, sv.values), "경보": T2.alert_line(yv, sv.values)}
        lr = LogisticRegression(C=1e6).fit(sv.values.reshape(-1, 1), yv)
        cal = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
        models = T2.final_models(d)
        MODEL_PKL.write_bytes(pickle.dumps(models))
        te = d[(d["date_s"] >= "2026-01-01") & d[T2.FEATS[0]].notna()]
        st = FA.combine(T2.predict(models, te), d, CFG)
    finally:
        feats = list(T2.FEATS)
        FA.configure({})

    s = pd.concat([sv[d.loc[sv.index, "date_s"] >= "2025-01-01"], st])
    rows = d.loc[s.index]
    bt = pd.DataFrame({"base_date": (rows["date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d").values,
                       "date": rows["date_s"].values, "stage": "15h", "grade": grade(s.values, th),
                       "prob": prob(s.values, cal).round(4), "score": s.values.round(4), "label": rows["y"].values})
    bt.to_csv(OUT_TAB / f"surge_{REGION}_15h.csv", index=False, encoding="utf-8-sig")

    # 성적표 (team_scorecard.py와 같은 지표, models.score_rows)
    sc = []
    val_rows = d.loc[sv.index]
    parts = {"2024~2025 검증 (문턱·보정 구간)": (yv, sv.values),
             "2025 검증": (yv[(val_rows["date_s"] >= "2025-01-01").values], sv.values[(val_rows["date_s"] >= "2025-01-01").values])}
    t = bt[(bt["date"] >= TEST[0]) & (bt["date"] <= TEST[1])].dropna(subset=["label"])
    parts["2026 시험 (2026-01-01 ~ 10-18, 참고: 후보 C 선택 전에 v2 성적을 본 구간)"] = (t["label"].astype(int).values, t["score"].values)
    for name, (y, sc_) in parts.items():
        sc += score_rows(name, np.asarray(y, int), np.asarray(sc_, float), prob(sc_, cal), th["주의"], th["경보"])
    pd.DataFrame(sc).round(4).to_csv(OUT_TAB / f"surge_score_{REGION}_15h.csv", index=False, encoding="utf-8-sig")

    choice = {
        "stage": "15h (전날 15시)", "버전": "v3 (2026-10-28 확정)",
        "모델": "팀 합의 급등 모델: 부호 고정 점수 + 스플라인 로지스틱 + LightGBM 그루터기(깊이 1) 확률 평균 "
                "(팀원 spike_simple3 재구현 + 흐린 평일·주간전망 입력 추가 + 휴무일 제외)",
        "피처": feats, "피처 수": len(feats),
        "부호 고정": {**T2.SIGNS, **FA.ADD},
        "부호 고정 점수": "학습 구간 평균·표준편차로 z (빈칸 = 학습 구간 중앙값) × 부호 → 단순 평균 → 1차원 로지스틱(C=1, balanced, 기울기 ≥ 0)",
        "스플라인 로지스틱": "학습 구간 중앙값으로 빈칸 → StandardScaler → SplineTransformer(3차, 매듭 4, 바깥 상수) → LogisticRegression(C=0.05, L2, balanced)",
        "그루터기": T2.STUMP, "결합": "세 확률(0~1) 단순 평균, 휴무일(주말·공휴일) = 0",
        "문턱": th,
        "문턱 규칙": "2024~2025 매달 앞으로만 검증 점수, 0.01 격자. 주의 = 급등 80% 이상 잡는 가장 높은 선, "
                  "경보 = 정밀도 50%·적중 2일 이상 중 가장 낮은 선 (없으면 정밀도 최고선) — 급락과 같은 규칙",
        "확률식": {"형태": "1/(1+exp(-(a·score+b))), score = 결합 점수 (2024~2025 검증 점수에 Platt 보정)", **cal},
        "학습": {"검증": "2024-01 ~ 2025-12 매달: 2023-01-01 ~ 그 전달 말 누적 학습", "최종": "2023-01-01 ~ 2025-12-31"},
        "선택 절차": "구성·세부 = 팀원 고정값(팀 합의). 헛경보 줄이기 후보 4개(PLAN.md ④-6)를 사전 등록 → 2024~2025 검증 주의 헛경보로 C 선택 "
                  "(342 → 218일, 재현율 80.8% 유지). 2026은 v2 성적을 이미 본 구간이라 참고",
        "입력 출처": "SMP = EPSIS + data.go.kr, 수요예측(D-1분까지) = data.go.kr, 특일정보, Open-Meteo 2일 전 실행 best_match(기온·운량·강수) "
                  "+ JMA(한낮 운량), 구역별 육상예보 11시분(오후 하늘), 주간 전력수요전망",
    }
    (OUT_TAB / f"surge_choice_{REGION}_15h.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"문턱 {th}, 확률식 {cal}")
    print(pd.DataFrame(sc)[["구간", "일수", "급등일", "PR-AUC", "PR-AUC 배수", "헛경보@재현율80%", "경보", "적중", "놓침",
                            "헛경보", "재현율", "정밀도"]].round(3).to_string(index=False))
    print(f"백테스트 {len(bt)}일 ({bt['date'].min()} ~ {bt['date'].max()})")


if __name__ == "__main__":
    main()
