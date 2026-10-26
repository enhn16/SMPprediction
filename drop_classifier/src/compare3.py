"""발표용 세 모델 비교 (육지 급락, 1차 15시) — 팀원 8개 결합 / 3개 결합(압축) / 현재 모델.

주의: 3개 결합의 2026 성적은 team_select.py 실행(2026-10-26) 뒤에 계산한 '참고' 값 (모델 선택에 쓰지 않음).
     팀원 8개 결합은 일별 예측 파일이 없어 팀원이 알려 준 값만 (나머지 빈칸).
입력: 현재 모델 = data/processed/smp_forecast_app/drop_badge.csv (2025 = 2024 학습 검증, 2026 = 2024~2025 학습)
     3개 결합 = team_select.py와 같은 데이터·규칙 (F0 피처, 2023~2024 학습 → 2025, 2023~2025 학습 → 2026)
지표: team_scorecard.py와 같은 정의, 휴일·주말만
출력: outputs/tables/19_compare3_육지.csv
실행: python -m drop_classifier.src.compare3
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from common.config import PROCESSED_DIR
from drop_classifier.src.final import threshold_for
from drop_classifier.src.team_select import BASES, F0, TEST, TRAIN_FIN, TRAIN_SEL, VALID, boot, data23, split
from drop_classifier.src.two_tier import OUT_TAB, strong_threshold
from smp_forecast_app.src.team_scorecard import drop_stats

warnings.filterwarnings("ignore")
FUSION = ["로지스틱", "랜덤포레스트(분류)", "LightGBM(분류)"]
H1 = ("2026-01-01", "2026-06-30")


def fusion(tr, te):
    t0 = time.time()
    s = np.mean([BASES[m](tr, te, F0) for m in FUSION], axis=0)
    return s, time.time() - t0


def summarize(name, d25, d26, warn, strong, prob26, extra):
    """d25·d26: date, label, s(클수록 위험) — 휴일·주말만."""
    y5, s5, y6, s6 = d25["label"].values, d25["s"].values, d26["label"].values, d26["s"].values
    v, t, bs = drop_stats(y5, s5), drop_stats(y6, s6), boot(y6, s6)
    h = d26["date"].between(*H1).values
    hit = lambda a, y: int((a & (y == 1)).sum())
    fa = lambda a, y: int((a & (y == 0)).sum())
    w5, w6, g5, g6 = s5 >= warn, s6 >= warn, s5 >= strong, s6 >= strong
    return {
        "모델": name,
        "PR-AUC 2026 시험": t["PR-AUC"], "PR-AUC 95% 구간": f"{bs['PR-AUC'].quantile(.025):.2f} ~ {bs['PR-AUC'].quantile(.975):.2f}",
        "ROC-AUC 2026": t["ROC-AUC"], "Brier 2026 (낮을수록 좋음)": brier_score_loss(y6, prob26),
        "헛경보@재현율80% 2026": t["헛경보@재현율80%"],
        "헛경보@재현율80% 95% 구간": f"{bs['헛경보@재현율80%'].quantile(.025):.0f} ~ {bs['헛경보@재현율80%'].quantile(.975):.0f}",
        "헛경보@재현율90% 2026": t["헛경보@재현율90%"],
        "주의 이상 2026 (적중/18 · 헛경보)": f"{hit(w6, y6)}/{int(y6.sum())} · {fa(w6, y6)}",
        "경보 2026 (적중/18 · 헛경보)": f"{hit(g6, y6)}/{int(y6.sum())} · {fa(g6, y6)}",
        "경보 정밀도 2026": hit(g6, y6) / max(g6.sum(), 1),
        "PR-AUC 2025 검증": v["PR-AUC"], "PR-AUC 검증→시험 변화": t["PR-AUC"] - v["PR-AUC"],
        "헛경보@재현율80% 2025 검증": v["헛경보@재현율80%"],
        "주의 이상 2025 (적중 · 헛경보)": f"{hit(w5, y5)} · {fa(w5, y5)}", "경보 2025 (적중 · 헛경보)": f"{hit(g5, y5)} · {fa(g5, y5)}",
        "PR-AUC 2026 1~6월": average_precision_score(y6[h], s6[h]), "PR-AUC 2026 7월~": average_precision_score(y6[~h], s6[~h]),
        **extra}


def main():
    df = data23()
    va, te = split(df, VALID), split(df, TEST)
    vo, to = va["offday"].values, te["offday"].values

    # 3개 결합: 2025 검증(두 학습 기간), 2026 시험
    s23, _ = fusion(split(df, TRAIN_SEL["2023~2024"]), va)
    s24, _ = fusion(split(df, TRAIN_SEL["2024"]), va)
    s26, sec = fusion(split(df, TRAIN_FIN), te)
    yv = va.loc[vo, "label"].values
    warn, strong = threshold_for(yv, s23[vo], 0.9), strong_threshold(yv, s23[vo])
    cal = LogisticRegression(C=1e6).fit(s23[vo].reshape(-1, 1), yv)
    d25 = va.loc[vo, ["date", "label"]].assign(s=s23[vo])
    d26 = te.loc[to, ["date", "label"]].assign(s=s26[to])
    rows = [summarize("3개 결합 (로지스틱+RF+LightGBM 분류)", d25, d26, warn, strong,
                      cal.predict_proba(s26[to].reshape(-1, 1))[:, 1],
                      {"PR-AUC 학습기간 흔들림 (2023~24 vs 2024 학습, 2025)":
                           f"{average_precision_score(yv, s23[vo]):.3f} vs {average_precision_score(yv, s24[vo]):.3f}",
                       "모델 수": 3, "예측 대상": "급락 확률 (0~1)", "문턱 단위": "확률 → 재학습하면 척도 바뀜",
                       "학습 기간 (시험용)": "2023~2025", "학습+예측 시간(초)": round(sec, 1),
                       "2026 채점 횟수": "1회 (선택 후 참고 계산)", "해석": "변수 영향 확인 어려움 (트리·부스팅 섞임)"})]

    # 현재 모델 (운영 파일)
    b = pd.read_csv(PROCESSED_DIR / "smp_forecast_app" / "drop_badge.csv", parse_dates=["date"])
    b = b.merge(df[["date", "offday"]], on="date")
    b = b[b["offday"] & b["label"].notna()].assign(s=lambda x: -x["pred_min"], label=lambda x: x["label"].astype(int))
    c25, c26 = b[b["date"].between(*VALID)], b[b["date"].between(*TEST)]
    tr = split(df, ("2024-01-01", "2025-12-31"))
    t0 = time.time()
    np.mean([BASES[m](tr, te, F0[:-1]) for m in ("Ridge", "EBM(회귀)", "랜덤포레스트(회귀)")], axis=0)
    rows.append(summarize("현재 모델 (Ridge+EBM+RF, 예상 한낮 최저)", c25, c26, -90.1, -75.3, c26["prob"].values,
                          {"PR-AUC 학습기간 흔들림 (2023~24 vs 2024 학습, 2025)": "0.495 vs 0.479 (같은 구성, team_select 실행)",
                           "모델 수": 3, "예측 대상": "내일 한낮 최저 SMP (원)", "문턱 단위": "원 (주의 90.1 · 경보 75.3) → 재학습해도 같은 뜻",
                           "학습 기간 (시험용)": "2024~2025", "학습+예측 시간(초)": round(time.time() - t0, 1),
                           "2026 채점 횟수": "1회", "해석": "Ridge 계수·EBM 그래프로 설명 가능, '예상 ○○원'으로 바로 읽힘"}))

    # 팀원 8개 결합 (보고 값만)
    rows.insert(0, {"모델": "팀원 8개 결합 (LGBM×2+LSTM+RF+ET+결합+로지스틱+나무)", "PR-AUC 2026 시험": 0.656,
                    "주의 이상 2026 (적중/18 · 헛경보)": "15/18 · 31", "주의 이상 2025 (적중 · 헛경보)": "– · 22",
                    "모델 수": 8, "예측 대상": "급락 확률", "문턱 단위": "확률",
                    "2026 채점 횟수": "2회 이상 (규제 강화 전후 34→31)", "해석": "어려움 (LSTM·결합 포함)"})
    res = pd.DataFrame(rows).set_index("모델").T
    res.to_csv(OUT_TAB / "19_compare3_육지.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(res.map(lambda x: round(x, 3) if isinstance(x, float) else x).to_string())


if __name__ == "__main__":
    main()
