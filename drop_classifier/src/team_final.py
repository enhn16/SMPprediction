"""팀 규칙 공식 결과 (육지 급락, 1차 15시) — 모델은 현재 구성 Ridge+EBM+랜덤포레스트 그대로, 학습 기간만 팀 규칙.

결정: PLAN.md 결정 기록 2026-10-26 (사용자 결정, team_select.py·ens8.py의 다른 모델은 비교용)
  모델: final_model.py와 같음 (zoo.BASES의 Ridge·EBM(회귀)·랜덤포레스트(회귀) 평균, 피처 BASE+SOLAR, 점수 = −예상 한낮 최저)
  데이터: team_select.data23() (2023-01-01부터. 단기 육상예보 2023-01~11은 2026-10-27 수집 — 그 전 v0은 빈칸을 중앙값으로 채워 학습)
  문턱: 2023~2024 학습 → 2025 휴일·주말 예측에서 주의 = 급락 90% 포착, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨
        (없으면 적중 3일 이상 중 정밀도가 가장 높은 문턱, 같으면 가장 느슨 — 결정 기록 2026-10-27)
  확률: 같은 2025 예측에 로지스틱 보정
  시험: 2023~2025 학습 → 2026-01-01 ~ 10-18 휴일·주말 (team_scorecard.py 지표)
출력 (outputs/tables, ROLES.md 4.1): drop_육지_15h.csv, drop_choice_육지.json, drop_score_육지.csv
실행: python -m drop_classifier.src.team_final
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from drop_classifier.src.final import threshold_for
from drop_classifier.src.final_model import COLS, MEMBERS
from drop_classifier.src.team_select import data23, score_rows
from drop_classifier.src.two_tier import OUT_TAB, strong_threshold
from drop_classifier.src.zoo import BASES

warnings.filterwarnings("ignore")
REGION = "육지"
TR_VAL, VALID = ("2023-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31")
TR_FIN, TEST = ("2023-01-01", "2025-12-31"), ("2026-01-01", "2026-10-18")
RECALL_WARN = 0.9
NAME = "Ridge+EBM+랜덤포레스트 (예상 한낮 최저)"


def score(tr, te) -> np.ndarray:
    """위험 점수 = −예상 한낮(11~15시) 최저 SMP (zoo.BASES 3종 평균)."""
    return np.mean([BASES[m](tr, te) for m in MEMBERS], axis=0)


def alarm_threshold(y, s):
    """경보 문턱: 정밀도 50%·적중 3일 이상 중 가장 느슨, 없으면 적중 3일 이상 중 정밀도 최고(같으면 가장 느슨)."""
    t = strong_threshold(y, s)
    if t is not None:
        return t, "정밀도 50%"
    best, best_p = None, -1.0
    for t in np.sort(np.unique(s)):  # 느슨한 → 엄격한 순
        a = s >= t
        tp = int((a & (y == 1)).sum())
        if tp >= 3 and tp / a.sum() > best_p + 1e-12:
            best, best_p = float(t), tp / a.sum()
    return best, f"대체 규칙 (정밀도 최고 {best_p:.0%})"


def split(df, rng):
    return df[df["date"].between(*rng)]


def main():
    t0 = time.time()
    df = data23()
    va, te = split(df, VALID), split(df, (TEST[0], "2099-12-31"))
    sv = score(split(df, TR_VAL), va)
    t_val = time.time() - t0
    st = score(split(df, TR_FIN), te)
    vo = va["offday"].values
    yv = va.loc[vo, "label"].values
    warn, (strong, how) = threshold_for(yv, sv[vo], RECALL_WARN), alarm_threshold(yv, sv[vo])
    cal = LogisticRegression(C=1e6).fit(sv[vo].reshape(-1, 1), yv)
    to_prob = lambda s: cal.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]
    print(f"문턱: 주의 {-warn:.1f}원, 경보 {-strong:.1f}원 ({how})", flush=True)

    to = te["offday"].values & te["date"].between(*TEST).values
    yt = te.loc[to, "label"].values
    sc = pd.DataFrame(score_rows("2025 검증", yv, sv[vo], to_prob(sv[vo]), warn, strong)
                      + score_rows(f"2026 시험 {TEST[0]} ~ {TEST[1]}", yt, st[to], to_prob(st[to]), warn, strong))
    sc.insert(0, "모델", NAME)
    sc.round(4).to_csv(OUT_TAB / f"drop_score_{REGION}.csv", index=False, encoding="utf-8-sig")

    # 4.1 일별 (2025 = 2023~2024 학습 예측, 2026~ = 2023~2025 학습 예측). label은 휴일·주말만, 미확정 빈칸
    both = pd.concat([va.assign(s=sv), te.assign(s=st)])
    lab = both["label"].where(both["min_smp_11_15"].notna())
    out = pd.DataFrame({"base_date": (both["date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
                        "date": both["date"].dt.strftime("%Y-%m-%d"),
                        "grade": np.where(both["s"] >= strong, "경보", np.where(both["s"] >= warn, "주의", "없음")),
                        "prob": to_prob(both["s"]).round(4), "score": (-both["s"]).round(1),
                        "label": np.where(both["offday"] & lab.notna(), lab.fillna(0).astype(int).astype(str), "")})
    out.to_csv(OUT_TAB / f"drop_{REGION}_15h.csv", index=False, encoding="utf-8-sig")

    choice = {"stage": "15h (전날 15시)", "모델": NAME, "구성": MEMBERS, "피처 묶음": "기본+주간전망+보강+태양광설비 (BASE+SOLAR)",
              "피처": COLS,
              "점수": "score = 예상 한낮(11~15시) 최저 SMP(원), 낮을수록 위험",
              "문턱": {"주의": float(-warn), "경보": float(-strong),
                     "규칙": "반올림 전 score ≤ 문턱이면 해당 단계 (문턱이 2025 어느 날의 예측값과 같아 반올림하면 경계일이 바뀜)"},
              "문턱 표시": {"주의": round(-warn, 1), "경보": round(-strong, 1)},
              "확률식": {"형태": "prob = 1/(1+exp(-(a·(−score)+b)))", "a": float(cal.coef_[0][0]), "b": float(cal.intercept_[0])},
              "학습": {"문턱·확률": "2023-01-01 ~ 2024-12-31 학습 → 2025 휴일·주말 예측",
                     "최종": "2023-01-01 ~ 2025-12-31 (고정, 2026 예측에 사용)"},
              "문턱 규칙": "주의 = 2025 휴일·주말 급락 90% 포착, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨 "
                       "(없으면 적중 3일 이상 중 정밀도 최고, 같으면 가장 느슨)",
              "경보 문턱 적용": how,
              "선택 절차": "현재 운영 구성 유지 (PLAN.md 결정 기록 2026-10-26). team_select.py·ens8.py 후보는 비교용",
              "2023 자료": "단기 육상예보 2023-01부터 수집(빈 날 1/1), 주간전망 2022-12 게시분부터. 남은 빈칸은 학습 구간 중앙값 (Ridge·RF), EBM은 빈칸 그대로",
              "급락 정의": "11~15시 최저 SMP 30원 미만, 휴일·주말만 채점"}
    (OUT_TAB / f"drop_choice_{REGION}.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")

    show = ["구간", "경보", "일수", "급락일", "PR-AUC", "PR-AUC 95%하한", "PR-AUC 95%상한", "ROC-AUC", "헛경보@재현율80%",
            "헛경보@재현율90%", "Brier", "적중", "놓침", "헛경보", "재현율", "정밀도"]
    pd.set_option("display.width", 250)
    print(sc[show].round(3).to_string(index=False))
    miss = te[to & (st < warn) & (te["label"] == 1)]
    print("2026 놓침(주의 미만):", miss["date"].dt.strftime("%m-%d").tolist())
    print(f"2025 예측(1회 학습) {t_val:.0f}초, 총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
