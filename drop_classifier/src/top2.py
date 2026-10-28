"""팀원 최고 조합 직접 재현: 상위 2개 평균 (LSTM + LightGBM(확장 변수)), 문턱 = 재현율 80% (발표 참고 비교, 모델 선택 아님).

조합은 팀원 2026 결과를 보고 정한 것 → 2026 시험 구간 사후 비교. 구성원 정의는 ens8.py와 같다 (규제 고정, 튜닝 없음).
  LSTM: 시간별 SMP 168시간 + 표 피처(BASE+SOLAR), 시드 3개 평균
  LightGBM(확장 변수): BASE+SOLAR + 연료비 + 최근 7일 SMP 수준, 분류
  결합: 두 확률의 같은 가중치 평균
팀 규칙: data23() (2023 실제 예보 포함), 2023~2024 학습 → 2025 휴일·주말로 문턱·확률 보정, 2023~2025 학습 → 2026-01-01 ~ 10-18 휴일·주말
문턱 두 가지로 나란히:
  ① 팀원 방식 = 2025 급락 80% 포착 (한 단계)
  ② 우리 규칙 = 주의 90%, 경보 정밀도 50%(없으면 정밀도 최고) — team_final.py와 같음
비교 기준선 = team_final.py 공식 결과 (Ridge+EBM+랜덤포레스트), 같은 두 문턱 방식으로 다시 계산
출력: outputs/tables/21_top2_vs_current_육지.csv, 21_top2_daily_육지.csv
실행: python -m drop_classifier.src.top2  [--members → 21_top2_members_daily_육지.csv: 구성원별 확률, 팀원 대조용]
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from drop_classifier.src.ens8 import MEMBERS, data
from drop_classifier.src.final import threshold_for
from drop_classifier.src.team_final import TEST, TR_FIN, TR_VAL, VALID, alarm_threshold
from drop_classifier.src.team_final import score as current_score
from drop_classifier.src.team_select import boot
from drop_classifier.src.two_tier import OUT_TAB
from smp_forecast_app.src.team_scorecard import drop_stats

warnings.filterwarnings("ignore")


def top2(tr, te):
    return np.mean([MEMBERS[k](tr, te) for k in ("LSTM", "LightGBM(확장 변수)")], axis=0)


def split(df, rng):
    return df[df["date"].between(*rng)]


def hit_fa(a, y):
    return int((a & (y == 1)).sum()), int((a & (y == 0)).sum())


def evaluate(name, fn, df, unit):
    t0 = time.time()
    va, te = split(df, VALID), split(df, TEST)
    sv, st = fn(split(df, TR_VAL), va), fn(split(df, TR_FIN), te)
    sec = time.time() - t0
    vo, to = va["offday"].values, te["offday"].values
    y5, s5, y6, s6 = va["label"].values[vo], sv[vo], te["label"].values[to], st[to]
    t80, t90, (tal, how) = threshold_for(y5, s5, 0.8), threshold_for(y5, s5, 0.9), alarm_threshold(y5, s5)
    cal = LogisticRegression(C=1e6).fit(s5.reshape(-1, 1), y5)
    v, t, bs = drop_stats(y5, s5), drop_stats(y6, s6), boot(y6, s6)
    row = {"모델": name, "2025 검증 PR-AUC": v["PR-AUC"], "2025 검증 헛경보@재현율80%": v["헛경보@재현율80%"],
           "PR-AUC": t["PR-AUC"], "PR-AUC 95% 구간": f"{bs['PR-AUC'].quantile(.025):.2f} ~ {bs['PR-AUC'].quantile(.975):.2f}",
           "ROC-AUC": t["ROC-AUC"], "Brier": brier_score_loss(y6, cal.predict_proba(s6.reshape(-1, 1))[:, 1]),
           "헛경보@재현율80%": t["헛경보@재현율80%"], "헛경보@재현율90%": t["헛경보@재현율90%"]}
    row["② 경보 문턱 규칙"] = how
    for lab, thr in (("① 재현율80% 한 단계", t80), ("② 주의(90%)", t90), ("② 경보", tal)):
        h, f = hit_fa(s6 >= thr, y6)
        h5, f5 = hit_fa(s5 >= thr, y5)
        row[f"{lab} 문턱"] = unit(thr)
        row[f"{lab} 2025 적중·헛경보"] = f"{h5}/{int(y5.sum())} · {f5}"
        row[f"{lab} 2026 적중·헛경보"] = f"{h}/{int(y6.sum())} · {f}"
        row[f"{lab} 2026 정밀도"] = h / max(h + f, 1)
    row["학습·예측 시간(초)"] = round(sec)
    print(f"{name}: {sec:.0f}초", flush=True)
    return row, te.loc[to, ["date", "label"]].assign(model=name, score=s6)


def members_daily(df) -> pd.DataFrame:
    """구성원별 확률 (2025 = 2023~2024 학습, 2026 = 2023~2025 학습), 휴일·주말만 — 팀원 대조용."""
    parts = []
    for (tr, rng), period in (((TR_VAL, VALID), "2025 검증"), ((TR_FIN, TEST), "2026 시험")):
        te = split(df, rng)
        p = {k: MEMBERS[k](split(df, tr), te) for k in ("LSTM", "LightGBM(확장 변수)")}
        d = pd.DataFrame({"date": te["date"].dt.strftime("%Y-%m-%d"), "구간": period,
                          "학습 기간": f"{tr[0]} ~ {tr[1]}", "lstm": p["LSTM"], "lgbm": p["LightGBM(확장 변수)"]})
        d["평균"] = (d["lstm"] + d["lgbm"]) / 2
        d["정답"] = te["label"].astype(int).values
        parts.append(d[te["offday"].values])
    out = pd.concat(parts)
    for period, g in out.groupby("구간"):
        print(period, {k: round(drop_stats(g["정답"].values, g[k].values)["PR-AUC"], 3) for k in ("lstm", "lgbm", "평균")})
    return out


def main():
    df = data(team=True)
    if "--members" in sys.argv:
        members_daily(df).round(5).to_csv(OUT_TAB / "21_top2_members_daily_육지.csv", index=False, encoding="utf-8-sig")
        return
    te = split(df, TEST)
    print(f"시간별 SMP 시퀀스 결측(2026 시험): {te['seq'].map(lambda s: np.isnan(s).all()).mean():.1%}", flush=True)
    rows, days = [], []
    for name, fn, unit in (("상위 2개 평균 (LSTM + LightGBM 확장 변수)", top2, lambda s: round(float(s), 3)),
                           ("현재 모델 (Ridge+EBM+RF, 공식)", current_score, lambda s: f"{-s:.1f}원")):
        r, d = evaluate(name, fn, df, unit)
        rows.append(r)
        days.append(d)
    res = pd.DataFrame(rows).set_index("모델").T
    res.to_csv(OUT_TAB / "21_top2_vs_current_육지.csv", encoding="utf-8-sig")
    pd.concat(days).round(4).to_csv(OUT_TAB / "21_top2_daily_육지.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res.map(lambda x: round(x, 3) if isinstance(x, float) else x).to_string())


if __name__ == "__main__":
    main()
