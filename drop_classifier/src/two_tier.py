"""정밀도 개선: 2단계 경보 + 매달 재학습 (육지, 판단 시각 D-1 15:00).

규칙은 2026년을 보기 전에 여기서 고정한다 (협의자료 Ⅴ-7 공정성).
  A) 매달 재학습: 매달 1일에 2024-01-01 ~ 전달 말일 자료로 다시 학습해 그달을 예측.
     채택 여부는 2025년에서만 판단 — 2025년 매달 재학습 예측의 선택점수(적중 80%·90%에 필요한
     헛경보 수 합)가 고정 모델(2024 학습)보다 작으면 채택.
  B) 2단계 경보 (채택된 방식의 2025년 예측으로 문턱 결정)
     - 주의: 2025년 급락의 80%를 잡는 문턱 (기존과 같은 방식)
     - 강한 경보: 2025년 정밀도 50% 이상이면서 적중 3일 이상인 가장 느슨한 문턱
       (2025→2026에 정밀도가 약 13%p 떨어진 만큼 목표 40%보다 여유를 둠)
  C) 2026-01-01 ~ 최신일을 한 번 채점.
실행: python -m drop_classifier.src.two_tier
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.config import ROOT
from drop_classifier.src.experiments import false_alarms_at
from drop_classifier.src.fair_select import CANDS, SETS, fcst15, metrics
from drop_classifier.src.final import threshold_for
from drop_classifier.src.models import split
from drop_classifier.src.search import day_table

warnings.filterwarnings("ignore")
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
COLS = SETS["기본+주간전망+보강"]
FN = CANDS["Ridge+EBM 회귀 평균"]
START = "2024-01-01"
STRONG_PREC, STRONG_MIN_TP = 0.5, 3


def static_scores(df, train, test):
    te = split(df, test).copy()
    te["score"] = FN(split(df, train), te, COLS)
    return te


def monthly_scores(df, test):
    """매달 1일 재학습: 학습 = START ~ 전달 말일."""
    te = split(df, test).copy()
    te["score"] = np.nan
    for m, idx in te.groupby(te["date"].dt.to_period("M")).groups.items():
        tr = df[(df["date"] >= START) & (df["date"] < m.start_time)]
        te.loc[idx, "score"] = FN(tr, te.loc[idx], COLS)
    return te


def sel_score(y, s):
    return false_alarms_at(y, s, 0.8)[0] + false_alarms_at(y, s, 0.9)[0]


def strong_threshold(y, s):
    """정밀도 STRONG_PREC 이상·적중 STRONG_MIN_TP 이상인 가장 느슨한(낮은) 점수 문턱."""
    best = None
    for t in np.sort(np.unique(s)):  # 느슨한 → 엄격한 순
        a = s >= t
        tp = int((a & (y == 1)).sum())
        if tp >= STRONG_MIN_TP and tp / a.sum() >= STRONG_PREC:
            best = float(t)
            break
    return best


def tier_rows(te, thr_warn, thr_strong, name):
    y, s = te["label"], te["score"]
    rows = [{"방식": name, "경보": "주의 이상", **metrics(y, (s >= thr_warn).astype(int), s)},
            {"방식": name, "경보": "강한 경보", **metrics(y, (s >= thr_strong).astype(int), s)}]
    mid = (s >= thr_warn) & (s < thr_strong)
    rows.append({"방식": name, "경보": "주의만(강한 경보 제외)", "일수": int(mid.sum()),
                 "적중": int((mid & (y == 1)).sum()), "헛경보": int((mid & (y == 0)).sum())})
    return rows


def main():
    df = day_table().merge(fcst15(), on="date", how="left")
    V, T = ("2025-01-01", "2025-12-31"), ("2026-01-01", "2099-12-31")

    # 1) 2025에서 재학습 채택 여부 판단
    v_static = static_scores(df, ("2024-01-01", "2024-12-31"), V)
    v_month = monthly_scores(df, V)
    y = v_static["label"].values
    sc_static, sc_month = sel_score(y, v_static["score"].values), sel_score(y, v_month["score"].values)
    use_monthly = sc_month < sc_static
    v = v_month if use_monthly else v_static
    print(f"[2025] 선택점수 고정 {sc_static} / 매달 재학습 {sc_month} → {'매달 재학습' if use_monthly else '고정'} 채택",
          flush=True)

    # 2) 문턱 (2025)
    yv, sv = v["label"].values, v["score"].values
    thr_warn, thr_strong = threshold_for(yv, sv, 0.8), strong_threshold(yv, sv)
    print(f"문턱: 주의 {-thr_warn:.1f}원, 강한 경보 {-thr_strong:.1f}원", flush=True)
    rows = tier_rows(v, thr_warn, thr_strong, "2025 검증(선택용)")

    # 3) 2026 한 번 채점 — 채택 방식, 그리고 참고로 다른 방식
    t_static = static_scores(df, ("2024-01-01", "2025-12-31"), T)
    t_month = monthly_scores(df, T)
    chosen, other = (t_month, t_static) if use_monthly else (t_static, t_month)
    cname = "매달 재학습" if use_monthly else "고정"
    oname = "고정" if use_monthly else "매달 재학습"
    rows += tier_rows(chosen, thr_warn, thr_strong, f"2026 시험: {cname} (채택)")
    rows += tier_rows(other, thr_warn, thr_strong, f"2026 참고: {oname}")
    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT_TAB / "10_two_tier_육지.csv", index=False, encoding="utf-8-sig")

    d = chosen[["date", "day_type", "label", "min_smp_11_15", "score"]].copy()
    d["pred"] = -d.pop("score")
    d["tier"] = np.where(d["pred"] <= -thr_strong, "강한 경보", np.where(d["pred"] <= -thr_warn, "주의", "없음"))
    d.to_csv(OUT_TAB / "10_two_tier_daily_육지.csv", index=False, encoding="utf-8-sig")
    info = {"재학습": cname, "주의 문턱": round(-thr_warn, 1), "강한 경보 문턱": round(-thr_strong, 1),
            "2025 선택점수": {"고정": sc_static, "매달 재학습": sc_month}}
    (OUT_TAB / "10_two_tier_choice.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 250)
    cols = ["방식", "경보", "일수", "급락일", "적중", "놓침", "헛경보", "재현율", "정밀도", "헛경보율", "ROC-AUC"]
    print("\n", res.reindex(columns=cols).round(3).to_string(index=False))
    for nm, t in (("채택", chosen), ("참고", other)):
        for half, g in t.groupby(t["date"] < "2026-07-01"):
            a = g["score"] >= thr_warn
            tp, fp = int((a & (g.label == 1)).sum()), int((a & (g.label == 0)).sum())
            print(f"{nm} {'1~6월' if half else '7월~'}: 주의 이상 적중 {tp}/{int(g.label.sum())}, 헛경보 {fp}, 정밀도 {tp / max(tp + fp, 1):.0%}")


if __name__ == "__main__":
    main()
