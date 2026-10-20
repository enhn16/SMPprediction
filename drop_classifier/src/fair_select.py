"""공정성 규칙에 맞춘 하루 전 모델 재선택과 2026년 최종 시험 (육지, 판단 시각 D-1 15:00).

규칙 (협의자료 Ⅴ-7): 시험 구간(2026-01-01 이후)으로 선택·튜닝 금지, 예측 시점에 알 수 없는 정보 금지.
  1) 선택: 2024년 학습 → 2025년 예측. 후보(모델 × 피처 묶음)를 2025년 성적으로만 비교
     기준 = 적중 80%·90% 달성에 필요한 헛경보 수의 합(작을수록 좋음), 해석 가능한 모델이 10% 이내면 우선
  2) 문턱: 선택된 후보의 2025년 예측에서 2025년 급락의 80%를 잡는 값
  3) 최종 시험: 2024~2025년으로 다시 학습 → 2026-01-01 ~ 최신일을 한 번만 채점

15시 판단에 쓰는 정보
- 날씨: 기상청 단기예보 구역별 육상예보, D-1 11시 발표분의 D일 오전·오후 (7개 지점 평균). 11시분이 없으면 05시분
- SMP·수요예측: D-1일분까지 (D일분은 평일 19~20시, 주말 14시 이전 공표라 제외)
- 주간 전력수요전망: 게시판 등록 다음 날부터
실행: python -m drop_classifier.src.fair_select
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common.config import ROOT
from common.collect.kma_regional_fcst import REGIONS
from common.loaders import load_kma_regional_fcst
from drop_classifier.src.experiments import false_alarms_at
from drop_classifier.src.features import CALENDAR, CALENDAR_X, DEMAND_PREV, LAG, LAG_X
from drop_classifier.src.final import threshold_for
from drop_classifier.src.models import split
from drop_classifier.src.outlook_features import OUTLOOK
from drop_classifier.src.search import CANDIDATES, INTERPRETABLE, day_table

warnings.filterwarnings("ignore")
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
TRAIN_SEL, VALID_SEL = ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31")
TRAIN_FIN, TEST = ("2024-01-01", "2025-12-31"), ("2026-01-01", "2099-12-31")
FCST15 = ["rf_sky_pm", "rf_sky_am", "rf_clear_pm", "rf_pop_pm", "rf_rain_pm", "rf_ta_pm"]
SETS = {
    "기본": CALENDAR + DEMAND_PREV + LAG + FCST15,
    "기본+주간전망": CALENDAR + DEMAND_PREV + LAG + FCST15 + OUTLOOK,
    "기본+주간전망+보강": CALENDAR + CALENDAR_X + DEMAND_PREV + LAG + LAG_X + FCST15 + OUTLOOK,
}
CANDS = dict(CANDIDATES)


def fcst15() -> pd.DataFrame:
    """D-1 11시(없으면 05시) 발표분의 D일 오전·오후 구간 → 7개 지점 평균."""
    f = load_kma_regional_fcst()
    f = f[f["reg"].isin(REGIONS)]
    f["date"] = f["tm_ef"].dt.normalize()
    f["seg"] = np.where(f["tm_ef"].dt.hour == 12, "pm", "am")
    f["issue_day"] = (f["date"] - f["tm_fc"].dt.normalize()).dt.days
    f = f[(f["issue_day"] == 1) & f["tm_fc"].dt.hour.isin([5, 11])]
    f = f.sort_values("tm_fc").groupby(["reg", "date", "seg"]).tail(1)  # 11시분 우선
    pm, am = f[f["seg"] == "pm"], f[f["seg"] == "am"]
    out = pd.DataFrame({
        "rf_sky_pm": pm.groupby("date")["sky"].mean(),
        "rf_clear_pm": pm.groupby("date")["sky"].apply(lambda s: (s <= 2).mean()),
        "rf_pop_pm": pm.groupby("date")["pop"].mean(),
        "rf_rain_pm": pm.groupby("date")["rain"].mean(),
        "rf_ta_pm": pm.groupby("date")["ta"].mean(),
        "rf_sky_am": am.groupby("date")["sky"].mean(),
        "rf_n_reg": pm.groupby("date")["reg"].nunique(),
        "rf_issue11": pm.groupby("date")["tm_fc"].apply(lambda s: (s.dt.hour == 11).mean()),
    })
    return out.reset_index()


def metrics(y, a, s) -> dict:
    tp, fn = int(((a == 1) & (y == 1)).sum()), int(((a == 0) & (y == 1)).sum())
    fp, tn = int(((a == 1) & (y == 0)).sum()), int(((a == 0) & (y == 0)).sum())
    return {"일수": len(y), "급락일": tp + fn, "적중": tp, "놓침": fn, "헛경보": fp,
            "재현율": tp / max(tp + fn, 1), "정밀도": tp / max(tp + fp, 1), "헛경보율": fp / max(fp + tn, 1),
            "ROC-AUC": roc_auc_score(y, s) if y.nunique() > 1 else np.nan,
            "PR-AUC": average_precision_score(y, s) if y.nunique() > 1 else np.nan}


def main():
    df = day_table().merge(fcst15(), on="date", how="left")
    cov = df[df["date"] >= "2024-01-01"]
    print(f"15시 예보 확보: {cov['rf_sky_pm'].notna().mean():.1%}, 11시 발표분 비율 {cov['rf_issue11'].mean():.1%}", flush=True)

    # 1) 선택: 2024 학습 → 2025 예측
    tr, va = split(df, TRAIN_SEL), split(df, VALID_SEL)
    rows, scores = [], {}
    for sname, cols in SETS.items():
        for mname, fn in CANDS.items():
            s = fn(tr, va, cols)
            scores[(sname, mname)] = s
            y = va["label"].values
            r = {"피처": sname, "모델": mname, "헛경보@80%": false_alarms_at(y, s, 0.8)[0],
                 "헛경보@90%": false_alarms_at(y, s, 0.9)[0], "PR-AUC": average_precision_score(y, s),
                 "ROC-AUC": roc_auc_score(y, s)}
            r["선택점수"] = r["헛경보@80%"] + r["헛경보@90%"]
            rows.append(r)
            print(f"[선택 2025] {sname} / {mname}: 헛경보@80% {r['헛경보@80%']}, @90% {r['헛경보@90%']}, PR-AUC {r['PR-AUC']:.3f}", flush=True)
    sel = pd.DataFrame(rows).sort_values(["선택점수", "PR-AUC"], ascending=[True, False])
    best = sel.iloc[0]
    interp = sel[sel["모델"].isin(INTERPRETABLE)].iloc[0]
    chosen = interp if interp["선택점수"] <= best["선택점수"] * 1.10 else best
    sel.round(3).to_csv(OUT_TAB / "09_fair_selection_2025_육지.csv", index=False, encoding="utf-8-sig")

    # 2) 문턱 (2025)
    cols, fn = SETS[chosen["피처"]], CANDS[chosen["모델"]]
    thr = threshold_for(va["label"].values, scores[(chosen["피처"], chosen["모델"])], 0.8)

    # 3) 최종 시험 2026 (한 번)
    tr2, te = split(df, TRAIN_FIN), split(df, TEST).copy()
    te["score"] = fn(tr2, te, cols)
    te["alarm"] = (te["score"] >= thr).astype(int)
    is_reg = "회귀" in chosen["모델"]
    parts = {"2026 전체": te, "1~6월": te[te["date"] < "2026-07-01"], "7월~": te[te["date"] >= "2026-07-01"],
             "휴무일": te[te["day_type"] != "평일"], "평일": te[te["day_type"] == "평일"]}
    res = pd.DataFrame([{"구간": k, **metrics(g["label"], g["alarm"], g["score"])} for k, g in parts.items()])
    # 기준선 규칙 (같은 15시 정보)
    rules = {
        "규칙: 휴무일이면 급락": (te["day_type"] != "평일").astype(int),
        "규칙: 휴무일+맑음 예보(오후)": ((te["day_type"] != "평일") & (te["rf_clear_pm"] >= 0.5)).astype(int),
    }
    rule_rows = [{"구간": k, **metrics(te["label"], v, v)} for k, v in rules.items()]
    res.round(4).to_csv(OUT_TAB / "09_fair_test2026_육지.csv", index=False, encoding="utf-8-sig")
    te.assign(pred=-te["score"] if is_reg else te["score"])[
        ["date", "day_type", "label", "min_smp_11_15", "pred", "alarm"] + FCST15].to_csv(
        OUT_TAB / "09_fair_test2026_daily_육지.csv", index=False, encoding="utf-8-sig")
    info = {"피처": chosen["피처"], "모델": chosen["모델"], "문턱": round(-thr, 1) if is_reg else round(thr, 3),
            "선택점수(2025)": int(chosen["선택점수"])}
    (OUT_TAB / "09_fair_choice.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 250)
    print("\n[2025 선택 상위 8]\n", sel.head(8).round(3).to_string(index=False))
    print("\n선택:", info)
    print("\n[2026 최종 시험]\n", res.round(3).to_string(index=False))
    print("\n[기준선 규칙 2026]\n", pd.DataFrame(rule_rows).round(3).to_string(index=False))
    c = te[te["date"].isin(pd.to_datetime(["2026-09-25", "2026-10-05", "2026-06-13"]))]
    print("\n", c[["date", "label", "min_smp_11_15", "score", "alarm", "rf_sky_pm"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
