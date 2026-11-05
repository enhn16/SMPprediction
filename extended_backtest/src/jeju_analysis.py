"""제주 급락 시험 결과 해석 (채점 뒤, 모델은 바꾸지 않음).

1) 놓친 날·헛경보 날 분류 — '가격이 떨어진(안 떨어진) 원인'과 '모델이 놓친 원인'을 나눔
   가격 원인: SMP는 전력거래소가 전날 오후 확정 → 그날 실제 계통 모양(순부하·새벽 풍력 등)을 거래소 전망의 대리값으로 봄
   모델 원인: 우리 전날 15시 전망(fc_*)과 실제(act_*)의 차이, 새벽(풍력) 사건, 연속의 첫날 등
2) 후보별 시험 성적 묶음 비교(피처 묶음·모델 종류) + 2024 고르기 순위와의 차이
3) 피처 기여: 선택 모델 기준 피처별 단변량 ROC-AUC(2024 고르기 기간 / 시험), LR1 계수 0 피처, LGB 중요도
저장: outputs/tables/J4_misses_jeju.csv, J4_false_alarms_jeju.csv, J4_miss_types_jeju.csv, J4_group_test_jeju.csv,
      J4_feature_auc_jeju.csv
실행(루트에서): python -m extended_backtest.src.jeju_analysis
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from extended_backtest.src.jeju_models import FEAT, MODELS, TRAIN0, load_table, sets, stage1

TAB = Path(__file__).resolve().parents[1] / "outputs" / "tables"
GAP = 80   # 전망 − 실제 순부하 차이(MW)가 이보다 크면 '전망 빗나감'


def classify_miss(r):
    if r["first_hour"] <= 7:
        return "새벽(풍력) 사건"
    if r["fc_net_min"] - r["act_net_min"] > GAP:
        return "전망보다 실제 순부하가 낮음(전망 빗나감)"
    if r["zero_28"] == 0:
        return "최근 28일 0원 없음(첫 사건)"
    return "전망·실제 모두 평범(다른 요인)"


def classify_fa(r):
    if r["act_net_min"] - r["fc_net_min"] > GAP:
        return "실제 순부하가 전망보다 높음(전망 빗나감)"
    if r["act_net_min"] <= 300:
        return "실제로도 순부하 낮음(0원까지는 안 감)"
    return "실제 순부하 보통(모델 과민)"


def main():
    d = pd.read_csv(FEAT, parse_dates=["date"])
    lab = pd.read_csv(TAB / "J1_labels_jeju.csv", parse_dates=["date"])[["date", "n_hours", "first_hour", "last_hour", "smp_min"]]
    g = pd.read_csv(TAB / "J3_daily_jeju.csv", parse_dates=["date"])
    x = g.merge(d[["date", "fc_net_min", "act_net_min", "fc_net_min_mid", "act_net_min_mid", "fc_sol_max", "act_sol_max",
                   "fc_wind_night", "act_wind_night", "zero_28", "month"]], on="date").merge(lab, on="date", how="left")
    x["전망-실제 순부하"] = x["fc_net_min"] - x["act_net_min"]
    miss = x[(x["event"] == 1) & (x["grade"] != "경보")].copy()
    miss["유형"] = miss.apply(classify_miss, axis=1)
    fa = x[(x["event"] == 0) & (x["grade"] != "없음")].copy()
    fa["유형"] = fa.apply(classify_fa, axis=1)
    miss.round(3).to_csv(TAB / "J4_misses_jeju.csv", index=False, encoding="utf-8-sig")
    fa.round(3).to_csv(TAB / "J4_false_alarms_jeju.csv", index=False, encoding="utf-8-sig")
    hit = x[(x["event"] == 1) & (x["grade"] == "경보")]
    t = pd.concat([miss.groupby(["grade", "유형"]).agg(일=("date", "size"), 구간=("n_intervals", "sum")).assign(구분="놓침(주의만·없음)"),
                   fa.groupby(["grade", "유형"]).agg(일=("date", "size"), 구간=("n_intervals", "sum")).assign(구분="헛경보")]).reset_index()
    t.to_csv(TAB / "J4_miss_types_jeju.csv", index=False, encoding="utf-8-sig")
    print(t.to_string())
    print("적중일 전망-실제 순부하 중앙", round(hit["전망-실제 순부하"].median(), 1),
          "/ 놓친 날", round(miss["전망-실제 순부하"].median(), 1), "/ 헛경보", round(fa["전망-실제 순부하"].median(), 1))
    print("놓친 날 휴일 비율", round(miss["offday"].mean(), 2), "적중일 휴일 비율", round(hit["offday"].mean(), 2))
    print("놓친 날 월", miss["month"].value_counts().sort_index().to_dict(), "헛경보 월", fa["month"].value_counts().sort_index().to_dict())

    # 2) 묶음별
    a = pd.read_csv(TAB / "J3_all_candidates_test_jeju.csv")
    a = a[a["문턱"] == "R2"].copy()
    a["피처"] = a["후보"].str.split("|").str[0].where(a["후보"].str.contains(r"\|"), "앙상블·규칙")
    a["모델"] = a["후보"].str.split("|").str[1].where(a["후보"].str.contains(r"\|"), a["후보"])
    a["성공+부분/2 비율"] = (a["성공"] + a["부분"] / 2) / a["구간"]
    a["2024 순위"] = a["2024 PR-AUC"].rank(ascending=False)
    a["시험 순위"] = a["PR-AUC(일)"].rank(ascending=False)
    grp = pd.concat([a.groupby("피처")[["2024 PR-AUC", "PR-AUC(일)", "성공+부분/2 비율", "헛경보 경보"]].median().assign(묶음="피처"),
                     a[a["피처"] != "앙상블·규칙"].groupby("모델")[["2024 PR-AUC", "PR-AUC(일)", "성공+부분/2 비율", "헛경보 경보"]].median().assign(묶음="모델")])
    grp.round(3).to_csv(TAB / "J4_group_test_jeju.csv", encoding="utf-8-sig")
    print(grp.round(3).to_string())
    print("2024 순위 vs 시험 순위 스피어만", round(a[["2024 순위", "시험 순위"]].corr(method="spearman").iloc[0, 1], 2))

    # 3) 피처별 단변량 ROC-AUC (부호 무관 max(auc, 1-auc)), 2024 학습 구간 vs 시험
    tbl = load_table()
    tbl["s1"] = stage1(tbl, pd.Timestamp("2025-01-01"))
    S = sets()
    cols = list(dict.fromkeys(S["PRS"] + S["S5P"]))
    tr = tbl[(tbl["date"] >= TRAIN0) & (tbl["date"] <= "2024-12-31")]
    te = tbl[(tbl["date"] >= "2025-01-01") & (tbl["date"] <= "2026-10-31")]
    rows = []
    for c in cols:
        r = {"피처": c, "묶음": "PRS" if c in S["PRS"] else "S5만"}
        for nm, part in [("2024", tr), ("시험", te)]:
            v = part[[c, "event"]].dropna()
            au = roc_auc_score(v["event"], v[c]) if v[c].nunique() > 1 else np.nan
            r[f"AUC {nm}"] = au
            r[f"방향 {nm}"] = "+" if au >= 0.5 else "−"
            r[f"분리력 {nm}"] = max(au, 1 - au) if au == au else np.nan
        rows.append(r)
    fa_ = pd.DataFrame(rows)
    # 2024 학습 LR1(PRS) 계수와 LGB(PRS) 중요도 (2025-01 학습 모델과 같은 자료)
    S_prs = S["PRS"]
    lr1 = MODELS["LR1"]().fit(tr[S_prs], tr["event"])
    lgb = MODELS["LGB"]().fit(tr[S_prs], tr["event"])
    fa_["LR1 계수(PRS)"] = fa_["피처"].map(dict(zip(S_prs, lr1[-1].coef_[0])))
    fa_["LGB 중요도(PRS)"] = fa_["피처"].map(dict(zip(S_prs, lgb[-1].feature_importances_)))
    fa_ = fa_.sort_values("분리력 2024", ascending=False)
    fa_.round(3).to_csv(TAB / "J4_feature_auc_jeju.csv", index=False, encoding="utf-8-sig")
    print(fa_.head(40).round(3).to_string())


if __name__ == "__main__":
    main()
