"""공급 측 피처 실험: 원전 가동 용량·태양광 설비용량·출력제어 (육지, 판단 시각 D-1 15:00).

피처와 확보 시점
  - 원전: D일 가동 가능 용량(GW)과 총용량 대비 비율. 한수원 계획예방정비 일정(사전 공개)으로 계산
  - 태양광 설비: 시장참여 설비용량 '태양'(GW). 공표 시점 미확인 → M월 값은 M+2월 1일부터
  - 출력제어: D-1 시행 여부, 최근 14일(D-14 ~ D-1) 시행일 수.
    D-1 출력제어는 전력거래소 '출력제어 전망'이 대개 D-2 저녁, 늦어도 D-1 오전에 게시돼 15시에 확보됨.
    D일 출력제어 전망은 대부분 D-1 15시 이후(저녁 18~21시) 게시 → 사용 안 함.
    2026-07-13 이후 게시판 실적이 아직 올라오지 않아 0으로 처리(최근 구간 과소 가능).

규칙 (2026 채점 전에 고정): fuel_level.py와 동일
  - 후보 = VARIANTS, 모델은 Ridge+EBM 회귀 평균
  - 선택: 2024 학습 → 2025 예측 선택점수(적중 80%·90%에 필요한 헛경보 합) 최소, 같으면 PR-AUC
  - 문턱: 주의 = 2025 급락 80%, 강한 경보 = 2025 정밀도 50%·적중 3일 이상 중 가장 느슨
  - 2026-01-01 ~ 최신일을 한 번 채점 (선택 후보와 기존 나란히)
실행: python -m drop_classifier.src.supply_level
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common.loaders import load_land_curtailment, load_market_capacity, load_nuclear_available
from drop_classifier.src.fair_select import fcst15, metrics
from drop_classifier.src.final import threshold_for
from drop_classifier.src.fuel_level import ABS, BASE, tier_metrics
from drop_classifier.src.models import split
from drop_classifier.src.search import day_table
from drop_classifier.src.two_tier import OUT_TAB, sel_score, strong_threshold

warnings.filterwarnings("ignore")
NUC = ["nuc_avail_gw", "nuc_avail_ratio"]
SOLAR = ["cap_solar_gw"]
CURT = ["curt_prev1", "curt_cnt14"]
VARIANTS = {
    "기존": BASE,
    "+원전": BASE + NUC,
    "+태양광설비": BASE + SOLAR,
    "+출력제어": BASE + CURT,
    "+원전+출력제어": BASE + NUC + CURT,
    "+원전+태양광설비+출력제어": BASE + NUC + SOLAR + CURT,
}


def supply_features(dates: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.to_datetime(dates.values)})
    n = load_nuclear_available()
    d = d.merge(n, on="date", how="left")
    d["nuc_avail_gw"] = d["nuc_avail_mw"] / 1000
    d["nuc_avail_ratio"] = d["nuc_avail_mw"] / d["nuc_total_mw"]
    cap = load_market_capacity().sort_values("usable_from")
    d = pd.merge_asof(d.sort_values("date"), cap[["usable_from", "cap_solar"]], left_on="date", right_on="usable_from")
    d["cap_solar_gw"] = d["cap_solar"] / 1000
    c = load_land_curtailment().set_index("date")["curt_solar"].clip(upper=1)
    flag = c.reindex(pd.date_range("2021-01-01", d["date"].max()), fill_value=0)
    d["curt_prev1"] = flag.shift(1).reindex(d["date"]).values
    d["curt_cnt14"] = flag.rolling(14).sum().shift(1).reindex(d["date"]).values
    return d.set_index("date").reindex(pd.to_datetime(dates.values))[NUC + SOLAR + CURT].reset_index(drop=True)


def main():
    df = day_table().merge(fcst15(), on="date", how="left")
    df = pd.concat([df.reset_index(drop=True), supply_features(df["date"])], axis=1)
    tr, va = split(df, ("2024-01-01", "2024-12-31")), split(df, ("2025-01-01", "2025-12-31"))
    y = va["label"].values

    rows, scores = [], {}
    for name, cols in VARIANTS.items():
        s = ABS(tr, va, cols)
        scores[name] = s
        warn, strong = threshold_for(y, s, 0.8), strong_threshold(y, s)
        tm = tier_metrics(va, s, warn, strong)
        rows.append({"변형": name, "선택점수": sel_score(y, s), "PR-AUC": average_precision_score(y, s),
                     "주의 적중": tm["주의 이상"]["적중"], "주의 헛경보": tm["주의 이상"]["헛경보"],
                     "강한 적중": tm["강한 경보"]["적중"], "강한 헛경보": tm["강한 경보"]["헛경보"]})
        print(f"[2025] {name}: 선택점수 {rows[-1]['선택점수']}, PR-AUC {rows[-1]['PR-AUC']:.3f}", flush=True)
    sel = pd.DataFrame(rows).sort_values(["선택점수", "PR-AUC"], ascending=[True, False])
    sel.round(3).to_csv(OUT_TAB / "12_supply_selection_2025_육지.csv", index=False, encoding="utf-8-sig")
    chosen = sel.iloc[0]["변형"]
    print("\n[2025 선택표]\n", sel.round(3).to_string(index=False), "\n선택:", chosen, flush=True)

    tr2, te = split(df, ("2024-01-01", "2025-12-31")), split(df, ("2026-01-01", "2099-12-31")).copy()
    out = []
    for name in dict.fromkeys([chosen, "기존"]):
        warn, strong = threshold_for(y, scores[name], 0.8), strong_threshold(y, scores[name])
        s = ABS(tr2, te, VARIANTS[name])
        for tier, m in tier_metrics(te, s, warn, strong).items():
            out.append({"변형": name + (" (선택)" if name == chosen else ""), "경보": tier, **m})
        if name == chosen:
            info = {"변형": name, "주의 문턱(원)": round(-warn, 1),
                    "강한 경보 문턱(원)": None if strong is None else round(-strong, 1)}
            (OUT_TAB / "12_supply_choice.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    res = pd.DataFrame(out)
    res.round(4).to_csv(OUT_TAB / "12_supply_test2026_육지.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n[2026 시험]\n", res[["변형", "경보", "적중", "놓침", "헛경보", "재현율", "정밀도", "헛경보율", "ROC-AUC"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
