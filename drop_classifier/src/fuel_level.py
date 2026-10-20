"""SMP 수준(연료비) 보정 실험 (육지, 판단 시각 D-1 15:00).

배경: 2026년 상반기 평소 SMP가 낮아(급락 없는 날 한낮 최저 중앙값 103원) 예측이 전반적으로 내려가
헛경보가 늘었다. 수준을 설명하는 두 가지 방법을 2025년 성적으로만 비교한다.
  1) SMP 수준 피처(연료비 없이): 최근 7일(D-7~D-1) 일 최고 SMP 평균·일 평균 SMP 평균.
     D-1 SMP는 D-2 저녁에 공표되므로 15시에 확보됨.
     + 상대 목표: 한낮 최저 SMP ÷ 최근 7일 최고 SMP 평균을 예측 (수준 변화와 무관한 '얼마나 꺼지나')
  2) 연료비 피처: EPSIS 월별 연료비단가(LNG·유연탄, 원/kWh). M월 값은 M+2월 1일부터 사용(공표 시점 보수 가정).

규칙 (2026 채점 전에 고정)
  - 후보 = 아래 VARIANTS, 모델은 모두 Ridge+EBM 회귀 평균
  - 선택: 2024 학습 → 2025 예측의 선택점수(적중 80%·90%에 필요한 헛경보 합)가 가장 작은 것, 같으면 PR-AUC
  - 문턱: two_tier.py와 같은 방식(주의 = 2025 급락 80%, 강한 경보 = 2025 정밀도 50%·적중 3일 이상 중 가장 느슨)
  - 2026-01-01 ~ 최신일을 한 번 채점
실행: python -m drop_classifier.src.fuel_level
"""
import json
import warnings

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

from common.config import SEED
from common.loaders import load_fuel_cost, load_smp_excel
from drop_classifier.src.fair_select import SETS, fcst15, metrics
from drop_classifier.src.final import threshold_for
from drop_classifier.src.models import split
from drop_classifier.src.search import _imp, day_table
from drop_classifier.src.two_tier import OUT_TAB, STRONG_PREC, sel_score, strong_threshold

warnings.filterwarnings("ignore")
BASE = SETS["기본+주간전망+보강"]
LEVEL = ["smp_max7", "smp_mean7"]
FUEL = ["fc_lng", "fc_coal", "fc_lng_coal_gap"]


def level_features(dates: pd.Series) -> pd.DataFrame:
    s = load_smp_excel("육지")
    daily = s.groupby("date")["smp"].agg(["max", "mean"])
    # D일 값 = D-7 ~ D-1 평균 (shift(1)로 D일 제외)
    roll = daily.rolling(7, min_periods=5).mean().shift(1)
    roll.columns = ["smp_max7", "smp_mean7"]
    return roll.reindex(pd.DatetimeIndex(dates)).reset_index(drop=True)


def fuel_features(dates: pd.Series) -> pd.DataFrame:
    f = load_fuel_cost().sort_values("usable_from")
    d = pd.DataFrame({"date": pd.to_datetime(dates.values)}).sort_values("date")
    m = pd.merge_asof(d, f[["usable_from", "fc_lng", "fc_coal"]], left_on="date", right_on="usable_from")
    m["fc_lng_coal_gap"] = m["fc_lng"] - m["fc_coal"]
    return m.set_index("date").reindex(pd.to_datetime(dates.values))[FUEL].reset_index(drop=True)


def ridge_ebm(target: str, clip):
    """Ridge(α=3)+EBM 회귀 평균. target='abs'면 한낮 최저 SMP, 'rel'이면 ÷ smp_max7. 점수 = −예측."""
    def f(tr, te, cols):
        y = tr["min_smp_11_15"] if target == "abs" else tr["min_smp_11_15"] / tr["smp_max7"]
        y = y.clip(*clip)
        ok = y.notna()
        preds = []
        for m in (_imp(Ridge(alpha=3)),
                  ExplainableBoostingRegressor(random_state=SEED, interactions=5, min_samples_leaf=5, max_bins=64, n_jobs=1)):
            preds.append(m.fit(tr.loc[ok, cols], y[ok]).predict(te[cols]))
        return -np.mean(preds, axis=0)
    return f


ABS, REL = ridge_ebm("abs", (0, 150)), ridge_ebm("rel", (0, 1.5))
VARIANTS = {
    "기존": (BASE, ABS),
    "+SMP 수준": (BASE + LEVEL, ABS),
    "+SMP 수준, 상대 목표": (BASE + LEVEL, REL),
    "+연료비": (BASE + FUEL, ABS),
    "+SMP 수준+연료비": (BASE + LEVEL + FUEL, ABS),
    "+SMP 수준+연료비, 상대 목표": (BASE + LEVEL + FUEL, REL),
}


def tier_metrics(te, s, warn, strong):
    """strong이 None(2025에 정밀도 기준을 넘는 문턱 없음)이면 강한 경보를 내지 않는다."""
    strong = np.inf if strong is None else strong
    y = te["label"]
    return {"주의 이상": metrics(y, (s >= warn).astype(int), s), "강한 경보": metrics(y, (s >= strong).astype(int), s)}


def main():
    df = day_table().merge(fcst15(), on="date", how="left")
    df = pd.concat([df.reset_index(drop=True), level_features(df["date"]), fuel_features(df["date"])], axis=1)
    tr, va = split(df, ("2024-01-01", "2024-12-31")), split(df, ("2025-01-01", "2025-12-31"))

    rows, scores = [], {}
    for name, (cols, fn) in VARIANTS.items():
        s = fn(tr, va, cols)
        scores[name] = s
        y = va["label"].values
        warn, strong = threshold_for(y, s, 0.8), strong_threshold(y, s)
        tm = tier_metrics(va, s, warn, strong)
        rows.append({"변형": name, "선택점수": sel_score(y, s), "PR-AUC": average_precision_score(y, s),
                     "주의 적중": tm["주의 이상"]["적중"], "주의 헛경보": tm["주의 이상"]["헛경보"],
                     "강한 적중": tm["강한 경보"]["적중"], "강한 헛경보": tm["강한 경보"]["헛경보"]})
        print(f"[2025] {name}: 선택점수 {rows[-1]['선택점수']}, PR-AUC {rows[-1]['PR-AUC']:.3f}", flush=True)
    sel = pd.DataFrame(rows).sort_values(["선택점수", "PR-AUC"], ascending=[True, False])
    sel.round(3).to_csv(OUT_TAB / "11_fuel_level_selection_2025_육지.csv", index=False, encoding="utf-8-sig")
    chosen = sel.iloc[0]["변형"]
    print("\n[2025 선택표]\n", sel.round(3).to_string(index=False), "\n선택:", chosen, flush=True)

    # 2026 한 번 채점: 선택된 변형 (기존과 나란히)
    tr2, te = split(df, ("2024-01-01", "2025-12-31")), split(df, ("2026-01-01", "2099-12-31")).copy()
    out = []
    for name in dict.fromkeys([chosen, "기존"]):
        cols, fn = VARIANTS[name]
        yv = va["label"].values
        warn, strong = threshold_for(yv, scores[name], 0.8), strong_threshold(yv, scores[name])
        s = fn(tr2, te, cols)
        for tier, m in tier_metrics(te, s, warn, strong).items():
            out.append({"변형": name + (" (선택)" if name == chosen else ""), "경보": tier, **m})
        for half, g in te.assign(s=s).groupby(te["date"] < "2026-07-01"):
            a = g["s"] >= warn
            tp, fp = int((a & (g.label == 1)).sum()), int((a & (g.label == 0)).sum())
            print(f"{name} {'1~6월' if half else '7월~'}: 주의 이상 적중 {tp}/{int(g.label.sum())}, 헛경보 {fp}")
        if name == chosen:
            info = {"변형": name, "주의 문턱(점수)": warn, "강한 경보 문턱(점수)": strong}
            (OUT_TAB / "11_fuel_level_choice.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    res = pd.DataFrame(out)
    res.round(4).to_csv(OUT_TAB / "11_fuel_level_test2026_육지.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n[2026 시험]\n", res[["변형", "경보", "적중", "놓침", "헛경보", "재현율", "정밀도", "헛경보율", "ROC-AUC"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
