"""시기별 성적 (분기). 매달 다시 학습·문턱을 다시 정하므로, 시기마다 모델이 본 사건 수·문턱·성적이 어떻게 바뀌는지 보인다.

칸: 분기 · 그 분기 모델이 학습한 사건일(첫 달~끝 달) · 주의/경보 문턱 범위 · 사건 구간 수 · 성공/부분/실패(구간) · 헛경보 주의/경보(날짜)
급락 = LSTM + LightGBM 평균, 급등 = B 새 레이블(대표)·A 지금 모델(비교).
실행: python -m extended_backtest.src.period_table → outputs/tables/P1_by_quarter.csv
"""
import numpy as np
import pandas as pd

from common.config import ROOT
from extended_backtest.src.score_tables import TEST

TAB = ROOT / "extended_backtest" / "outputs" / "tables"


def train_events(label: pd.Series, dates: pd.Series) -> pd.Series:
    """달마다 '그 전달 말까지 학습한 사건일 수' (2020-01부터)."""
    s = pd.Series(label.values, index=pd.DatetimeIndex(dates)).sort_index()
    months = pd.period_range("2025-01", "2026-10", freq="M")
    return pd.Series({m: int(s[s.index < m.start_time].sum()) for m in months})


def quarter_rows(event, model, daily, kind, n_train, th_warn, th_alarm, fmt):
    d = daily[daily["date"].between(*TEST)].copy()
    d["q"] = d["date"].dt.to_period("Q")
    iv = pd.read_csv(TAB / f"E1_intervals_{kind}.csv", parse_dates=["date"])
    iv = iv[iv["date"].between(*TEST)].merge(d[["date", "grade", "q"]], on="date")
    rows = []
    for q, x in d.groupby("q"):
        ms = pd.period_range(q.start_time, min(q.end_time, pd.Timestamp(TEST[1])), freq="M")
        xi = iv[iv["q"] == q]
        n = len(xi)
        k = {g: int((xi["grade"] == g).sum()) for g in ("경보", "주의", "없음")}
        rows.append({
            "사건": event, "모델": model, "분기": f"{q.year} {q.quarter}분기",
            "학습한 사건일": f"{n_train[ms[0]]}~{n_train[ms[-1]]}",
            "주의 문턱": fmt(th_warn[ms]), "경보 문턱": fmt(th_alarm[ms]),
            "구간 수": n,
            "성공(경보)": f"{k['경보']} ({k['경보'] / n:.0%})" if n else "-",
            "부분(주의)": f"{k['주의']} ({k['주의'] / n:.0%})" if n else "-",
            "실패": f"{k['없음']} ({k['없음'] / n:.0%})" if n else "-",
            "주의 이상 잡음": f"{(k['경보'] + k['주의']) / n:.0%}" if n else "-",
            "헛경보 주의/경보": f"{int(((x['event'] == 0) & (x['grade'] == '주의')).sum())} / "
                           f"{int(((x['event'] == 0) & (x['grade'] == '경보')).sum())}",
            "날": len(x)})
    return rows


def main():
    rows = []
    # 급락
    full = pd.read_csv(TAB / "D1_drop_walk_daily.csv", parse_dates=["date"])
    feat = pd.read_pickle(ROOT / "data" / "processed" / "extended_backtest" / "drop_features_육지.pkl")
    n_tr = train_events(feat["label"], feat["date"])
    th = pd.read_csv(TAB / "D1_drop_walk_thresholds.csv").set_index(pd.PeriodIndex(
        pd.read_csv(TAB / "D1_drop_walk_thresholds.csv")["month"], freq="M"))
    g = pd.read_csv(TAB / "D1_drop_walk_graded.csv", parse_dates=["date"]).assign(event=lambda x: x["label"])
    won = lambda s: f"{s.min():.0f}~{s.max():.0f}원" if round(s.min()) != round(s.max()) else f"{s.min():.0f}원"
    rows += quarter_rows("육지 급락", "LSTM + LightGBM 평균", g, "land_drop", n_tr,
                         th["주의_예상최저이하"], th["경보_예상최저이하"], won)
    # 급등
    sf = pd.read_pickle(ROOT / "data" / "processed" / "extended_backtest" / "surge_features_육지.pkl")
    sth = pd.read_csv(TAB / "S1_surge_walk_thresholds.csv")
    sth.index = pd.PeriodIndex(sth["month"], freq="M")
    sg = pd.read_csv(TAB / "S1_surge_walk_graded.csv", parse_dates=["date"])
    pr = lambda s: f"{s.min():.2f}~{s.max():.2f}" if round(s.min(), 2) != round(s.max(), 2) else f"{s.min():.2f}"
    for v, label, name in (("B", "y_new", "B 새 레이블 (대표)"), ("A", "y", "A 지금 모델 (비교)")):
        lab = sf[label].dropna()
        n_tr = train_events(lab, sf.loc[lab.index, "date"])
        rows += quarter_rows("육지 급등", name, sg.assign(grade=sg[f"grade_{v}"], event=sg["y_new"]), "land_surge",
                             n_tr, sth[f"주의_{v}"], sth[f"경보_{v}"], pr)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "P1_by_quarter.csv", index=False, encoding="utf-8-sig")
    for (e, m), x in out.groupby(["사건", "모델"], sort=False):
        print(f"\n=== {e} — {m} ===")
        print(x.drop(columns=["사건", "모델"]).to_string(index=False))


if __name__ == "__main__":
    main()
