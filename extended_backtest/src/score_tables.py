"""새 판정 기준 채점표 (판정기준 v1.1 6절 형식). 시험 2025-01-01 ~ 2026-10-31, 예측 시점 전날 15시.

표 1 사건 판정 (구간 단위): 구간 수 · 성공(그날 경보) · 부분(주의만) · 실패(없음), 휴일/평일, 건수(비율)
표 2 헛경보 (날짜 단위): 주의·경보를 냈는데 그날 사건이 없던 날 수, 휴일/평일/계, 주의·경보 따로
표 3 혼동행렬 (날짜 단위): 판정(경보·주의·없음) × 실제(사건 있음·없음)
구간: 시험 전체 / 2025 / 2026(2차 시험). 휴일 = 주말·공휴일.

실행: python -m extended_backtest.src.score_tables → outputs/tables/T_*.csv, 화면 요약
"""
import pandas as pd

from common.config import ROOT
from extended_backtest.src.data import offday_flags

TAB = ROOT / "extended_backtest" / "outputs" / "tables"
TEST = ("2025-01-01", "2026-10-31")
PERIODS = {"시험 전체 (2025-01~2026-10)": TEST, "2025": ("2025-01-01", "2025-12-31"),
           "2026 (2차 시험)": ("2026-01-01", "2026-10-31")}
GRADES = ["경보", "주의", "없음"]


def sources():
    """[(사건 이름, 모델 이름, 날짜별 판정 [date, grade, event], 구간 파일 kind)]"""
    out = []
    p = TAB / "D1_drop_walk_graded.csv"
    if p.exists():
        d = pd.read_csv(p, parse_dates=["date"])
        out.append(("육지 급락", "LSTM + LightGBM 평균 (운영 구성, 매달 재학습)",
                    d.assign(event=d["label"])[["date", "grade", "event"]], "land_drop"))
    s = pd.read_csv(TAB / "S1_surge_walk_graded.csv", parse_dates=["date"])
    for v, name in (("A", "지금 모델 (옛 레이블 ×1.25로 학습)"), ("B", "새 레이블 (+20원)로 다시 학습")):
        out.append(("육지 급등", name, s.assign(grade=s[f"grade_{v}"], event=s["y_new"])[["date", "grade", "event"]],
                    "land_surge"))
    return out


def tables(event, model, daily, kind, rng):
    daily = daily[daily["date"].between(*rng)].copy()
    daily["휴일"] = offday_flags(daily["date"]).values
    iv = pd.read_csv(TAB / f"E1_intervals_{kind}.csv", parse_dates=["date"])
    iv = iv[iv["date"].between(*rng)].merge(daily[["date", "grade"]], on="date", how="left")
    assert iv["grade"].notna().all(), "판정 없는 구간"
    t1, t2 = [], []
    for off, lab in ((True, "휴일"), (False, "평일")):
        x = iv[iv["is_offday"] == off]
        n = len(x)
        r = {"사건": event, "모델": model, "구분": lab, "구간 수": n}
        for g, col in (("경보", "성공"), ("주의", "부분"), ("없음", "실패")):
            k = int((x["grade"] == g).sum())
            r[col] = f"{k} ({k / n:.0%})" if n else "0"
            r[col + "_n"] = k
        t1.append(r)
    for lab, mask in (("휴일", daily["휴일"]), ("평일", ~daily["휴일"]), ("계", daily["휴일"] | True)):
        x = daily[mask & (daily["event"] == 0)]
        w, a = int((x["grade"] == "주의").sum()), int((x["grade"] == "경보").sum())
        t2.append({"사건": event, "모델": model, "구분": lab, "주의": w, "경보": a, "표시": f"주의 {w} / 경보 {a}",
                   "사건 없는 날": len(x)})
    cm = (daily.assign(실제=daily["event"].map({1: "사건 있음", 0: "사건 없음"}))
          .pivot_table(index="grade", columns="실제", values="date", aggfunc="count", fill_value=0)
          .reindex(index=GRADES, columns=["사건 있음", "사건 없음"], fill_value=0))
    cm = cm.rename_axis("판정").reset_index().assign(사건=event, 모델=model)
    return pd.DataFrame(t1), pd.DataFrame(t2), cm


def main():
    all1, all2, all3 = [], [], []
    for event, model, daily, kind in sources():
        for pname, rng in PERIODS.items():
            t1, t2, cm = tables(event, model, daily, kind, rng)
            for t, acc in ((t1, all1), (t2, all2), (cm, all3)):
                acc.append(t.assign(기간=pname))
    t1, t2, t3 = (pd.concat(a, ignore_index=True) for a in (all1, all2, all3))
    t1.to_csv(TAB / "T1_event_judgement.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(TAB / "T2_false_alarms.csv", index=False, encoding="utf-8-sig")
    t3.to_csv(TAB / "T3_confusion.csv", index=False, encoding="utf-8-sig")
    for pname in PERIODS:
        print(f"\n=== {pname} ===")
        print(t1[t1["기간"] == pname][["사건", "모델", "구분", "구간 수", "성공", "부분", "실패"]].to_string(index=False))
        print(t2[t2["기간"] == pname][["사건", "모델", "구분", "표시", "사건 없는 날"]].to_string(index=False))
    print("\n=== 혼동행렬 (시험 전체, 날짜) ===")
    print(t3[t3["기간"] == list(PERIODS)[0]].to_string(index=False))


if __name__ == "__main__":
    main()
