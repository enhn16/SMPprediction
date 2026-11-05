"""새 판정 기준(2026-11-04 판정기준 v1.1)으로 사건 구간을 센다.

- 육지 급등: 실적 SMP ≥ 같은 시각 직전 28일(그날 제외) 실적 중앙값 + 20원
- 육지 급락: 11~15시 실적 SMP < 30원
- 제주 급락: 제주 SMP ≤ 0원 (음수 포함 — 부서 집계와 같음, 2026-11-05 확인)
- 기준을 넘은 시간이 이어지면 1구간 (하루 안에서 셈). 휴일 = 주말·공휴일.
- 판정 기간 2020-01-29 ~ 2026-10-31.
실행: python -m extended_backtest.src.events
"""
import pandas as pd

from extended_backtest.src.data import load_smp_all, offday_flags
from common.config import ROOT

START, END = pd.Timestamp("2020-01-29"), pd.Timestamp("2026-10-31")
OUT = ROOT / "extended_backtest" / "outputs" / "tables"


def hourly_flags(kind: str) -> pd.DataFrame:
    region = "제주" if kind == "jeju_zero" else "육지"
    s = load_smp_all(region)
    wide = s.pivot(index="date", columns="hour", values="smp").sort_index()
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max()))
    if kind == "land_surge":
        # 같은 시각 직전 28일 중앙값 (그날 제외 → shift(1)), 28일 모두 있어야 판정
        base = wide.shift(1).rolling(28, min_periods=28).median()
        flag = wide >= base + 20
    elif kind == "land_drop":
        base = None
        flag = (wide < 30) & wide.columns.isin(range(11, 16))[None, :]
    else:
        base = None
        flag = wide <= 0
    flag = flag & wide.notna()
    long = flag.stack().rename("flag").reset_index()
    long.columns = ["date", "hour", "flag"]
    long = long.merge(wide.stack().rename("smp").reset_index().set_axis(["date", "hour", "smp"], axis=1), on=["date", "hour"])
    if base is not None:
        long = long.merge(base.stack().rename("base").reset_index().set_axis(["date", "hour", "base"], axis=1),
                          on=["date", "hour"], how="left")
    return long[(long["date"] >= START) & (long["date"] <= END)].reset_index(drop=True)


def intervals(h: pd.DataFrame) -> pd.DataFrame:
    """하루 안에서 연속된 flag 시간 묶음 → [date, start_hour, end_hour, n_hours, extreme_smp]."""
    h = h.sort_values(["date", "hour"]).copy()
    new = h["flag"] & ~(h.groupby("date")["flag"].shift(1, fill_value=False))
    h["iid"] = new.cumsum().where(h["flag"])
    g = h.dropna(subset=["iid"]).groupby("iid")
    out = g.agg(date=("date", "first"), start_hour=("hour", "min"), end_hour=("hour", "max"),
                n_hours=("hour", "size"), smp_max=("smp", "max"), smp_min=("smp", "min")).reset_index(drop=True)
    out["is_offday"] = offday_flags(out["date"]).values
    return out


def main():
    rows = []
    for kind, name in [("land_surge", "육지 급등"), ("land_drop", "육지 급락"), ("jeju_zero", "제주 0원 이하")]:
        h = hourly_flags(kind)
        iv = intervals(h)
        iv.to_csv(OUT / f"E1_intervals_{kind}.csv", index=False, encoding="utf-8-sig")
        iv["year"] = iv["date"].dt.year
        t = iv.groupby(["year", "is_offday"]).size().unstack(fill_value=0).reindex(columns=[True, False], fill_value=0)
        t.loc["합계"] = t.sum()
        for y, r in t.iterrows():
            rows.append({"사건": name, "연도": y, "휴일": r[True], "평일": r[False]})
        days = iv["date"].nunique()
        judged = h["date"].nunique()
        print(f"{name}: 구간 {len(iv)} (휴일 {iv.is_offday.sum()} / 평일 {(~iv.is_offday).sum()}), 사건일 {days} / 판정일 {judged}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "E1_event_counts.csv", index=False, encoding="utf-8-sig")
    print(res.pivot_table(index="연도", columns="사건", values=["휴일", "평일"], aggfunc="first").to_string())


if __name__ == "__main__":
    main()
