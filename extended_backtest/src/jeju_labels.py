"""제주 급락 새 레이블: 날짜별 SMP ≤ 0원 시간 유무·구간 수 (판정 2020-01-29 ~ 2026-10-31).

판정: 하루 경보는 그날 모든 구간에 적용 → 모델은 '날짜' 단위로 학습·예측하고, 채점은 구간 수로 센다.
저장: outputs/tables/J1_labels_jeju.csv [date, is_offday, event, n_intervals, n_hours, smp_min, first_hour, last_hour]
실행(루트에서): python -m extended_backtest.src.jeju_labels
"""
import pandas as pd

from extended_backtest.src.data import load_smp_all, offday_flags
from extended_backtest.src.events import START, END
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs" / "tables" / "J1_labels_jeju.csv"


def build() -> pd.DataFrame:
    s = load_smp_all("제주")
    s = s[(s["date"] >= START) & (s["date"] <= END)].copy()
    s["z"] = s["smp"] <= 0
    s = s.sort_values(["date", "hour"])
    # 같은 날 안에서 연속된 0원 이하 시간 = 1구간
    s["start"] = s["z"] & ~s.groupby("date")["z"].shift(1, fill_value=False)
    g = s.groupby("date")
    zh = s[s["z"]].groupby("date")["hour"]
    out = pd.DataFrame({"n_intervals": g["start"].sum(), "n_hours": g["z"].sum(), "smp_min": g["smp"].min(),
                        "first_hour": zh.min(), "last_hour": zh.max()}).reset_index()
    out["event"] = (out["n_hours"] > 0).astype(int)
    out["is_offday"] = offday_flags(out["date"]).values
    out = out[["date", "is_offday", "event", "n_intervals", "n_hours", "smp_min", "first_hour", "last_hour"]]
    for c in ["n_intervals", "n_hours"]:
        out[c] = out[c].astype(int)
    return out


def main():
    lab = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(OUT, index=False, encoding="utf-8-sig")
    # E1 사건 구간과 대조
    e1 = pd.read_csv(OUT.parent / "E1_intervals_jeju_zero.csv", parse_dates=["date"])
    print(f"날 {len(lab)}, 사건일 {lab['event'].sum()}, 구간 {lab['n_intervals'].sum()} (E1: 날 {e1['date'].nunique()}, 구간 {len(e1)})")
    m = e1.groupby("date").size().rename("e1").to_frame().join(lab.set_index("date")["n_intervals"], how="outer").fillna(0)
    print("E1과 다른 날:", int((m["e1"] != m["n_intervals"]).sum()))


if __name__ == "__main__":
    main()
