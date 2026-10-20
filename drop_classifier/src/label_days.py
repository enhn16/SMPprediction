"""① 급락일 레이블링.

- 일 단위 레이블: 한낮 창(기본 11~15시) SMP 최저값 < 30원 → 1
- 창 확정용으로 10~16시 최저값, 24시간 전체의 저가 시간 분포도 함께 집계
- 시간 표기는 원본 그대로 1~24 (12 = 11:00~12:00 구간)

실행: python -m drop_classifier.src.label_days [계통]   (기본 육지)
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from common.loaders import load_smp_excel
from common.plot import setup_korean_font

THRESHOLD = 30.0
WINDOW = (11, 15)       # 잠정 레이블 창
WIDE_WINDOW = (10, 16)  # 창 확정용
TRAIN = ("2024-01-01", "2026-06-30")
VALID_START = "2026-07-01"

OUT_DATA = PROCESSED_DIR / "drop_classifier"
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
OUT_FIG = ROOT / "drop_classifier" / "outputs" / "figures"

# 색: 급락=주황, 정상=파랑, 보조=회색 (dataviz 기본 팔레트)
C_DROP, C_NORMAL, C_MUTED, C_INK, C_INK2 = "#eb6834", "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e"


def window_min(smp: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """창 안 최저 SMP와 그 시각(동률이면 이른 시각)."""
    w = smp[smp["hour"].between(lo, hi)]
    idx = w.groupby("date")["smp"].idxmin()
    return w.loc[idx, ["date", "smp", "hour"]].set_index("date")


def build_daily(region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    smp = load_smp_excel(region)
    m = window_min(smp, *WINDOW).add_prefix("min_").rename(columns={"min_smp": "min_smp_11_15", "min_hour": "min_hour_11_15"})
    w = window_min(smp, *WIDE_WINDOW).rename(columns={"smp": "min_smp_10_16", "hour": "min_hour_10_16"})
    noon = smp[smp["hour"] == 12].set_index("date")["smp"].rename("smp_12h")
    low_cnt = (smp[smp["hour"].between(*WIDE_WINDOW)]
               .assign(low=lambda d: d["smp"] < THRESHOLD)
               .groupby("date")["low"].sum().rename("n_low_hours_10_16"))
    daily = pd.concat([m, w, noon, low_cnt], axis=1).reset_index()
    daily["label"] = (daily["min_smp_11_15"] < THRESHOLD).astype(int)
    daily.insert(1, "region", region)

    cal = build_calendar(daily["date"].min(), daily["date"].max())
    daily = daily.merge(cal, on="date", how="left")
    daily["split"] = "참고(~2023)"
    daily.loc[daily["date"].between(*TRAIN), "split"] = "학습"
    daily.loc[daily["date"] >= VALID_START, "split"] = "검증"
    return daily, smp


def count_table(daily: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    t = daily.groupby(by)["label"].agg(일수="size", 급락일="sum").reset_index()
    t["급락비율(%)"] = (t["급락일"] / t["일수"] * 100).round(1)
    return t


def save_tables(daily: pd.DataFrame, smp: pd.DataFrame, region: str):
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    daily = daily.assign(연도=daily["date"].dt.year, 월=daily["date"].dt.month)
    tables = {
        "by_year": count_table(daily, ["연도"]),
        "by_month": count_table(daily, ["월"]),
        "by_daytype": count_table(daily, ["day_type"]),
        "by_year_daytype": count_table(daily, ["연도", "day_type"]),
        "by_split": count_table(daily, ["split"]),
        "by_split_daytype": count_table(daily, ["split", "day_type"]),
    }
    for name, t in tables.items():
        t.to_csv(OUT_TAB / f"01_label_counts_{name}_{region}.csv", index=False, encoding="utf-8-sig")

    # 임계값 민감도: 같은 창에서 기준을 바꾸면 급락일 수가 어떻게 달라지는가
    sens = pd.DataFrame([
        {"기준(원)": th,
         "11~15시 급락일": int((daily["min_smp_11_15"] < th).sum()),
         "10~16시 급락일": int((daily["min_smp_10_16"] < th).sum())}
        for th in (1, 10, 30, 50, 70)
    ])
    sens.to_csv(OUT_TAB / f"01_threshold_sensitivity_{region}.csv", index=False, encoding="utf-8-sig")

    # 24시간 전체에서 SMP < 기준인 시간의 분포 (창 밖 저가 확인용)
    low = smp[smp["smp"] < THRESHOLD]
    hour_dist = low.groupby("hour").size().reindex(range(1, 25), fill_value=0).rename("저가 시간 수").reset_index()
    hour_dist.to_csv(OUT_TAB / f"01_low_hour_distribution_{region}.csv", index=False, encoding="utf-8-sig")

    drops = daily[daily["label"] == 1][["date", "weekday_ko", "day_type", "holiday_name", "off_block_len",
                                        "off_block_day", "is_myeongjeol", "min_smp_11_15", "min_hour_11_15",
                                        "smp_12h", "n_low_hours_10_16", "split"]]
    drops.to_csv(OUT_TAB / f"01_drop_days_{region}.csv", index=False, encoding="utf-8-sig")
    return tables, sens, hour_dist, low


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def save_figures(daily: pd.DataFrame, hour_dist: pd.DataFrame, region: str):
    setup_korean_font()
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    # 1) 저가(<30원) 시간 분포: 레이블 창 강조
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = [C_DROP if WINDOW[0] <= h <= WINDOW[1] else C_MUTED for h in hour_dist["hour"]]
    ax.bar(hour_dist["hour"], hour_dist["저가 시간 수"], color=colors, width=0.7, edgecolor="white", linewidth=1)
    for h, v in zip(hour_dist["hour"], hour_dist["저가 시간 수"]):
        if v:
            ax.text(h, v, f"{v}", ha="center", va="bottom", fontsize=8, color=C_INK2)
    ax.set_xticks(range(1, 25))
    ax.set_xlabel("시간 (12 = 11~12시 구간)", color=C_INK2)
    ax.set_ylabel("시간 수", color=C_INK2)
    ax.set_title(f"{region} SMP {THRESHOLD:.0f}원 미만 시간의 시간대 분포 (주황 = 레이블 창 {WINDOW[0]}~{WINDOW[1]}시)",
                 loc="left", fontsize=11, color=C_INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"01_low_hour_distribution_{region}.png", dpi=150)
    plt.close(fig)

    # 2) 연-월별 급락일 수
    ym = daily.groupby(daily["date"].dt.to_period("M"))["label"].sum()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = range(len(ym))
    ax.bar(x, ym.values, color=C_DROP, width=0.7, edgecolor="white", linewidth=1)
    ticks = [i for i, p in enumerate(ym.index) if p.month in (1, 7)]
    ax.set_xticks(ticks, [ym.index[i].strftime("%Y-%m") for i in ticks])
    valid_i = next(i for i, p in enumerate(ym.index) if str(p) >= VALID_START[:7])
    ax.set_ylim(0, ym.max() * 1.25)
    ax.axvline(valid_i - 0.5, color=C_INK2, linewidth=1, linestyle="--")
    ax.text(valid_i - 0.8, ym.max() * 1.15, "← 학습 | 검증 →", fontsize=9, color=C_INK2, ha="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_ylabel("급락일 수", color=C_INK2)
    ax.set_title(f"{region} 월별 급락일 수 (한낮 {WINDOW[0]}~{WINDOW[1]}시 최저 SMP < {THRESHOLD:.0f}원)",
                 loc="left", fontsize=11, color=C_INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"01_monthly_drop_count_{region}.png", dpi=150)
    plt.close(fig)

    # 3) 요일유형별 급락 비율 (학습+검증 기간, 2024~)
    d = daily[daily["split"] != "참고(~2023)"]
    order = ["평일", "주말", "공휴일", "연휴"]
    t = d.groupby("day_type")["label"].agg(["size", "sum"]).reindex(order)
    rate = t["sum"] / t["size"] * 100
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(order, rate.values, color=C_DROP, width=0.55, edgecolor="white", linewidth=1)
    for i, k in enumerate(order):
        ax.text(i, rate[k], f"{rate[k]:.1f}%\n({int(t.loc[k, 'sum'])}/{int(t.loc[k, 'size'])}일)",
                ha="center", va="bottom", fontsize=9, color=C_INK)
    ax.set_ylim(0, rate.max() * 1.35)
    ax.set_ylabel("급락 비율 (%)", color=C_INK2)
    ax.set_title(f"{region} 요일유형별 급락 비율 ({TRAIN[0][:7]} ~ {daily['date'].max():%Y-%m-%d})", loc="left", fontsize=11, color=C_INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"01_drop_rate_by_daytype_{region}.png", dpi=150)
    plt.close(fig)


def main(region: str = "육지"):
    daily, smp = build_daily(region)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    out = OUT_DATA / f"daily_label_{region}.csv"
    daily.to_csv(out, index=False, encoding="utf-8-sig")
    tables, sens, hour_dist, low = save_tables(daily, smp, region)
    save_figures(daily, hour_dist, region)

    print(f"저장: {out} ({len(daily)}일, {daily['date'].min():%Y-%m-%d} ~ {daily['date'].max():%Y-%m-%d})")
    for name in ("by_year", "by_split", "by_daytype", "by_split_daytype"):
        print(f"\n[{name}]\n{tables[name].to_string(index=False)}")
    print(f"\n[임계값 민감도]\n{sens.to_string(index=False)}")
    out_of_window = low[~low["hour"].between(*WIDE_WINDOW)]
    print(f"\n[{THRESHOLD:.0f}원 미만 시간] 전체 {len(low)}시간, 10~16시 밖 {len(out_of_window)}시간")
    print(hour_dist[hour_dist["저가 시간 수"] > 0].to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:2])
