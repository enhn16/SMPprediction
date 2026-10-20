"""발표용 그래프 (outputs/figures).

01_day_{날짜}.png      대표일 24시간 예측 + 80% 범위 + 실적 (자체 15시 모델)
02_coverage_by_hour.png 2026 시간대별 포함률·평균 폭 (자체 모델 / 담당자 모델)
03_coverage_by_month.png 2026 월별 포함률
04_timeline_2026.png   2026 일별 한낮(13시) 예측·범위·실적 흐름
05_horizon.png         주간 지평(D+1~D+6)별 포함률·폭·MAE (2026)
실행: python -m smp_forecast_app.src.figures
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from common.plot import setup_korean_font

FIG = ROOT / "smp_forecast_app" / "outputs" / "figures"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
DAYS = {"2026-09-25": "추석 연휴 (한낮 0원)", "2026-10-05": "흐린 연휴 (급락 없음)", "2026-09-30": "평일"}
C_BAND, C_LINE, C_BAD = "#2563c9", "#2563c9", "#d1452f"


def day_plot(b, date, title):
    g = b[b["date"] == date].sort_values("hour")
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axvspan(10.5, 15.5, color="0.92", zorder=0)
    ax.fill_between(g["hour"], g["lo"], g["hi"], color=C_BAND, alpha=0.18, label="80% 범위")
    ax.plot(g["hour"], g["pred"], color=C_LINE, lw=2, label="예측 (전날 15시)")
    has = g["y"].notna()
    out = has & ((g["y"] < g["lo"]) | (g["y"] > g["hi"]))
    ax.scatter(g.loc[has & ~out, "hour"], g.loc[has & ~out, "y"], color="black", s=22, zorder=3, label="실적 (범위 안)")
    ax.scatter(g.loc[out, "hour"], g.loc[out, "y"], color=C_BAD, s=26, zorder=3, label="실적 (범위 밖)")
    inn = int((has & ~out).sum())
    ax.set_title(f"{date:%Y-%m-%d} {title} — 실적 {inn}/{int(has.sum())}시간 범위 안")
    ax.set_xlabel("시간 (1~24시, 12시 = 11:00~12:00)")
    ax.set_ylabel("SMP (원/kWh)")
    ax.set_xticks(range(1, 25, 2))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / f"01_day_{date:%Y-%m-%d}.png", dpi=150)
    plt.close(fig)


def main():
    setup_korean_font()
    FIG.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(PROCESSED_DIR / "smp_forecast_app" / "B_predictions_hourly.csv", parse_dates=["date"])
    for d, t in DAYS.items():
        day_plot(b, pd.Timestamp(d), t)

    # 시간대별 포함률
    hb = pd.read_csv(TAB / "B3_test2026_by_hour.csv")
    ha = pd.read_csv(TAB / "A2_test2026_by_hour.csv")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, col, lab in ((a1, "포함률", "포함률"), (a2, "평균폭", "평균 폭 (원)")):
        if col == "포함률":
            ax.axhspan(0.75, 0.85, color="0.9", label="목표 75~85%")
        ax.plot(hb["시간"], hb[col], "o-", color=C_LINE, label="자체 모델 (전날 15시)")
        ax.plot(ha["시간"], ha[col], "s--", color="0.45", label="담당자 모델 (참고, 23:30)")
        ax.set_ylabel(lab)
    a1.set_ylim(0.5, 1.0)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.legend(fontsize=8, loc="lower left")
    a1.set_title("2026년 시간대별 80% 범위 포함률과 폭 (2025년까지 자료로 범위 결정)")
    a2.set_xlabel("시간")
    a2.set_xticks(range(1, 25))
    fig.tight_layout()
    fig.savefig(FIG / "02_coverage_by_hour.png", dpi=150)
    plt.close(fig)

    # 월별 포함률
    mb = pd.read_csv(TAB / "B3_test2026_by_month.csv")
    ma = pd.read_csv(TAB / "A2_test2026_by_month.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhspan(0.75, 0.85, color="0.9", label="목표 75~85%")
    x = np.arange(len(mb))
    ax.bar(x - 0.2, mb["포함률"], 0.4, color=C_BAND, label="자체 모델")
    am = ma.set_index("월").reindex(mb["월"])
    ax.bar(x + 0.2, am["포함률"].values, 0.4, color="0.6", label="담당자 모델 (참고)")
    ax.set_xticks(x, [m[5:] + "월" for m in mb["월"]])
    ax.set_ylim(0.4, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("2026년 월별 80% 범위 포함률")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "03_coverage_by_month.png", dpi=150)
    plt.close(fig)

    # 2026 13시 흐름
    g = b[(b["date"] >= "2026-01-01") & (b["hour"] == 13)].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(g["date"], g["lo"], g["hi"], color=C_BAND, alpha=0.18, label="80% 범위")
    ax.plot(g["date"], g["pred"], color=C_LINE, lw=1, label="예측")
    out = (g["y"] < g["lo"]) | (g["y"] > g["hi"])
    ax.scatter(g.loc[~out, "date"], g.loc[~out, "y"], s=6, color="black", label="실적 (범위 안)")
    ax.scatter(g.loc[out, "date"], g.loc[out, "y"], s=9, color=C_BAD, label="실적 (범위 밖)")
    ax.set_title("2026년 매일 13시(12~13시) 예측·80% 범위·실적 — 자체 모델(전날 15시)")
    ax.set_ylabel("SMP (원/kWh)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "04_timeline_2026.png", dpi=150)
    plt.close(fig)
    # 지평별
    wp = TAB / "W2_test2026.csv"
    if wp.exists():
        w = pd.read_csv(wp)
        w = w[w["구간"] == "전체"].sort_values("H")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        a1.axhspan(0.75, 0.85, color="0.9", label="목표 75~85%")
        a1.bar(w["H"], w["포함률"], color=C_BAND, width=0.6)
        for x, v in zip(w["H"], w["포함률"]):
            a1.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
        a1.set_ylim(0.5, 1.0)
        a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        a1.set_xticks(w["H"], [f"D+{h}" for h in w["H"]])
        a1.set_title("80% 범위 포함률")
        a1.legend(fontsize=8, loc="lower right")
        a2.plot(w["H"], w["평균폭"], "o-", color=C_LINE, label="범위 평균 폭")
        a2.plot(w["H"], w["MAE"], "s--", color="0.4", label="점예측 MAE")
        a2.set_xticks(w["H"], [f"D+{h}" for h in w["H"]])
        a2.set_ylabel("원/kWh")
        a2.set_title("멀수록 범위가 넓어짐")
        a2.legend(fontsize=8)
        fig.suptitle("2026년 주간 지평별 성적 — 자체 모델(기준일 15시 판단)", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / "05_horizon.png", dpi=150)
        plt.close(fig)
    print("그래프 저장:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
