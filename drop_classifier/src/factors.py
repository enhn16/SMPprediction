"""② 요인 분석 및 비교 그래프 (육지).

※ 이 단계의 기상·태양광 값은 실측이다 (기상 단기예보 과거분 확보 전).
   수요는 하루전 수요예측(D-1 확보 가능). 그래프·표에 '실측' 여부를 표기한다.

한낮 = 11~15시 (SMP 표기 기준, 12 = 11~12시 구간). 각 값은 한낮 5개 시간의 평균.
- demand_fcst : 하루전 수요예측(MW)
- solar, wind : 시장 참여 태양광·풍력 발전 실적(MWh/h ≈ MW), 2025년까지만 있음
- net_load    : demand_fcst − solar − wind
- icsr        : 일사량(MJ/m²/h), 육지 ASOS 11개 지점 평균 → 2026년 태양광 실적 대체 지표
- ss, cloud, ta : 일조(시간), 전운량(0~10), 기온(℃)

실행: python -m drop_classifier.src.factors
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from common.loaders import load_asos, load_renewables, load_smp_forecast_api
from common.plot import setup_korean_font

REGION = "육지"
MID = (11, 15)
START = "2023-01-01"
JEJU_STN = {184, 185, 188, 189}
CASES = {"2026-09-25": "9/25 추석", "2026-10-05": "10/5 대체공휴일"}

OUT_DATA = PROCESSED_DIR / "drop_classifier"
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
OUT_FIG = ROOT / "drop_classifier" / "outputs" / "figures"

C_DROP, C_NORMAL, C_MUTED, C_INK, C_INK2 = "#eb6834", "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e"

VAR_LABEL = {
    "demand_fcst": "하루전 수요예측 (MW)",
    "net_load": "순부하 (MW, 수요예측-태양광·풍력 실적)",
    "solar": "태양광 실적 (MW)",
    "icsr": "일사량 (MJ/m²/h, 실측)",
    "ss": "일조 (시간/h, 실측)",
    "cloud": "전운량 (0~10, 실측)",
    "ta": "기온 (℃, 실측)",
}
GROUPS = ["휴무일·급락", "휴무일·정상", "평일·급락", "평일·정상"]


def build_factors() -> pd.DataFrame:
    label = pd.read_csv(OUT_DATA / f"daily_label_{REGION}.csv", parse_dates=["date"])
    label = label[label["date"] >= START]

    def mid_mean(df, cols):
        return df[df["hour"].between(*MID)].groupby("date")[cols].mean()

    fc = load_smp_forecast_api()
    fc = mid_mean(fc[fc["region"] == REGION], ["demand_fcst"])
    re = load_renewables()
    re = mid_mean(re[re["region"] == REGION], ["solar", "wind"])
    asos = load_asos()
    asos = asos[~asos["stn_id"].isin(JEJU_STN)]
    # 지점별 한낮 평균 → 지점 간 평균
    wx = (asos[asos["hour"].between(*MID)].groupby(["date", "stn_id"])[["icsr", "ss", "cloud", "ta"]].mean()
          .groupby("date").mean())

    df = label.set_index("date").join([fc, re, wx], how="left").reset_index()
    df["net_load"] = df["demand_fcst"] - df["solar"] - df["wind"]
    df["offday"] = np.where(df["day_type"] == "평일", "평일", "휴무일")
    df["group"] = df["offday"] + "·" + np.where(df["label"] == 1, "급락", "정상")
    df["month"] = df["date"].dt.month
    return df


def save_tables(df: pd.DataFrame) -> dict:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    cols = list(VAR_LABEL)
    stats = df.groupby("group")[cols].agg(["median", "mean"]).round(2)
    stats.columns = [f"{VAR_LABEL[v].split(' (')[0]}_{s}" for v, s in stats.columns]
    stats.insert(0, "일수", df.groupby("group").size())
    stats = stats.reindex(GROUPS)
    stats.to_csv(OUT_TAB / f"02_group_stats_{REGION}.csv", encoding="utf-8-sig")

    # 일사량이 태양광 실적을 얼마나 대신하는가 (2023~2025)
    both = df.dropna(subset=["solar", "icsr"])
    corr = {y: round(both[both["date"].dt.year == y][["solar", "icsr"]].corr().iloc[0, 1], 3)
            for y in sorted(both["date"].dt.year.unique())}

    # 개별 확인 대상: 평일 급락일, 실패 사례
    pick = df[((df["offday"] == "평일") & (df["label"] == 1)) | df["date"].isin(pd.to_datetime(list(CASES)))]
    detail_cols = ["date", "weekday_ko", "day_type", "holiday_name", "label", "min_smp_11_15"] + cols
    detail = pick[detail_cols].copy()
    # 같은 요일유형(휴무일/평일) 안에서의 백분위
    for c in cols:
        detail[f"{c}_백분위"] = [
            round((df.loc[df["offday"] == o, c] < v).mean() * 100) if pd.notna(v) else np.nan
            for o, v in zip(pick["offday"], pick[c])
        ]
    detail.round(2).to_csv(OUT_TAB / f"02_case_detail_{REGION}.csv", index=False, encoding="utf-8-sig")
    return {"stats": stats, "corr": corr, "detail": detail}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def _scatter_panels(df, x, y, title, fname, note):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, grp in zip(axes, ["휴무일", "평일"]):
        d = df[(df["offday"] == grp)].dropna(subset=[x, y])
        n, dr = d[d["label"] == 0], d[d["label"] == 1]
        ax.scatter(n[x], n[y], s=14, color=C_NORMAL, alpha=0.35, linewidths=0, label=f"정상 ({len(n)}일)")
        ax.scatter(dr[x], dr[y], s=40, color=C_DROP, edgecolors="white", linewidths=1.2, label=f"급락 ({len(dr)}일)", zorder=3)
        for day, name in CASES.items():
            r = d[d["date"] == pd.Timestamp(day)]
            if len(r):
                ax.scatter(r[x], r[y], s=150, facecolors="none", edgecolors=C_INK, linewidths=1.5, zorder=4)
                ax.annotate(name, (r[x].iloc[0], r[y].iloc[0]), xytext=(10, 8), textcoords="offset points",
                            fontsize=9, color=C_INK, fontweight="bold")
        ax.set_title(f"{grp}", loc="left", fontsize=11, color=C_INK)
        ax.set_xlabel(VAR_LABEL[x], color=C_INK2)
        _style(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].set_ylabel(VAR_LABEL[y], color=C_INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=C_INK)
    fig.text(0.01, 0.01, note, fontsize=8.5, color=C_INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_FIG / fname, dpi=150)
    plt.close(fig)


def save_figures(df: pd.DataFrame):
    setup_korean_font()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    period = f"{df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m-%d}"

    # 1) 핵심 산점도: 일사량(실측) × 수요예측 — 전체 기간, 실패 사례 포함
    _scatter_panels(df, "icsr", "demand_fcst",
                    f"{REGION} 한낮 일사량과 하루전 수요예측에 따른 급락 여부 ({period})",
                    f"02_scatter_icsr_demand_{REGION}.png",
                    "한낮 = 11~15시 평균. 일사량은 실측(ASOS 육지 11개 지점 평균)이라 상한선 분석용. 수요예측은 전날 확보 가능.")

    # 2) 태양광 실적 × 수요예측 — 2025년까지
    _scatter_panels(df[df["date"] <= "2025-12-31"], "solar", "demand_fcst",
                    f"{REGION} 한낮 태양광 실적과 하루전 수요예측에 따른 급락 여부 (2023-01 ~ 2025-12)",
                    f"02_scatter_solar_demand_{REGION}.png",
                    "태양광은 전력시장 참여 설비 실적(자가용·한전PPA 제외). 2026년 실적은 미공개.")

    # 3) 그룹별 박스플롯 (소표본 그룹은 점을 겹쳐 표시)
    vars_ = list(VAR_LABEL)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    colors = {"휴무일·급락": C_DROP, "휴무일·정상": C_NORMAL, "평일·급락": C_DROP, "평일·정상": C_NORMAL}
    rng = np.random.default_rng(42)
    for ax, v in zip(axes.flat, vars_):
        data = [df.loc[df["group"] == g, v].dropna() for g in GROUPS]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=C_INK, linewidth=1.5),
                        whiskerprops=dict(color=C_MUTED), capprops=dict(color=C_MUTED))
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.35 if "정상" in g else 0.6)
            patch.set_edgecolor(colors[g])
        for i, (g, d) in enumerate(zip(GROUPS, data), start=1):
            if "급락" in g:
                ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=12, color=colors[g],
                           edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xticks(range(1, 5), [f"{g}\n(n={len(d)})" for g, d in zip(GROUPS, data)], fontsize=8)
        ax.set_title(VAR_LABEL[v], loc="left", fontsize=10, color=C_INK)
        _style(ax)
        ax.xaxis.grid(False)
    axes.flat[-1].axis("off")
    fig.suptitle(f"{REGION} 요일유형 × 급락 여부별 한낮(11~15시) 요인 분포 ({period}, 태양광·순부하는 2025년까지)",
                 x=0.01, ha="left", fontsize=12, color=C_INK)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"02_boxplot_groups_{REGION}.png", dpi=150)
    plt.close(fig)

    # 4) 월별(1~12월) 급락일 수: 휴무일/평일 누적
    m = df.pivot_table(index="month", columns="offday", values="label", aggfunc="sum").reindex(range(1, 13)).fillna(0)
    days = df.groupby(["month", "offday"]).size().unstack().reindex(range(1, 13))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(m.index, m["휴무일"], color=C_DROP, width=0.65, edgecolor="white", linewidth=2, label="휴무일 급락")
    ax.bar(m.index, m["평일"], bottom=m["휴무일"], color=C_NORMAL, width=0.65, edgecolor="white", linewidth=2, label="평일 급락")
    for mo in m.index:
        tot = m.loc[mo].sum()
        rate = m.loc[mo, "휴무일"] / days.loc[mo, "휴무일"] * 100
        ax.text(mo, tot + 0.2, f"{int(tot)}\n휴무일 {rate:.0f}%", ha="center", va="bottom", fontsize=8, color=C_INK2)
    ax.set_xticks(range(1, 13), [f"{i}월" for i in range(1, 13)])
    ax.set_ylim(0, m.sum(axis=1).max() * 1.4)
    ax.set_ylabel("급락일 수", color=C_INK2)
    ax.set_title(f"{REGION} 월별 급락일 수 ({period}, 숫자 아래 % = 그 달 휴무일 중 급락 비율)", loc="left", fontsize=11, color=C_INK)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncols=2)
    _style(ax)
    ax.xaxis.grid(False)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"02_monthly_drop_by_month_{REGION}.png", dpi=150)
    plt.close(fig)


def main():
    df = build_factors()
    df.to_csv(OUT_DATA / f"daily_factors_{REGION}.csv", index=False, encoding="utf-8-sig")
    res = save_tables(df)
    save_figures(df)
    pd.set_option("display.width", 250)
    print(f"저장: daily_factors_{REGION}.csv ({len(df)}일)")
    print("\n결측:", df[list(VAR_LABEL)].isna().sum().to_dict())
    print("\n일사량-태양광 실적 상관 (연도별):", res["corr"])
    s = res["stats"]
    print("\n[그룹별 중앙값]\n", s[["일수"] + [c for c in s.columns if c.endswith("median")]].to_string())
    d = res["detail"]
    print("\n[개별 사례]\n", d[["date", "weekday_ko", "day_type", "label", "min_smp_11_15", "demand_fcst", "demand_fcst_백분위",
                             "icsr", "icsr_백분위", "cloud", "ta", "solar"]].to_string(index=False))


if __name__ == "__main__":
    main()
