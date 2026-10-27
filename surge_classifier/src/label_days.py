"""① 급등일 레이블링.

- 일 단위 레이블: 그날 최고 SMP ≥ 직전 28일 '일 최고 SMP' 중앙값 × 1.25 → 1 (현업 기준안)
- 민감도 확인용으로 +20%·+30% 레이블도 함께 저장
- 시간 표기는 원본 그대로 1~24 (12 = 11:00~12:00 구간)
- SMP는 팀 데이터 목록의 전력거래소 원본(load_smp_actual)만 쓴다. 2022년은 기준값 계산용(참고), 레이블은 2023년부터 사용.
  (현업 표 재현은 2026-10-26에 담당자 파일로 한 번 확인 — PLAN.md 결정 기록)

실행: python -m surge_classifier.src.label_days [계통]   (기본 육지)
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from common.loaders import load_smp_actual
from common.plot import setup_korean_font

RATIO = 1.25                 # 기준안: 직전 28일 중앙값보다 25% 이상
SENS_RATIOS = (1.20, 1.25, 1.30)
BASE_DAYS = 28
# 팀 공통 규칙: 2023~2024 학습 → 2025 검증으로 선택, 2023~2025 재학습 → 2026-01-01 ~ 10-18 한 번 채점
TRAIN = ("2023-01-01", "2024-12-31")
VALID = ("2025-01-01", "2025-12-31")
TEST = ("2026-01-01", "2026-10-18")

OUT_DATA = PROCESSED_DIR / "surge_classifier"
OUT_TAB = ROOT / "surge_classifier" / "outputs" / "tables"
OUT_FIG = ROOT / "surge_classifier" / "outputs" / "figures"

# 색: 급등=빨강, 보조=회색 (과제1 그래프와 같은 톤)
C_SURGE, C_MUTED, C_INK, C_INK2 = "#d6453d", "#b9b8b2", "#0b0b0b", "#52514e"


def daily_max(region: str) -> pd.DataFrame:
    """[date] → max_smp, max_hour, mean_smp, min_smp (24시간이 다 있는 날만)."""
    smp = load_smp_actual(region)
    w = smp.pivot(index="date", columns="hour", values="smp")
    w = w[w.notna().sum(axis=1) == 24]
    return pd.DataFrame({
        "max_smp": w.max(axis=1),
        "max_hour": w.idxmax(axis=1),          # 동률이면 이른 시각
        "mean_smp": w.mean(axis=1).round(2),
        "min_smp": w.min(axis=1),
    })


def base_28d(max_smp: pd.Series) -> pd.Series:
    """그날을 빼고 직전 28일의 일 최고 SMP 중앙값 (날짜 연속 가정)."""
    return max_smp.shift(1).rolling(BASE_DAYS, min_periods=BASE_DAYS).median()


def build_daily(region: str) -> pd.DataFrame:
    daily = daily_max(region)
    # 기준값: 그날을 빼고 직전 28일의 일 최고 SMP 중앙값 (D-1 15시에 이미 아는 값)
    daily["base_28d"] = base_28d(daily["max_smp"])
    daily["ratio"] = (daily["max_smp"] / daily["base_28d"]).round(4)
    for r in SENS_RATIOS:
        daily[f"label_{round(r * 100) - 100}"] = (daily["ratio"] >= r).astype(int)
    daily["label"] = (daily["ratio"] >= RATIO).astype(int)
    daily = daily[daily["base_28d"].notna()].reset_index()
    daily.insert(1, "region", region)

    # 최고가 시간대 구분 (발생 시각은 따로 기록)
    daily["max_period"] = pd.cut(daily["max_hour"], [0, 6, 12, 17, 24], labels=["새벽(1~6시)", "오전(7~12시)", "오후(13~17시)", "저녁(18~24시)"])
    # 매월 1~3일: 연료비 단가 변경으로 가격 수준 자체가 오를 수 있음 → 따로 표시
    daily["is_fuel_change_window"] = daily["date"].dt.day <= 3

    cal = build_calendar(daily["date"].min(), daily["date"].max())
    daily = daily.merge(cal, on="date", how="left")
    daily["split"] = "참고(2022)"
    daily.loc[daily["date"].between(*TRAIN), "split"] = "학습"
    daily.loc[daily["date"].between(*VALID), "split"] = "검증"
    daily.loc[daily["date"].between(*TEST), "split"] = "시험"
    daily.loc[daily["date"] > TEST[1], "split"] = "시험 이후"
    return daily


def count_table(daily: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    t = daily.groupby(by, observed=True)["label"].agg(일수="size", 급등일="sum").reset_index()
    t["급등비율(%)"] = (t["급등일"] / t["일수"] * 100).round(1)
    return t


def save_tables(daily: pd.DataFrame, region: str) -> dict:
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    daily = daily.assign(연도=daily["date"].dt.year, 월=daily["date"].dt.month)
    tables = {
        "by_year": count_table(daily, ["연도"]),
        "by_month": count_table(daily, ["월"]),
        "by_daytype": count_table(daily, ["day_type"]),
        "by_split": count_table(daily, ["split"]),
        "by_period": count_table(daily, ["max_period"]),
    }
    for name, t in tables.items():
        t.to_csv(OUT_TAB / f"01_label_counts_{name}_{region}.csv", index=False, encoding="utf-8-sig")

    # 기준 민감도: 구간별 급등일 수
    sens = pd.DataFrame([
        {"기준": f"+{round(r * 100) - 100}%",
         **{f"{sp}": int((daily.loc[daily["split"] == sp, "ratio"] >= r).sum())
            for sp in ("참고(2022)", "학습", "검증", "시험", "시험 이후")}}
        for r in SENS_RATIOS
    ])
    sens.to_csv(OUT_TAB / f"01_threshold_sensitivity_{region}.csv", index=False, encoding="utf-8-sig")

    surges = daily[daily["label"] == 1][["date", "weekday_ko", "day_type", "holiday_name", "max_smp", "max_hour",
                                         "max_period", "base_28d", "ratio", "mean_smp", "is_fuel_change_window", "split"]]
    surges.to_csv(OUT_TAB / f"01_surge_days_{region}.csv", index=False, encoding="utf-8-sig")
    tables["sens"] = sens
    tables["surges"] = surges
    return tables


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def save_figures(daily: pd.DataFrame, region: str):
    setup_korean_font()
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    # 1) 일 최고 SMP와 기준선(직전 28일 중앙값 × 1.25), 급등일 표시
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["max_smp"], color=C_MUTED, linewidth=0.8, label="일 최고 SMP")
    ax.plot(daily["date"], daily["base_28d"] * RATIO, color=C_INK2, linewidth=1, linestyle="--",
            label=f"기준선 (직전 {BASE_DAYS}일 중앙값 × {RATIO})")
    s = daily[daily["label"] == 1]
    ax.scatter(s["date"], s["max_smp"], color=C_SURGE, s=14, zorder=3, label=f"급등일 ({len(s)}일)")
    ax.set_ylabel("원/kWh", color=C_INK2)
    ax.set_title(f"{region} 일 최고 SMP와 급등일", loc="left", fontsize=11, color=C_INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"01_daily_max_surge_{region}.png", dpi=150)
    plt.close(fig)

    # 2) 급등일 최고가 발생 시각 분포
    hd = s["max_hour"].value_counts().reindex(range(1, 25), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(hd.index, hd.values, color=C_SURGE, width=0.7, edgecolor="white", linewidth=1)
    for h, v in hd.items():
        if v:
            ax.text(h, v, f"{v}", ha="center", va="bottom", fontsize=8, color=C_INK2)
    ax.set_xticks(range(1, 25))
    ax.set_xlabel("시간 (12 = 11~12시 구간)", color=C_INK2)
    ax.set_ylabel("급등일 수", color=C_INK2)
    ax.set_title(f"{region} 급등일의 최고가 발생 시각", loc="left", fontsize=11, color=C_INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"01_surge_hour_distribution_{region}.png", dpi=150)
    plt.close(fig)


def main(region: str = "육지"):
    daily = build_daily(region)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    out = OUT_DATA / f"daily_label_{region}.csv"
    daily.to_csv(out, index=False, encoding="utf-8-sig")
    tables = save_tables(daily, region)
    save_figures(daily, region)

    print(f"저장: {out} ({len(daily)}일, {daily['date'].min():%Y-%m-%d} ~ {daily['date'].max():%Y-%m-%d})")
    print(f"\n[기준 민감도]\n{tables['sens'].to_string(index=False)}")
    for name in ("by_year", "by_split", "by_daytype", "by_period", "by_month"):
        print(f"\n[{name}]\n{tables[name].to_string(index=False)}")
    print(f"\n[2024년 이후 급등일]\n{tables['surges'][tables['surges']['date'] >= '2024-01-01'].to_string(index=False)}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
