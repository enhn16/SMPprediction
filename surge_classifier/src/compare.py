"""② 급등일 vs 일반일 비교 (2023~2025만, 2026 시험 구간은 보지 않음).

- 피처별 급등일·일반일 중앙값, 단일 피처 순위 성능(ROC-AUC)
- 최고가 시간대(오전·오후·저녁)별로 급등일 특징이 다른지
- 그래프: 핵심 피처 박스플롯

실행: python -m surge_classifier.src.compare
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.plot import setup_korean_font
from surge_classifier.src.features import (CALENDAR, DEMAND_PREV, OPENMETEO, OUTLOOK, SMP_HIST, SUPPLY, WEATHER)
from surge_classifier.src.label_days import C_INK, C_INK2, C_MUTED, C_SURGE, OUT_DATA, OUT_FIG, OUT_TAB, _style

NAMES = {
    "is_offday": "휴무일", "is_after_offday": "휴무일 다음 평일", "is_summer": "여름(7~9월)", "is_winter": "겨울(12~2월)",
    "is_fuel_change_window": "매월 1~3일", "log_base": "기준값(로그)", "prev_ratio": "전날 최고 ÷ 기준",
    "prev_mean_ratio": "전날 평균 ÷ 기준", "prev_spike": "전날 최고 ÷ 전날 평균", "max7_ratio": "직전 7일 최고 ÷ 기준",
    "same_dow_ratio": "지난주 같은 요일 ÷ 기준", "surges_last14": "직전 14일 급등 수", "ratio_std14": "직전 14일 비율 변동",
    "base_trend": "기준값 28일 변화", "dem_max_prev": "전날 수요예측 최대(MW)", "dem_max_prev_rel": "전날 수요예측 최대(상대)",
    "dem_am_prev_rel": "전날 오전 수요(상대)", "dem_eve_prev_rel": "전날 저녁 수요(상대)", "kpx_peak": "주간전망 최대부하(GW)",
    "kpx_peak_rel": "주간전망 최대부하(주 내 상대)", "kpx_peak_vs_prev": "주간전망 ÷ 전날 수요예측", "sky_pm": "오후 하늘(1맑음~4흐림)",
    "sky_am": "오전 하늘", "pop_pm": "오후 강수확률", "rain_pm": "오후 강수 유무", "ta_pm": "오후 기온", "ta_am": "오전 기온",
    "cdd": "폭염도(오후 28℃ 초과분)", "hdd": "한파도(오전 0℃ 미만분)", "wk_sky_pm": "평일 × 오후 하늘", "wk_cdd": "평일 × 폭염도",
    "rad_mid": "한낮 일사 예보", "cloud_mid": "한낮 운량 예보", "wk_cloud_mid": "평일 × 한낮 운량", "cap_solar_gw": "태양광 설비(GW)",
    "fc_lng": "LNG 연료비", "lng_coal_ratio": "LNG ÷ 유연탄 연료비",
}


def main():
    df = pd.read_csv(OUT_DATA / "features_육지.csv", parse_dates=["date"])
    d = df[df["date"].between("2023-01-01", "2025-12-31") & df["label"].notna()].copy()
    lab = pd.read_csv(OUT_DATA / "daily_label_육지.csv", parse_dates=["date"])[["date", "max_period"]]
    d = d.merge(lab, on="date", how="left")
    cols = [c for c in CALENDAR + SMP_HIST + DEMAND_PREV + OUTLOOK + WEATHER + OPENMETEO + SUPPLY if c in NAMES]

    rows = []
    for c in cols:
        x = d[[c, "label"]].dropna()
        if x["label"].nunique() < 2:
            continue
        auc = roc_auc_score(x["label"], x[c])
        rows.append({"피처": c, "이름": NAMES[c], "표본": len(x), "급등일 수": int(x["label"].sum()),
                     "급등일 중앙값": x.loc[x["label"] == 1, c].median(), "일반일 중앙값": x.loc[x["label"] == 0, c].median(),
                     "ROC-AUC": auc, "방향": "높을수록 급등" if auc >= 0.5 else "낮을수록 급등", "분리력": abs(auc - 0.5) * 2})
    t = pd.DataFrame(rows).sort_values("분리력", ascending=False)
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(OUT_TAB / "02_feature_compare_육지.csv", index=False, encoding="utf-8-sig")

    # 최고가 시간대별 급등일 특징
    key = ["is_offday", "prev_ratio", "dem_max_prev_rel", "sky_pm", "ta_pm", "ta_am", "cloud_mid", "is_summer", "is_winter"]
    s = d[d["label"] == 1]
    per = s.groupby("max_period", observed=True)[key].median()
    per.insert(0, "급등일", s.groupby("max_period", observed=True).size())
    per.loc["일반일"] = [len(d) - len(s)] + d.loc[d["label"] == 0, key].median().tolist()
    per.round(3).to_csv(OUT_TAB / "02_by_period_육지.csv", encoding="utf-8-sig")

    # 그래프: 분리력 상위 6개 피처 박스플롯
    setup_korean_font()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    top = [c for c in t["피처"] if d[c].nunique() > 2][:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.flat, top):
        x = d[[c, "label"]].dropna()
        bp = ax.boxplot([x.loc[x["label"] == 0, c], x.loc[x["label"] == 1, c]], tick_labels=["일반일", "급등일"],
                        widths=0.5, patch_artist=True, showfliers=False, medianprops={"color": C_INK})
        for patch, col in zip(bp["boxes"], [C_MUTED, C_SURGE]):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        auc = t.set_index("피처").loc[c, "ROC-AUC"]
        ax.set_title(f"{NAMES[c]} (ROC {auc:.2f})", fontsize=10, color=C_INK, loc="left")
        _style(ax)
    fig.suptitle("육지 급등일 vs 일반일 — 2023~2025 (판단 시각 15시에 아는 값)", x=0.01, ha="left", fontsize=12, color=C_INK)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "02_feature_box_육지.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 250)
    print("[피처별 비교 — 2023~2025, 분리력 순]")
    print(t[["이름", "표본", "급등일 수", "급등일 중앙값", "일반일 중앙값", "ROC-AUC"]].round(3).to_string(index=False))
    print("\n[최고가 시간대별 급등일 중앙값]")
    print(per.round(2).to_string())


if __name__ == "__main__":
    main()
