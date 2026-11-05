"""제주 2020 ~ 2024 '낮아지는 흐름과 원인' 분석 (학습 구간 자료만, 서술용).

1) 반기별: 낮(11~15시) 최저 SMP, 연료비(LNG) 대비 비율, 태양광·풍력·수요·순부하, 순부하 300MW 이하 시간, 출력제어 일수, 0원 일수
2) 잉여일(출력제어 또는 0원)과 보통 날의 계통 모양 비교 (2021-01 ~ 2024-05, 실제값)
3) 1단계 로지스틱(잉여일) 표준화 계수 — 원인 방향·크기
저장: outputs/tables/J5_trend_halfyear_jeju.csv, J5_surplus_vs_normal_jeju.csv, J5_stage1_coef_jeju.csv, outputs/figures/J5_trend_jeju.png
실행(루트에서): python -m extended_backtest.src.jeju_trend
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from extended_backtest.src.data import load_smp_all
from extended_backtest.src.jeju_features import MID, hourly_actual
from extended_backtest.src.jeju_models import FEAT, S1_COLS, pipe

TAB = Path(__file__).resolve().parents[1] / "outputs" / "tables"
FIG = Path(__file__).resolve().parents[1] / "outputs" / "figures"
END = "2024-12-31"
NAMES = {"act_net_min": "하루 최저 순부하", "act_net_min_mid": "한낮 최저 순부하", "act_net_min_night": "새벽 최저 순부하",
         "act_sol_max": "태양광 최대", "act_wind_mean": "풍력 평균", "act_wind_night": "새벽 풍력",
         "act_dem_min_mid": "한낮 최저 수요", "act_dem_mean": "하루 평균 수요", "act_re_max": "재생에너지 비중 최대",
         "offday": "휴일", "m_sin": "달(sin)", "m_cos": "달(cos)"}


def main():
    d = pd.read_csv(FEAT, parse_dates=["date"])
    d = d[d["date"] <= END].copy()
    d["half"] = d["date"].dt.year.astype(str) + np.where(d["date"].dt.month <= 6, " 상", " 하")
    s = load_smp_all("제주")
    s = s[(s["date"] >= "2020-01-29") & (s["date"] <= END)]
    mid = s[s["hour"].isin(MID)].groupby("date")["smp"].min().rename("mid_min")
    d = d.merge(mid.reset_index(), on="date", how="left")
    a = hourly_actual()
    a = a[(a["date"] >= "2020-01-29") & (a["date"] <= END)].copy()
    a["net"] = a["dem"] - a["sol"] - a["wind"]
    a["half"] = a["date"].dt.year.astype(str) + np.where(a["date"].dt.month <= 6, " 상", " 하")
    low = a.groupby("half").agg(순부하300이하_시간=("net", lambda x: int((x <= 300).sum())))
    t = d.groupby("half").agg(
        낮최저SMP_중앙=("mid_min", "median"), 낮최저SMP_하위5=("mid_min", lambda x: x.quantile(0.05)),
        LNG연료비=("fuel_lng", "mean"), 태양광최대_중앙=("act_sol_max", "median"), 풍력평균=("act_wind_mean", "mean"),
        수요평균=("act_dem_mean", "mean"), 한낮최저순부하_중앙=("act_net_min_mid", "median"),
        출력제어일=("curt_hours", lambda x: int((x > 0).sum())), 영원일=("event", "sum"), 날수=("date", "size"))
    t["낮최저SMP/LNG"] = t["낮최저SMP_중앙"] / t["LNG연료비"]
    t = t.join(low).reset_index().rename(columns={"half": "반기", "영원일": "0원일"})
    t.loc[t["반기"] < "2021", "출력제어일"] = np.nan              # 2020은 출력제어 기록 없음
    t.loc[t["반기"] == "2024 하", "출력제어일"] = np.nan          # 2024-06-03 이후 기록 없음
    t.round(3).to_csv(TAB / "J5_trend_halfyear_jeju.csv", index=False, encoding="utf-8-sig")
    print(t.round(2).to_string())

    c = d[d["surplus"].notna()]
    cols = [x for x in S1_COLS if x.startswith("act_")]
    cmp_ = c.groupby("surplus")[cols].median().T.rename(columns={0.0: "보통 날", 1.0: "잉여일"})
    cmp_.index = [NAMES[i] for i in cmp_.index]
    cmp_.round(1).to_csv(TAB / "J5_surplus_vs_normal_jeju.csv", encoding="utf-8-sig")
    print(cmp_.round(1).to_string())

    m = pipe(LogisticRegression(C=0.3, max_iter=2000)).fit(c[S1_COLS], c["surplus"])
    coef = pd.Series(m[-1].coef_[0], index=[NAMES[i] for i in S1_COLS]).sort_values()
    coef.rename("표준화 계수(+ = 잉여 쪽)").round(3).to_csv(TAB / "J5_stage1_coef_jeju.csv", encoding="utf-8-sig")
    print(coef.round(2).to_string())

    import matplotlib.pyplot as plt
    from common.plot import setup_korean_font
    setup_korean_font()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(t))
    ax[0].plot(x, t["낮최저SMP/LNG"], marker="o")
    ax[0].set_title("낮 최저 SMP ÷ LNG 연료비 (반기 중앙값)")
    ax[1].bar(x - 0.2, t["태양광최대_중앙"], 0.4, label="태양광 하루 최대(중앙값)")
    ax[1].bar(x + 0.2, t["한낮최저순부하_중앙"], 0.4, label="한낮 최저 순부하(중앙값)")
    ax[1].set_title("제주 계통 (MW)")
    ax[1].legend(fontsize=8)
    ax[2].bar(x - 0.2, t["출력제어일"].fillna(0), 0.4, label="출력제어일")
    ax[2].bar(x + 0.2, t["0원일"], 0.4, label="0원 이하일")
    ax[2].set_title("재생에너지 잉여 처리 (일)")
    ax[2].legend(fontsize=8)
    for a_ in ax:
        a_.set_xticks(x, t["반기"], rotation=45, fontsize=8)
    fig.suptitle("제주 2020~2024: 태양광이 늘며 낮 순부하가 내려가고, 잉여가 출력제어 → 0원으로 옮겨감 (2020 출력제어 기록 없음, 2024 하반기 기록 없음)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "J5_trend_jeju.png", dpi=150)


if __name__ == "__main__":
    main()
