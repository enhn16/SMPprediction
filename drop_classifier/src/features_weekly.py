"""일주일 모델 피처 테이블 (육지).

판단일 I의 18시(중기예보 18시 발표 직후)에, 대상일 D = I + h (h = 1~7)의 급락 여부를 판단한다.
판단 시점에 공공데이터로 알 수 있는 값만 쓴다.
- 달력: D일 (사전 확정)
- SMP 실적: I일까지 (I일 SMP는 I-1일 23시 공개)
- 수요: I일까지 공개된 하루전 수요예측 (I일분은 I-1일 23시 공개). D일 수요는 최근 같은 유형일 수요예측 중앙값으로 대신함
- 날씨: I일 18시 발표 중기예보의 D일 예보 (리드 3~7일). 1~2일 뒤는 중기예보가 없어 날씨 피처 없음

행 하나 = (판단일 I, 리드 h). 실행: python -m drop_classifier.src.features_weekly
"""
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR
from common.loaders import load_mid_forecast, load_smp_forecast_api

REGION = "육지"
START = "2024-01-01"
SOLAR_REGIONS = ["전라남도", "전라북도", "충청남도", "경상북도", "경상남도"]
OUT_DATA = PROCESSED_DIR / "drop_classifier"

W_CALENDAR = ["is_offday", "is_weekend_only", "is_holiday_single", "is_long_holiday", "off_block_len", "off_block_day",
              "is_myeongjeol", "is_labor_day", "month_sin", "month_cos", "is_bridge", "is_after_offday", "trend_years"]
W_STATE = ["w_prev_same_type_min_smp", "w_last_min_smp", "w_min_smp_last7", "w_drops_last14",
           "w_demand_proxy", "w_demand_proxy_rel", "w_demand_last"]
W_MID = ["mf_sky_mean", "mf_clear_share", "mf_pop_mean", "mf_rain_share"]

FEATURE_KO_W = {
    "w_prev_same_type_min_smp": "판단일까지 직전 같은 유형일 최저 SMP", "w_last_min_smp": "판단일 최저 SMP",
    "w_min_smp_last7": "판단일까지 7일 최저 SMP", "w_drops_last14": "판단일까지 14일 급락 횟수",
    "w_demand_proxy": "최근 같은 유형일 수요예측(대상일 수요 대리)", "w_demand_proxy_rel": "수요 대리값(28일 대비)",
    "w_demand_last": "판단일 수요예측", "mf_sky_mean": "중기 하늘상태 예보", "mf_clear_share": "중기 맑음 비율",
    "mf_pop_mean": "중기 강수확률", "mf_rain_share": "중기 강수 예보 비율", "lead": "며칠 뒤",
}


def mid_features() -> pd.DataFrame:
    """I일 18시 발표분의 대상일별 요약 (태양광 많은 5개 광역, 오전·오후 평균)."""
    mf = load_mid_forecast()
    mf = mf[(mf["issue_hour"] == 18) & mf["region"].isin(SOLAR_REGIONS)]
    g = mf.groupby(["issue_date", "target_date"]).agg(
        mf_sky_mean=("sky", "mean"),
        mf_clear_share=("sky", lambda s: (s == 1).mean()),
        mf_pop_mean=("pop", "mean"),
        mf_rain_share=("rain", "mean"),
    ).reset_index()
    return g


def build() -> pd.DataFrame:
    daily = pd.read_csv(OUT_DATA / f"features_{REGION}.csv", parse_dates=["date"])
    label = pd.read_csv(OUT_DATA / f"daily_label_{REGION}.csv", parse_dates=["date"])  # 2022~ (시차 계산용)
    fc = load_smp_forecast_api()
    dem = fc[(fc["region"] == REGION) & fc["hour"].between(11, 15)].groupby("date")["demand_fcst"].mean()
    hist = label[["date", "min_smp_11_15", "label", "day_type"]].copy()
    hist["is_offday"] = (hist["day_type"] != "평일").astype(int)
    hist["demand_fcst"] = hist["date"].map(dem)
    hist = hist.sort_values("date").reset_index(drop=True)
    pos = {d: i for i, d in enumerate(hist["date"])}
    smp, lab, off, dmd = (hist[c].values for c in ("min_smp_11_15", "label", "is_offday", "demand_fcst"))

    mid = mid_features().set_index(["issue_date", "target_date"])
    keep = ["date", "label", "min_smp_11_15", "day_type", "weekday_ko", "holiday_name"] + W_CALENDAR
    rows = []
    for _, t in daily[keep].iterrows():
        for h in range(1, 8):
            issue = t["date"] - pd.Timedelta(days=h)
            k = pos.get(issue)
            if k is None or k < 60:
                continue
            past = np.arange(k - 55, k + 1)
            same = past[off[past] == t["is_offday"]]
            proxy = np.median(dmd[same[-4:]]) if len(same) else np.nan
            r = {**t.to_dict(), "issue_date": issue, "lead": h,
                 "w_prev_same_type_min_smp": smp[same[-1]] if len(same) else np.nan,
                 "w_last_min_smp": smp[k],
                 "w_min_smp_last7": smp[k - 6:k + 1].min(),
                 "w_drops_last14": lab[k - 13:k + 1].sum(),
                 "w_demand_proxy": proxy,
                 "w_demand_proxy_rel": proxy / np.median(dmd[k - 27:k + 1]),
                 "w_demand_last": dmd[k]}
            key = (issue, t["date"])
            if key in mid.index:
                r.update(mid.loc[key].to_dict())
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    df = build()
    out = OUT_DATA / f"features_weekly_{REGION}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out} ({len(df)}행)")
    cov = df.groupby("lead")["mf_sky_mean"].apply(lambda s: s.notna().mean()).round(3)
    print("리드별 중기예보 확보 비율:", cov.to_dict())


if __name__ == "__main__":
    main()
