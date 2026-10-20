"""갈래 B — 자체 D+1 시간별 예측용 피처 테이블 (육지, 판단 시각 D-1 15:00).

행 = 대상일 D × 시간 h(1~24). 모든 피처는 D-1 15:00에 알 수 있는 값만 쓴다.
| 묶음 | 피처 | 확보 시점 |
|---|---|---|
| 달력 | 시간·요일·월·연중일, 휴무일·요일유형·연휴 길이/며칠째·명절, 전날/다음날 휴무 | 사전 확정 (common/calendar) |
| SMP | D-1·D-2·D-7 같은 시간, D-1 일평균/최저/최고/한낮최저, 7일 같은 시간 평균, 28일 평균, 직전 같은 유형일(휴무/평일) 값 | D-1 SMP는 D-2 저녁 공표 |
| 수요 | D-1·D-7 같은 시간 하루전 수요예측, D-1 일평균, 직전 같은 유형일 값 | D-1분은 D-2 저녁 공표 (D일분은 D-1 19~20시라 미사용) |
| 주간전망 | 전력거래소 주간 전력수요전망 D일 최대부하·전망 내 비율 | 등록일 다음 날부터 (posted ≤ D-2) |
| 연료비 | EPSIS LNG·유연탄 연료비단가 | M월 값은 M+2월 1일부터 |
| 설비 | 태양광 시장참여 설비용량 | M월 값은 M+2월 1일부터 |
| 날씨 | D-1 11시(없으면 05시) 발표 단기 육상예보 하늘상태·강수확률·기온, Open-Meteo D-2 실행 한낮 일사·운량 | 15시 이전 발표분 |
저장: data/processed/smp_forecast_app/features_hourly_육지.csv
실행: python -m smp_forecast_app.src.features
"""
import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR
from common.loaders import (load_fuel_cost, load_kma_regional_fcst, load_market_capacity, load_mid_forecast,
                            load_openmeteo_prev2, load_smp_excel, load_smp_forecast_api)
from drop_classifier.src import outlook_features
from common.collect.kma_regional_fcst import REGIONS
from drop_classifier.src.fair_select import fcst15

OUT = PROCESSED_DIR / "smp_forecast_app" / "features_hourly_육지.csv"
DAYTYPE_CODE = {"평일": 0, "주말": 1, "공휴일": 2, "연휴": 3}

CAL = ["hour", "dow", "month", "doy_sin", "doy_cos", "offday", "daytype", "off_block_len", "off_block_day",
       "is_myeongjeol", "is_labor_day", "prev_offday", "next_offday", "type_change"]
SMP = ["s_d1", "s_d2", "s_d7", "s_d1_mean", "s_d1_min", "s_d1_max", "s_d1_midmin", "s_ma7_h", "s_ma28",
       "s_same_h", "s_same_mean", "s_same_midmin", "s_d1_shape"]
DEM = ["d_d1_h", "d_d7_h", "d_d1_mean", "d_same_h", "d_d1_shape"]
LEVEL = ["fc_lng", "fc_coal", "cap_solar"]
OUTLOOK = ["kpx_peak", "kpx_peak_rel", "kpx_peak_vs_d1"]
WEATHER = ["rf_sky_pm", "rf_pop_pm", "rf_ta_pm", "rf_sky_am", "om_rad_mid", "om_cloud_mid"]
FEATURE_SETS = {"기본": CAL + SMP + DEM + LEVEL, "기본+전망·날씨": CAL + SMP + DEM + LEVEL + OUTLOOK + WEATHER}


def _wide(df, col):
    return df.pivot_table(index="date", columns="hour", values=col)


def _monthly_usable(dates: pd.Series, table: pd.DataFrame, cols: list[str], lead: int = 1) -> pd.DataFrame:
    """판단일(D-H)에 쓸 수 있는 가장 최근 월값 (usable_from ≤ D-H)."""
    t = table.sort_values("usable_from")[["usable_from"] + cols]
    dec = pd.DataFrame({"dec": dates - pd.Timedelta(days=lead)}).reset_index()
    m = pd.merge_asof(dec.sort_values("dec"), t, left_on="dec", right_on="usable_from")
    return m.sort_values("index").set_index("index")[cols]


def build(lead: int = 1) -> pd.DataFrame:
    """lead = H: 판단일 I = D-H의 15시에 D를 예측. H=1이 하루 전(기본). 피처 이름은 H와 관계없이 같게 둔다
    (s_d1 = '판단 시점의 가장 최근 확정일(D-H)' 같은 시간 SMP)."""
    H = lead
    smp = load_smp_excel("육지")
    dem = load_smp_forecast_api()
    dem = dem[dem["region"] == "육지"]
    S, D = _wide(smp, "smp"), _wide(dem, "demand_fcst")
    last = S.index.max()
    days = pd.date_range("2022-01-08", last + pd.Timedelta(days=H))   # 마지막 실적일 + H = 아직 오지 않은 날 예측용
    S, D = S.reindex(pd.date_range("2021-12-01", days[-1])), D.reindex(pd.date_range("2021-12-01", days[-1]))
    cal = build_calendar(days[0] - pd.Timedelta(days=60), days[-1] + pd.Timedelta(days=1)).set_index("date")

    def shift(W, k):  # 대상일 D의 행에 D-k의 값
        return W.shift(k).reindex(days)

    feats = {}
    feats["s_d1"], feats["s_d2"], feats["s_d7"] = shift(S, H), shift(S, H + 1), shift(S, 7 if H <= 7 else 14)
    feats["s_ma7_h"] = sum(shift(S, k) for k in range(H, H + 7)) / 7
    feats["d_d1_h"], feats["d_d7_h"] = shift(D, H), shift(D, 7 if H <= 7 else 14)
    S1, D1 = shift(S, H), shift(D, H)
    daily = {
        "s_d1_mean": S1.mean(axis=1), "s_d1_min": S1.min(axis=1), "s_d1_max": S1.max(axis=1),
        "s_d1_midmin": S1[[11, 12, 13, 14, 15]].min(axis=1), "s_ma28": S.mean(axis=1).rolling(28).mean().shift(H).reindex(days),
        "d_d1_mean": D1.mean(axis=1),
    }

    # 직전 같은 유형일(휴무일/평일): D-H 이전에서 offday가 D와 같은 가장 가까운 날
    off = cal["is_offday"]
    same_idx = []
    for d in days:
        k = H
        while off.get(d - pd.Timedelta(days=k), False) != off[d]:
            k += 1
        same_idx.append(d - pd.Timedelta(days=k))
    same_idx = pd.DatetimeIndex(same_idx)
    Ssame, Dsame = S.reindex(same_idx).set_axis(days), D.reindex(same_idx).set_axis(days)
    feats["s_same_h"], feats["d_same_h"] = Ssame, Dsame
    daily["s_same_mean"] = Ssame.mean(axis=1)
    daily["s_same_midmin"] = Ssame[[11, 12, 13, 14, 15]].min(axis=1)
    feats["s_d1_shape"] = S1.div(S1.mean(axis=1).clip(lower=20), axis=0)
    feats["d_d1_shape"] = D1.div(D1.mean(axis=1), axis=0)

    # long 형식으로
    long = pd.concat({k: v.stack(dropna=False) for k, v in feats.items()}, axis=1)
    long.index.names = ["date", "hour"]
    long = long.reset_index()
    long = long.merge(pd.DataFrame(daily).rename_axis("date").reset_index(), on="date", how="left")
    long["hour"] = long["hour"].astype(int)

    c = cal.reindex(days)
    cd = pd.DataFrame({
        "date": days, "dow": c["weekday"].values, "month": days.month,
        "doy_sin": np.sin(2 * np.pi * days.dayofyear / 365.25), "doy_cos": np.cos(2 * np.pi * days.dayofyear / 365.25),
        "offday": c["is_offday"].astype(int).values, "day_type": c["day_type"].values,
        "daytype": c["day_type"].map(DAYTYPE_CODE).values, "off_block_len": c["off_block_len"].values,
        "off_block_day": c["off_block_day"].values, "is_myeongjeol": c["is_myeongjeol"].astype(int).values,
        "is_labor_day": c["is_labor_day"].astype(int).values,
        "prev_offday": cal["is_offday"].shift(1).reindex(days).astype(float).values,
        "next_offday": cal["is_offday"].shift(-1).reindex(days).astype(float).values,
    })
    cd["type_change"] = (cd["offday"] != cd["prev_offday"]).astype(int)

    # 월값: 연료비, 태양광 설비
    fc = load_fuel_cost()
    cd[["fc_lng", "fc_coal"]] = _monthly_usable(cd["date"], fc, ["fc_lng", "fc_coal"], H).values
    cap = load_market_capacity()
    cd[["cap_solar"]] = _monthly_usable(cd["date"], cap, ["cap_solar"], H).values

    # 주간 전력수요전망 (판단일 I = D-H 기준 전날까지 등록분: posted ≤ I-1)
    ol = outlook_features.attach(cd["date"], cd["date"] - pd.Timedelta(days=H))
    cd["kpx_peak"], cd["kpx_peak_rel"] = ol["kpx_peak"].values, ol["kpx_peak_rel"].values
    d1max = D1.max(axis=1).reindex(days).values / 1000
    cd["kpx_peak_vs_d1"] = cd["kpx_peak"] / d1max

    # 날씨 예보 (15시 이전 발표)
    w = weather_for_lead(H)
    cd = cd.merge(w, on="date", how="left")
    if H == 1:  # Open-Meteo D-2 실행분은 하루 전 판단에만 (H≥2면 판단 시각 이후 실행분일 수 있음)
        om = load_openmeteo_prev2().groupby("date")[["rad_mid", "cloud_mid"]].mean()
        om = om.rename(columns={"rad_mid": "om_rad_mid", "cloud_mid": "om_cloud_mid"}).reset_index()
        cd = cd.merge(om, on="date", how="left")
    else:
        cd["om_rad_mid"], cd["om_cloud_mid"] = np.nan, np.nan

    out = long.merge(cd, on="date", how="left")
    y = smp.rename(columns={"smp": "y"})
    out = out.merge(y, on=["date", "hour"], how="left")
    return out.sort_values(["date", "hour"]).reset_index(drop=True)


def out_path(lead: int = 1):
    return OUT if lead == 1 else OUT.with_name(f"features_hourly_육지_h{lead}.csv")


def load_features(lead: int = 1) -> pd.DataFrame:
    return pd.read_csv(out_path(lead), parse_dates=["date"])


def weather_for_lead(H: int) -> pd.DataFrame:
    """판단일 I = D-H의 15시 이전 발표 날씨 예보 → D일 [rf_sky_pm, rf_pop_pm, rf_ta_pm, rf_sky_am].
    H=1: fcst15 (D-1 11시, 없으면 05시 단기 육상예보) — 과제1과 같음
    H=2~3: I일 11시(없으면 05시) 발표 단기 육상예보의 D일 구간 (발표 후 3일째까지 제공)
    H≥4: I일 06시 발표 중기 육상예보 (육지 9개 권역, 기온 없음)"""
    cols = ["date", "rf_sky_pm", "rf_pop_pm", "rf_ta_pm", "rf_sky_am"]
    if H == 1:
        return fcst15()[cols]
    if H <= 3:
        f = load_kma_regional_fcst()
        f = f[f["reg"].isin(REGIONS)]
        f["date"] = f["tm_ef"].dt.normalize()
        f["seg"] = np.where(f["tm_ef"].dt.hour == 12, "pm", "am")
        f = f[((f["date"] - f["tm_fc"].dt.normalize()).dt.days == H) & f["tm_fc"].dt.hour.isin([5, 11])]
        f = f.sort_values("tm_fc").groupby(["reg", "date", "seg"]).tail(1)
        pm, am = f[f["seg"] == "pm"], f[f["seg"] == "am"]
        return pd.DataFrame({"rf_sky_pm": pm.groupby("date")["sky"].mean(), "rf_pop_pm": pm.groupby("date")["pop"].mean(),
                             "rf_ta_pm": pm.groupby("date")["ta"].mean(), "rf_sky_am": am.groupby("date")["sky"].mean()}
                            ).rename_axis("date").reset_index()
    m = load_mid_forecast()
    m = m[(m["issue_hour"] == 6) & (m["lead"] == H) & (m["region"] != "제주도")]
    pm, am = m[m["ampm"] == "오후"], m[m["ampm"] == "오전"]
    out = pd.DataFrame({"rf_sky_pm": pm.groupby("target_date")["sky"].mean(), "rf_pop_pm": pm.groupby("target_date")["pop"].mean(),
                        "rf_sky_am": am.groupby("target_date")["sky"].mean()}).rename_axis("date").reset_index()
    out["rf_ta_pm"] = np.nan
    return out[cols]


def main(leads=(1,)):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for H in leads:
        save(H)


def save(H: int):
    p = out_path(H)
    df = build(H)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    cov = df[df["date"] >= "2024-01-01"]
    print(f"[H={H}] 저장: {p} ({len(df):,}행, {df.date.min().date()} ~ {df.date.max().date()})")
    print("2024~ 결측률:", cov[FEATURE_SETS["기본+전망·날씨"]].isna().mean().round(3).loc[lambda s: s > 0].to_dict())


if __name__ == "__main__":
    main()
