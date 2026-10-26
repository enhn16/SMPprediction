"""③ 피처 테이블 생성 (육지).

판단 시각: D-1 17시. 각 피처가 언제 확보되는지 FEATURE_INFO에 적어 둔다.
한낮 = 11~15시 (SMP 표기 기준).

피처 묶음
- CALENDAR : 요일유형·연휴·명절·월 (사전 확정)
- DEMAND   : D일 하루전 수요예측 (KPX 공개 시각 확인 필요, 아래 주의)
- DEMAND_PREV : D-1일 하루전 수요예측 (D-2에 공개 → 확실히 확보)
- LAG      : D-1 이전 SMP 실적
- FCST     : D-1 17시 발표 단기예보
- ACTUAL   : ASOS 실측 (상한선 전용, 실제 예측에는 사용 불가)

주의: data.go.kr API의 하루전 수요예측은 D-1 23시경 갱신된다. 발전사가 D-1 17시 이전에
같은 값을 받을 수 있는지 확인되지 않았으므로, DEMAND를 뺀 묶음(DEMAND_PREV 사용)도 함께 평가한다.

실행: python -m drop_classifier.src.features
"""
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR
from common.loaders import load_asos, load_short_forecast, load_smp_forecast_api
from common.stations import FORECAST_STATIONS

REGION = "육지"
MID = (11, 15)
START = "2024-01-01"
OUT_DATA = PROCESSED_DIR / "drop_classifier"

CALENDAR = ["is_offday", "is_weekend_only", "is_holiday_single", "is_long_holiday", "off_block_len", "off_block_day",
            "is_myeongjeol", "is_labor_day", "month_sin", "month_cos"]
DEMAND = ["demand_fcst", "demand_rel28"]
DEMAND_PREV = ["demand_fcst_prev", "demand_prev_rel28"]
LAG = ["prev_same_type_min_smp", "prev_day_min_smp"]
# fc_clear_share는 fc_sky_mean과 상관 -0.98이라 모델 피처에서 제외 (규칙 기준선에서만 사용)
FCST = ["fc_sky_mean", "fc_pop_mean", "fc_rain_share", "fc_tmp_mean", "fc_wsd_mean"]
ACTUAL = ["act_cloud", "act_ss", "act_icsr", "act_ta"]

# 개선 후보 (2026-10-19 추가, 모두 D-1 17시 확보 가능)
DEMAND_X = ["demand_fcst_min", "demand_fcst_13", "demand_rel_sametype"]
LAG_X = ["drops_last14", "min_smp_last7"]
FCST_X = ["fc_sky_honam", "fc_sky_max", "fc_sky_min", "fc_tmp_sq"]
CALENDAR_X = ["is_bridge", "is_after_offday", "trend_years"]
HONAM = [146, 156, 165]  # 전주, 광주, 목포

FEATURE_INFO = {
    "is_offday": "휴무일(토·일·공휴일) 여부 — 사전 확정",
    "is_weekend_only": "연휴가 아닌 일반 주말 — 사전 확정",
    "is_holiday_single": "연휴가 아닌 단독 공휴일 — 사전 확정",
    "is_long_holiday": "3일 이상 연휴 — 사전 확정",
    "off_block_len": "연속 휴무일 길이 — 사전 확정",
    "off_block_day": "연속 휴무일 며칠째 — 사전 확정",
    "is_myeongjeol": "설날·추석 연휴 — 사전 확정",
    "is_labor_day": "2025년까지 근로자의날(5/1) — 사전 확정",
    "month_sin": "월 주기(sin) — 사전 확정",
    "month_cos": "월 주기(cos) — 사전 확정",
    "demand_fcst": "D일 한낮 하루전 수요예측 평균(MW) — data.go.kr 기준 D-1 23시 공개, 17시 확보 여부 확인 필요",
    "demand_rel28": "demand_fcst ÷ 직전 28일(D-28~D-1) 한낮 수요예측 중앙값 — 위와 같음",
    "demand_fcst_prev": "D-1일 한낮 하루전 수요예측(MW) — D-2 23시 공개",
    "demand_prev_rel28": "demand_fcst_prev ÷ 직전 28일(D-29~D-2) 중앙값 — D-2 23시 공개",
    "prev_same_type_min_smp": "D-1 이전 가장 가까운 같은 유형일(휴무일/평일)의 한낮 최저 SMP — D-1 이전 확정",
    "prev_day_min_smp": "D-1일 한낮 최저 SMP — D-2 23시 공개",
    "fc_sky_mean": "D-1 17시 발표 예보 한낮 하늘상태 평균(1 맑음~4 흐림), 육지 7개 지점",
    "fc_clear_share": "같은 예보에서 하늘상태 '맑음' 비율",
    "fc_pop_mean": "같은 예보 강수확률 평균(%)",
    "fc_rain_share": "같은 예보 강수형태 있음(비·눈) 비율",
    "fc_tmp_mean": "같은 예보 기온 평균(℃)",
    "fc_wsd_mean": "같은 예보 풍속 평균(m/s)",
    "act_cloud": "ASOS 실측 한낮 전운량 — 상한선 전용",
    "act_ss": "ASOS 실측 한낮 일조 — 상한선 전용",
    "act_icsr": "ASOS 실측 한낮 일사량 — 상한선 전용",
    "act_ta": "ASOS 실측 한낮 기온 — 상한선 전용",
    "demand_fcst_min": "D일 한낮(11~15시) 수요예측 최저값 — demand_fcst와 같은 시각 공개",
    "demand_fcst_13": "D일 13시(12~13시 구간) 수요예측 — 급락이 가장 잦은 시각",
    "demand_rel_sametype": "demand_fcst ÷ 직전 56일 중 같은 유형일(휴무일/평일) 수요예측 중앙값",
    "drops_last14": "D-14~D-1 급락일 수 — D-1 이전 확정",
    "min_smp_last7": "D-7~D-1 한낮 최저 SMP의 최솟값 — D-1 이전 확정",
    "fc_sky_honam": "호남 3개 지점(전주·광주·목포) 한낮 하늘상태 예보 평균",
    "fc_sky_max": "7개 지점 중 가장 흐린 지점의 하늘상태 예보",
    "fc_sky_min": "7개 지점 중 가장 맑은 지점의 하늘상태 예보",
    "fc_tmp_sq": "(기온 예보 − 15℃)² — 냉·난방 수요의 비선형 효과",
    "is_bridge": "징검다리 평일(전날·다음 날 모두 휴무일) — 사전 확정",
    "is_after_offday": "휴무일 다음 첫 평일 — 사전 확정",
    "trend_years": "2024-01-01부터 경과 연수 (태양광 설비 증가 추세 대리)",
}


# 그래프·보고서용 한국어 이름
FEATURE_KO = {
    "is_offday": "휴무일", "is_weekend_only": "일반 주말", "is_holiday_single": "단독 공휴일",
    "is_long_holiday": "3일 이상 연휴", "off_block_len": "연휴 길이", "off_block_day": "연휴 며칠째",
    "is_myeongjeol": "명절", "is_labor_day": "근로자의날", "month_sin": "월(sin)", "month_cos": "월(cos)",
    "demand_fcst": "수요예측", "demand_rel28": "수요예측(28일 대비)",
    "demand_fcst_prev": "전날 수요예측", "demand_prev_rel28": "전날 수요예측(28일 대비)",
    "prev_same_type_min_smp": "직전 같은 유형일 최저 SMP", "prev_day_min_smp": "전날 최저 SMP",
    "fc_sky_mean": "하늘상태 예보(1맑음~4흐림)", "fc_clear_share": "맑음 예보 비율", "fc_pop_mean": "강수확률 예보",
    "fc_rain_share": "강수 예보 비율", "fc_tmp_mean": "기온 예보", "fc_wsd_mean": "풍속 예보",
    "act_cloud": "실측 전운량", "act_ss": "실측 일조", "act_icsr": "실측 일사량", "act_ta": "실측 기온",
    "demand_fcst_min": "한낮 최저 수요예측", "demand_fcst_13": "13시 수요예측", "demand_rel_sametype": "수요예측(같은 유형일 대비)",
    "drops_last14": "최근 14일 급락 횟수", "min_smp_last7": "최근 7일 최저 SMP",
    "fc_sky_honam": "호남 하늘상태 예보", "fc_sky_max": "가장 흐린 지점 하늘상태", "fc_sky_min": "가장 맑은 지점 하늘상태",
    "fc_tmp_sq": "기온 예보 편차²", "is_bridge": "징검다리 평일", "is_after_offday": "휴무일 다음 평일", "trend_years": "추세(연)",
}


def ko(name: str) -> str:
    """피처 이름(상호작용 'a & b' 포함)을 한국어로."""
    return " × ".join(FEATURE_KO.get(p.strip(), p.strip()) for p in name.split("&"))


def build(start: str = START) -> pd.DataFrame:
    """start: 남길 첫 날짜 (기본 2024-01-01 = 기존 동작, 팀 규칙 재선택은 2023-01-01)."""
    label = pd.read_csv(OUT_DATA / f"daily_label_{REGION}.csv", parse_dates=["date"])
    stations = FORECAST_STATIONS[REGION]

    # 수요예측: 한낮 평균, 직전 28일 대비 상대값 (2026년 수준 변화 대응)
    fc = load_smp_forecast_api()
    dem = fc[(fc["region"] == REGION) & fc["hour"].between(*MID)].groupby("date")["demand_fcst"].mean()
    dem = dem.asfreq("D")
    base28 = dem.shift(1).rolling(28, min_periods=20).median()
    demand = pd.DataFrame({"demand_fcst": dem, "demand_rel28": dem / base28})
    demand["demand_fcst_prev"] = demand["demand_fcst"].shift(1)
    demand["demand_prev_rel28"] = demand["demand_rel28"].shift(1)
    fmid = fc[(fc["region"] == REGION) & fc["hour"].between(*MID)]
    demand["demand_fcst_min"] = fmid.groupby("date")["demand_fcst"].min()
    demand["demand_fcst_13"] = fc[(fc["region"] == REGION) & (fc["hour"] == 13)].set_index("date")["demand_fcst"]

    # 단기예보: D-1 17시 발표분의 한낮 지점 평균
    sf = load_short_forecast(stations)
    sf = sf[sf["hour"].between(*MID)]
    fcst = sf.groupby("date").agg(
        fc_sky_mean=("SKY", "mean"),
        fc_clear_share=("SKY", lambda s: (s == 1).mean()),
        fc_pop_mean=("POP", "mean"),
        fc_rain_share=("PTY", lambda s: (s > 0).mean()),
        fc_tmp_mean=("TMP", "mean"),
        fc_wsd_mean=("WSD", "mean"),
        fc_n_station=("stn_id", "nunique"),
    )
    by_stn = sf.groupby(["date", "stn_id"])["SKY"].mean().unstack()
    fcst["fc_sky_honam"] = by_stn[HONAM].mean(axis=1)
    fcst["fc_sky_max"] = by_stn.max(axis=1)
    fcst["fc_sky_min"] = by_stn.min(axis=1)
    fcst["fc_tmp_sq"] = (fcst["fc_tmp_mean"] - 15) ** 2

    # 실측(상한선): 같은 7개 지점
    a = load_asos()
    a = a[a["stn_id"].isin(stations) & a["hour"].between(*MID)]
    actual = (a.groupby(["date", "stn_id"])[["cloud", "ss", "icsr", "ta"]].mean().groupby("date").mean()
              .add_prefix("act_"))

    df = label.set_index("date").join([demand, fcst, actual], how="left").reset_index()

    # 달력
    df["is_offday"] = (df["day_type"] != "평일").astype(int)
    df["is_weekend_only"] = (df["day_type"] == "주말").astype(int)
    df["is_holiday_single"] = (df["day_type"] == "공휴일").astype(int)
    df["is_long_holiday"] = (df["day_type"] == "연휴").astype(int)
    df["is_myeongjeol"] = df["is_myeongjeol"].astype(int)
    df["is_labor_day"] = df["is_labor_day"].astype(int)
    m = df["date"].dt.month
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)

    # 시차 SMP: D-1 이전만 사용
    df = df.sort_values("date").reset_index(drop=True)
    df["prev_day_min_smp"] = df["min_smp_11_15"].shift(1)
    prev_same = []
    last = {}
    for off, v in zip(df["is_offday"], df["min_smp_11_15"]):
        prev_same.append(last.get(off, np.nan))
        last[off] = v
    df["prev_same_type_min_smp"] = prev_same

    # 최근 시장 흐름 (D-1 이전만)
    df["drops_last14"] = df["label"].shift(1).rolling(14, min_periods=1).sum()
    df["min_smp_last7"] = df["min_smp_11_15"].shift(1).rolling(7, min_periods=1).min()

    # 같은 유형일 대비 수요예측: 직전 56일 중 같은 휴무일/평일 유형의 중앙값
    rel = []
    for i, (d, off, v) in enumerate(zip(df["date"], df["is_offday"], df["demand_fcst"])):
        past = df.iloc[max(0, i - 56):i]
        ref = past.loc[past["is_offday"] == off, "demand_fcst"].median()
        rel.append(v / ref if pd.notna(ref) else np.nan)
    df["demand_rel_sametype"] = rel

    # 달력 보강
    off = df["is_offday"]
    df["is_bridge"] = ((off == 0) & (off.shift(1) == 1) & (off.shift(-1) == 1)).astype(int)
    df["is_after_offday"] = ((off == 0) & (off.shift(1) == 1)).astype(int)
    df["trend_years"] = (df["date"] - pd.Timestamp(START)).dt.days / 365.25

    df = df[df["date"] >= start].reset_index(drop=True)
    return df


def main():
    df = build()
    out = OUT_DATA / f"features_{REGION}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    pd.Series(FEATURE_INFO, name="설명").rename_axis("피처").to_csv(
        OUT_DATA / "feature_info.csv", encoding="utf-8-sig")
    cols = CALENDAR + DEMAND + DEMAND_PREV + LAG + FCST + ACTUAL
    print(f"저장: {out} ({len(df)}일, {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d})")
    print("결측:", {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()})
    print("예보 지점 수 분포:", df["fc_n_station"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
