"""운영용 과제1 급락 판단 — 과제1 최종 모델(Ridge+EBM+랜덤포레스트, 피처 '기본+주간전망+보강+태양광설비')을 내일 날짜까지 다시 계산.

과제1 파일(drop_classifier/outputs, data/processed/drop_classifier)은 건드리지 않는다.
피처는 과제1 features.build + fair_select.fcst15 + outlook_features.attach + supply_level.supply_features와 같은 정의로
과제2 쪽에서 다시 만들되, 실적이 없는 '내일' 행도 만든다 (달력은 사전 확정, 나머지는 D-1 이전 값).
검증: python -m smp_forecast_app.src.ops_drop  → 과제1 피처 테이블과 2024~ 값이 같은지 비교
"""
import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.loaders import load_market_capacity, load_smp_actual, load_smp_forecast_api
from drop_classifier.src import outlook_features
from drop_classifier.src.fair_select import fcst15
from drop_classifier.src.fuel_level import BASE
from drop_classifier.src.supply_level import SOLAR

COLS = BASE + SOLAR
import json as _json
from common.config import ROOT as _ROOT
_CH = _json.loads((_ROOT / "drop_classifier" / "outputs" / "tables" / "13_final_choice.json").read_text(encoding="utf-8"))
WARN, STRONG = _CH["주의 문턱"], _CH["강한 경보 문턱"]   # 과제1 최종 문턱 (2026-10-21: 90.1·75.3원, 재현율 90%)
MEMBERS = ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)"]   # 과제1 최종 모델 구성 (final_model.MEMBERS)


def predict_min(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """과제1 최종 모델(zoo.BASES 3종 평균)로 내일 한낮 최저 SMP 예측(원)."""
    from drop_classifier.src.zoo import BASES
    return -np.mean([BASES[m](train, target) for m in MEMBERS], axis=0)
MID = [11, 12, 13, 14, 15]


def drop_features(start: str = "2024-01-01") -> pd.DataFrame:
    """start: 남길 첫 날짜 (기본 2024-01-01 = 기존 동작, 팀 규칙 predict()는 2023-01-01)."""
    s = load_smp_actual("육지")
    w = s.pivot_table(index="date", columns="hour", values="smp")
    last = w.index.max()
    days = pd.date_range("2022-01-01", last + pd.Timedelta(days=1))   # 마지막 실적일 + 1 = 내일
    mn = w[MID].min(axis=1).reindex(days)
    lab = (mn < 30).astype(float).where(mn.notna())
    cal = build_calendar(days[0], days[-1] + pd.Timedelta(days=1)).set_index("date")
    c = cal.reindex(days)
    df = pd.DataFrame(index=days)
    dt = c["day_type"]
    df["is_offday"] = (dt != "평일").astype(int)
    df["is_weekend_only"] = (dt == "주말").astype(int)
    df["is_holiday_single"] = (dt == "공휴일").astype(int)
    df["is_long_holiday"] = (dt == "연휴").astype(int)
    df["off_block_len"], df["off_block_day"] = c["off_block_len"], c["off_block_day"]
    df["is_myeongjeol"] = c["is_myeongjeol"].astype(int)
    df["is_labor_day"] = c["is_labor_day"].astype(int)
    m = days.month
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    off_all = (cal["day_type"] != "평일").astype(int)
    df["is_bridge"] = ((off_all == 0) & (off_all.shift(1) == 1) & (off_all.shift(-1) == 1)).astype(int).reindex(days)
    df["is_after_offday"] = ((off_all == 0) & (off_all.shift(1) == 1)).astype(int).reindex(days)
    df["trend_years"] = (days - pd.Timestamp("2024-01-01")).days / 365.25

    fc = load_smp_forecast_api()
    dem = fc[(fc["region"] == "육지") & fc["hour"].between(11, 15)].groupby("date")["demand_fcst"].mean().asfreq("D")
    rel = dem / dem.shift(1).rolling(28, min_periods=20).median()
    df["demand_fcst_prev"] = dem.shift(1).reindex(days)
    df["demand_prev_rel28"] = rel.shift(1).reindex(days)

    df["prev_day_min_smp"] = mn.shift(1)
    prev_same, lastv = [], {}
    for off, v in zip(df["is_offday"], mn):
        prev_same.append(lastv.get(off, np.nan))
        if pd.notna(v):
            lastv[off] = v
    df["prev_same_type_min_smp"] = prev_same
    df["drops_last14"] = lab.shift(1).rolling(14, min_periods=1).sum()
    df["min_smp_last7"] = mn.shift(1).rolling(7, min_periods=1).min()

    df = df.rename_axis("date").reset_index()
    df = df.merge(fcst15()[["date", "rf_sky_pm", "rf_sky_am", "rf_clear_pm", "rf_pop_pm", "rf_rain_pm", "rf_ta_pm"]],
                  on="date", how="left")
    ol = outlook_features.attach(df["date"], df["date"] - pd.Timedelta(days=1))
    df["kpx_peak"], df["kpx_peak_rel"] = ol["kpx_peak"].values, ol["kpx_peak_rel"].values
    cap = load_market_capacity().sort_values("usable_from")
    df = pd.merge_asof(df.sort_values("date"), cap[["usable_from", "cap_solar"]], left_on="date", right_on="usable_from")
    df["cap_solar_gw"] = df["cap_solar"] / 1000
    df["min_smp_11_15"], df["label"] = mn.values, lab.values
    return df[df["date"] >= start][["date"] + COLS + ["min_smp_11_15", "label"]].reset_index(drop=True)


def grade_of(pred_min):
    return np.where(pred_min <= STRONG, "강한 경보", np.where(pred_min <= WARN, "주의", "없음"))


# ---------------------------------------------------------------- 팀 규칙 1차(15시) 급락 경보 (R1, ROLES.md 4.1)
TEAM_START = "2023-01-01"


def predict(bases, expanding: bool = False, dfe: pd.DataFrame | None = None) -> pd.DataFrame:
    """기준일(base_date) 15시 판단 → 다음 날 급락 경보. 열: base_date, date, grade(없음/주의/경보), prob, score(예상 한낮 최저 원).

    모델·문턱·확률식 = drop_classifier/outputs/tables/drop_choice_육지.json (team_final.py).
    학습: 기본 = 대상일 전년도 말까지 고정 (2026 대상 → 2023~2025, 2025 대상 → 2023~2024, 백테스트와 같음).
          expanding=True면 2023-01-01 ~ 기준일까지 확정된 날 전부 (매일 재학습).
    같은 학습 구간끼리 한 번만 학습. 피처가 없는 날(내일 행 미생성)은 빠진다.
    """
    from drop_classifier.src.zoo import BASES
    ch = _json.loads((_ROOT / "drop_classifier" / "outputs" / "tables" / "drop_choice_육지.json").read_text(encoding="utf-8"))
    warn, strong = ch["문턱"]["주의"], ch["문턱"]["경보"]
    a, b = ch["확률식"]["a"], ch["확률식"]["b"]
    if dfe is None:
        dfe = drop_features(TEAM_START)
    have = set(dfe["date"])
    jobs = {}
    for I in pd.to_datetime(pd.Series(bases)).drop_duplicates():
        d1 = I + pd.Timedelta(days=1)
        if d1 not in have:
            continue
        end = I if expanding else min(I, pd.Timestamp(f"{d1.year - 1}-12-31"))
        jobs.setdefault(end, []).append((I, d1))
    rows = []
    for end, pairs in jobs.items():
        tr = dfe[dfe["label"].notna() & dfe["date"].between(TEAM_START, end)]
        te = dfe[dfe["date"].isin([d for _, d in pairs])].set_index("date")
        pm = pd.Series(-np.mean([BASES[m](tr, te.reset_index()) for m in MEMBERS], axis=0), index=te.index)
        for I, d1 in pairs:
            p = float(pm[d1])
            rows.append({"base_date": I.strftime("%Y-%m-%d"), "date": d1.strftime("%Y-%m-%d"),
                         "grade": "경보" if p <= strong else "주의" if p <= warn else "없음",
                         "prob": round(1 / (1 + np.exp(-(a * -p + b))), 4), "score": round(p, 1)})
    return pd.DataFrame(rows, columns=["base_date", "date", "grade", "prob", "score"])


def main():
    """과제1 피처와 같은지 확인 (2024-01-01 ~ 과제1 피처 마지막 날)."""
    from drop_classifier.src.search import day_table
    from drop_classifier.src.supply_level import supply_features
    t1 = day_table().merge(fcst15(), on="date", how="left")
    t1 = pd.concat([t1.reset_index(drop=True), supply_features(t1["date"])], axis=1)
    mine = drop_features()
    m = t1[["date"] + COLS].merge(mine[["date"] + COLS], on="date", suffixes=("_t1", "_ops"))
    bad = {}
    for c in COLS:
        a, b = m[f"{c}_t1"].astype(float), m[f"{c}_ops"].astype(float)
        diff = ~((a - b).abs().fillna(0) < 1e-6) | (a.isna() != b.isna())
        if diff.any():
            bad[c] = int(diff.sum())
    print(f"비교 {len(m)}일, 다른 칸: {bad or '없음'}")
    print("내일 행:", mine.tail(1)[["date", "is_offday", "prev_day_min_smp", "rf_sky_pm", "kpx_peak", "cap_solar_gw"]].to_dict("records"))


if __name__ == "__main__":
    main()
