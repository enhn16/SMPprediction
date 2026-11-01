"""④ 팀 합의 급등 모델 재구현 v2 — 팀원 spike_simple3.py(reference/급등_코드)를 그대로 옮기고 입력만 우리 자료로 바꿈.

같게 둔 것 (팀원 코드 그대로): 피처 37개(기본 30 + 가격 계단 7)와 계산식, 부호 고정 점수(19개 입력·부호, 1차원 balanced 로지스틱
  눈금, 기울기 ≥ 0), 스플라인(표준화 → 3차·매듭 4 → 로지스틱 C=0.05 balanced), 그루터기(LightGBM 깊이 1·200그루·학습률 0.03·
  잎 20·subsample·colsample 0.8·balanced·seed 42), 세 확률의 단순 평균, 2024-01 ~ 2025-12 매달 앞으로만 검증(그 전달까지 누적,
  학습 2023-01-01~), 문턱 = 0.01 격자에서 주의 재현율 80% 최고선·경보 정밀도 50%(적중 2일 이상) 최저선(없으면 정밀도 최고선),
  최종 2023~2025 재학습 → 2026.
바꾼 것 (팀 13종): SMP = EPSIS 확정값 + data.go.kr 공표값(담당자 파일 대신), 수요예측 = data.go.kr API, 휴일 = 특일정보
  (common.calendar), 날씨 = Open-Meteo 2일 전 실행 best_match 7개 도시 기온·운량·강수(collect_extra.openmeteo).
시험: 2026-01-01 ~ 10-18 (공식) + ~10-02 (팀원 기간).

실행: python -m surge_classifier.src.team_model2   → outputs/tables/05_team2_{daily,metrics}_육지.csv, 05_team2_choice_육지.json
"""
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from common.calendar import build_calendar
from common.loaders import load_smp_actual, load_smp_forecast_api
from surge_classifier.src.collect_extra import OUT as EXTRA_RAW
from surge_classifier.src.label_days import OUT_TAB

warnings.filterwarnings("ignore")

SPIKE_RATIO, REF_DAYS = 1.25, 28
DEV_FROM, DEV_TO, VALID_FROM, TEST_FROM = "2023-01-01", "2025-12-31", "2024-01-01", "2026-01-01"
TEST, TEST_TEAM = ("2026-01-01", "2026-10-18"), ("2026-01-01", "2026-10-02")
EVENING = range(17, 22)
BASE = ["dow", "is_off", "is_holiday", "is_major", "after_off", "off_run_len", "month_sin", "month_cos", "ref_max",
        "prev_max", "prev_max_rel", "prev_mean_rel", "prev_evening_rel", "max7_rel", "vol7", "spikes_28d", "spikes_90d",
        "days_since_spike", "prev_dem_max", "prev_dem_rel28", "same_type_dem_rel28", "dem_trend7"]
WEATHER = ["wx_tmax", "wx_tmin", "wx_tmean", "wx_hdd", "wx_cdd", "wx_tchg", "wx_cloud", "wx_precip"]
LADDER = ["peak60_rel", "near_peak_28", "prev_max_vs_q90", "prev_high_hours", "ref_change28", "ref_vs_year", "prev_excess"]
FEATS = BASE + WEATHER + LADDER
KEYS = ["sign", "spline", "stump"]
NAMES = {"sign": "부호 고정 점수", "spline": "스플라인 로지스틱", "stump": "LightGBM 그루터기", "avg": "세 모델 평균"}
SIGNS = {
    "prev_max_rel": +1, "prev_evening_rel": +1, "max7_rel": +1, "vol7": +1, "spikes_28d": +1, "spikes_90d": +1,
    "days_since_spike": -1, "prev_dem_rel28": +1, "same_type_dem_rel28": +1, "dem_trend7": +1,
    "is_off": -1, "is_holiday": -1, "wx_hdd": +1, "wx_cdd": +1,
    "peak60_rel": +1, "near_peak_28": +1, "prev_max_vs_q90": +1, "prev_high_hours": +1, "prev_excess": +1,
}
STUMP = dict(n_estimators=200, learning_rate=0.03, max_depth=1, num_leaves=2, min_child_samples=20,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", random_state=42, verbose=-1)
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


# ---------------------------------------------------------------- 입력 (팀원 data_spike.build_daily + add_ladder와 같은 식)
def load_hourly() -> pd.DataFrame:
    s = load_smp_actual("육지")[["date", "hour", "smp"]]
    f = load_smp_forecast_api()
    f = f[f["region"] == "육지"][["date", "hour", "demand_fcst"]]
    h = s.merge(f, on=["date", "hour"], how="left").rename(columns={"hour": "hour_n", "smp": "smp_land",
                                                                    "demand_fcst": "demand_fcst_land"})
    h["date"] = h["date"].dt.strftime("%Y-%m-%d")
    return h.sort_values(["date", "hour_n"]).reset_index(drop=True)


def weather_daily() -> pd.DataFrame:
    from common.collect.openmeteo_prevruns import CITIES
    rows = []
    for p in sorted((EXTRA_RAW / "openmeteo_best").glob("prev2_best_*.json"), key=lambda p: p.stat().st_mtime):
        for site, loc in zip(CITIES, json.loads(p.read_text(encoding="utf-8"))):
            w = pd.DataFrame(loc["hourly"])
            w.columns = [c.replace("_previous_day2", "") for c in w.columns]
            rows.append(w.assign(site=site))
    w = pd.concat(rows).drop_duplicates(["time", "site"], keep="last")   # 같은 시각은 나중에 받은 파일 값
    w["date"] = w["time"].str[:10]
    site = w.groupby(["date", "site"]).agg(tmax=("temperature_2m", "max"), tmin=("temperature_2m", "min"),
                                           tmean=("temperature_2m", "mean"), cloud=("cloud_cover", "mean"),
                                           precip=("precipitation", "sum"), n=("temperature_2m", "count"))
    site = site[site["n"] >= 20]
    g = site.groupby("date")
    out = pd.DataFrame({"wx_tmax": g["tmax"].mean(), "wx_tmin": g["tmin"].mean(), "wx_tmean": g["tmean"].mean(),
                        "wx_cloud": g["cloud"].mean(), "wx_precip": g["precip"].mean()})
    out["wx_hdd"] = (18 - out["wx_tmean"]).clip(lower=0)
    out["wx_cdd"] = (out["wx_tmean"] - 24).clip(lower=0)
    return out


def build_daily(extend_days: int = 1) -> pd.DataFrame:
    """대상일별 행. 마지막 SMP 날 + extend_days까지 (운영: 내일 행)."""
    h = load_hourly()
    g = h.groupby("date")
    day = pd.DataFrame({"n": g["smp_land"].count(), "day_max": g["smp_land"].max(), "day_mean": g["smp_land"].mean(),
                        "dem_max": g["demand_fcst_land"].max(),
                        "eve_max": h[h["hour_n"].isin(EVENING)].groupby("date")["smp_land"].max()})
    end = pd.Timestamp(h["date"].max()) + pd.Timedelta(days=extend_days)
    d = pd.DataFrame({"date": pd.date_range(h["date"].min(), end, freq="D")})
    d["date_s"] = d["date"].dt.strftime("%Y-%m-%d")
    d = d.merge(day, left_on="date_s", right_index=True, how="left")
    full = d["n"] == 24
    for c in ("day_max", "day_mean", "eve_max"):
        d.loc[~full, c] = np.nan
    d["ref_max"] = d["day_max"].shift(1).rolling(REF_DAYS, min_periods=14).median()
    d["y"] = np.where(full & d["ref_max"].notna(), (d["day_max"] >= SPIKE_RATIO * d["ref_max"]).astype(float), np.nan)

    cal = build_calendar(d["date"].min(), d["date"].max()).set_index("date")
    cal.index = pd.to_datetime(cal.index).strftime("%Y-%m-%d")
    d["holiday_name"] = d["date_s"].map(cal["holiday_name"].where(cal["is_holiday"].astype(bool))).fillna("")
    d["is_holiday"] = d["date_s"].map(cal["is_holiday"]).fillna(False).astype(int)
    d["dow"] = d["date"].dt.weekday
    d["is_off"] = ((d["dow"] >= 5) | (d["is_holiday"] == 1)).astype(int)
    d["after_off"] = ((d["is_off"] == 0) & (d["is_off"].shift(1) == 1)).astype(int)
    run = (d["is_off"] != d["is_off"].shift()).cumsum()
    d["off_run_len"] = np.where(d["is_off"] == 1, d.groupby(run)["is_off"].transform("size"), 0)
    major = d.groupby(run)["holiday_name"].transform(lambda s: s.str.contains("설날|추석").any())
    d["is_major"] = ((d["is_off"] == 1) & major).astype(int)
    m = d["date"].dt.month
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * (m - 1) / 12), np.cos(2 * np.pi * (m - 1) / 12)

    d["prev_max"] = d["day_max"].shift(1)
    d["prev_max_rel"] = d["prev_max"] / d["ref_max"]
    d["prev_mean_rel"] = d["day_mean"].shift(1) / d["ref_max"]
    d["prev_evening_rel"] = d["eve_max"].shift(1) / d["ref_max"]
    d["max7_rel"] = d["day_max"].shift(1).rolling(7, min_periods=4).max() / d["ref_max"]
    r7 = d["day_max"].shift(1).rolling(7, min_periods=4)
    d["vol7"] = r7.std() / r7.mean()
    yl = d["y"].shift(1)
    d["spikes_28d"] = yl.rolling(28, min_periods=14).sum()
    d["spikes_90d"] = yl.rolling(90, min_periods=45).sum()
    last, since = np.nan, []
    for i, v in enumerate(d["y"].values):
        since.append(i - last if last == last else np.nan)
        if v == 1:
            last = i
    d["days_since_spike"] = np.clip(since, None, 365)
    dem28 = d["dem_max"].shift(1).rolling(28, min_periods=14).mean()
    d["prev_dem_max"] = d["dem_max"].shift(1)
    d["prev_dem_rel28"] = d["prev_dem_max"] / dem28
    d["dem_trend7"] = d["prev_dem_max"] / d["dem_max"].shift(8)
    last_type, st = {0: np.nan, 1: np.nan}, []
    for r in d.itertuples():
        st.append(last_type[r.is_off])
        if r.dem_max == r.dem_max:
            last_type[r.is_off] = r.dem_max
    d["same_type_dem_rel28"] = np.array(st, dtype=float) / dem28

    d = d.merge(weather_daily(), left_on="date_s", right_index=True, how="left")
    d["wx_tchg"] = d["wx_tmean"] - d["wx_tmean"].shift(1)

    # 가격 계단 (팀원 spike_improve.add_ladder)
    dm = d["day_max"]
    prev = dm.shift(1)
    d["peak60_rel"] = prev.rolling(60, min_periods=30).max() / d["ref_max"]
    near = (dm >= 0.97 * dm.shift(1).rolling(60, min_periods=30).max()).astype(float).where(dm.notna())
    d["near_peak_28"] = near.shift(1).rolling(28, min_periods=14).sum()
    d["prev_max_vs_q90"] = prev / prev.rolling(90, min_periods=45).quantile(0.9)
    hh = h.pivot_table(index="date", columns="hour_n", values="smp_land")
    ref_prev = d.set_index("date_s")["ref_max"].shift(1)
    high = hh.ge(1.15 * ref_prev.reindex(hh.index), axis=0).sum(axis=1).where(hh.notna().sum(axis=1) == 24)
    d["prev_high_hours"] = d["date_s"].map(high).shift(1)
    d["ref_change28"] = d["ref_max"] / d["ref_max"].shift(28)
    d["ref_vs_year"] = d["ref_max"] / prev.rolling(365, min_periods=180).median()
    d["prev_excess"] = prev - d["ref_max"]
    return d


# ---------------------------------------------------------------- 세 모델 (팀원 코드 그대로)
class SignScore:
    def fit(self, x, y):
        cols = list(SIGNS)
        self.med = x[cols].median()
        z = x[cols].fillna(self.med)
        self.mu, self.sd = z.mean(), z.std().replace(0, 1)
        self.cal = LogisticRegression(class_weight="balanced").fit(self.raw(x).reshape(-1, 1), y)
        if self.cal.coef_[0, 0] < 0:
            self.cal.coef_[0, 0] = 0.0
        return self

    def raw(self, x):
        z = (x[list(SIGNS)].fillna(self.med) - self.mu) / self.sd
        return (z * pd.Series(SIGNS)).mean(axis=1).values

    def predict(self, x):
        return self.cal.predict_proba(self.raw(x).reshape(-1, 1))[:, 1]


class Spline:
    def fit(self, x, y):
        self.med = x.median()
        self.m = make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3, extrapolation="constant"),
                               LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))
        self.m.fit(x.fillna(self.med), y)
        return self

    def predict(self, x):
        return self.m.predict_proba(x.fillna(self.med))[:, 1]


class Stump:
    def fit(self, x, y):
        self.m = lgb.LGBMClassifier(**STUMP).fit(x, y)
        return self

    def predict(self, x):
        return self.m.predict_proba(x)[:, 1]


MAKE = {"sign": SignScore, "spline": Spline, "stump": Stump}


def fit_all(train):
    y = train["y"].astype(int)
    return {k: MAKE[k]().fit(train[FEATS], y) for k in KEYS}


def predict(models, x) -> pd.DataFrame:
    out = pd.DataFrame({k: m.predict(x[FEATS]) for k, m in models.items()}, index=x.index)
    out["avg"] = out[KEYS].mean(axis=1)
    return out


# ---------------------------------------------------------------- 문턱 (팀원 step3_ensemble 그대로)
def counts(y, p, t):
    a = p >= t
    return int((a & (y == 1)).sum()), int((a & (y == 0)).sum())


def caution_line(y, p, recall=0.8):
    need = np.ceil(recall * y.sum())
    ok = [t for t in GRID if counts(y, p, t)[0] >= need]
    return float(max(ok)) if ok else float(GRID[0])


def alert_line(y, p, precision=0.5):
    best = None
    for t in GRID:
        tp, fp = counts(y, p, t)
        if tp >= 2:
            prec = tp / (tp + fp)
            if prec >= precision:
                return float(t)
            if best is None or prec > best[0]:
                best = (prec, float(t))
    return best[1] if best else 0.5


def walk_forward(d) -> pd.DataFrame:
    lab = d[d["y"].notna()]
    dev = lab[(lab["date_s"] >= DEV_FROM) & (lab["date_s"] <= DEV_TO)]
    val = dev[dev["date_s"] >= VALID_FROM]
    parts = []
    for month, rows in val.groupby(val["date"].dt.to_period("M")):
        parts.append(predict(fit_all(dev[dev["date"] < month.start_time]), rows))
    return pd.concat(parts)


def final_models(d):
    lab = d[d["y"].notna()]
    return fit_all(lab[(lab["date_s"] >= DEV_FROM) & (lab["date_s"] <= DEV_TO)])


def metric_rows(name, y, p, th):
    y, p = np.asarray(y, int), np.asarray(p, float)
    base = {"구간": name, "일수": len(y), "급등일": int(y.sum()), "PR-AUC": average_precision_score(y, p),
            "PR-AUC 배수": average_precision_score(y, p) / y.mean(), "ROC-AUC": roc_auc_score(y, p),
            "상위 10일 적중": int(y[np.argsort(-p, kind="stable")[:10]].sum())}
    out = []
    for line in ("주의", "경보"):
        tp, fp = counts(y, p, th[line])
        out.append({**base, "기준선": line, "문턱": th[line], "적중": tp, "놓침": int(y.sum()) - tp, "헛경보": fp,
                    "재현율": tp / max(y.sum(), 1), "정밀도": tp / max(tp + fp, 1)})
    return out


def main():
    d = build_daily()
    val = walk_forward(d)
    v = d.loc[val.index]
    th = {k: {"주의": caution_line(v["y"].values, val[k].values), "경보": alert_line(v["y"].values, val[k].values)}
          for k in KEYS + ["avg"]}
    te_rows = d[(d["date_s"] >= TEST_FROM) & d[FEATS[0]].notna()]
    test = predict(final_models(d), te_rows)
    daily = pd.concat([val.assign(period="검증 2024~2025"), test.assign(period="시험 2026")])
    daily = d.loc[daily.index, ["date_s", "y", "is_off", "day_max", "ref_max"]].join(daily).rename(columns={"date_s": "date"})
    daily.to_csv(OUT_TAB / "05_team2_daily_육지.csv", index=False, encoding="utf-8-sig")

    rows = []
    sets = {"검증 2024~2025": daily[daily["period"] == "검증 2024~2025"]}
    for nm, (a, b) in (("시험 2026 (공식 ~10-18)", TEST), ("시험 2026 (팀원 기간 ~10-02)", TEST_TEAM)):
        sets[nm] = daily[(daily["date"] >= a) & (daily["date"] <= b)].dropna(subset=["y"])
    for nm, part in sets.items():
        for k in KEYS + ["avg"]:
            rows += [{"모델": NAMES[k], **r} for r in metric_rows(nm, part["y"], part[k], th[k])]
    met = pd.DataFrame(rows)
    met.to_csv(OUT_TAB / "05_team2_metrics_육지.csv", index=False, encoding="utf-8-sig")
    (OUT_TAB / "05_team2_choice_육지.json").write_text(json.dumps(
        {"모델": "팀 합의 급등 모델 v2 (팀원 spike_simple3 재구현, 입력 = 팀 13종)", "피처": FEATS, "부호": SIGNS,
         "그루터기": STUMP, "스플라인": "StandardScaler → SplineTransformer(n_knots=4, degree=3, constant) → LogisticRegression(C=0.05, balanced)",
         "결합": "세 확률(0~1) 단순 평균", "문턱": th, "검증": "2024-01 ~ 2025-12 매달 앞으로만 (학습 2023-01-01 ~ 전달 말)",
         "최종 학습": [DEV_FROM, DEV_TO]}, ensure_ascii=False, indent=1), encoding="utf-8")
    e = met[met["모델"] == NAMES["avg"]]
    print(e.round(3).to_string(index=False))
    print("문턱:", {k: v for k, v in th.items()})
    return daily, met, th


if __name__ == "__main__":
    main()
