"""한 달 예측 mx3 — 시간별 표 모델 걷기 전진 (기준일 2024-01-01 ~, 분기 첫날 재학습: 시간별 행이 24배라 mx_seq와 같은 주기). mx_tab과 같은 규칙을 시간별로.

학습: 대상일 2023-01-01 이후 중 재학습일 전에 실적이 확정된 행 (date < 재학습일). 기준일은 하루 걸러 씀 (이웃 기준일은 거의 같은 표본).
목표는 '일 기준 수준 대비 비율' (시간별 same4는 휴무일 한낮에 0 근처라 기준으로 쓰지 않음): 예측 = 기준 × 비율
  hlgb_r : 기준 = 일평균 same4
  hlgb_a : 기준 = 일평균 same4 × √(일평균 연료 정규화 계절비)
  hlgbx_r·hlgbx_a·hresx_{tag} : 같은 모델 + 추가 피처 묶음(mx3_data --extra: 최근 분위수·급락 빈도·작년 하루 모양·시간별 수요 계절비·
                예상 순부하·태양광 비율·변동성·가까운 지평 전망/예보·유연탄·LNG 3개월 변화). 묶음 전체를 한 번에 비교 (하나씩 넣고 빼지 않음)
  hq10·hq90 : hlgb_r와 같은 입력·목표(same4 비율)의 분위수 회귀 (10%·90%) → 범위 후보 (점예측과의 차를 폭으로 옮기고 칸별 보정)
  hshape·hshapex : 하루 모양 전용 — 목표 = 실적 ÷ 그날 실제 일평균 (수준이 섞이지 않은 모양). 저장값은 모양(비율), 쓸 때 하루 평균 1로 정규화
  hres_{tag} : Chronos 시간별 중앙값(tag)의 잔차 y − Chronos 를 같은 피처 + Chronos 파생으로 (최종 = Chronos + λ × 잔차)
수준 피처는 기준으로 나눈 값으로 넣는다 (n_*). 달력·하루 모양·계절비·태양광·수요·연료는 그대로.
출력: data/processed/smp_forecast_app/mx/mx3_tab_{name}.csv.gz [base_date, H, hour, pred]
실행: python -m smp_forecast_app.src.mx3_tab hlgb_r hlgb_a hres_hc2s ...   (이미 있으면 건너뜀)
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

from common.config import SEED
from smp_forecast_app.src.mx3_data import load_hourly
from smp_forecast_app.src.mx_features import MX

warnings.filterwarnings("ignore")
START, TRAIN_FROM = "2024-01-01", "2023-01-01"
LGB = dict(n_estimators=600, learning_rate=0.04, num_leaves=63, min_child_samples=200, subsample=0.7, subsample_freq=1,
           colsample_bytree=0.7, objective="l1", random_state=SEED, verbose=-1, n_jobs=6)
CAL = ["hour", "H", "dow", "month", "doy_sin", "doy_cos", "offday", "hol_cls", "off_block_len", "off_block_day", "is_myeongjeol",
       "prev_offday", "next_offday", "dom", "mon_cross"]
LEVEL = ["h_same4", "h_same8", "h_l7", "h_l28", "h_ly", "mean_same4", "mean_same8", "mean_l7", "mean_l28", "mean_l91", "mean_ly",
         "mean_fuelimp", "mid_same4", "eve_same4", "midmin_same4"]
XTRA = ["x_q10", "x_q90", "x_min28", "x_low28", "x_sd8", "x_ly_shape", "x_seasM", "x_dem_seas", "x_netload", "x_pv_ratio", "x_vol28",
        "kpx_peak_rel", "rf_sky_pm", "rf_pop_pm", "fc_coal", "fc_lng_chg3"]
RAW = ["h_shape4", "h_shape28", "h_seasN", "mean_seasN", "mean_seasM", "mean_trend", "mean_r_now", "seas_dem_m", "dem_l28",
       "pv_exp", "pv_share", "pvmid_exp", "h_pv_exp", "cap_solar", "fc_lng", "fc_lng_chg1"]


def base_level(d, kind):
    b = d["mean_same4"].clip(lower=10)
    if kind == "a":
        b = b * np.sqrt(d["mean_seasN"].clip(0.6, 1.6).fillna(1))
    return b


def design(d, b):
    X = d[CAL + RAW + [c for c in XTRA if c in d]].copy()
    for c in ("x_q10", "x_q90", "x_min28"):        # 수준 피처는 기준으로 나눔
        if c in X:
            X[c] = X[c] / b
    for c in LEVEL:
        X[f"n_{c}"] = d[c] / b
    X["lvl"] = b
    return X


def chronos_hourly(tag):
    return pd.read_csv(MX / f"mx3_chr_{tag}.csv.gz", parse_dates=["base_date"])[["base_date", "H", "hour", "q50", "q10", "q90"]]


def xy(d, name, chr_=None):
    """모델 name의 (X, 목표, 쓸 행, 기준, 방식). 방식: ratio → 기준 × 예측, raw → 예측 그대로(잔차·모양).
    hres_*는 chr_ (Chronos 시간별 [base_date, H, hour, q10, q50, q90])를 붙여 씀."""
    if name.startswith("hres"):
        d = d.merge(chr_, on=["base_date", "H", "hour"], how="left")
        b = base_level(d, "r")
        X = design(d, b)
        X["c_ratio"], X["c_width"] = d["q50"] / b, (d["q90"] - d["q10"]) / b
        X["c_vs_same4"] = d["q50"] - d["h_same4"]
        return X, d["y"] - d["q50"], d["q50"].notna(), b, "raw"
    if name.startswith("hshape"):
        b = base_level(d, "r")
        ya = d.groupby(["base_date", "H"])["y"].transform("mean")
        return design(d, b), (d["y"] / ya.clip(lower=10)).clip(-0.5, 3), b.notna(), b, "raw"
    b = base_level(d, name[-1])
    return design(d, b), (d["y"] / b).clip(-0.5, 3), b.notna(), b, "ratio"


def params(name):
    return dict(LGB, objective="quantile", alpha=int(name[2:]) / 100) if name.startswith("hq") else LGB


def run(name):
    from lightgbm import LGBMRegressor
    d = load_hourly(extra=name.split("_")[0].endswith("x"))     # hlgbx_* · hresx_* = 추가 피처 포함
    d = d[d["base_date"] >= "2022-12-01"].reset_index(drop=True)
    X, yt, ok, b, mode = xy(d, name, chronos_hourly(name.split("_", 1)[1]) if name.startswith("hres") else None)
    alt = (d["base_date"] - pd.Timestamp("2022-12-01")).dt.days % 2 == 0
    pred = pd.Series(np.nan, index=d.index)
    ps = list(pd.date_range(START, d["base_date"].max(), freq="QS")) + [pd.Timestamp("2100-01-01")]
    t0 = time.time()
    for a, e in zip(ps[:-1], ps[1:]):
        tr = (d["date"] >= TRAIN_FROM) & (d["date"] < a) & yt.notna() & ok & alt
        te = (d["base_date"] >= a) & (d["base_date"] < e) & ok
        if te.sum() == 0:
            continue
        pred[te] = LGBMRegressor(**params(name)).fit(X[tr], yt[tr]).predict(X[te])
        print(f"{name} {a.date()} 학습 {tr.sum():,}행 ({time.time()-t0:.0f}s)", flush=True)
    out = d[["base_date", "H", "hour"]].copy()
    out["pred"] = pred * b if mode == "ratio" else pred
    out = out[out["base_date"] >= START]
    tmp = MX / f"mx3_tab_{name}.tmp.csv.gz"           # 다 쓴 뒤 이름 바꿈 (쓰는 중에 다른 프로세스가 읽지 않게)
    out.round(3).to_csv(tmp, index=False, compression="gzip")
    tmp.replace(MX / f"mx3_tab_{name}.csv.gz")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        if (MX / f"mx3_tab_{n}.csv.gz").exists():
            print(f"{n}: 이미 있음 → 건너뜀")
            continue
        run(n)
