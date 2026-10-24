"""제주 일주일 예측 — 잔차 보정 ('기준 예측이 틀린 만큼'을 제주 정보로 예측). 육지 wz_resid와 같은 방식.

  최종 = 기준 예측 + λ × 잔차 예측   (λ는 jz_ens가 2025 상반기로 정함)
기준 예측 (학습 이력 2023~)
  land  : 육지 일주일 예측 (2025~ 육지 최종, 2023~2024 육지 Chronos-2 small)
  jc2sx : 제주 Chronos-2 small + 육지 공변량      jc2s : 제주 Chronos-2 small
피처 묶음
  제주외부 : 달력 + 제주 수요예측·날씨 예보·Open-Meteo + 제주−육지 관계 + 기준 파생
  전체     : jz_tab COLS 전체 + 기준 파생
  제주외부+계통 · 전체+계통 : 위 묶음 + 제주 5분 계통 실적(판단일 14:25까지 풍력·태양광·수요·순부하, 2024-01~) (2026-10-25 추가)
  기준 파생: 기준 예측값, 분위 폭, 어제·같은 유형일·28일 평균 대비 차이, 육지 예측과의 차이
목적함수: l1 (육지와 같음), l2 (급락처럼 큰 오차를 더 따라가도록)
출력: data/processed/smp_forecast_app/jeju/jzoo_resid/{기준}_{묶음}_{목적}.csv.gz [date, hour, H, rhat]
실행: python -m smp_forecast_app.src.jz_resid [묶음 이름 …]   (먼저 jz_chronos --hist jc2s jc2sx, 묶음을 주면 그 묶음만)
"""
import time
import warnings

import lightgbm as lgb
import pandas as pd

from smp_forecast_app.src.jz_common import HS, load_pred, rolling_fit
from smp_forecast_app.src.jz_features import CAL, DEM, JDIR, OM, SPREAD, SUKUB, WEATHER
from smp_forecast_app.src.jz_tab import COLS, table
from smp_forecast_app.src.model import PARAMS

warnings.filterwarnings("ignore")

RES = JDIR / "jzoo_resid"
BASES = ["land", "jc2sx", "jc2s"]
CD = ["b_pred", "b_spread", "b_vs_d1", "b_vs_same", "b_vs_ma28", "b_vs_land"]
SETS = {"제주외부": CAL + DEM + SPREAD + WEATHER + OM + CD, "전체": COLS + CD,
        "제주외부+계통": CAL + DEM + SPREAD + WEATHER + OM + SUKUB + CD, "전체+계통": COLS + SUKUB + CD}
OBJ = ("l1", "l2")
LGB = dict(PARAMS, n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=200,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=3)


def base_all(name):
    """기준 예측 [date, hour, H, b_pred, b_lo, b_hi] 2023~."""
    if name == "land":
        from smp_forecast_app.src.jz_common import land_pred
        d = land_pred()
        return d.rename(columns={"L_base": "b_pred", "L_qlo": "b_lo", "L_qhi": "b_hi"})[["date", "hour", "H", "b_pred", "b_lo", "b_hi"]]
    d = pd.concat([load_pred(name, hist=True), load_pred(name)], ignore_index=True)
    return d.rename(columns={"pred": "b_pred", "qlo": "b_lo", "qhi": "b_hi"})


def rtable(H, b):
    d = table(H).merge(b[b["H"] == H].drop(columns="H"), on=["date", "hour"])
    d["b_spread"] = d["b_hi"] - d["b_lo"]
    d["b_vs_d1"] = d["b_pred"] - d["s_d1"]
    d["b_vs_same"] = d["b_pred"] - d["s_same_h"]
    d["b_vs_ma28"] = d["b_pred"] - d["s_ma28"]
    d["b_vs_land"] = d["b_pred"] - d["L_base"]
    d["r"] = d["y"] - d["b_pred"]
    return d


def fit_predict(cols, obj):
    def f(tr, te):
        tr = tr[tr["r"].notna()]
        return lgb.LGBMRegressor(**dict(LGB, objective=obj)).fit(tr[cols], tr["r"]).predict(te[cols])
    return f


def main(sets=None):
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for bn in BASES:
        b = base_all(bn)
        out = {(k, o): [] for k in (sets or SETS) for o in OBJ}
        for H in HS:
            d = rtable(H, b)
            for (k, o) in out:
                out[(k, o)].append(rolling_fit(d, H, "MS", fit_predict(SETS[k], o)).rename(columns={"pred": "rhat"}))
            print(f"잔차 보정 {bn} H={H} ({time.time()-t0:.0f}s)", flush=True)
        for (k, o), parts in out.items():
            pd.concat(parts).round(3).to_csv(RES / f"{bn}_{k}_{o}.csv.gz", index=False, encoding="utf-8")
    print(f"저장: {RES}", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
