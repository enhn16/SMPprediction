"""일주일 예측 — Chronos-2 small 잔차 보정 (Chronos가 모르는 외부 정보로 'Chronos가 틀린 만큼'을 예측).

  최종 예측 = Chronos-2 small + λ × 잔차 예측   (λ는 wz_ens가 2025 상반기로 정함)
  잔차 = 실적 − Chronos-2 small, 모델 = LightGBM(L1), 며칠 뒤 H마다 따로, 매월 1일 재학습 (실적은 재학습일 − H일까지)
  학습 이력: Chronos 과거 예측 2023~2024 (wz_chronos --hist) + 2025~ (wzoo/c2s)
피처 묶음 (2025 하반기에 wz_ens가 비교)
  외부만   : 달력·휴무 + 연료비·태양광 설비·주간 수요전망·기온/강수/일사/운량 예보 + Chronos 파생
  계절제외 : '기본+전망·날씨' 전체에서 월·연중 위치만 뺀 것 + Chronos 파생
  전체피처 : '기본+전망·날씨' 전체 + Chronos 파생
  Chronos 파생: 예측값, 분위 폭(0.9−0.1), 어제·최근 같은 유형일·28일 평균 대비 차이
출력: data/processed/smp_forecast_app/wzoo_resid/{묶음}.csv.gz [date, hour, H, rhat]
실행: python -m smp_forecast_app.src.wz_resid   (먼저 python -m smp_forecast_app.src.wz_chronos --hist c2s)
"""
import time
import warnings

import lightgbm as lgb
import pandas as pd

from common.config import PROCESSED_DIR
from smp_forecast_app.src.model import PARAMS
from smp_forecast_app.src.wz_chronos import HIST
from smp_forecast_app.src.wz_common import COLS, HS, feats, load_pred, rolling_fit

warnings.filterwarnings("ignore")

RES = PROCESSED_DIR / "smp_forecast_app" / "wzoo_resid"
CD = ["pred", "c_spread", "c_vs_d1", "c_vs_same", "c_vs_ma28"]
EXO = ["hour", "dow", "offday", "daytype", "prev_offday", "next_offday", "type_change", "is_myeongjeol",
       "fc_lng", "fc_coal", "cap_solar", "kpx_peak", "kpx_peak_rel", "kpx_peak_vs_d1",
       "rf_sky_pm", "rf_pop_pm", "rf_ta_pm", "rf_sky_am", "om_rad_mid", "om_cloud_mid"]
SETS = {"외부만": EXO + CD,
        "계절제외": [c for c in COLS if c not in ("month", "doy_sin", "doy_cos")] + CD,
        "전체피처": COLS + CD}
LGB = dict(PARAMS, objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=200,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)


def chronos_all():
    """Chronos-2 small 예측 2023~ (과거 + 백테스트)."""
    hist = pd.read_csv(HIST / "c2s.csv.gz", parse_dates=["date"])
    return pd.concat([hist, load_pred("c2s")], ignore_index=True)


def table(H, c=None, f=None):
    """H일 뒤 피처 표(f, 기본 저장본) + Chronos 파생 + 잔차 r. c = Chronos-2 small [date, hour, H, pred, qlo, qhi]."""
    c = chronos_all() if c is None else c
    f = feats(H) if f is None else f
    d = f.merge(c[c["H"] == H][["date", "hour", "pred", "qlo", "qhi"]], on=["date", "hour"])
    d["c_spread"] = d["qhi"] - d["qlo"]
    d["c_vs_d1"] = d["pred"] - d["s_d1"]
    d["c_vs_same"] = d["pred"] - d["s_same_h"]
    d["c_vs_ma28"] = d["pred"] - d["s_ma28"]
    d["r"] = d["y"] - d["pred"]
    return d


def fit_predict(cols):
    def f(tr, te):
        tr = tr[tr["r"].notna()]
        return lgb.LGBMRegressor(**LGB).fit(tr[cols], tr["r"]).predict(te[cols])
    return f


def main():
    RES.mkdir(parents=True, exist_ok=True)
    c = chronos_all()
    t0, out = time.time(), {k: [] for k in SETS}
    for H in HS:
        d = table(H, c)
        for k, cols in SETS.items():
            out[k].append(rolling_fit(d, H, "MS", fit_predict(cols)).rename(columns={"pred": "rhat"}))
        print(f"잔차 보정 H={H} ({time.time()-t0:.0f}s)", flush=True)
    for k, parts in out.items():
        pd.concat(parts).round(3).to_csv(RES / f"{k}.csv.gz", index=False, encoding="utf-8")
    print(f"저장: {RES} ({', '.join(SETS)})", flush=True)


if __name__ == "__main__":
    main()
