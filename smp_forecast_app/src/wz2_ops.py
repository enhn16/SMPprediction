"""일주일 예측 2차 (21시) — 운영 함수. 모델·근거: docs/WEEK21.md, outputs/tables/WZ2_choice.json

  weekly(bases, stage="21h") -> DataFrame [base_date, H, date, hour, pred, lo, hi, offday, stage, pub, c2s, rhat]
    기준일 I마다 I+1 공표 SMP(24시간)가 있으면 2차, 없으면 1차 운영 기록(ops/hourly_log.csv)의 I, H=2~7 값 (pub=0).
    2차 = Chronos-2 small(I+1까지 SMP, ops.chronos_rows([I+1])를 H+1로) + λ × 2차 잔차 보정, 범위 = WZ2_choice 지평별 방법.
    기록 파일은 쓰지 않는다 (운영 기록은 총괄 ops.py).
  train()  2차 잔차 보정 H별 LightGBM을 공표 실적 전부로 학습 → data/processed/smp_forecast_app/wz2/wz2_resid.pkl
    (일요일 1차 train과 함께. 모델이 없으면 weekly가 처음 한 번 학습)
판단 정보: 1차(I일 15시) 정보 + I+1 공표 SMP. 날씨·전망·연료비는 1차와 같은 build(H).
실행 (시험): SMP_NO_PUSH=1 python -m smp_forecast_app.src.wz2_ops [기준일 ...]   → 결과 요약만 출력
"""
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from common.config import ROOT
from common.loaders import load_smp_actual
from smp_forecast_app.src.intervals import add_keys
from smp_forecast_app.src.model import cqr
from smp_forecast_app.src.wz2_backtest import HS2, SETS2, W2, shift_chronos

TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
MODEL = W2 / "wz2_resid.pkl"
COLS_OUT = ["base_date", "H", "date", "hour", "pred", "lo", "hi", "offday", "stage", "pub", "c2s", "rhat"]


def _choice():
    return json.loads((TAB / "WZ2_choice.json").read_text(encoding="utf-8"))


def _ops_log():
    from smp_forecast_app.src.ops import OPS
    p = OPS / "hourly_log.csv"
    return pd.read_csv(p, parse_dates=["date", "base_date"]) if p.exists() else pd.DataFrame()


def _chronos_hist2():
    """2차 잔차 학습·범위 이력용 당긴 Chronos [date, hour, H, pred, qlo, qhi(c2s), qlo_c2, qhi_c2]:
    백테스트 캐시(wz2/) + 1차 운영 기록(c2s·Chronos-2 분위수)을 하루 당긴 것."""
    c = pd.read_csv(W2 / "c2s.csv.gz", parse_dates=["date"])
    q = pd.read_csv(W2 / "c2.csv.gz", parse_dates=["date"]).rename(columns={"qlo": "qlo_c2", "qhi": "qhi_c2"})
    c = c.merge(q[["date", "hour", "H", "qlo_c2", "qhi_c2"]], on=["date", "hour", "H"], how="left")
    ol = _ops_log()
    if len(ol) and "c2s" in ol:
        ol = ol[ol["c2s"].notna()]
        ol = pd.DataFrame({"date": ol["date"], "hour": ol["hour"], "H": ol["H"], "pred": ol["c2s"], "qlo": ol["c2s_qlo"],
                           "qhi": ol["c2s_qhi"], "qlo_c2": ol["qlo"], "qhi_c2": ol["qhi"]})
        c = pd.concat([c, shift_chronos(ol)]).drop_duplicates(["date", "hour", "H"], keep="last")
    return c


def train(until=None, path=None):
    """until(검증용): 백테스트처럼 대상일 < until − (H−1)일 실적만 학습. 기본 = 공표 실적 전부."""
    import joblib
    import lightgbm as lgb
    from smp_forecast_app.src.features import build
    from smp_forecast_app.src.wz_resid import LGB, table
    t0 = time.time()
    path = path or MODEL
    cols = SETS2[_choice()["후보 코드"]]
    c = _chronos_hist2()
    models = {}
    for H in HS2:
        d = table(H, c, build(H))          # (b): 1차 '외부만' 정의 (Chronos 파생은 I일 SMP 기준)
        d = d[d["r"].notna() & (d["date"] >= "2022-01-01")]
        if until is not None:
            d = d[d["date"] < pd.Timestamp(until) - pd.Timedelta(days=H - 1)]
        models[H] = lgb.LGBMRegressor(**LGB).fit(d[cols], d["r"])
    joblib.dump({"models": models, "cols": cols, "trained_at": datetime.now().isoformat(timespec="minutes"),
                 "last_date": str(d["date"].max().date())}, path)
    print(f"2차 잔차 보정 학습 → {path} (실적 ~{d['date'].max().date()}, {time.time()-t0:.0f}s)", flush=True)


def _interval(nh, H, method, hist, act, cal):
    """지평 H의 범위. nh = [date, hour, pred, qlo_c2, qhi_c2, rhat(λ 곱한 값)]."""
    if method == "Chronos-2 분위수 + 잔차 이동":
        return (nh["qlo_c2"] + nh["rhat_l"]).clip(lower=0).values, (nh["qhi_c2"] + nh["rhat_l"]).values
    if method == "Chronos-2 분위수 그대로":
        return nh["qlo_c2"].clip(lower=0).values, nh["qhi_c2"].values
    if method.startswith("보정분위수회귀(Chronos-2 분위수)"):
        hh = hist[(hist["H"] == H) & ~hist["date"].isin(nh["date"])].merge(act, on=["date", "hour"])
        hh = hh.assign(qlo=hh["qlo_c2"], qhi=hh["qhi_c2"])[["date", "hour", "pred", "qlo", "qhi", "y"]]
        d = pd.concat([hh, nh.assign(qlo=nh["qlo_c2"], qhi=nh["qhi_c2"], y=np.nan)[["date", "hour", "pred", "qlo", "qhi", "y"]]],
                      ignore_index=True)
        d["offday"] = d["date"].map(cal).astype(bool)
        d = add_keys(d.sort_values(["date", "hour"]).reset_index(drop=True))
        mask = d["y"].isna().values
        m = re.search(r"갱신(\d+)일", method)
        lo, hi = cqr(d, mask, window=int(m.group(1)) if m else None, key="k_band_off" if "시간대묶음" in method else None, gap=H - 1)
        g = d[mask][["date", "hour"]].assign(lo=np.clip(lo, 0, None), hi=hi)
        r = nh[["date", "hour"]].merge(g, on=["date", "hour"], how="left")
        return r["lo"].values, r["hi"].values
    raise ValueError(f"운영에서 지원하지 않는 범위 방법: {method}")


def weekly(bases, stage="21h"):
    import joblib
    from common.calendar import build_calendar
    from smp_forecast_app.src.features import build
    from smp_forecast_app.src.ops import chronos_rows
    from smp_forecast_app.src.wz_resid import table
    if stage != "21h":
        raise ValueError("wz2_ops는 2차(21h)만. 1차는 ops.weekly")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    ch = _choice()
    lam, methods = float(ch["잔차 보정 λ"]), ch["범위"]
    bases = [pd.Timestamp(b).normalize() for b in bases]
    s = load_smp_actual("육지")[["date", "hour", "smp"]]
    full = set(s.groupby("date").size().loc[lambda x: x == 24].index)
    pub = [I for I in bases if I + pd.Timedelta(days=1) in full]
    out = []
    if pub:
        if not MODEL.exists():
            train()
        m = joblib.load(MODEL)
        new = chronos_rows([I + pd.Timedelta(days=1) for I in pub])
        new = shift_chronos(new.rename(columns={"qlo": "qlo_c2", "qhi": "qhi_c2"}))
        new["base_date"] = new["date"] - pd.to_timedelta(new["H"], unit="D")
        act = s.rename(columns={"smp": "y"})
        hist = _chronos_hist2()
        cal = build_calendar(hist["date"].min(), new["date"].max() + pd.Timedelta(days=1)).set_index("date")["is_offday"]
        for H in HS2:
            nh = new[new["H"] == H].copy()
            c = nh[["date", "hour", "H", "pred", "c2s_qlo", "c2s_qhi"]].set_axis(["date", "hour", "H", "pred", "qlo", "qhi"], axis=1)
            d = table(H, c, build(H))
            d = d[d["date"].isin(nh["date"])]
            r = pd.Series(m["models"][H].predict(d[m["cols"]]), index=pd.MultiIndex.from_frame(d[["date", "hour"]]))
            nh["rhat"] = nh.set_index(["date", "hour"]).index.map(r).astype(float)
            if nh["rhat"].isna().any():
                print(f"2차 잔차 보정 H={H}: 피처 없는 {nh['rhat'].isna().sum()}행은 Chronos 그대로", flush=True)
            nh["rhat"] = nh["rhat"].fillna(0)
            nh["c2s"], nh["rhat_l"] = nh["pred"], lam * nh["rhat"]
            nh["pred"] = nh["c2s"] + nh["rhat_l"]
            nh["lo"], nh["hi"] = _interval(nh, H, methods[str(H)], hist, act, cal)
            nh["lo"] = np.minimum(nh["lo"], nh["pred"]).clip(lower=0)
            nh["hi"] = np.maximum(nh["hi"], nh["pred"])
            out.append(nh.assign(pub=1))
    rest = [I for I in bases if I not in pub]
    if rest:
        ol = _ops_log()
        if len(ol):
            ol = ol[ol["base_date"].isin(rest) & ol["H"].between(2, 7)]
            ol = ol.sort_values("run_at").drop_duplicates(["base_date", "H", "date", "hour"], keep="last") if "run_at" in ol else ol
            out.append(ol[["base_date", "H", "date", "hour", "pred", "lo", "hi", "c2s", "rhat"]].assign(pub=0))
        miss = [I for I in rest if not len(ol) or I not in set(ol["base_date"])]
        if miss:
            print("2차: 공표 SMP·1차 기록 모두 없는 기준일 → 건너뜀: " + ", ".join(str(I.date()) for I in miss), flush=True)
    if not out:
        return pd.DataFrame(columns=COLS_OUT)
    r = pd.concat(out, ignore_index=True)
    cal = build_calendar(r["date"].min(), r["date"].max()).set_index("date")["is_offday"]
    r["offday"] = r["date"].map(cal).astype(int)
    r["stage"] = stage
    return r[COLS_OUT].sort_values(["base_date", "H", "date", "hour"]).reset_index(drop=True)


if __name__ == "__main__":
    bs = sys.argv[1:] or [str(pd.Timestamp.today().date())]
    t0 = time.time()
    w = weekly(bs)
    print(f"weekly({bs}) {len(w)}행, {time.time()-t0:.0f}s")
    print(w.groupby(["base_date", "pub", "H"]).agg(pred=("pred", "mean"), lo=("lo", "mean"), hi=("hi", "mean")).round(1).to_string())
