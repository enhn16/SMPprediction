"""한 달 예측 mx3 (시간별) — 운영 함수. 모델·근거: docs/MONTH.md 0-3절, outputs/tables/MX3_choice.json, mx/mx3_coef.json

  monthly(bases, hist=None, hist_hourly=None, return_hourly=False)
    -> DataFrame [base_date, H, date, day_type, offday, pred_{지표}, lo_{지표}, hi_{지표}, dc2_{지표}, dc2_qlo_{지표}, dc2_qhi_{지표}]
       (mx_ops.monthly와 같은 열. 4지표는 시간별 예측을 묶은 것)
       return_hourly=True면 (일별, 시간별) — 시간별 [base_date, H, hour, date, pred, lo, hi, hlgb_r, hq10, hq90]
    예측 = 계열 대표 4개 단순 평균 (mx3_ens.recompute, 계수 mx3_coef.json):
      LightGBM 시간별 2종 평균 · Chronos-2 small 시간별 + λ × 잔차 보정 · v2 일평균 구성 × 모양 · 일평균 앙상블 × 모양
    시간별 범위 = LightGBM 분위수 회귀 보정(갱신 365일), 4지표 범위 = MX3_choice '4지표 범위' 방법.
    오차 이력 = 백테스트(MX3_predictions_hourly·daily + hlgb_r·hq10·hq90) + hist·hist_hourly(총괄 운영 기록, 같은 열). 실적은 매번 새로 붙임.
    기록 파일은 쓰지 않는다 (시간별 기록을 남기면 범위 이력이 운영 예측으로 이어짐).
  train(until_h=None, until_d=None, until_seq=None)  시간별 LightGBM 6종 + 일별 표 모델 4종(일평균) + LSTM·GRU → mx/mx3_models.pkl, mx/mx3_seq.pt
    until_*: 검증용 학습 마감 (백테스트와 같게: 시간별 = 분기 첫날, 일별 표 = 달 첫날, LSTM·GRU = 분기 첫날). 기본 = 확정 실적 전부
실행 (시험): python -m smp_forecast_app.src.mx3_ops [기준일 ...]   |   python -m smp_forecast_app.src.mx3_ops --train
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from common.config import SEED
from common.loaders import load_smp_actual
from smp_forecast_app.src import mx3_data, mx_features
from smp_forecast_app.src.features_daily import daily_stats
from smp_forecast_app.src.intervals import BAND
from smp_forecast_app.src.mx3_ens import BANDS, TAB, WEEK, cqr_interval, recompute, to_daily
from smp_forecast_app.src.mx3_tab import LGB, TRAIN_FROM, params, xy
from smp_forecast_app.src.mx_features import MX, STATS

HMODELS = ("hlgb_r", "hlgb_a", "hres_hc2s", "hshape", "hq10", "hq90")
DTABS = ("lgb_r", "lgb_a", "lgb_d", "et_r")
SEQ = ("lstm", "gru")
MODEL, SEQP = MX / "mx3_models.pkl", MX / "mx3_seq.pt"


def _cut(x, until):
    return pd.Series(True, index=x.index) if until is None else x < pd.Timestamp(until)


def train(until_h=None, until_d=None, until_seq=None, path=MODEL, seq_path=SEQP):
    import joblib
    import torch
    from lightgbm import LGBMRegressor
    from smp_forecast_app.src.mx3_tab import chronos_hourly
    from smp_forecast_app.src.mx_seq import TRAIN_FROM as SEQ_FROM
    from smp_forecast_app.src.mx_tab import START, model, spec, target
    from smp_forecast_app.src.mz import seq_data, seq_fit
    t0 = time.time()
    F = mx_features.build_csv()
    h = mx3_data.build(F)
    h = h[h["base_date"] >= "2022-12-01"].reset_index(drop=True)
    alt = (h["base_date"] - pd.Timestamp("2022-12-01")).dt.days % 2 == 0
    models = {}
    for name in HMODELS:
        X, yt, ok, b, mode = xy(h, name, chronos_hourly("hc2s") if name.startswith("hres") else None)
        tr = (h["date"] >= TRAIN_FROM) & yt.notna() & ok & alt & _cut(h["date"], until_h)
        models[name] = (LGBMRegressor(**params(name)).fit(X[tr], yt[tr]), list(X.columns))
    for name in DTABS:                                    # 일별 표 모델 (일평균만: 수준 앙상블 구성)
        cols, base, mode = spec(F, "mean", name)
        yt = target(F["y_mean"], base, mode)
        tr = (F["date"] >= "2023-01-01") & yt.notna() & base.notna() & _cut(F["date"], until_d)
        med = F.loc[F["date"] < START, cols].median() if name.startswith("et") else None
        X = F.loc[tr, cols].fillna(med) if med is not None else F.loc[tr, cols]
        models[f"d_{name}"] = (model(name).fit(X, yt[tr]), cols, med)
    last = F.loc[F["y_mean"].notna(), "date"].max()
    joblib.dump({"models": models, "trained_at": datetime.now().isoformat(timespec="minutes"), "last_date": str(last.date())}, path)
    Xp, Xf, Y, L, B = seq_data()
    cut = pd.Timestamp(until_seq) if until_seq is not None else last + pd.Timedelta(days=1)
    tr = np.where((B + pd.Timedelta(days=34) < cut) & (B + pd.Timedelta(days=7) >= SEQ_FROM))[0]
    states = {k: [seq_fit(k, Xp, Xf, Y, tr, SEED + sd).state_dict() for sd in range(3)] for k in SEQ}
    torch.save({"states": states, "nf": Xf.shape[2], "trained_at": datetime.now().isoformat(timespec="minutes"), "n_train": len(tr)}, seq_path)
    print(f"한 달(mx3) 학습 → {path.name}, {seq_path.name} (시간별 {len(HMODELS)}종 + 일별 {len(DTABS)}종 + LSTM·GRU {len(tr)}개 기준일, "
          f"{time.time()-t0:.0f}s)", flush=True)


def _components(bases, path, seq_path):
    """기준일들의 시간별 표 + 모든 구성 예측 (recompute에 필요한 열)."""
    import joblib
    from smp_forecast_app.src.mx3_chronos import run as chr_run
    from smp_forecast_app.src.mx_chronos import predict as chr_daily
    from smp_forecast_app.src.mx_ops import _seq_preds
    from smp_forecast_app.src.mx_tab import finish, spec
    F = mx_features.build_csv()
    Fb = F[F["base_date"].isin(bases)].reset_index(drop=True)
    d = mx3_data.build(Fb).reset_index(drop=True)
    s = load_smp_actual("육지")[["date", "hour", "smp"]]
    cs = chr_run("hc2s", bases, s, save=False)
    cl = chr_run("hc2", bases, s, save=False)[["base_date", "H", "hour", "q50"]].rename(columns={"q50": "c_hc2"})
    d = d.merge(cs.rename(columns={"q50": "c_hc2s", "q10": "q10_hc2s", "q90": "q90_hc2s"}), on=["base_date", "H", "hour"], how="left")
    d = d.merge(cl, on=["base_date", "H", "hour"], how="left")
    dc2 = chr_daily("dc2", bases)
    d = d.merge(dc2[["base_date", "H", "pred_mean"]].rename(columns={"pred_mean": "l_dc2"}), on=["base_date", "H"], how="left")
    for k, f in _seq_preds(bases, seq_path).items():
        d = d.merge(f[["base_date", "H", "pred_mean"]].rename(columns={"pred_mean": f"l_{k}"}), on=["base_date", "H"], how="left")
    mm = joblib.load(path)["models"]
    for name in DTABS:
        m, cols, med = mm[f"d_{name}"]
        _, base, mode = spec(Fb, "mean", name)
        X = Fb[cols].fillna(med) if med is not None else Fb[cols]
        p = Fb[["base_date", "H"]].assign(**{f"l_{name}": finish(pd.Series(m.predict(X), index=Fb.index), base, mode).values})
        d = d.merge(p, on=["base_date", "H"], how="left")
    chr_ = cs[["base_date", "H", "hour", "q10", "q50", "q90"]]
    for name in HMODELS:
        m, cols = mm[name]
        X, _, ok, b, mode = xy(d, name, chr_ if name.startswith("hres") else None)
        p = pd.Series(m.predict(X[cols]), index=d.index)
        d[name] = (p * b if mode == "ratio" else p).where(ok)
    d["l_hc2s"] = d.groupby(["base_date", "H"])["c_hc2s"].transform("mean")
    d["l_hc2"] = d.groupby(["base_date", "H"])["c_hc2"].transform("mean")
    d["W"], d["band"] = WEEK(d["H"]), d["hour"].map(BAND).map(BANDS)
    return d, Fb, dc2


def _hist_hourly(hist_hourly):
    """시간별 범위 이력: 백테스트 예측 + LightGBM 분위수 (+ 운영 기록). 실적은 새로 붙임."""
    h = pd.read_csv(MX / "MX3_predictions_hourly.csv.gz", parse_dates=["base_date", "date"])[["base_date", "H", "hour", "date", "offday", "pred"]]
    for n in ("hlgb_r", "hq10", "hq90"):
        h = h.merge(pd.read_csv(MX / f"mx3_tab_{n}.csv.gz", parse_dates=["base_date"]).rename(columns={"pred": n}),
                    on=["base_date", "H", "hour"], how="left")
    if hist_hourly is not None and len(hist_hourly):
        hh = hist_hourly.copy()
        for k in ("base_date", "date"):
            hh[k] = pd.to_datetime(hh[k])
        h = pd.concat([h, hh[[c for c in h.columns if c in hh]]]).drop_duplicates(["base_date", "H", "hour"], keep="last")
    s = load_smp_actual("육지")[["date", "hour", "smp"]].rename(columns={"smp": "y"})
    return h.merge(s, on=["date", "hour"], how="left")


def _hist_daily(hist):
    h = pd.read_csv(MX / "MX3_predictions_daily.csv", parse_dates=["base_date", "date"])
    h = h[["base_date", "H", "date", "offday"] + [f"pred_{s}" for s in STATS]]
    if hist is not None and len(hist):
        hh = hist.copy()
        for k in ("base_date", "date"):
            hh[k] = pd.to_datetime(hh[k])
        h = pd.concat([h, hh[[c for c in h.columns if c in hh]]]).drop_duplicates(["base_date", "H"], keep="last")
    st = daily_stats().rename(columns={s: f"y_{s}" for s in STATS})
    return h.merge(st, left_on="date", right_index=True, how="left")


def monthly(bases, hist=None, hist_hourly=None, return_hourly=False, path=MODEL, seq_path=SEQP):
    from smp_forecast_app.src.mx_ens import interval_methods as daily_methods
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    bases = [pd.Timestamp(b).normalize() for b in bases]
    if not path.exists() or not seq_path.exists():
        train(path=path, seq_path=seq_path)
    prm = json.loads((MX / "mx3_coef.json").read_text(encoding="utf-8"))
    choice = json.loads((TAB / "MX3_choice.json").read_text(encoding="utf-8"))
    d, Fb, dc2 = _components(bases, path, seq_path)
    if d.empty:
        return pd.DataFrame()
    d["pred"] = recompute(d, prm)
    # 시간별 범위
    hs = _hist_hourly(hist_hourly)
    hs = hs[~hs["base_date"].isin(bases)]
    nw = d[["base_date", "H", "hour", "date", "offday", "pred", "hlgb_r", "hq10", "hq90"]].assign(y=np.nan)
    D = pd.concat([hs, nw], ignore_index=True)
    D = D[D["pred"].notna()].reset_index(drop=True)
    D["W"], D["band"] = WEEK(D["H"]), D["hour"].map(BAND).map(BANDS)
    mask = D["base_date"].isin(bases).values
    lo, hi = cqr_interval(D, mask, 365, ["W", "offday", "band"], "lgb")
    hr = D[mask][["base_date", "H", "hour", "date", "pred", "hlgb_r", "hq10", "hq90"]].assign(lo=lo, hi=hi)
    # 4지표 (시간별 예측을 묶음) + 지표별 범위
    out = Fb[["base_date", "H", "date", "day_type", "offday"]].merge(to_daily(hr, cols=("pred",)), on=["base_date", "H"], how="left")
    H0 = _hist_daily(hist)
    H0 = H0[~H0["base_date"].isin(bases)]
    ms = daily_methods()
    for s in STATS:
        method = choice["4지표 범위(팀 비교표용)"][{"mean": "일평균", "mid": "한낮 11~15시", "eve": "저녁 18~22시", "midmin": "한낮 최저"}[s]]
        Dd = pd.concat([H0[["base_date", "H", "date", "offday", f"y_{s}", f"pred_{s}"]],
                        out[["base_date", "H", "date", "offday", f"pred_{s}"]].assign(**{f"y_{s}": np.nan})], ignore_index=True)
        Dd = Dd.rename(columns={f"y_{s}": "y", f"pred_{s}": "pred"})
        Dd = Dd[Dd["pred"].notna()].sort_values(["base_date", "H"]).reset_index(drop=True)
        mk = Dd["base_date"].isin(bases).values
        if method.startswith("고정"):     # 고정 1년 = 대상 연도 시작 전 1년 오차 (백테스트와 같게 그 해 전체를 평가 구간으로 넣고 새 행만)
            parts = []
            for yr in sorted(Dd.loc[mk, "date"].dt.year.unique()):
                ev = ((Dd["date"].dt.year == yr) & (Dd["base_date"] <= Dd.loc[mk, "base_date"].max())).values
                l_, h_ = ms[method](Dd, ev)
                g = Dd[ev][["base_date", "H"]].assign(lo=l_, hi=h_)
                parts.append(g[g["base_date"].isin(bases) & (Dd.loc[ev, "date"].dt.year == yr).values])
            r = pd.concat(parts)
        else:
            l_, h_ = ms[method](Dd, mk)
            r = Dd[mk][["base_date", "H"]].assign(lo=l_, hi=h_)
        out = out.merge(r.rename(columns={"lo": f"lo_{s}", "hi": f"hi_{s}"}), on=["base_date", "H"], how="left")
        c = dc2[["base_date", "H", f"pred_{s}", f"qlo_{s}", f"qhi_{s}"]].rename(
            columns={f"pred_{s}": f"dc2_{s}", f"qlo_{s}": f"dc2_qlo_{s}", f"qhi_{s}": f"dc2_qhi_{s}"})
        out = out.merge(c, on=["base_date", "H"], how="left")
    keep = ["base_date", "H", "date", "day_type", "offday"] + [f"{p}_{s}" for s in STATS for p in ("pred", "lo", "hi")] + \
           [f"dc2_{p}{s}" for s in STATS for p in ("", "qlo_", "qhi_")]
    out = out[keep].sort_values(["base_date", "H"]).reset_index(drop=True)
    if return_hourly:
        return out, hr.sort_values(["base_date", "H", "hour"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--train" in a:
        train()
    else:
        bs = a or [str(pd.Timestamp.today().date())]
        t0 = time.time()
        w, h = monthly(bs, return_hourly=True)
        print(f"monthly({bs}) 일별 {len(w)}행·시간별 {len(h)}행, {time.time()-t0:.0f}s")
        print(w.groupby("base_date")[[f"pred_{s}" for s in STATS] + ["lo_mean", "hi_mean"]].mean().round(1).to_string())
