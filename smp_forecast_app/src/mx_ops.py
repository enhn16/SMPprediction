"""한 달 예측 개선(mx) — 운영 함수. 모델·근거: docs/MONTH.md, outputs/tables/MX_choice{_v2}.json, mx/mx_ens_coef{_v2}.json

  monthly(bases, hist=None, ver="v2") -> DataFrame [base_date, H, date, day_type, offday, pred_{지표}, lo_{지표}, hi_{지표}, dc2_{지표}, dc2_qlo_{지표}, dc2_qhi_{지표}]
    기준일 I(15시 판단)마다 7~34일 뒤 일별 4지표(mean·mid·eve·midmin) 예측과 80% 범위.
    예측 = 주차별 선형식(mx_ens_coef: 기준값·Chronos·표 모델·LSTM·GRU의 계수) — 백테스트 선택 결과를 그대로 재현.
    범위 = MX_choice 지표별 방법. 오차 이력 = 백테스트(MX_predictions_daily + chr_dc2) + hist(총괄 운영 기록, 같은 열).
    dc2_* 열은 다음 날부터 범위 이력으로 쓰이므로 운영 기록에 함께 남길 것. 기록 파일은 쓰지 않는다.
  train(ver="v2")  계수에 쓰이는 표 모델(lgb_*·et_r·resid_*)과 LSTM·GRU(v2)를 대상일 2023-01~ 확정 실적으로 학습
    → mx/mx_models{_v2}.pkl, mx/mx_seq_v2.pt
  ver: "v2" = 최종 (LSTM·GRU 포함, 한 달 5차 시험), "v1" = 4차 시험 모델
실행 (시험): SMP_NO_PUSH=1 python -m smp_forecast_app.src.mx_ops [기준일 ...]
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from common.config import ROOT, SEED
from smp_forecast_app.src import mx_features
from smp_forecast_app.src.model_monthly import WEEK
from smp_forecast_app.src.mx_ens import interval_methods
from smp_forecast_app.src.mx_features import MX, STATS
from smp_forecast_app.src.mx_tab import START, TRAIN_FROM, finish, model, spec, target

TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
CHR = ("hc2s", "hc2", "dc2")
SEQ = ("lstm", "gru")
SUF = {"v1": "", "v2": "_v2"}


def _paths(ver):
    s = SUF[ver]
    return {"coef": MX / f"mx_ens_coef{s}.json", "choice": TAB / f"MX_choice{s}.json", "hist": MX / f"MX_predictions_daily{s}.csv",
            "model": MX / f"mx_models{s}.pkl", "seq": MX / f"mx_seq{s}.pt"}


def _coef(ver):
    return json.loads(_paths(ver)["coef"].read_text(encoding="utf-8"))


def _comps(ver):
    return {c for s in _coef(ver).values() for w in s.values() for c in w["계수"]}


def needed(ver="v2"):
    """계수에 쓰이는 표 모델 이름 (Chronos·기준값·LSTM·GRU 제외)."""
    return sorted(_comps(ver) - set(CHR) - set(SEQ) - {"anchor", "same4", "same8"})


def _with_chronos(d, tags, preds=None):
    """d에 Chronos 예측 열(tag) 붙임. preds = {tag: DataFrame} (없으면 백테스트 파일)."""
    for t in tags:
        c = preds[t] if preds else pd.read_csv(MX / f"chr_{t}.csv.gz", parse_dates=["base_date", "date"])
        d = d.merge(c.drop(columns="date"), on=["base_date", "H"], how="left", suffixes=("", f"__{t}"))
        for s in STATS:
            d = d.rename(columns={f"pred_{s}": f"{t}_{s}", f"qlo_{s}": f"{t}_qlo_{s}", f"qhi_{s}": f"{t}_qhi_{s}"})
    return d


def train(until=None, path=None, ver="v2", until_seq=None, seq_path=None):
    """until·until_seq(검증용): 백테스트처럼 표 모델은 대상일 < until, LSTM·GRU는 34일 목표가 모두 < until_seq인 기준일만 학습.
    기본 = 확정 실적 전부."""
    import joblib
    t0 = time.time()
    p = _paths(ver)
    path, seq_path = path or p["model"], seq_path or p["seq"]
    d = mx_features.build_csv()
    d = _with_chronos(d, ["hc2s"])
    models = {}
    for name in needed(ver):
        for s in STATS:
            x = d.assign(**{f"chr_{s}": d.get(f"{name.split('_', 1)[1]}_{s}")}) if name.startswith("resid_") else d
            cols, base, mode = spec(x, s, name)
            yt = target(x[f"y_{s}"], base, mode)
            tr = (x["date"] >= TRAIN_FROM) & yt.notna() & base.notna()
            if until is not None:
                tr &= x["date"] < pd.Timestamp(until)
            med = x.loc[x["date"] < START, cols].median() if name.startswith("et") else None   # 백테스트(mx_tab)와 같은 결측 대체값
            X = x.loc[tr, cols].fillna(med) if med is not None else x.loc[tr, cols]
            models[(name, s)] = (model(name).fit(X, yt[tr]), cols, med)
    joblib.dump({"models": models, "trained_at": datetime.now().isoformat(timespec="minutes"),
                 "last_date": str(d.loc[d["y_mean"].notna(), "date"].max().date())}, path)
    msg = f"표 모델 {', '.join(needed(ver))}"
    if set(SEQ) & _comps(ver):
        import torch
        from smp_forecast_app.src.mx_seq import TRAIN_FROM as SEQ_FROM
        from smp_forecast_app.src.mz import seq_data, seq_fit
        Xp, Xf, Y, L, B = seq_data()
        last = d.loc[d["y_mean"].notna(), "date"].max()
        cut = pd.Timestamp(until_seq) if until_seq is not None else last + pd.Timedelta(days=1)
        tr = np.where((B + pd.Timedelta(days=34) < cut) & (B + pd.Timedelta(days=7) >= SEQ_FROM))[0]
        states = {k: [seq_fit(k, Xp, Xf, Y, tr, SEED + sd).state_dict() for sd in range(3)] for k in SEQ}   # mx_seq와 같은 시드 3개
        torch.save({"states": states, "nf": Xf.shape[2], "trained_at": datetime.now().isoformat(timespec="minutes"),
                    "n_train": len(tr)}, seq_path)
        msg += f" + LSTM·GRU({len(tr)}개 기준일)"
    print(f"한 달(mx {ver}) 학습 → {path.name}{', ' + seq_path.name if set(SEQ) & _comps(ver) else ''} ({msg}, {time.time()-t0:.0f}s)",
          flush=True)


def _seq_preds(bases, seq_path):
    """LSTM·GRU (시드 3개 평균) → {kind: DataFrame[base_date, H, pred_{지표}]}."""
    import torch
    from smp_forecast_app.src.mz import seq_data, seq_net, seq_predict
    ck = torch.load(seq_path, weights_only=False)
    Xp, Xf, Y, L, B = seq_data()
    idx = np.array([int(np.where(B == I)[0][0]) for I in bases if (B == I).any()])
    out = {}
    for kind in SEQ:
        ms = []
        for st in ck["states"][kind]:
            m = seq_net(kind, ck["nf"])
            m.load_state_dict(st)
            m.eval()
            ms.append(m)
        p = seq_predict(ms, Xp, Xf, idx) * L[idx, None, None]
        out[kind] = pd.DataFrame([{"base_date": B[i], "H": h + 7, **{f"pred_{s}": float(p[j, h, k]) for k, s in enumerate(STATS)}}
                                  for j, i in enumerate(idx) for h in range(28)])
    return out


def _history(hist, ver):
    """범위 오차 이력: 백테스트 예측 + 실적 + Chronos-2 일별 분위수 (+ 운영 기록)."""
    h = pd.read_csv(_paths(ver)["hist"], parse_dates=["base_date", "date"])
    c = pd.read_csv(MX / "chr_dc2.csv.gz", parse_dates=["base_date", "date"]).drop(columns="date")
    c = c.rename(columns={f"{p}_{s}": f"dc2_{p}_{s}" if p != "pred" else f"dc2_{s}" for p in ("pred", "qlo", "qhi") for s in STATS})
    h = h.merge(c, on=["base_date", "H"], how="left")
    if hist is not None and len(hist):
        hh = hist.copy()
        for k in ("base_date", "date"):
            hh[k] = pd.to_datetime(hh[k])
        h = pd.concat([h, hh]).drop_duplicates(["base_date", "H"], keep="last")
    return h


def monthly(bases, hist=None, ver="v2", model_path=None, seq_path=None):
    import joblib
    from smp_forecast_app.src.mx_chronos import predict
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    p = _paths(ver)
    model_path, seq_path = model_path or p["model"], seq_path or p["seq"]
    bases = [pd.Timestamp(b).normalize() for b in bases]
    coef, choice = _coef(ver), json.loads(p["choice"].read_text(encoding="utf-8"))["지표별"]
    F = mx_features.build_csv()
    ys = F[["base_date", "H"] + [f"y_{s}" for s in STATS]]
    d = F[F["base_date"].isin(bases)].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    d = _with_chronos(d, CHR, {t: predict(t, bases) for t in CHR})
    if not model_path.exists() or (set(SEQ) & _comps(ver) and not seq_path.exists()):
        train(ver=ver, path=model_path, seq_path=seq_path)
    mm = joblib.load(model_path)["models"]
    for name in needed(ver):
        for s in STATS:
            x = d.assign(**{f"chr_{s}": d[f"{name.split('_', 1)[1]}_{s}"]}) if name.startswith("resid_") else d
            m, cols, med = mm[(name, s)]
            _, base, mode = spec(x, s, name)
            X = x[cols].fillna(med) if med is not None else x[cols]
            d[f"{name}_{s}"] = finish(pd.Series(m.predict(X), index=d.index), base, mode)
    if set(SEQ) & _comps(ver):
        for kind, f in _seq_preds(bases, seq_path).items():
            d = d.merge(f.rename(columns={f"pred_{s}": f"{kind}_{s}" for s in STATS}), on=["base_date", "H"], how="left")
    d["W"] = WEEK(d["H"])
    out = d[["base_date", "H", "date", "day_type", "offday"]].copy()
    H0 = _history(hist, ver).drop(columns=[f"y_{s}" for s in STATS], errors="ignore").merge(ys, on=["base_date", "H"], how="left")
    ms = interval_methods()
    for s in STATS:
        comp = {"anchor": d[f"{s}_same4"].clip(lower=10) * np.sqrt(d[f"{s}_seasN"].clip(0.6, 1.6).fillna(1)),
                "same4": d[f"{s}_same4"], "same8": d[f"{s}_same8"]}
        pred = pd.Series(np.nan, index=d.index)
        for w, cw in coef[s].items():
            m = d["W"] == int(w)
            v = cw["절편"]
            for c, b in cw["계수"].items():
                v = v + b * (comp[c] if c in comp else d[f"{c}_{s}"])[m]
            pred[m] = v
        out[f"pred_{s}"] = pred
        out[f"dc2_{s}"], out[f"dc2_qlo_{s}"], out[f"dc2_qhi_{s}"] = d[f"dc2_{s}"], d[f"dc2_qlo_{s}"], d[f"dc2_qhi_{s}"]
        # 범위: 이력 + 새 행
        hs = H0[~H0["base_date"].isin(bases)][["base_date", "H", "date", "offday", f"y_{s}", f"pred_{s}", f"dc2_{s}", f"dc2_qlo_{s}", f"dc2_qhi_{s}"]]
        nw = out[["base_date", "H", "date", "offday", f"pred_{s}", f"dc2_{s}", f"dc2_qlo_{s}", f"dc2_qhi_{s}"]].assign(**{f"y_{s}": np.nan})
        D = pd.concat([hs, nw], ignore_index=True).rename(columns={f"y_{s}": "y", f"pred_{s}": "pred", f"dc2_{s}": "dc2",
                                                                      f"dc2_qlo_{s}": "qlo_dc2", f"dc2_qhi_{s}": "qhi_dc2"})
        D = D[D["pred"].notna()].sort_values(["base_date", "H"]).reset_index(drop=True)
        mask = D["base_date"].isin(bases).values
        method = choice[s]["범위"]
        if method.startswith("고정"):
            # 고정형 = 대상 연도 시작 전 1년 오차로 한 해 고정 (백테스트와 같게: 그 해 대상일 전체를 평가 구간으로 넣고 새 행만 꺼냄)
            parts = []
            for yr in sorted(D.loc[mask, "date"].dt.year.unique()):
                ev = ((D["date"].dt.year == yr) & (D["base_date"] <= D.loc[mask, "base_date"].max())).values
                lo, hi = ms[method](D, ev)
                g = D[ev][["base_date", "H"]].assign(lo=lo, hi=hi)
                parts.append(g[g["base_date"].isin(bases) & (D.loc[ev, "date"].dt.year == yr).values])
            r = pd.concat(parts)
        else:
            lo, hi = ms[method](D, mask)
            r = D[mask][["base_date", "H"]].assign(lo=lo, hi=hi)
        out = out.merge(r.rename(columns={"lo": f"lo_{s}", "hi": f"hi_{s}"}), on=["base_date", "H"], how="left")
    keep = ["base_date", "H", "date", "day_type", "offday"] + [f"{p}_{s}" for s in STATS for p in ("pred", "lo", "hi")] + \
           [f"dc2_{p}{s}" for s in STATS for p in ("", "qlo_", "qhi_")]
    return out[keep].sort_values(["base_date", "H"]).reset_index(drop=True)


if __name__ == "__main__":
    bs = sys.argv[1:] or [str(pd.Timestamp.today().date())]
    t0 = time.time()
    w = monthly(bs)
    print(f"monthly({bs}) {len(w)}행, {time.time()-t0:.0f}s")
    print(w.groupby("base_date")[[f"pred_{s}" for s in STATS] + ["lo_mean", "hi_mean"]].mean().round(1).to_string())
