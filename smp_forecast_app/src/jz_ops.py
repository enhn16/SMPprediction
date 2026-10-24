"""제주 운영 예측 — 과제2 채팅이 ops.py에서 호출 (육지 일주일 예측을 먼저 산출한 뒤).

  weekly(bases) → [base_date, H, date, hour, y, pred, lo, hi, offday]   (JWZ_predictions_hourly.csv와 같은 형식)
     pred = 육지 일주일 예측(육지 운영 기록 pred) + λ × 제주 잔차 보정 (JWZ_choice.json: 기준·피처 묶음·목적·λ)
     범위 = JWZ_choice의 지평별 방법 (Chronos-2 small 제주 분위수 그대로 / 보정 분위수 회귀 / 오차 분위수), 하한 0원 자르기 없음
  drops(bases) → [base_date, date, grade, pred_min, prob]   (기준일 I → 내일 I+1)
     JD_choice.json의 모델(피처 묶음·모델·문턱·확률식)을 2024 ~ 2025 자료로 학습한 그대로 사용 (2026 시험과 같은 '고정' 방식)
  train() : 잔차 보정 모델(H별, 공표 실적 전부로 학습)과 급락 모델을 models/에 저장 — 매주 일요일 육지 train과 함께 호출 권장
판단 시각: 기준일 I의 값은 I일 15시에 알 수 있는 자료만 씀(피처가 날짜에 고정) → 늦게 실행해도 같은 값.
          단, 육지 예측은 육지 운영 기록(ops/hourly_log.csv)에서 같은 기준일 행을 읽으므로 육지 weekly가 먼저 돌아야 함.
기록: data/processed/smp_forecast_app/jeju/ops/jwz_log.csv (범위의 오차 이력용, 육지 ops/ 폴더는 건드리지 않음)
실행 시험: python -m smp_forecast_app.src.jz_ops [YYYY-MM-DD ...]   (기준일, 생략 시 제주 마지막 실적일)
"""
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.loaders import load_smp_actual
from smp_forecast_app.src.intervals import add_keys, methods
from common.config import ROOT
from smp_forecast_app.src.jz_common import HS, JDIR, land_pred, load_pred

TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
MODELS = JDIR / "models"
JOPS = JDIR / "ops"
QTAG = {"Chronos-2 small 분위수": "jc2s", "Chronos-2 small+육지 분위수": "jc2sx", "Chronos-2+육지 분위수": "jc2x",
        "LightGBM 분위수": "jlgbq"}


def log(msg):
    print(f"[제주 {datetime.now():%H:%M:%S}] {msg}", flush=True)


def _choice(name):
    return json.loads((TAB / name).read_text(encoding="utf-8"))


def _pick():
    """JWZ_choice → (기준, 피처 묶음, 목적, λ). 지원: 잔차 보정 단일 후보(R_기준_묶음_목적)."""
    ch = _choice("JWZ_choice.json")
    codes = ch["구성 코드"]
    if len(codes) != 1 or not codes[0].startswith("R_"):
        raise NotImplementedError(f"운영 함수는 '기준 + λ × 잔차 보정' 단일 후보만 지원: {codes}")
    _, base, fset, obj = codes[0].split("_")
    return base, fset, obj, float(ch["잔차 보정 λ(2025 전체)"][codes[0]]), ch


def _land_features():
    from smp_forecast_app.src.features import build
    return {H: build(H) for H in HS}


def _tables(land_feats):
    """H별 제주 피처 표 (육지 피처는 새로 계산)."""
    from smp_forecast_app.src.jz_features import build
    return {H: build(H, land_feats[H]) for H in HS}


def train(feats_h=None):
    """잔차 보정 모델(H별 LightGBM, 공표 실적이 있는 행 전부)과 급락 모델 저장."""
    import joblib
    import lightgbm as lgb
    from smp_forecast_app.src.jz_resid import LGB, SETS
    t0 = time.time()
    base, fset, obj, lam, _ = _pick()
    feats_h = feats_h or _tables(_land_features())
    rt = _rtables_fresh(base, feats_h)
    models = {}
    for H in HS:
        d = rt[H][rt[H]["r"].notna()]
        models[H] = lgb.LGBMRegressor(**dict(LGB, objective=obj)).fit(d[SETS[fset]], d["r"])
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "cols": SETS[fset], "base": base, "lam": lam, "trained_at": datetime.now().isoformat(timespec="minutes"),
                 "last_date": str(d["date"].max().date())}, MODELS / "jz_resid.pkl")
    log(f"잔차 보정 학습 완료 (실적 ~{d['date'].max().date()}, {time.time()-t0:.0f}s)")
    train_drop()


def _rtables_fresh(base, feats_h):
    """새로 만든 제주 피처 표 + 육지 예측 + 기준 예측 → 잔차 표 (jz_resid.rtable과 같은 열)."""
    from smp_forecast_app.src.jz_resid import base_all
    b = base_all(base)
    lp = land_pred()
    out = {}
    for H in HS:
        f = feats_h[H].merge(lp[lp["H"] == H].drop(columns="H"), on=["date", "hour"], how="left")
        f["L_spread"] = f["L_qhi"] - f["L_qlo"]
        d = f.merge(b[b["H"] == H].drop(columns="H"), on=["date", "hour"])
        d["b_spread"] = d["b_hi"] - d["b_lo"]
        d["b_vs_d1"] = d["b_pred"] - d["s_d1"]
        d["b_vs_same"] = d["b_pred"] - d["s_same_h"]
        d["b_vs_ma28"] = d["b_pred"] - d["s_ma28"]
        d["b_vs_land"] = d["b_pred"] - d["L_base"]
        d["r"] = d["y"] - d["b_pred"]
        out[H] = d
    return out


def _quantiles(tag, bases):
    """범위용 분위수: 백테스트 파일 + 운영 기준일은 새로 계산."""
    from smp_forecast_app.src.jz_chronos import forecast, land_future_table
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    hist = load_pred(tag)
    hist = hist.assign(base=hist["date"] - pd.to_timedelta(hist["H"], unit="D"))
    need = [b for b in bases if b not in set(hist["base"])]
    new = forecast(tag, need, land_future_table(land_pred()), log=log) if need else hist.iloc[:0]
    return pd.concat([hist.drop(columns="base"), new], ignore_index=True).drop_duplicates(["date", "hour", "H"], keep="last")


def weekly(bases):
    import joblib
    bases = [pd.Timestamp(b) for b in bases]
    base, fset, obj, lam, ch = _pick()
    if not (MODELS / "jz_resid.pkl").exists():
        train()
    m = joblib.load(MODELS / "jz_resid.pkl")
    feats_h = _tables(_land_features())
    rt = _rtables_fresh(base, feats_h)
    act = load_smp_actual("제주")[["date", "hour", "smp"]].rename(columns={"smp": "y"})
    qtags = {QTAG[k] for k in QTAG if any(k in v for v in ch["범위"].values())}
    qs = {t: _quantiles(t, bases) for t in qtags}
    hist = pd.read_csv(JDIR / "JWZ_predictions_hourly.csv", parse_dates=["base_date", "date"])[["date", "hour", "H", "pred"]]
    ol = JOPS / "jwz_log.csv"
    if ol.exists():
        o = pd.read_csv(ol, parse_dates=["date"])[["date", "hour", "H", "pred"]]
        hist = pd.concat([hist, o]).drop_duplicates(["date", "hour", "H"], keep="last")
    cal = build_calendar(hist["date"].min(), max(bases) + pd.Timedelta(days=8)).set_index("date")["is_offday"]
    res = []
    for H in HS:
        targets = [b + pd.Timedelta(days=H) for b in bases]
        d = rt[H][rt[H]["date"].isin(targets)]
        if d.empty:
            log(f"H={H}: 기준 예측(육지 운영 기록)이 없어 건너뜀")
            continue
        rhat = m["models"][H].predict(d[m["cols"]])
        nh = d[["date", "hour"]].assign(H=H, pred=d["b_pred"].values + lam * rhat)
        method = ch["범위"][str(H)]
        qn = next((QTAG[k] for k in QTAG if k in method), None)
        hh = hist[(hist["H"] == H) & ~hist["date"].isin(nh["date"])]
        dd = pd.concat([hh, nh], ignore_index=True).merge(act, on=["date", "hour"], how="left")
        dd.loc[dd["date"].isin(nh["date"]), "y"] = np.nan          # 예측 대상 행은 오차 이력에서 제외
        if qn:
            dd = dd.merge(qs[qn][qs[qn]["H"] == H][["date", "hour", "qlo", "qhi"]], on=["date", "hour"], how="left")
        dd["offday"] = dd["date"].map(cal).astype(bool)
        dd = add_keys(dd.sort_values(["date", "hour"]).reset_index(drop=True))
        dd = dd.assign(err=dd["err"].fillna(0), rerr=dd["rerr"].fillna(0))
        mask = dd["date"].isin(nh["date"]).values
        if "그대로" in method:
            lo, hi = dd.loc[mask, "qlo"].values, dd.loc[mask, "qhi"].values
        elif method.startswith("보정분위수회귀"):
            from smp_forecast_app.src.model import cqr
            w = int(re.search(r"갱신(\d+)일", method).group(1))
            lo, hi = cqr(dd, mask, window=w, key="k_band_off", gap=H, clip0=False)
        else:
            lo, hi = methods(gap=H, clip0=False)[method.replace("오차분위수·", "")](dd, mask)
        res.append(dd[mask][["date", "hour", "pred", "offday"]].assign(H=H, lo=lo, hi=hi))
    if not res:
        return pd.DataFrame(columns=["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday"])
    r = pd.concat(res)
    r["base_date"] = r["date"] - pd.to_timedelta(r["H"], unit="D")
    r = r.merge(act, on=["date", "hour"], how="left")
    r = r[["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday"]].sort_values(["base_date", "H", "hour"])
    JOPS.mkdir(parents=True, exist_ok=True)
    w = r.assign(run_at=datetime.now().strftime("%Y-%m-%d %H:%M"))
    if ol.exists():
        w = pd.concat([pd.read_csv(ol, parse_dates=["base_date", "date"]), w]).drop_duplicates(["base_date", "H", "date", "hour"], keep="last")
    w.round(3).to_csv(ol, index=False, encoding="utf-8-sig")
    return r.round(2).reset_index(drop=True)


# ---------------------------------------------------------------- 급락
def train_drop():
    """JD_choice의 피처·모델을 2024-01 ~ 2025-12로 학습 (2026 시험과 같은 고정 모델)."""
    import joblib
    from drop_classifier.src.jeju_drop import day_table
    ch = _choice("JD_choice.json")
    d = day_table()
    tr = d[d["label"].notna() & (d["date"] < "2026-01-01")]
    joblib.dump({"train": tr, "cols": ch["피처 목록"], "model": ch["모델"]}, MODELS / "jz_drop.pkl")
    log(f"급락 모델 학습 자료 저장 ({tr['date'].min().date()} ~ {tr['date'].max().date()}, {len(tr)}일)")


def drops(bases):
    """기준일 I → 내일 I+1 등급·확률. 피처: 제주 Chronos-2 small(육지 공변량) H=1은 운영 기준일만 새로 계산."""
    import joblib
    from drop_classifier.src import jeju_drop as jd
    bases = [pd.Timestamp(b) for b in bases]
    ch = _choice("JD_choice.json")
    if not (MODELS / "jz_drop.pkl").exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        train_drop()
    saved = joblib.load(MODELS / "jz_drop.pkl")
    from smp_forecast_app.src.features import build as land_build
    from smp_forecast_app.src.jz_features import build
    d = jd.day_table(f=build(1, land_build(1)), jc2sx=_quantiles("jc2sx", bases))
    te = d[d["date"].isin([b + pd.Timedelta(days=1) for b in bases])]
    if te.empty:
        return pd.DataFrame(columns=["base_date", "date", "grade", "pred_min", "prob"])
    s = jd.score_fn(saved["model"], saved["cols"])(saved["train"], te)
    reg = jd.is_reg(saved["model"])
    sc = s                              # 위험 점수 (회귀면 −예상 한낮 최저)
    warn = -ch["주의 문턱"] if reg else ch["주의 문턱"]
    strong = None if ch["강한 경보 문턱"] is None else (-ch["강한 경보 문턱"] if reg else ch["강한 경보 문턱"])
    grade = np.where(sc >= (strong if strong is not None else np.inf), "강한 경보", np.where(sc >= warn, "주의", "없음"))
    prob = 1 / (1 + np.exp(-(ch["a"] + ch["b"] * sc)))
    return pd.DataFrame({"base_date": te["date"].values - pd.Timedelta(days=1), "date": te["date"].values, "grade": grade,
                         "pred_min": (-sc if reg else np.full(len(te), np.nan)).round(2), "prob": prob.round(4)})


def monthly(bases):
    """기준일 I → 7~34일 뒤 제주 일별 참고 전망 [base_date, H, date, day_type, offday, pred_*, lo_*, hi_*] (JMZ_predictions_daily.csv 형식).
    육지 운영 기록(ops/daily_log.csv)의 같은 기준일 행을 읽고 JMZ_choice의 '최근 제주−육지 차이'를 더함 → 육지 monthly 다음에 호출."""
    from smp_forecast_app.src import jz_month as jm
    bases = [pd.Timestamp(b) for b in bases]
    ch = _choice("JMZ_choice.json")["지표별"]
    land = pd.read_csv(ROOT / "data" / "processed" / "smp_forecast_app" / "ops" / "daily_log.csv", parse_dates=["base_date", "date"])
    land = land[land["base_date"].isin(bases)].drop_duplicates(["base_date", "H"], keep="last")
    if land.empty:
        log("육지 한 달 운영 기록에 해당 기준일 없음 → 제주 한 달 건너뜀")
        return pd.DataFrame()
    return jm.apply(land, {s: ch[s]["점예측"] for s in jm.STATS}, {s: ch[s]["범위"] for s in jm.STATS}).round(2)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    bs = [pd.Timestamp(a) for a in sys.argv[1:]] or [load_smp_actual("제주")["date"].max()]
    pd.set_option("display.width", 200)
    w = weekly(bs)
    print(w.groupby(["base_date", "H"])[["pred", "lo", "hi"]].mean().round(1).to_string())
    print(drops(bs).to_string(index=False))
