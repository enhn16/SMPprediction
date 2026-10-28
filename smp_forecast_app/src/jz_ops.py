"""제주 운영 예측 — 과제2 채팅이 ops.py에서 호출 (육지 일주일 예측을 먼저 산출한 뒤).

  weekly(bases) → [base_date, H, date, hour, y, pred, lo, hi, offday]   (JWZ_predictions_hourly.csv와 같은 형식)
     pred = 육지 일주일 예측(육지 운영 기록 pred) + λ × 제주 잔차 보정 (JWZ_choice.json: 기준·피처 묶음·목적·λ)
     범위 = JWZ_choice의 지평별 방법 (Chronos-2 small 제주 분위수 그대로 / 보정 분위수 회귀 / 오차 분위수), 하한 0원 자르기 없음
  drops(bases, model="team") → [base_date, date, grade, pred_min, prob]   (기준일 I → 내일 I+1)
     2026-10-28~ 팀 규칙 재선택 모델(drop_choice_제주.json = JD_choice.json, 2023~2025 고정 학습) = predict(bases).
     model="v1": 이전 S5 KNN (JD_choice_v1.json, 2024~2025)
  predict(bases) → [base_date, date, grade, prob, score]   (ROLES.md 4.1)
  weekly(bases, stage="21h", hist=None) → 일주일 2차 (D+2~7, JW2_choice.json)
  monthly(bases) → 한 달 (JMZ_choice.json = 육지 한 달 v2 기준, 2026-10-28~)
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
    train_drop_team()


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


def weekly(bases, stage="15h", hist=None):
    """stage = "15h"(기본, 1차 = 지금 동작) / "21h"(2차, _weekly21). hist는 21h에서만 씀."""
    if stage == "21h":
        return _weekly21(bases, hist)
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


# ---------------------------------------------------------------- 일주일 2차 (21시, D+2~7) — JW2_choice.json, docs/JEJU.md 11절
def _weekly21(bases, hist=None):
    """기준일 I 21시 → [base_date, H, date, hour, y, pred, lo, hi, offday, stage, pub] (H = 2~7, JW2_predictions_hourly.csv 형식).
    I+1 제주 공표 SMP(24시간)와 육지 2차(wz2_ops.weekly, pub=1)가 모두 있으면 2차 = 육지 2차 + λ × 제주 1차 잔차 보정,
    없으면 제주 1차 운영 기록(jwz_log)의 I, H=2~7 값 (pub=0). 기록 파일은 쓰지 않는다.
    hist: 지난 2차 운영 기록 [date, hour, H, pred] (범위 오차 이력용, 없으면 백테스트 jz2/JW2_hist.csv.gz만).
    호출 순서: 21:30 육지 wz2_ops.weekly 뒤 (육지 1차 운영 기록 hourly_log와 제주 1차 잔차 모델이 필요)."""
    import joblib
    from smp_forecast_app.src import jz2, wz2_ops
    cols = ["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday", "stage", "pub"]
    ch = _choice("JW2_choice.json")
    if ch["후보 코드"] != "b":
        raise NotImplementedError(f"운영 2차는 '(b) 육지 2차 + 제주 1차 잔차 보정'만 지원: {ch['후보 코드']}")
    lam2, meth = float(ch["잔차 보정 λ(2025 전체)"]), ch["범위"]
    bases = [pd.Timestamp(b).normalize() for b in bases]
    act = load_smp_actual("제주")[["date", "hour", "smp"]].rename(columns={"smp": "y"})
    full = set(act.groupby("date").size().loc[lambda x: x == 24].index)
    L2 = wz2_ops.weekly(bases)
    lpub = set(L2.loc[L2["pub"] == 1, "base_date"])
    pub = [I for I in bases if I + pd.Timedelta(days=1) in full and I in lpub]
    out = []
    if pub:
        base, fset, obj, lam1, _ = _pick()
        if not (MODELS / "jz_resid.pkl").exists():
            train()
        m = joblib.load(MODELS / "jz_resid.pkl")
        rt = _rtables_fresh(base, _tables(_land_features()))
        q = jz2.shift(_quantiles("jc2s", [I + pd.Timedelta(days=1) for I in pub]))[["date", "hour", "H", "qlo", "qhi"]]
        qh = jz2.jeju_chronos("jc2s")[["date", "hour", "H", "qlo", "qhi"]]
        q = pd.concat([qh, q]).drop_duplicates(["date", "hour", "H"], keep="last")
        hh = pd.read_csv(jz2.OUT / "JW2_hist.csv.gz", parse_dates=["date"])
        if hist is not None and len(hist):
            hh = pd.concat([hh, hist[["date", "hour", "H", "pred"]]]).drop_duplicates(["date", "hour", "H"], keep="last")
        cal = build_calendar(hh["date"].min(), max(pub) + pd.Timedelta(days=8)).set_index("date")["is_offday"]
        for H in jz2.HS2:
            tg = {I + pd.Timedelta(days=H): I for I in pub}
            d = rt[H][rt[H]["date"].isin(tg)]
            rh = pd.Series(m["models"][H].predict(d[m["cols"]]), index=pd.MultiIndex.from_frame(d[["date", "hour"]]))
            nh = L2[(L2["pub"] == 1) & (L2["H"] == H) & L2["base_date"].isin(pub)][["date", "hour", "pred"]].copy()
            nh["rh"] = nh.set_index(["date", "hour"]).index.map(rh).astype(float)
            if nh["rh"].isna().any():
                log(f"2차 H={H}: 제주 잔차 피처 없는 {nh['rh'].isna().sum()}행은 육지 2차 그대로")
            nh["pred"] = nh["pred"] + lam2 * nh["rh"].fillna(0)
            method, g = meth[str(H)], H - 1
            past = hh[(hh["H"] == H) & ~hh["date"].isin(nh["date"])]
            dd = pd.concat([past, nh[["date", "hour", "pred"]]], ignore_index=True).merge(act, on=["date", "hour"], how="left")
            dd.loc[dd["date"].isin(nh["date"]), "y"] = np.nan
            dd = dd.merge(q[q["H"] == H].drop(columns="H"), on=["date", "hour"], how="left")
            dd["offday"] = dd["date"].map(cal).astype(bool)
            dd = add_keys(dd.sort_values(["date", "hour"]).reset_index(drop=True))
            dd = dd.assign(err=dd["err"].fillna(0), rerr=dd["rerr"].fillna(0))
            mask = dd["date"].isin(nh["date"]).values
            if "그대로" in method:
                lo, hi = dd.loc[mask, "qlo"].values, dd.loc[mask, "qhi"].values
            elif method.startswith("보정분위수회귀"):
                from smp_forecast_app.src.model import cqr
                w = int(re.search(r"갱신(\d+)일", method).group(1))
                lo, hi = cqr(dd, mask, window=w, key="k_band_off", gap=g, clip0=False)
            else:
                lo, hi = methods(gap=g, clip0=False)[method.replace("오차분위수·", "")](dd, mask)
            r = dd[mask][["date", "hour", "pred", "offday"]].assign(H=H, lo=lo, hi=hi)
            r["base_date"] = r["date"].map(tg)
            out.append(r.assign(pub=1))
    rest = [I for I in bases if I not in pub]
    if rest:
        ol = JOPS / "jwz_log.csv"
        one = pd.read_csv(ol, parse_dates=["base_date", "date"]) if ol.exists() else pd.DataFrame(columns=cols)
        bt = pd.read_csv(JDIR / "JWZ_predictions_hourly.csv", parse_dates=["base_date", "date"])
        one = pd.concat([bt, one]).drop_duplicates(["base_date", "H", "date", "hour"], keep="last")
        one = one[one["base_date"].isin(rest) & one["H"].between(2, 7)]
        if len(one):
            out.append(one[["base_date", "H", "date", "hour", "pred", "lo", "hi", "offday"]].assign(pub=0))
        miss = [I for I in rest if I not in set(one["base_date"])]
        if miss:
            log("2차: 공표 SMP·1차 기록 모두 없는 기준일 → 건너뜀: " + ", ".join(str(I.date()) for I in miss))
    if not out:
        return pd.DataFrame(columns=cols)
    r = pd.concat(out, ignore_index=True).drop(columns="y", errors="ignore").merge(act, on=["date", "hour"], how="left")
    r["offday"] = r["offday"].astype(int)
    r["stage"] = "21h"
    return r[cols].sort_values(["base_date", "H", "date", "hour"]).reset_index(drop=True).round(2)


# ---------------------------------------------------------------- 급락
def train_drop():
    """(이전 모델 v1) JD_choice_v1의 피처·모델을 2024-01 ~ 2025-12로 학습 (2026 시험과 같은 고정 모델)."""
    import joblib
    from drop_classifier.src.jeju_drop import day_table
    ch = _choice("JD_choice_v1.json")
    d = day_table()
    tr = d[d["label"].notna() & (d["date"] < "2026-01-01")]
    joblib.dump({"train": tr, "cols": ch["피처 목록"], "model": ch["모델"]}, MODELS / "jz_drop.pkl")
    log(f"급락 모델 학습 자료 저장 ({tr['date'].min().date()} ~ {tr['date'].max().date()}, {len(tr)}일)")


def drops(bases, model="team"):
    """기준일 I → 내일 I+1 [base_date, date, grade, pred_min, prob]. 피처: 제주 Chronos-2 small(육지 공변량) H=1은 운영 기준일만 새로 계산.
    model: "team"(기본, 2026-10-28~ 팀 규칙 재선택 = predict, JD_choice.json) / "v1"(이전 S5 KNN 2024~2025, JD_choice_v1.json)."""
    if model == "team":
        r = predict(bases)
        return pd.DataFrame({"base_date": pd.to_datetime(r["base_date"]), "date": pd.to_datetime(r["date"]), "grade": r["grade"],
                             "pred_min": np.nan, "prob": r["prob"]})
    import joblib
    from drop_classifier.src import jeju_drop as jd
    bases = [pd.Timestamp(b) for b in bases]
    ch = _choice("JD_choice_v1.json")
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


# ---------------------------------------------------------------- 급락 1차 (팀 규칙 재선택, ROLES.md 4.1) — drops와 별개
def train_drop_team():
    """drop_choice_제주.json 모델의 학습 자료(2023-01 ~ 2025-12, 고정)를 models/jz_drop_team.pkl에 저장."""
    import joblib
    from drop_classifier.src import jeju_team as jt
    d = jt.table()
    tr = d[d["date"].between(*jt.TR_FIN) & d["label"].notna()]
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"train": tr}, MODELS / "jz_drop_team.pkl")
    log(f"급락(팀 규칙) 학습 자료 저장 ({tr['date'].min().date()} ~ {tr['date'].max().date()}, {len(tr)}일)")


def predict(bases):
    """기준일 I(15시) → 내일 I+1 [base_date, date, grade, prob, score] (ROLES.md 4.1, drop_제주_15h.csv와 같은 열, label 제외).
    모델·문턱·확률식 = drop_classifier/outputs/tables/drop_choice_제주.json. drops와 같이 제주 Chronos H=1을 운영 기준일만 새로 계산."""
    import joblib
    from drop_classifier.src import jeju_drop as jd
    from drop_classifier.src import jeju_team as jt
    bases = [pd.Timestamp(b) for b in bases]
    ch = json.loads((jd.TAB / "drop_choice_제주.json").read_text(encoding="utf-8"))
    if not (MODELS / "jz_drop_team.pkl").exists():
        train_drop_team()
    tr = joblib.load(MODELS / "jz_drop_team.pkl")["train"]
    from smp_forecast_app.src.features import build as land_build
    from smp_forecast_app.src.jz_features import build
    d = jd.day_table(f=build(1, land_build(1)), jc2sx=_quantiles("jc2sx", bases), start=jt.START)
    d["has_om"] = d["jo_rad_mid"].notna().astype(float)
    te = d[d["date"].isin([b + pd.Timedelta(days=1) for b in bases])]
    cols = ["base_date", "date", "grade", "prob", "score"]
    if te.empty:
        return pd.DataFrame(columns=cols)
    s = jt.fit_score(ch["모델"], tr, te, ch["피처"])
    warn, strong = ch["문턱"]["주의"], ch["문턱"]["경보"]
    a, b = ch["확률식"]["a"], ch["확률식"]["b"]
    return pd.DataFrame({"base_date": (te["date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d").values,
                         "date": te["date"].dt.strftime("%Y-%m-%d").values,
                         "grade": np.where(s >= strong, "경보", np.where(s >= warn, "주의", "없음")),
                         "prob": (1 / (1 + np.exp(-(a * s + b)))).round(4), "score": np.round(s, 4)})


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
