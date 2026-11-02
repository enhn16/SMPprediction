"""운영(매일 예측) — 작업 스케줄러가 호출. 웹앱 모델(2026-10-22 선택)을 그대로 쓴다.

  python -m smp_forecast_app.src.ops train    매주 일요일: 일주일 잔차 보정(LightGBM)·한 달 mx v2(표 모델·LSTM·GRU) 재학습·저장
  python -m smp_forecast_app.src.ops daily    매일 15:30(+로그온·절전 해제·매시 재시도): 수집 → 밀린 기준일까지 예측(육지 → 제주) → 기록 → 웹 → 푸시
                                              15:30 전에는 어제 기준일까지만 산출, 산출할 것이 없으면 바로 끝남
  python -m smp_forecast_app.src.ops actuals  매일 21:30: 공표 SMP 수집 → 웹 → 푸시

모델 (선택 근거: docs/01_결과.md 10~11절, outputs/tables/WZ_choice.json·MZ_choice.json)
  일주일 (1~7일 뒤, 시간별): Chronos-2 small 중앙값 + λ × 잔차 보정(외부 정보 LightGBM, wz_resid). 범위 = WZ_choice의 지평별 방법
        (Chronos-2 분위수 그대로 / 보정 분위수 회귀(Chronos-2 분위수, 90일 갱신, 시간대묶음×휴무일))
  한 달 (7~34일 뒤, 일별 4개 지표): R3 mx v2 — Chronos 3종·LSTM·GRU·표 모델 조합 (mx_ops, docs/MONTH.md, MX_choice_v2.json)
  내일 급락 경보: R1 산출 ops_drop.predict (Ridge+EBM+랜덤포레스트, 2023~전년도 말 고정 학습, drop_choice_육지.json 문턱·확률)
판단 시각 규칙: 기준일 I의 예측은 I일 15시에 알 수 있는 값만 (I일 SMP는 I-1일 저녁 공표). 늦게 켜져도 같은 값,
  실제 산출 시각(run_at)을 기록. 밀린 기준일은 최대 14일까지 따라잡음.
  제주: jz_ops.weekly·drops·monthly (모델·근거는 docs/JEJU.md). 육지 운영 기록을 읽으므로 육지 다음에 실행
기록: data/processed/smp_forecast_app/ops/{hourly,daily,drop}_log.csv, run_log.txt
      제주 data/processed/smp_forecast_app/jeju/ops/{jwz,jdaily,jdrop}_log.csv
"""
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT, SEED
from common.loaders import load_smp_actual
from smp_forecast_app.src import export_web, features_daily
from smp_forecast_app.src.intervals import add_keys
from smp_forecast_app.src.model import cqr

DATA = PROCESSED_DIR / "smp_forecast_app"
OPS = DATA / "ops"
MODELS = DATA / "models"
WZOO = DATA / "wzoo"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
PY = sys.executable
MAX_CATCHUP = 14
CUTOFF = (15, 30)   # 매일 예측 예약 시각. 이 시각 이후에만 오늘 기준일을 산출
JDIR = DATA / "jeju"
JOPS = JDIR / "ops"   # 제주 운영 기록 (jwz_log = 일주일, jdrop_log = 급락, jdaily_log = 한 달)
CHRONOS = {"c2s": "autogluon/chronos-2-small", "c2": "amazon/chronos-2"}
CTX = 2048


def log(msg):
    OPS.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(OPS / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, timeout=900):
    """수집 명령. 실패해도 멈추지 않고 기록만 (빠진 자료는 결측으로 예측)."""
    try:
        r = subprocess.run([PY, "-W", "ignore", "-m"] + cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        tail = (r.stdout + r.stderr).strip().splitlines()[-2:]
        log(f"수집 {' '.join(cmd)} → 종료코드 {r.returncode} {' | '.join(tail)}")
    except Exception as e:  # noqa: BLE001
        log(f"수집 {' '.join(cmd)} 실패: {e}")


def _read(name, **kw):
    p = OPS / name
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def _append(name, new, keys):
    old = _read(name)
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    out.drop_duplicates(keys, keep="last").to_csv(OPS / name, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- 학습 (매주 일요일)
def _chronos_hist():
    """잔차 보정 학습용 Chronos-2 small 예측: 과거(2023~2024) + 백테스트 + 운영 기록(c2s 열이 있는 행)."""
    from smp_forecast_app.src.wz_resid import chronos_all
    c = chronos_all()
    ol = _read("hourly_log.csv", parse_dates=["date"])
    if len(ol) and "c2s" in ol:
        ol = ol[ol["c2s"].notna()].rename(columns={"c2s": "pred_c", "c2s_qlo": "qlo_c", "c2s_qhi": "qhi_c"})
        ol = ol[["date", "hour", "H", "pred_c", "qlo_c", "qhi_c"]].set_axis(["date", "hour", "H", "pred", "qlo", "qhi"], axis=1)
        c = pd.concat([c, ol]).drop_duplicates(["date", "hour", "H"], keep="last")
    return c


def train_resid():
    """일주일 잔차 보정 모델 (H별 LightGBM, 공표 실적이 있는 행 전부로 학습)."""
    import joblib
    import lightgbm as lgb
    from smp_forecast_app.src.features import build
    from smp_forecast_app.src.wz_resid import LGB, SETS, table
    t0 = time.time()
    ch = json.loads((TAB / "WZ_choice.json").read_text(encoding="utf-8"))
    cols = SETS[ch.get("잔차 보정 피처 묶음", "외부만")]
    c = _chronos_hist()
    models = {}
    for H in range(1, 8):
        d = table(H, c, build(H))
        d = d[d["r"].notna()]
        models[H] = lgb.LGBMRegressor(**LGB).fit(d[cols], d["r"])
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "cols": cols, "trained_at": datetime.now().isoformat(timespec="minutes"),
                 "last_date": str(d["date"].max().date())}, MODELS / "wz_resid.pkl")
    log(f"잔차 보정 학습 완료 → {MODELS / 'wz_resid.pkl'} (실적 ~{d['date'].max().date()}, {time.time()-t0:.0f}s)")


def train():
    """매주 일요일: 일주일 잔차 보정 + 한 달 예측(mx v2: 표 모델·LSTM·GRU, 약 5분) + 제주."""
    from smp_forecast_app.src import mx_ops
    train_resid()
    t0 = time.time()
    log("학습 시작 (한 달 mx v2)")
    mx_ops.train()
    log(f"한 달 학습 완료 ({time.time()-t0:.0f}s)")
    try:
        from smp_forecast_app.src import jz_ops
        jz_ops.train()   # 제주 잔차 보정(H별) + 급락 고정 모델
        log("제주 학습 완료")
    except Exception:  # noqa: BLE001
        log("제주 학습 실패:\n" + traceback.format_exc())


# ---------------------------------------------------------------- 예측
def chronos_rows(bases):
    """기준일별 Chronos-2 small 중앙값·분위수 + Chronos-2 분위수(0.1·0.9) → [date, hour, H, base_date, pred, qlo, qhi, c2s_qlo, c2s_qhi]."""
    import torch
    from chronos import BaseChronosPipeline
    from common.calendar import build_calendar
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    s = load_smp_actual("육지").sort_values(["date", "hour"]).reset_index(drop=True)
    vals = s["smp"].values.astype("float32")
    days = pd.DatetimeIndex(s["date"].values[::24])
    cal = build_calendar(days[0], days[-1] + pd.Timedelta(days=10)).set_index("date")
    hourly = {"offday": np.repeat(cal["is_offday"].astype(float).values, 24).astype("float32"),
              "daytype": np.repeat(cal["day_type"].map({"평일": 0, "주말": 1, "공휴일": 2, "연휴": 3}).astype(float).values, 24).astype("float32"),
              "myeongjeol": np.repeat(cal["is_myeongjeol"].astype(float).values, 24).astype("float32")}
    pos = {d: i for i, d in enumerate(days)}
    out = {}
    for tag, name in CHRONOS.items():
        pipe = BaseChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)
        inputs = []
        for I in bases:
            end = (pos[I] + 1) * 24
            ctx = vals[max(0, end - CTX):end]
            c0 = (I - cal.index[0]).days * 24 + 24 - len(ctx)
            inputs.append({"target": ctx, "past_covariates": {k: v[c0:c0 + len(ctx)] for k, v in hourly.items()},
                           "future_covariates": {k: v[c0 + len(ctx):c0 + len(ctx) + 168] for k, v in hourly.items()}})
        q, _ = pipe.predict_quantiles(inputs, prediction_length=168, quantile_levels=[0.1, 0.5, 0.9])
        out[tag] = [(qq[0] if qq.dim() == 3 else qq).numpy() for qq in q]
    rows = []
    for k, I in enumerate(bases):
        for t in range(168):
            H, h = t // 24 + 1, t % 24 + 1
            rows.append((I + pd.Timedelta(days=H), h, H, I, out["c2s"][k][t, 1], out["c2"][k][t, 0], out["c2"][k][t, 2],
                         out["c2s"][k][t, 0], out["c2s"][k][t, 2]))
    return pd.DataFrame(rows, columns=["date", "hour", "H", "base_date", "pred", "qlo", "qhi", "c2s_qlo", "c2s_qhi"])


def resid_correct(new):
    """Chronos-2 small 중앙값(pred) + λ × 잔차 예측 → pred. 원래 중앙값은 c2s 열에 남김. 모델·피처가 없으면 보정 0."""
    import joblib
    from smp_forecast_app.src.features import build
    from smp_forecast_app.src.wz_resid import table
    ch = json.loads((TAB / "WZ_choice.json").read_text(encoding="utf-8"))
    lam = float(ch.get("잔차 보정 λ", 0))
    new = new.assign(c2s=new["pred"], rhat=0.0)
    if lam == 0:
        return new
    if not (MODELS / "wz_resid.pkl").exists():
        train_resid()
    m = joblib.load(MODELS / "wz_resid.pkl")
    parts = []
    for H in range(1, 8):
        nh = new[new["H"] == H]
        c = nh[["date", "hour", "H", "c2s", "c2s_qlo", "c2s_qhi"]].set_axis(["date", "hour", "H", "pred", "qlo", "qhi"], axis=1)
        d = table(H, c, build(H))
        d = d[d["date"].isin(nh["date"])]
        r = pd.Series(m["models"][H].predict(d[m["cols"]]), index=pd.MultiIndex.from_frame(d[["date", "hour"]]))
        nh = nh.assign(rhat=nh.set_index(["date", "hour"]).index.map(r).astype(float))
        miss = nh["rhat"].isna().sum()
        if miss:
            log(f"잔차 보정 H={H}: 피처 없는 {miss}행은 Chronos 그대로")
        parts.append(nh.assign(rhat=nh["rhat"].fillna(0)))
    new = pd.concat(parts)
    return new.assign(pred=new["c2s"] + lam * new["rhat"])


def weekly(bases, run_at):
    from common.calendar import build_calendar
    ch = json.loads((TAB / "WZ_choice.json").read_text(encoding="utf-8"))["범위"]
    new = resid_correct(chronos_rows(bases))
    act = load_smp_actual("육지")[["date", "hour", "smp"]].rename(columns={"smp": "y"})
    # 오차 이력: 백테스트(Chronos-2 small 예측 + Chronos-2 분위수) + 운영 기록
    hist = pd.read_csv(WZOO / "c2s.csv.gz", parse_dates=["date"])[["date", "hour", "H", "pred"]].merge(
        pd.read_csv(WZOO / "c2.csv.gz", parse_dates=["date"])[["date", "hour", "H", "qlo", "qhi"]], on=["date", "hour", "H"])
    ol = _read("hourly_log.csv", parse_dates=["date", "base_date"])
    if len(ol):
        hist = pd.concat([hist, ol[["date", "hour", "H", "pred", "qlo", "qhi"]]]).drop_duplicates(["date", "hour", "H"], keep="last")
    cal = build_calendar(hist["date"].min(), new["date"].max()).set_index("date")["is_offday"]
    res = []
    for H in range(1, 8):
        nh = new[new["H"] == H]
        method = ch[str(H)]
        if "그대로" in method:
            res.append(nh.assign(lo=nh["qlo"].clip(lower=0), hi=nh["qhi"]))
            continue
        hh = hist[(hist["H"] == H) & ~hist["date"].isin(nh["date"])].merge(act, on=["date", "hour"])
        d = pd.concat([hh, nh[["date", "hour", "H", "pred", "qlo", "qhi"]].assign(y=np.nan)], ignore_index=True)
        d["offday"] = d["date"].map(cal).astype(bool)
        d = add_keys(d.sort_values(["date", "hour"]).reset_index(drop=True))
        d = d.assign(err=d["err"].fillna(0), rerr=d["rerr"].fillna(0))
        mask = d["y"].isna().values
        m = re.search(r"갱신(\d+)일", method)
        lo, hi = cqr(d, mask, window=int(m.group(1)) if m else None, key="k_band_off" if "시간대묶음" in method else None, gap=H)
        g = d[mask][["date", "hour"]].assign(lo=np.clip(lo, 0, None), hi=hi)
        res.append(nh.merge(g, on=["date", "hour"]))
    r = pd.concat(res).assign(run_at=run_at)
    for c in ("base_date", "date"):
        r[c] = r[c].dt.strftime("%Y-%m-%d")
    _append("hourly_log.csv", r[["run_at", "base_date", "H", "date", "hour", "pred", "lo", "hi", "qlo", "qhi",
                                 "c2s", "c2s_qlo", "c2s_qhi", "rhat"]].round(3),
            ["base_date", "H", "date", "hour"])


def monthly(bases, run_at):
    """한 달 예측 = R3 mx v2 (2026-10-27 교체, docs/MONTH.md 0·4절). 지난 운영 기록(daily_log)을 범위 이력으로 넘김."""
    from smp_forecast_app.src import mx_ops
    hist = _read("daily_log.csv", parse_dates=["base_date", "date"])
    r = mx_ops.monthly(bases, hist=hist if len(hist) else None)
    if r is None or r.empty:
        return
    r = r.assign(run_at=run_at)
    for c in ("base_date", "date"):
        r[c] = pd.to_datetime(r[c]).dt.strftime("%Y-%m-%d")
    _append("daily_log.csv", r[["run_at"] + [c for c in r.columns if c != "run_at"]].round(3), ["base_date", "H"])


def drops(bases, run_at):
    """내일 급락 경보 (R1 산출, 2026-10-28 교체): ops_drop.predict_team — LSTM + LightGBM(확장 변수) 평균(팀 합의),
    모든 날 대상, 2023~전년도 말 고정 학습, 하루 약 25초. 문턱·확률식은 drop_classifier/outputs/tables/drop_choice_육지.json.
    이전 모델(Ridge+EBM+RF) 운영 기록은 ops/drop_log_v1_ridge_ebm_rf.csv. 등급 값 없음/주의/경보."""
    from smp_forecast_app.src.ops_drop import predict_team
    r = predict_team(bases)
    if len(r):
        r = r.rename(columns={"score": "pred_min"}).assign(run_at=run_at)
        _append("drop_log.csv", r[["run_at", "base_date", "date", "pred_min", "grade", "prob"]], ["date"])


def surges(bases, run_at):
    """내일 급등 경보 (R4 v3, 2026-10-28): surge_classifier.src.ops_surge.predict — 부호 고정 점수 + 스플라인 로지스틱 +
    LightGBM 그루터기 평균(팀 합의), 15시 하나, 휴무일은 주의·경보 없음. 입력 = 공용 수집 + collect_extra openmeteo. 하루 약 14초."""
    from surge_classifier.src.ops_surge import predict
    r = predict(list(bases))
    if len(r):
        r = r.assign(run_at=run_at)
        for c in ("base_date", "date"):
            r[c] = pd.to_datetime(r[c]).dt.strftime("%Y-%m-%d")
        _append("surge_log.csv", r[["run_at", "base_date", "date", "grade", "prob", "score"]], ["date"])


def reasons(bases, run_at):
    """경보 근거 문장 (R5 과제3, 2026-11-02): 급락·급등 예측 뒤 판단일마다 다음 날 하루분을 계산해 R5_why_육지.json에 바꿔 끼움 (하루 약 40초)."""
    for b in bases:
        run(["smp_forecast_app.src.reason_web", "--ops", b.strftime("%Y-%m-%d")])


# ---------------------------------------------------------------- 제주 (제주 채팅이 만든 jz_ops 호출, 육지 산출 뒤)
def collect_jeju(today):
    run(["common.collect.kpx_jeju_sukub", (today - pd.Timedelta(days=3)).strftime("%Y%m%d")])   # 5분 실적: 판단일 14:25까지 필요
    run(["common.collect.kma_regional_fcst", (today - pd.Timedelta(days=3)).strftime("%Y%m"), today.strftime("%Y%m"), "jeju"])
    run(["common.collect.openmeteo_prevruns", (today - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
         (today + pd.Timedelta(days=3)).strftime("%Y-%m-%d"), "jeju"])
    run(["common.collect.openmeteo_prevruns", (today - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
         (today + pd.Timedelta(days=8)).strftime("%Y-%m-%d"), "jeju_ext"])


def _jread(name, **kw):
    p = JOPS / name
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def _jappend(name, new, keys):
    old = _jread(name)
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    JOPS.mkdir(parents=True, exist_ok=True)
    out.drop_duplicates(keys, keep="last").to_csv(JOPS / name, index=False, encoding="utf-8-sig")


def jeju_bases(today):
    """제주 산출할 기준일: 제주 백테스트·운영 기록 다음 날부터, 육지 일주일이 산출된 기준일까지 (최대 14일)."""
    done = pd.read_csv(JDIR / "JWZ_predictions_hourly.csv", usecols=["base_date"], parse_dates=["base_date"])["base_date"]
    jl = _jread("jwz_log.csv", usecols=["base_date"], parse_dates=["base_date"])
    last = max(done.max(), jl["base_date"].max()) if len(jl) else done.max()
    land = set(_read("hourly_log.csv", usecols=["base_date"], parse_dates=["base_date"])["base_date"])
    start = max(last + pd.Timedelta(days=1), today - pd.Timedelta(days=MAX_CATCHUP - 1))
    return [d for d in pd.date_range(start, today) if d in land]


def jeju(today, run_at):
    from smp_forecast_app.src import jz_ops
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    bases = jeju_bases(today)
    if not bases:
        log("제주: 새 기준일 없음")
        return
    log(f"제주: 기준일 {bases[0].date()} ~ {bases[-1].date()} ({len(bases)}일) 산출")
    steps = (("일주일", jz_ops.weekly, None, None),                       # jwz_log.csv는 jz_ops가 직접 기록
             ("급락 경고", jz_ops.drops, "jdrop_log.csv", ["date"]),
             ("한 달", jz_ops.monthly, "jdaily_log.csv", ["base_date", "H"]))
    for name, fn, out, keys in steps:
        try:
            r = fn(bases)
            if out and len(r):
                r = r.assign(run_at=run_at)
                for c in ("base_date", "date"):
                    r[c] = pd.to_datetime(r[c]).dt.strftime("%Y-%m-%d")
                _jappend(out, r, keys)
            log(f"제주 {name} 완료 ({len(r)}행)")
        except Exception:  # noqa: BLE001
            log(f"제주 {name} 실패:\n" + traceback.format_exc())


def collect_accum():
    """매일 쌓아야 사라지지 않는 자료 (팀 공통 데이터 목록 13종 중, 아직 피처 미사용). daily·actuals(21:30·23:50) 모두 호출.
    전력수급예보(kpx_demand_fcst)·단기예보 격자(kma_vilage_fcst)는 팀 목록 밖이라 2026-10-26부터 수집하지 않음."""
    run(["common.collect.kpx_sukub"])              # 육지 5분 수급 — API가 오늘분만 제공 (23:50 실행이 하루 끝까지 받음)


def last_due():
    """지금 산출할 수 있는 마지막 기준일. 15:30 전에는 오늘 판단 자료(11시 단기예보 등)가 다 모이지 않았다고 보고 어제까지."""
    now = datetime.now()
    today = pd.Timestamp(now.date())
    return today if (now.hour, now.minute) >= CUTOFF else today - pd.Timedelta(days=1)


def daily():
    # 예약 시각 외에 로그온·절전 해제 때도 호출되므로, 산출할 기준일이 없으면 수집 없이 바로 끝낸다
    done = _read("hourly_log.csv", parse_dates=["base_date"])
    due = last_due()
    if len(done) and done["base_date"].max() >= due and not jeju_bases(due):
        print(f"기준일 {due.date()}까지 육지·제주 모두 이미 산출됨 — 건너뜀", flush=True)
        return
    t0 = time.time()
    run_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    log("===== 매일 예측 시작 =====")
    today = due
    run(["common.collect.smp_forecast", "--recent", "5"])
    run(["common.collect.kma_regional_fcst", (today - pd.Timedelta(days=3)).strftime("%Y%m"), today.strftime("%Y%m")])
    run(["common.collect.mid_fcst"])
    run(["common.collect.openmeteo_prevruns", (today - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
         (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")])   # D+1 일사·운량 (D-2 실행분, H=1 피처)
    run(["common.collect.kpx_weekly_outlook"])
    run(["surge_classifier.src.collect_extra", "openmeteo"])   # 급등 v3 입력: Open-Meteo 2일 전 실행 best_match 기온·운량·강수
    collect_accum()
    collect_jeju(today)
    if today.day <= 10:
        run(["common.collect.epsis_fuel_cost", "202301"])
        run(["common.collect.epsis_capacity"])
    if today.month == 12:
        run(["common.collect.holidays", str(today.year), str(today.year + 1)])
    features_daily.main()
    last_act = load_smp_actual("육지")["date"].max()
    done = _read("hourly_log.csv", parse_dates=["base_date"])
    start = done["base_date"].max() + pd.Timedelta(days=1) if len(done) else today
    start = max(start, today - pd.Timedelta(days=MAX_CATCHUP - 1))
    bases = [d for d in pd.date_range(start, today) if d <= last_act]
    if not bases:
        log("새 기준일 없음 (이미 산출됨)")
    else:
        log(f"기준일 {bases[0].date()} ~ {bases[-1].date()} ({len(bases)}일) 산출" + (" — 지연 산출 포함" if len(bases) > 1 else ""))
        for name, fn in (("일주일", weekly), ("한 달", monthly), ("급락 경고", drops), ("급등 경보", surges), ("경보 근거", reasons)):
            try:
                fn(bases, run_at)
                log(f"{name} 완료")
            except Exception:  # noqa: BLE001
                log(f"{name} 실패:\n" + traceback.format_exc())
    jeju(today, run_at)
    export_web.main()
    push(f"웹 데이터 갱신 {today:%Y-%m-%d} (예측)")
    log(f"===== 매일 예측 완료 ({time.time()-t0:.0f}s) =====")


def actuals():
    log("===== 공표 SMP 반영 =====")
    run(["common.collect.smp_forecast", "--recent", "3"])
    collect_accum()
    run(["common.collect.kpx_pv_region"], timeout=1800)   # 지역별 태양광 (약 7주 늦게 갱신, 새 날짜만)
    export_web.main()
    push(f"웹 데이터 갱신 {pd.Timestamp.today():%Y-%m-%d} (공표 SMP)")


def push(msg):
    """웹 데이터 파일만 커밋·푸시. 실패해도 예측은 로컬에 남음."""
    f = ["smp_forecast_app/web/data/forecast_data.js", "smp_forecast_app/web/data/forecast_data_jeju.js"]
    if os.environ.get("SMP_NO_PUSH"):   # 수동 시험용
        log("푸시 건너뜀 (SMP_NO_PUSH)")
        return
    try:
        f = [x for x in f if (ROOT / x).exists()]
        subprocess.run(["git", "add", *f], cwd=ROOT, check=True, capture_output=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet", "--", *f], cwd=ROOT).returncode == 0:
            log("푸시: 변경 없음")
            return
        subprocess.run(["git", "commit", "-m", msg, "--", *f], cwd=ROOT, check=True, capture_output=True)
        r = subprocess.run(["git", "push"], cwd=ROOT, capture_output=True, text=True, timeout=180)
        log(f"푸시: {'성공' if r.returncode == 0 else '실패 ' + r.stderr.strip()[-200:]}")
    except Exception as e:  # noqa: BLE001
        log(f"푸시 실패: {e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    try:
        {"train": train, "daily": daily, "actuals": actuals}[cmd]()
    except Exception:
        log("오류:\n" + traceback.format_exc())
        raise
