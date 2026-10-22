"""운영(매일 예측) — 작업 스케줄러가 호출. 웹앱 모델(2026-10-22 선택)을 그대로 쓴다.

  python -m smp_forecast_app.src.ops train    매주 일요일: 일주일 잔차 보정(LightGBM)·한 달 LSTM·GRU 재학습·저장
  python -m smp_forecast_app.src.ops daily    매일 15:30: 수집 → 밀린 기준일까지 예측 → 기록 → 웹 → 푸시
  python -m smp_forecast_app.src.ops actuals  매일 21:30: 공표 SMP 수집 → 웹 → 푸시

모델 (선택 근거: docs/01_결과.md 10~11절, outputs/tables/WZ_choice.json·MZ_choice.json)
  일주일 (1~7일 뒤, 시간별): Chronos-2 small 중앙값 + λ × 잔차 보정(외부 정보 LightGBM, wz_resid). 범위 = WZ_choice의 지평별 방법
        (Chronos-2 분위수 그대로 / 보정 분위수 회귀(Chronos-2 분위수, 90일 갱신, 시간대묶음×휴무일))
  한 달 (7~34일 뒤, 일별 4개 지표): 기준(최근 같은 유형일 4일 평균) + λ × (LSTM·GRU 평균 − 기준), λ·범위는 MZ_choice
  내일 급락 경고: 과제1 최종 모델(Ridge+EBM+랜덤포레스트)을 매일 판단일까지의 자료로 학습해 계산 → 확률 보정(D_choice)
판단 시각 규칙: 기준일 I의 예측은 I일 15시에 알 수 있는 값만 (I일 SMP는 I-1일 저녁 공표). 늦게 켜져도 같은 값,
  실제 산출 시각(run_at)을 기록. 밀린 기준일은 최대 14일까지 따라잡음.
기록: data/processed/smp_forecast_app/ops/{hourly,daily,drop}_log.csv, run_log.txt
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
from smp_forecast_app.src.features_daily import STATS, load_daily
from smp_forecast_app.src.intervals import add_keys
from smp_forecast_app.src.model import cqr
from smp_forecast_app.src.mz_ens import INTV
from smp_forecast_app.src.model_monthly import interval as m_interval
from smp_forecast_app.src.ops_drop import drop_features, grade_of, predict_min

DATA = PROCESSED_DIR / "smp_forecast_app"
OPS = DATA / "ops"
MODELS = DATA / "models"
WZOO = DATA / "wzoo"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
PY = sys.executable
MAX_CATCHUP = 14
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
    import torch
    from smp_forecast_app.src.mz import seq_data, seq_fit
    train_resid()
    t0 = time.time()
    log("학습 시작 (한 달 LSTM·GRU)")
    Xp, Xf, Y, L, B = seq_data()
    today = pd.Timestamp.today().normalize()
    tr = np.where(B + pd.Timedelta(days=34) < today)[0]       # 목표 28일이 모두 공표된 기준일만
    states = {kind: [seq_fit(kind, Xp, Xf, Y, tr, SEED + sd).state_dict() for sd in range(3)] for kind in ("lstm", "gru")}
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"states": states, "nf": Xf.shape[2], "trained_at": datetime.now().isoformat(timespec="minutes"),
                "last_base": str(B[tr].max().date())}, MODELS / "mz_seq.pt")
    log(f"학습 완료 → {MODELS / 'mz_seq.pt'} (기준일 ~{B[tr].max().date()}, {time.time()-t0:.0f}s)")


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


def _lambda(pick):
    """MZ_choice 점예측 이름 → (λ, 보정 모델 종류). 지원: '기준 ... 그대로', '기준 + λ × (시계열 신경망 평균 − 기준)'."""
    if pick.endswith("그대로"):
        return 0.0, None
    m = re.match(r"기준 \+ ([0-9.]+) × \((.+) − 기준\)", pick)
    if m and m.group(2) == "시계열 신경망 평균":
        return float(m.group(1)), "seq"
    log(f"한 달 운영 미지원 구성 '{pick}' → 기준값으로 대체")
    return 0.0, None


def monthly(bases, run_at):
    import torch
    from smp_forecast_app.src.mz import seq_data, seq_net, seq_predict
    ch = json.loads((TAB / "MZ_choice.json").read_text(encoding="utf-8"))["지표별"]
    dd = load_daily()
    new = dd[dd["base_date"].isin(bases)][["base_date", "H", "date", "day_type", "offday"] + [f"{s}_same4" for s in STATS]].copy()
    if new.empty:
        return
    ck = torch.load(MODELS / "mz_seq.pt", weights_only=False)
    models = []
    for kind in ("lstm", "gru"):
        for st in ck["states"][kind]:
            m = seq_net(kind, ck["nf"])
            m.load_state_dict(st)
            m.eval()
            models.append(m)
    Xp, Xf, Y, L, B = seq_data()
    idx = [int(np.where(B == I)[0][0]) for I in bases if (B == I).any()]
    sp = seq_predict(models, Xp, Xf, np.array(idx)) * L[idx, None, None]
    seq = pd.DataFrame([{"base_date": B[i], "H": h + 7, **{f"seq_{s}": float(sp[j, h, k]) for k, s in enumerate(STATS)}}
                        for j, i in enumerate(idx) for h in range(28)])
    new = new.merge(seq, on=["base_date", "H"], how="left")
    hist = pd.read_csv(DATA / "MZ_predictions_daily.csv", parse_dates=["base_date", "date"])
    ol = _read("daily_log.csv", parse_dates=["base_date", "date"])
    if len(ol):
        hist = pd.concat([hist, ol]).drop_duplicates(["base_date", "H"], keep="last")
    hist = hist[~hist["base_date"].isin(bases)]
    ys = dd.set_index(["base_date", "H"])[[f"y_{s}" for s in STATS]]
    hist = hist.drop(columns=[f"y_{s}" for s in STATS], errors="ignore").join(ys, on=["base_date", "H"])
    out = new[["base_date", "H", "date", "day_type", "offday"]].copy()
    for s in STATS:
        lam, kind = _lambda(ch[s]["점예측"])
        base = new[f"{s}_same4"]
        out[f"pred_{s}"] = base + lam * (new[f"seq_{s}"] - base) if kind == "seq" else base
        out[f"y_{s}"] = np.nan
    allm = pd.concat([hist, out], ignore_index=True)
    mask = allm["base_date"].isin(bases).values
    for s in STATS:
        w, form = INTV[ch[s]["범위"]]
        lo, hi = m_interval(allm, mask, s, f"pred_{s}", w, form)
        allm.loc[mask, f"lo_{s}"], allm.loc[mask, f"hi_{s}"] = lo, hi
    r = allm[mask].assign(run_at=run_at)
    for c in ("base_date", "date"):
        r[c] = pd.to_datetime(r[c]).dt.strftime("%Y-%m-%d")
    keep = ["run_at", "base_date", "H", "date", "day_type", "offday"] + [f"{p}_{s}" for s in STATS for p in ("pred", "lo", "hi")]
    _append("daily_log.csv", r[keep].round(3), ["base_date", "H"])


def drops(bases, run_at):
    dch = json.loads((TAB / "D_choice.json").read_text(encoding="utf-8"))
    dfe = drop_features()
    rows = []
    for I in bases:
        d1 = I + pd.Timedelta(days=1)
        if d1 not in set(dfe["date"]):
            continue
        tr = dfe[dfe["label"].notna() & (dfe["date"] <= I)]   # 판단일까지 확정된 날로 학습 (2024~)
        pm = float(predict_min(tr, dfe[dfe["date"] == d1])[0])
        prob = 1 / (1 + np.exp(-(dch["a"] + dch["b"] * pm)))
        rows.append({"run_at": run_at, "base_date": I.strftime("%Y-%m-%d"), "date": d1.strftime("%Y-%m-%d"), "pred_min": round(pm, 2),
                     "grade": str(grade_of(np.array([pm]))[0]), "prob": round(float(prob), 4)})
    if rows:
        _append("drop_log.csv", pd.DataFrame(rows), ["date"])


def daily():
    t0 = time.time()
    run_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    log("===== 매일 예측 시작 =====")
    today = pd.Timestamp.today().normalize()
    run(["common.collect.smp_forecast", "--recent", "5"])
    run(["common.collect.kma_regional_fcst", (today - pd.Timedelta(days=3)).strftime("%Y%m"), today.strftime("%Y%m")])
    run(["common.collect.mid_fcst"])
    run(["common.collect.openmeteo_prevruns", (today - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
         (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")])   # D+1 일사·운량 (D-2 실행분, H=1 피처)
    run(["common.collect.kpx_weekly_outlook"])
    run(["common.collect.kpx_demand_fcst"])
    if today.day <= 10:
        run(["common.collect.epsis_fuel_cost", "202301"])
        run(["common.collect.epsis_capacity"])
    if today.month == 12:
        run(["common.collect.holidays", str(today.year), str(today.year + 1)])
    if not (MODELS / "mz_seq.pt").exists():
        train()
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
        for name, fn in (("일주일", weekly), ("한 달", monthly), ("급락 경고", drops)):
            try:
                fn(bases, run_at)
                log(f"{name} 완료")
            except Exception:  # noqa: BLE001
                log(f"{name} 실패:\n" + traceback.format_exc())
    export_web.main()
    push(f"웹 데이터 갱신 {today:%Y-%m-%d} (예측)")
    log(f"===== 매일 예측 완료 ({time.time()-t0:.0f}s) =====")


def actuals():
    log("===== 공표 SMP 반영 =====")
    run(["common.collect.smp_forecast", "--recent", "3"])
    export_web.main()
    push(f"웹 데이터 갱신 {pd.Timestamp.today():%Y-%m-%d} (공표 SMP)")


def push(msg):
    """웹 데이터 파일만 커밋·푸시. 실패해도 예측은 로컬에 남음."""
    f = "smp_forecast_app/web/data/forecast_data.js"
    if os.environ.get("SMP_NO_PUSH"):   # 수동 시험용
        log("푸시 건너뜀 (SMP_NO_PUSH)")
        return
    try:
        subprocess.run(["git", "add", f], cwd=ROOT, check=True, capture_output=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet", "--", f], cwd=ROOT).returncode == 0:
            log("푸시: 변경 없음")
            return
        subprocess.run(["git", "commit", "-m", msg, "--", f], cwd=ROOT, check=True, capture_output=True)
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
