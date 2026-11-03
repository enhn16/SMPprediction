"""일주일 예측 K3 (육지, 1~7일 뒤 시간별) — 운영 함수. 모델·근거: docs/WEEK_V3.md ⑥·⑦, 운영 안내: docs/WEEK_K3_OPS.md

K3 = 평일 행: Chronos-2 small 전체 미세조정판(1년 1회, 달력 공변량) + 0.5 × 잔차 보정
     휴무일 행: Chronos-2 small 제로샷(달력 + 주간 최대부하·하늘·강수확률·연료비 공변량) + 0.5 × 잔차 보정
     잔차 보정 = LightGBM(L1), 평일용·휴무일용 × 지평 H 1~7 = 14개, 대상일의 달마다 다시 학습 (실적은 그달 1일 − (H−1)일 전까지)
     80% 범위 = Chronos-2(대형) 분위수 0.1·0.9를 점예측만큼 옮긴 뒤 대칭 보정 (직전 365일, 시간대묶음 × 휴무일 칸, 오차 80% 분위수)
미세조정판은 기준일로 고른다: 마감 {Y}-12-31 판 ft{Y}f는 기준일 {Y}-12-31부터 다음 판 전까지 (마감 뒤 실적은 학습에 없음).

  weekly(bases, stage="15h") -> DataFrame [base_date, H, date, hour, y, pred, lo, hi, offday, stage] + 구성값
        (ckpt, c, rhat, ft_pred, ext_pred, c2_qlo, c2_qhi). 기준일 15시까지 자료만. 구성값은 wz3/ops/k3_parts.csv.gz에 쌓아
        다음 학습·범위 이력으로 쓴다 (운영 기록 hourly_log는 건드리지 않음).
  train(month)         그달 대상일용 잔차 보정 14개 → wz3/ops/resid_{YYYY-MM}.pkl. weekly가 없으면 스스로 부른다
  finetune_year(year)  Chronos-2 small 전체 미세조정 1판 (마감 {year}-12-31, 2022-01부터, CPU 약 1시간) → wz3/ft/ft{year}f
  backtest()           누수 없는 K3 백테스트(대상일 2025-01-01 ~) · 성적표 · 설정 파일, k3_parts 초기값 (한 번)
실행: python -m smp_forecast_app.src.wz3_ops [기준일 ...]  |  --train [YYYY-MM]  |  --finetune 2026  |  --backtest
"""
import json
import sys
import time
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.loaders import load_smp_actual
from smp_forecast_app.src import range_adjust
from smp_forecast_app.src import wz3_resid as resid
from smp_forecast_app.src.final_metrics import hourly_segments, metrics
from smp_forecast_app.src.wz3_chronos import CH, EXT
from smp_forecast_app.src.wz3_chronos import run as chronos_run
from smp_forecast_app.src.wz3_select import DATA, SEL, TAB, TEST, W3, c2_quantiles, conformal, final_predictions
from smp_forecast_app.src.wz_chronos import HIST
from smp_forecast_app.src.wz_resid import CD, EXO, LGB

OPSD = W3 / "ops"
PARTS = OPSD / "k3_parts.csv.gz"
CHOICE = TAB / "WZ3_K3_choice.json"
BACKTEST = W3 / "WZ3_K3_backtest_hourly.csv"
SCORE = TAB / "WZ3_K3_score.csv"
SPEC = ["mix|fts|ext", "one", False]
WINDOW = 365
COLS = EXO + CD
PCOLS = ["date", "hour", "H", "base_date", "ckpt", "ft_pred", "ft_qlo", "ft_qhi", "ext_pred", "ext_qlo", "ext_qhi",
         "c2_qlo", "c2_qhi", "rhat_ft", "rhat_ext", "offday", "c", "rhat", "pred", "lo", "hi", "run_at"]


def _lam():
    return float(json.loads(CHOICE.read_text(encoding="utf-8"))["λ"])


def _base(d):
    return d["date"] - pd.to_timedelta(d["H"], unit="D")


# ---------------------------------------------------------------- 미세조정판
def ckpt_year(I) -> int:
    """기준일 I에 써야 할 판의 마감 연도 (12/31 기준일은 그해 판)."""
    return I.year if (I.month, I.day) == (12, 31) else I.year - 1


def ckpt_for(I):
    """기준일 I에 쓸 수 있는 가장 새 판 (마감 ≤ I). 2023-12-30 이전 기준일은 제로샷(None)."""
    for y in range(ckpt_year(I), 2022, -1):
        if (W3 / "ft" / f"ft{y}f" / "finetuned-ckpt" / "model.safetensors").exists():
            if y != ckpt_year(I):
                print(f"[주의] 기준일 {I.date()}: ft{ckpt_year(I)}f 판이 없어 ft{y}f로 대신함 (finetune_year({ckpt_year(I)}) 필요)", flush=True)
            return f"ft{y}f"
    return None


def finetune_year(year):
    from smp_forecast_app.src.wz3_finetune import CUTS, finetune
    CUTS[f"ft{year}"] = f"{year}-12-31"
    finetune(f"ft{year}f")


# ---------------------------------------------------------------- 구성값 이력
def parts() -> pd.DataFrame:
    if not PARTS.exists():
        seed_parts()
    return pd.read_csv(PARTS, parse_dates=["date", "base_date"], low_memory=False)


def _hist_chronos(p, kind):
    """잔차 학습용 Chronos 표 [date, hour, H, pred, qlo, qhi] (kind = ft | ext)."""
    x = p[["date", "hour", "H", f"{kind}_pred", f"{kind}_qlo", f"{kind}_qhi"]].dropna()
    return x.set_axis(["date", "hour", "H", "pred", "qlo", "qhi"], axis=1)


# ---------------------------------------------------------------- 잔차 보정
def _model_path(month):
    return OPSD / f"resid_{month:%Y-%m}.pkl"


def train(month=None, p=None, F=None):
    """대상일이 month(그달 1일)인 행에 쓸 잔차 보정 14개. 실적은 대상일 < month − (H−1)일 (백테스트 매월 재학습과 같은 규칙)."""
    from smp_forecast_app.src.features import build
    month = pd.Timestamp(month or datetime.now().strftime("%Y-%m-01")).replace(day=1)
    p = parts() if p is None else p
    t0 = time.time()
    models = {}
    for kind, cfg in (("ft", "ftf"), ("ext", "ext")):
        c = _hist_chronos(p, kind)
        for H in range(1, 8):
            f = F[H] if F is not None else build(H)
            d, _ = resid.table(H, cfg, c, f=f)
            tr = d[(d["date"] >= resid.TRAIN_START) & (d["date"] < month - pd.Timedelta(days=H - 1)) & d["r"].notna()]
            models[(kind, H)] = lgb.LGBMRegressor(**LGB).fit(tr[COLS], tr["r"])
    OPSD.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "cols": COLS, "month": f"{month:%Y-%m}",
                 "trained_at": datetime.now().isoformat(timespec="minutes")}, _model_path(month))
    print(f"잔차 보정 학습 {month:%Y-%m} → {_model_path(month).name} ({time.time() - t0:.0f}s)", flush=True)
    return _model_path(month)


# ---------------------------------------------------------------- 예측
def weekly(bases, stage="15h"):
    from smp_forecast_app.src.features import build
    t0 = time.time()
    bases = sorted({pd.Timestamp(b).normalize() for b in bases})
    last = load_smp_actual("육지")["date"].max()
    if any(I > last for I in bases):
        print(f"[주의] SMP가 {last.date()}까지라 그 뒤 기준일은 건너뜀: {[str(I.date()) for I in bases if I > last]}", flush=True)
        bases = [I for I in bases if I <= last]
    if not bases:
        return pd.DataFrame()
    p = parts()
    F = {H: build(H) for H in range(1, 8)}
    fdaily = {H: f[f["hour"] == 12].set_index("date")[EXT] for H, f in F.items()}
    lam = _lam()

    # 1) Chronos 3개: 평일용 미세조정판(판마다 묶어서) · 휴무일용 공변량판 · 범위용 대형판
    run = lambda tag, cov, bs, **kw: chronos_run(tag, cov, "2000-01-01", "2100-01-01", bases_only=bs, **kw)
    groups = {}
    for I in bases:
        groups.setdefault(ckpt_for(I), []).append(I)
    ft = pd.concat([run("c2s", "cal", bs, ckpt=ck).assign(ckpt=ck or "제로샷") for ck, bs in groups.items()])
    ex = run("c2s", "ext", bases, fdaily=fdaily)
    c2 = run("c2", "cal", bases)
    key = ["date", "hour", "H"]
    new = (ft.rename(columns={"pred": "ft_pred", "qlo": "ft_qlo", "qhi": "ft_qhi"})
           .merge(ex.rename(columns={"pred": "ext_pred", "qlo": "ext_qlo", "qhi": "ext_qhi"}), on=key)
           .merge(c2[key + ["qlo", "qhi"]].rename(columns={"qlo": "c2_qlo", "qhi": "c2_qhi"}), on=key))
    new["base_date"] = _base(new)
    t_ch = time.time() - t0

    # 2) 잔차 보정 (대상일의 달 모델, 없으면 학습)
    hist_p = pd.concat([p[~p.set_index(key).index.isin(new.set_index(key).index)], new], ignore_index=True)
    new["month"] = new["date"].dt.to_period("M").dt.to_timestamp()
    for m in new["month"].unique():
        if not _model_path(pd.Timestamp(m)).exists():
            train(m, p=hist_p, F=F)
    out = []
    for m, g in new.groupby("month"):
        mm = joblib.load(_model_path(pd.Timestamp(m)))["models"]
        for H, gh in g.groupby("H"):
            for kind, cfg in (("ft", "ftf"), ("ext", "ext")):
                c = gh[["date", "hour", "H", f"{kind}_pred", f"{kind}_qlo", f"{kind}_qhi"]].set_axis(
                    ["date", "hour", "H", "pred", "qlo", "qhi"], axis=1)
                d, _ = resid.table(H, cfg, c, f=F[H])
                r = pd.Series(mm[(kind, H)].predict(d[COLS]), index=pd.MultiIndex.from_frame(d[["date", "hour"]]))
                gh = gh.assign(**{f"rhat_{kind}": gh.set_index(["date", "hour"]).index.map(r).astype(float)})
            out.append(gh)
    new = pd.concat(out).drop(columns="month")
    miss = new[["rhat_ft", "rhat_ext"]].isna().any(axis=1).sum()
    if miss:
        print(f"[주의] 피처가 없는 {miss}행은 잔차 보정 0 (Chronos 그대로)", flush=True)
    cal = build_calendar(min(p["date"].min(), new["date"].min()), new["date"].max()).set_index("date")["is_offday"]
    new["offday"] = new["date"].map(cal).astype(bool)
    new["c"] = np.where(new["offday"], new["ext_pred"], new["ft_pred"])
    new["rhat"] = np.where(new["offday"], new["rhat_ext"], new["rhat_ft"])
    new["pred"] = new["c"] + lam * new["rhat"].fillna(0)

    # 3) 범위: 대형판 분위수를 점예측만큼 옮김 → 직전 365일 오차로 대칭 보정 (지평마다)
    act = load_smp_actual("육지")[["date", "hour", "smp"]].rename(columns={"smp": "y"})
    hist = p[p["pred"].notna() & ~p.set_index(key).index.isin(new.set_index(key).index)]
    hist = hist[hist["date"] >= new["date"].min() - pd.Timedelta(days=WINDOW + 8)]
    res = []
    for H in range(1, 8):
        d = pd.concat([hist[hist["H"] == H].assign(new=False), new[new["H"] == H].assign(new=True)], ignore_index=True)
        d = d.drop(columns=["y"], errors="ignore").merge(act, on=["date", "hour"], how="left")
        d.loc[d["new"], "y"] = np.nan        # 새 예측 행의 실적은 범위 계산에 쓰지 않음 (백테스트 재현 때도 같게)
        d["offday"] = d["offday"].astype(bool)
        d = d.sort_values(["date", "hour"]).reset_index(drop=True)
        shift = d["pred"].values - d["c"].values
        ev = d["new"].values
        lo, hi = conformal(d, d["c2_qlo"].values + shift, d["c2_qhi"].values + shift, d["pred"].values, ev, WINDOW, "sym")
        res.append(d[ev].assign(lo=lo[ev], hi=hi[ev]).drop(columns=["y", "new"]))
    new = pd.concat(res).assign(run_at=datetime.now().isoformat(timespec="minutes"))

    # 4) 구성값 쌓기 → 결과
    keep = pd.concat([p[~p.set_index(key).index.isin(new.set_index(key).index)], new[PCOLS]], ignore_index=True)
    keep.sort_values(["date", "hour", "H"]).round(3).to_csv(PARTS, index=False, encoding="utf-8")
    r = new.merge(act, on=["date", "hour"], how="left").assign(stage=stage)
    r = r[["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday", "stage",
           "ckpt", "c", "rhat", "ft_pred", "ext_pred", "c2_qlo", "c2_qhi"]].sort_values(["base_date", "H", "hour"])
    print(f"K3 일주일 {len(bases)}개 기준일 {bases[0].date()}~{bases[-1].date()}: {len(r):,}행 "
          f"(Chronos {t_ch:.0f}s, 전체 {time.time() - t0:.0f}s)", flush=True)
    return r.reset_index(drop=True)


# ---------------------------------------------------------------- 백테스트 (한 번)
def assemble_fts():
    """평일 Chronos 미세조정판을 기준일 기준으로 이어 붙임 → wz3/chronos/c2s_fts_2023-01-01 (대상일 2023 ~ 2026-10-18).
    기준일 ~2023-12-30 제로샷 · [2023-12-31, 2024-12-31) ft2023f · [2024-12-31, 2025-12-31) ft2024f · 2025-12-31 ~ ft2025f."""
    rd = lambda n: pd.read_csv(CH / f"{n}.csv.gz", parse_dates=["date"])
    hist = pd.read_csv(HIST / "c2s.csv.gz", parse_dates=["date"])
    spans = [(hist, None, "2023-12-31"), (rd("ftq_ft2023f"), "2023-12-31", "2024-12-31"),
             (rd("ftq_ft2024f"), "2024-12-31", "2025-12-31"), (rd("c2s_cal_2026-01-01_ft2025f"), "2025-12-31", None)]
    out = []
    for x, a, b in spans:
        bd = _base(x)
        out.append(x[(bd >= (a or "1900-01-01")) & (bd < (b or "2100-01-01"))])
    out = pd.concat(out, ignore_index=True)
    out = out[out["date"] >= "2023-01-01"].sort_values(["date", "hour", "H"])
    assert not out.duplicated(["date", "hour", "H"]).any()
    out.to_csv(CH / "c2s_fts_2023-01-01.csv.gz", index=False, encoding="utf-8")
    print(f"저장 c2s_fts_2023-01-01.csv.gz ({len(out):,}행, 대상일 ~{out['date'].max().date()})", flush=True)


def seed_parts():
    """k3_parts 초기값 = 백테스트의 구성값 (Chronos 3개·잔차 예측은 모두 그때그때 표본 밖)."""
    key = ["date", "hour", "H"]
    ft = resid.chronos("fts").rename(columns={"pred": "ft_pred", "qlo": "ft_qlo", "qhi": "ft_qhi"})
    ex = resid.chronos("ext").rename(columns={"pred": "ext_pred", "qlo": "ext_qlo", "qhi": "ext_qhi"})
    c2 = c2_quantiles().rename(columns={"qlo": "c2_qlo", "qhi": "c2_qhi"})
    rf = pd.read_csv(resid.RES3 / "fts.csv.gz", parse_dates=["date"])[key + ["rhat"]].rename(columns={"rhat": "rhat_ft"})
    re_ = pd.read_csv(resid.RES3 / "ext.csv.gz", parse_dates=["date"])[key + ["rhat"]].rename(columns={"rhat": "rhat_ext"})
    end = ft["date"].max()          # 평일·휴무일 구성이 모두 있는 마지막 대상일
    p = ft.merge(ex, on=key).merge(c2, on=key, how="left").merge(rf, on=key, how="left").merge(re_, on=key, how="left")
    p = p[p["date"] <= end]
    p["base_date"] = _base(p)
    p["ckpt"] = "백테스트"
    cal = build_calendar(p["date"].min(), p["date"].max()).set_index("date")["is_offday"]
    p["offday"] = p["date"].map(cal).astype(bool)
    p["c"] = np.where(p["offday"], p["ext_pred"], p["ft_pred"])
    p["rhat"] = np.where(p["offday"], p["rhat_ext"], p["rhat_ft"])
    p["pred"] = p["c"] + _lam() * p["rhat"]          # 잔차 예측이 없는 2023 행은 빈칸 (잔차 학습에만 씀)
    bt = pd.read_csv(BACKTEST, parse_dates=["date"])[key + ["lo", "hi"]]
    p = p.merge(bt, on=key, how="left").assign(run_at="")
    OPSD.mkdir(parents=True, exist_ok=True)
    p[PCOLS].sort_values(key).round(3).to_csv(PARTS, index=False, encoding="utf-8")
    print(f"저장 {PARTS.name} ({len(p):,}행, 대상일 {p['date'].min().date()} ~ {end.date()})", flush=True)


def backtest():
    assemble_fts()
    resid.run("fts")
    ch = json.loads((TAB / "WZ3_choice.json").read_text(encoding="utf-8"))     # 범위 A1 (파일은 고치지 않음)
    ch["점예측"] = {"spec": SPEC}
    w, info = final_predictions(ch)
    end = resid.chronos("fts")["date"].max()
    w = w[w["date"] <= end].assign(stage="15h")
    w.round(2).to_csv(BACKTEST, index=False, encoding="utf-8-sig")
    lam = info["λ"]["전체"]
    CHOICE.write_text(json.dumps({
        "모델": "K3 — 평일 Chronos-2 small 전체 미세조정(1년 1회) + 잔차, 휴무일 Chronos-2 small 공변량 확장 + 잔차",
        "spec": SPEC, "λ": lam, "λ 맞춤": "2024~2025 (2024만으로 맞춰도 0.5)",
        "잔차 보정": {"모델": "LightGBM L1 (wz_resid.LGB)", "피처": COLS, "학습": "대상일의 달마다, 2023-01-01 ~ 그달 1일 − (H−1)일 실적",
                  "평일 행": "미세조정판 Chronos 기준 잔차", "휴무일 행": "공변량 확장판 Chronos 기준 잔차"},
        "Chronos": {"평일": "autogluon/chronos-2-small 전체 미세조정 lr 1e-6·500단계·배치 32·과거 2,048시간, 공변량 휴무일·요일유형·명절",
                    "휴무일": "autogluon/chronos-2-small 제로샷, 공변량 달력 + " + "·".join(EXT),
                    "범위용": "amazon/chronos-2 제로샷 분위수 0.1·0.9 (달력 공변량)",
                    "미세조정판 고르기": "마감 {Y}-12-31 판은 기준일 {Y}-12-31부터 (기준일 기준, 누수 없음)"},
        "범위": {"방법": "A1 대칭 보정", "분위수": "Chronos-2 대형판 0.1·0.9 + (점예측 − Chronos)", "보정": "직전 365일 오차의 80% 분위수",
               "칸": "시간대묶음 × 휴무일", "휴일·주말 한낮 하한 보정(range_adjust)": "적용하지 않음"},
        "선택 근거": "WEEK_V3.md ⑥(2024·2025), 2026은 한 번 채점", "백테스트": str(BACKTEST.relative_to(DATA.parent.parent.parent)),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 {BACKTEST.name} ({len(w):,}행, {w['date'].min().date()} ~ {w['date'].max().date()}), λ = {lam}", flush=True)
    score(w)
    if PARTS.exists():
        print(f"{PARTS.name}: 이미 있음 → 운영 이력을 지키려고 덮어쓰지 않음 (처음부터 만들려면 파일을 옮긴 뒤 다시)", flush=True)
    else:
        seed_parts()


def score(w):
    """지금 운영 모델과 같은 날짜·시간끼리: 기간(2025 검증·2026 시험) × 구간 × 지평."""
    cur = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["base_date", "date"])
    cur = range_adjust.apply(cur[cur["y"].notna()].reset_index(drop=True), "육지")
    k = w[["date", "hour", "H", "pred", "lo", "hi"]].rename(columns={"pred": "pred_k3", "lo": "lo_k3", "hi": "hi_k3"})
    m = cur.merge(k, on=["date", "hour", "H"])
    act = load_smp_actual("육지")[["date", "hour", "smp"]]
    m = m.merge(act.assign(date=act["date"] + pd.Timedelta(days=7)).rename(columns={"smp": "lag7"}), on=["date", "hour"], how="left")
    m["offday"] = m["offday"].astype(bool)
    rows = []
    for per, (a, b) in (("2025 하반기 검증", SEL), ("2026 시험", TEST)):   # 지금 모델 백테스트가 2025-07-01부터
        x = m[m["date"].between(a, b) & m["y"].notna()].reset_index(drop=True)
        segs = hourly_segments(x) | {f"D+{h}": (x["H"] == h).values for h in range(1, 8)}
        for seg, s in segs.items():
            g = x[s]
            for who, lab in (("", "지금 모델"), ("_k3", "K3")):
                rows.append({"기간": per, "구분": seg, "모델": lab,
                             **metrics(g["y"], g[f"pred{who}"], g[f"lo{who}"], g[f"hi{who}"], g["lag7"])})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(SCORE, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 200)
    print(t[["기간", "구분", "모델", "표본수", "MAE", "RMSE", "포함률", "평균 폭", "구간점수"]].round(3).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--backtest":
        backtest()
    elif a and a[0] == "--train":
        train(a[1] + "-01" if len(a) > 1 else None)
    elif a and a[0] == "--finetune":
        finetune_year(int(a[1]))
    else:
        bases = a or [load_smp_actual("육지")["date"].max()]
        r = weekly(bases)
        print(r.groupby("H")[["pred", "lo", "hi"]].mean().round(1).to_string())
