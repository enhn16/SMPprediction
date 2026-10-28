"""일주일 예측 2차 (기준일 I의 21시 판단, 대상 D = I+H, H=2~7) — 백테스트. 설계: docs/WEEK21.md

21시 정보 = 1차(15시) 정보 + I+1일 공표 SMP·하루전 수요예측. 날씨·전망·연료비·달력은 1차와 같음.
Chronos: 입력이 SMP·달력뿐이라 2차(I, H) = 1차(I+1, H−1) → 기존 wzoo·wzoo_hist 예측을 하루 당겨 재사용.
잔차 보정 후보 (LightGBM L1, H별·매월 재학습, 실적은 재학습일 − H일까지 — wz_common.rolling_fit)
  b : 1차 '외부만' 피처 그대로 (Chronos 파생은 I일 SMP 기준)
  c : b + 내일 공표 SMP 피처 (Chronos 파생을 I+1 기준으로 다시 + p1_*)
  d : c + 내일 수요예측 피처 (dm_p1_*, kpx_peak_vs_p1)
  dall (d+) : d + 1차 '기본+전망·날씨' 전체 (SMP·수요 수준 피처 포함, 2026 채점 전 추가)
피처 표 = h{H} (판단일 I) + h{H-1}의 SMP·수요 열 (판단일 I+1). 2026-10-27부터 2023 보강 자료로 다시 만든 wz2/features/ 사용 (--features).
참고: 1차 잔차 보정도 같은 보강 피처로 다시 학습해 resid1_backfill.csv.gz (같은 자료 기준 1차 비교용). 공용 코드 무수정.
출력: data/processed/smp_forecast_app/wz2/{c2s,c2}.csv.gz (당긴 Chronos), resid_{b,c,d}.csv.gz [date, hour, H, rhat]
실행: python -m smp_forecast_app.src.wz2_backtest          (이미 있는 파일은 건너뜀, --force 다시)
      python -m smp_forecast_app.src.wz2_backtest --check  (몇 기준일 Chronos를 실제로 다시 돌려 당긴 값과 비교)
      python -m smp_forecast_app.src.wz2_backtest --features  (보강 자료로 피처 H=1~7 다시 생성 → wz2/features/)
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR
from smp_forecast_app.src.features import load_features
from smp_forecast_app.src.wz_common import COLS, OUT as WZOO, rolling_fit
from smp_forecast_app.src.wz_resid import CD, EXO, LGB, chronos_all, table

warnings.filterwarnings("ignore")

W2 = PROCESSED_DIR / "smp_forecast_app" / "wz2"
HS2 = range(2, 8)
P1 = ["p1_h", "p1_mean", "p1_min", "p1_max", "p1_midmin", "p1_vs_d0", "p1_surprise"]
DM = ["dm_p1_h", "dm_p1_mean", "dm_p1_max", "dm_p1_shape", "dm_p1_vs_d0", "kpx_peak_vs_p1"]
SETS2 = {"b": EXO + CD, "c": EXO + CD + P1, "d": EXO + CD + P1 + DM,
         "dall": list(dict.fromkeys(COLS + CD + P1 + DM))}   # d+ : 1차 '기본+전망·날씨' 전체(SMP·수요 수준 포함) + d


def shift_chronos(c):
    """1차 Chronos [date, hour, H, pred, qlo, qhi] → 2차 (H+1). 기준일 I+1의 H−1일 뒤 = 기준일 I의 H일 뒤."""
    c = c[c["H"] <= 6].copy()
    c["H"] = c["H"] + 1
    return c


def chronos2():
    """당긴 Chronos-2 small (2023~, 잔차 학습용) · Chronos-2 (2025~, 범위용). 캐시 wz2/."""
    W2.mkdir(parents=True, exist_ok=True)
    out = {}
    for tag in ("c2s", "c2"):
        p = W2 / f"{tag}.csv.gz"
        if not p.exists():
            src = chronos_all() if tag == "c2s" else pd.read_csv(WZOO / "c2.csv.gz", parse_dates=["date"])
            shift_chronos(src).round(3).to_csv(p, index=False, encoding="utf-8")
        out[tag] = pd.read_csv(p, parse_dates=["date"])
    return out


FEAT = W2 / "features"   # 2023 보강 자료로 다시 만든 피처 (공용 features_hourly_육지_h*.csv는 덮어쓰지 않음)


def feats2(H):
    """H일 뒤 피처: wz2/features/h{H}.csv.gz가 있으면 그것(2023 보강 반영), 없으면 공용 저장본."""
    p = FEAT / f"h{H}.csv.gz"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else load_features(H)


def build_features():
    """features.build(H) (H=1~7)를 보강된 원본으로 다시 만들어 wz2/features/에 저장."""
    from smp_forecast_app.src.features import build
    FEAT.mkdir(parents=True, exist_ok=True)
    for H in range(1, 8):
        t0 = time.time()
        build(H).to_csv(FEAT / f"h{H}.csv.gz", index=False, encoding="utf-8", compression="gzip")
        print(f"피처 H={H} 저장 ({time.time()-t0:.0f}s)", flush=True)


def resid1():
    """참고: 1차 잔차 보정('외부만')을 같은 보강 피처로 다시 학습 (H=1~7, 매월) → wz2/resid1_backfill.csv.gz.
    2차와 1차를 같은 자료 기준으로 비교하기 위한 것 (운영 1차 모델·wzoo_resid는 그대로)."""
    from smp_forecast_app.src.wz_resid import SETS
    p = W2 / "resid1_backfill.csv.gz"
    if p.exists():
        return
    c, parts, t0 = chronos_all(), [], time.time()
    for H in range(1, 8):
        d = table(H, c, feats2(H))
        parts.append(rolling_fit(d, H, "MS", fit_predict(SETS["외부만"])).rename(columns={"pred": "rhat"}))
        print(f"1차 재학습(보강) H={H} ({time.time()-t0:.0f}s)", flush=True)
    pd.concat(parts).round(3).to_csv(p, index=False, encoding="utf-8")


def table2(H, c2, c1=None):
    """2차 H일 뒤 잔차 학습 표. c2 = 당긴 Chronos-2 small, c1 = 1차 Chronos-2 small(내일 '놀람' 계산용)."""
    c1 = chronos_all() if c1 is None else c1
    f = feats2(H)                                # 판단일 I: 날씨·전망·연료비·달력, 1차 SMP 피처
    g = feats2(H - 1)                            # 판단일 I+1: s_* = I+1까지 SMP, d_* = I+1까지 수요예측
    d = table(H, c2, f)                          # b: 1차 정의 그대로 (Chronos 파생은 I일 SMP 기준)
    g = g[["date", "hour", "s_d1", "s_d1_mean", "s_d1_min", "s_d1_max", "s_d1_midmin", "s_same_h", "s_ma28",
           "d_d1_h", "d_d1_mean", "d_d1_shape"]].add_prefix("g_").rename(columns={"g_date": "date", "g_hour": "hour"})
    d = d.merge(g, on=["date", "hour"], how="left")
    # c: Chronos 파생을 I+1 기준으로 + 내일 공표 SMP
    d["c_vs_d1"] = d["pred"] - d["g_s_d1"]
    d["c_vs_same"] = d["pred"] - d["g_s_same_h"]
    d["c_vs_ma28"] = d["pred"] - d["g_s_ma28"]
    d["p1_h"], d["p1_mean"], d["p1_min"] = d["g_s_d1"], d["g_s_d1_mean"], d["g_s_d1_min"]
    d["p1_max"], d["p1_midmin"] = d["g_s_d1_max"], d["g_s_d1_midmin"]
    d["p1_vs_d0"] = d["g_s_d1_mean"] - d["s_d1_mean"]
    # 놀람: I+1 공표 일평균 − 1차(기준일 I 15시) Chronos가 본 I+1 일평균
    m1 = c1[c1["H"] == 1].groupby("date")["pred"].mean()
    p1day = d["date"] - pd.to_timedelta(H - 1, unit="D")
    d["p1_surprise"] = d["p1_mean"] - p1day.map(m1).values
    # d: 내일 수요예측
    dmax = d.groupby("date")["g_d_d1_h"].transform("max")
    d["dm_p1_h"], d["dm_p1_mean"], d["dm_p1_max"], d["dm_p1_shape"] = d["g_d_d1_h"], d["g_d_d1_mean"], dmax, d["g_d_d1_shape"]
    d["dm_p1_vs_d0"] = d["g_d_d1_mean"] / d["d_d1_mean"]
    d["kpx_peak_vs_p1"] = d["kpx_peak"] / (dmax / 1000)
    return d


def b_table(d):
    """b는 1차 정의의 Chronos 파생을 써야 하므로 table()이 만든 원래 값으로 되돌린 사본."""
    b = d.copy()
    b["c_vs_d1"] = b["pred"] - b["s_d1"]
    b["c_vs_same"] = b["pred"] - b["s_same_h"]
    b["c_vs_ma28"] = b["pred"] - b["s_ma28"]
    return b


def fit_predict(cols):
    def f(tr, te):
        tr = tr[tr["r"].notna()]
        return lgb.LGBMRegressor(**LGB).fit(tr[cols], tr["r"]).predict(te[cols])
    return f


def main(force=False):
    ch = chronos2()
    c1 = chronos_all()
    todo = [k for k in SETS2 if force or not (W2 / f"resid_{k}.csv.gz").exists()]
    if not todo:
        print("잔차 보정 b·c·d: 이미 있음 → 건너뜀")
        return
    t0, out = time.time(), {k: [] for k in todo}
    for H in HS2:
        d = table2(H, ch["c2s"], c1)
        for k in todo:
            src = b_table(d) if k == "b" else d
            out[k].append(rolling_fit(src, H, "MS", fit_predict(SETS2[k])).rename(columns={"pred": "rhat"}))
        print(f"2차 잔차 보정 H={H} ({time.time()-t0:.0f}s)", flush=True)
    for k, parts in out.items():
        pd.concat(parts).round(3).to_csv(W2 / f"resid_{k}.csv.gz", index=False, encoding="utf-8")
    print(f"저장: {W2} resid_{{{','.join(todo)}}}", flush=True)


def check(bases=("2025-03-10", "2025-08-15", "2026-02-03")):
    """당긴 Chronos가 실제로 I+1까지 넣고 돌린 값과 같은지 (ops.chronos_rows로 기준일 I+1 실행)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from smp_forecast_app.src.ops import chronos_rows
    ch = chronos2()
    t0 = time.time()
    run = chronos_rows([pd.Timestamp(b) + pd.Timedelta(days=1) for b in bases])
    print(f"Chronos 2종 {len(bases)}기준일: {time.time()-t0:.0f}s")
    run = shift_chronos(run.rename(columns={"pred": "p_c2s", "qlo": "qlo_c2", "qhi": "qhi_c2"}))
    m = run.merge(ch["c2s"][["date", "hour", "H", "pred"]], on=["date", "hour", "H"]).merge(
        ch["c2"][["date", "hour", "H", "qlo", "qhi"]], on=["date", "hour", "H"])
    print(f"비교 {len(m)}행 | c2s 중앙값 최대 차 {np.abs(m['p_c2s'] - m['pred']).max():.3f}원, "
          f"c2 분위수 최대 차 {max(np.abs(m['qlo_c2'] - m['qlo']).max(), np.abs(m['qhi_c2'] - m['qhi']).max()):.3f}원")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--check" in a:
        check()
    elif "--features" in a:
        build_features()
    else:
        main(force="--force" in a)
        resid1()
