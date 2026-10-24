"""갈래 B — 자체 D+1 시간별 예측(15시 판단) + 80% 범위 방법 비교.

학습 방식: 매월 1일 재학습 (대상 월 이전의 모든 실적으로 학습 → 그 달 예측). 운영의 주기 재학습을 단순화한 것.
점예측 후보 (2025년 MAE로 선택): 학습 시작 {2022, 2023} × 피처 {기본, 기본+전망·날씨} × 목표 {원값, 직전 같은 유형일 대비 차이}
범위 후보 (2025년 성적으로 선택, intervals.select 규칙)
  - 오차 분위수형 (intervals.methods): 점예측 오차의 10%·90% 분위수를 더함
  - 분위수 회귀: LightGBM quantile(0.1, 0.9) 그대로
  - 보정 분위수 회귀(CQR): 분위수 회귀 범위를 직전 오차로 넓히거나 좁힘 (고정 1년 / 매일 갱신 56·90일)
시험: 2026-01-01 ~ 최신일 한 번 채점. 2024 예측은 2025 범위의 오차 이력으로만 쓴다.
출력: outputs/tables/B1_point_selection_2025.csv, B2_interval_selection_2025.csv, B3_test2026.csv,
      B3_test2026_by_hour.csv, B3_test2026_by_month.csv, B4_weekly_ref.csv, B_choice.json,
      data/processed/smp_forecast_app/B_predictions_hourly.csv (2024~ 예측·범위 전 후보)
실행: python -m smp_forecast_app.src.model
"""
import glob
import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT, SEED
from smp_forecast_app.src.features import FEATURE_SETS, load_features
from smp_forecast_app.src.intervals import (ALPHA, BAND, add_keys, methods, report, score, segments, select)

warnings.filterwarnings("ignore")
OUT_TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
OUT_DATA = PROCESSED_DIR / "smp_forecast_app"
PARAMS = dict(n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1)


def fit_predict(df, months, start, cols, target, objective="regression", alpha=None, lead=1):
    """대상 월마다 그 달 이전 자료로 학습해 그 달을 예측.
    lead = H: 그 달 첫날의 판단일이 m0-H이므로 실적은 m0-H일까지만 학습에 쓴다 (H=1이면 m0 전날까지)."""
    pred = pd.Series(np.nan, index=df.index)
    base = df["s_same_h"] if target == "diff" else 0.0
    yt = df["y"] - base
    p = dict(PARAMS, objective=objective, **({"alpha": alpha} if alpha else {}))
    for m in months:
        m0, m1 = pd.Timestamp(m), pd.Timestamp(m) + pd.offsets.MonthBegin(1)
        tr = (df["date"] >= start) & (df["date"] < m0 - pd.Timedelta(days=lead - 1)) & df["y"].notna() & yt.notna()
        te = (df["date"] >= m0) & (df["date"] < m1)
        if te.sum() == 0:
            continue
        model = lgb.LGBMRegressor(**p).fit(df.loc[tr, cols], yt[tr])
        pred[te] = model.predict(df.loc[te, cols])
    return pred + base


def months_of(y0, y1_end):
    return pd.date_range(y0, y1_end, freq="MS")


def point_metrics(y, p, naive) -> dict:
    ok = y.notna() & p.notna()
    mae = (y[ok] - p[ok]).abs().mean()
    nmae = (y[ok] - naive[ok]).abs().mean()
    mape = ((y[ok] - p[ok]).abs() / y[ok].clip(lower=20)).mean()
    return {"시간수": int(ok.sum()), "MAE": mae, "naive_MAE": nmae, "rMAE": mae / nmae, "MAPE(하한20)": mape,
            "편의(원)": (p[ok] - y[ok]).mean()}


def cqr(d, eval_mask, window=None, key=None, gap=1, clip0=True):
    """보정 분위수 회귀: 점수 e = max(qlo − y, y − qhi)의 (1−α) 분위수 c만큼 양쪽을 넓힘(음수면 좁힘).
    window=None → 평가 구간 직전 365일로 고정, 정수 → 매일 직전 window일."""
    d = d.copy()
    d["cs"] = np.maximum(d["qlo"] - d["y"], d["y"] - d["qhi"])
    d["g"] = "all" if key is None else d[key]
    ev = d[eval_mask]
    c = pd.Series(np.nan, index=ev.index)

    def qtab(rows):
        q = rows.groupby("g")["cs"].quantile(1 - ALPHA)
        return q.reindex(ev["g"].unique()).fillna(rows["cs"].quantile(1 - ALPHA))

    if window is None:
        start = ev["date"].min() - pd.Timedelta(days=gap - 1)
        rows = d[(d["date"] < start) & (d["date"] >= start - pd.Timedelta(days=365))]
        c[:] = ev["g"].map(qtab(rows)).values
    else:
        for day, part in ev.groupby("date"):
            end = day - pd.Timedelta(days=gap - 1)
            rows = d[(d["date"] < end) & (d["date"] >= end - pd.Timedelta(days=window))]
            c[part.index] = part["g"].map(qtab(rows)).values
    lo = (ev["qlo"] - c).clip(lower=0) if clip0 else ev["qlo"] - c   # 제주는 clip0=False
    hi = ev["qhi"] + c
    return np.minimum(lo, ev["pred"]).values, np.maximum(hi, ev["pred"]).values


def interval_methods(gap: int = 1) -> dict:
    g = gap
    m = {f"오차분위수·{k}": (lambda f: lambda d, e: f(d, e))(f) for k, f in methods(g).items()}
    m["분위수회귀"] = lambda d, e: (d.loc[e, "qlo"].clip(lower=0).values, d.loc[e, "qhi"].values)
    m["보정분위수회귀·고정1년"] = lambda d, e: cqr(d, e, gap=g)
    m["보정분위수회귀·고정1년·시간대묶음×휴무일"] = lambda d, e: cqr(d, e, key="k_band_off", gap=g)
    for w in (56, 90):
        m[f"보정분위수회귀·갱신{w}일"] = (lambda w: lambda d, e: cqr(d, e, window=w, gap=g))(w)
        m[f"보정분위수회귀·갱신{w}일·시간대묶음×휴무일"] = (lambda w: lambda d, e: cqr(d, e, window=w, key="k_band_off", gap=g))(w)
    return m


def weekly_d1() -> pd.DataFrame:
    """담당자 운영 주간 예측(15시)의 D+1 (data/smp_edu_data, 읽기 전용). 같은 대상일이 여러 번이면 마지막 산출분."""
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/smp_edu_data/outputs/predict/SMP주간예측_*.csv"))):
        d = pd.read_csv(f, parse_dates=["기준일", "대상일"])
        rows.append(d[d["지평"] == "D+1"])
    w = pd.concat(rows).sort_values("기준일").drop_duplicates(["대상일", "거래시간"], keep="last")
    return w.rename(columns={"대상일": "date", "거래시간": "hour", "예측": "pred_weekly"})[["date", "hour", "pred_weekly"]]


def main():
    t0 = time.time()
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    df = load_features()
    last_actual = df.loc[df["y"].notna(), "date"].max()
    print(f"피처 {len(df):,}행, 실적 최신 {last_actual.date()}", flush=True)

    # 1) 점예측 선택 (2025, 매월 재학습)
    m25 = months_of("2025-01-01", "2025-12-01")
    is25 = df["date"].dt.year == 2025
    rows, best, best_mae = [], None, np.inf
    for start in ("2022-01-01", "2023-01-01"):
        for fs, cols in FEATURE_SETS.items():
            for target in ("raw", "diff"):
                p = fit_predict(df, m25, start, cols, target)
                r = {"학습시작": start[:4], "피처": fs, "목표": target, **point_metrics(df.loc[is25, "y"], p[is25], df.loc[is25, "s_d1"])}
                rows.append(r)
                print(f"[점예측 2025] {start[:4]} / {fs} / {target}: MAE {r['MAE']:.2f}, rMAE {r['rMAE']:.3f} ({time.time()-t0:.0f}s)", flush=True)
                if r["MAE"] < best_mae:
                    best, best_mae = (start, fs, target), r["MAE"]
    pd.DataFrame(rows).round(4).to_csv(OUT_TAB / "B1_point_selection_2025.csv", index=False, encoding="utf-8-sig")
    start, fs, target = best
    cols = FEATURE_SETS[fs]
    print(f"\n선택 점예측: 학습시작 {start[:4]}, 피처 {fs}, 목표 {target}\n", flush=True)

    # 2) 선택 구성으로 2024-01 ~ 최신월 예측 (점 + 분위수 0.1·0.9)
    mall = months_of("2024-01-01", df["date"].max().strftime("%Y-%m-01"))
    df["pred"] = fit_predict(df, mall, start, cols, target)
    df["qlo"] = fit_predict(df, mall, start, cols, target, "quantile", ALPHA / 2)
    df["qhi"] = fit_predict(df, mall, start, cols, target, "quantile", 1 - ALPHA / 2)
    print(f"전 기간 예측 완료 ({time.time()-t0:.0f}s)", flush=True)
    d = df[(df["date"] >= "2024-01-01")].copy()
    d["offday"] = d["offday"].astype(bool)
    known = add_keys(d[d["y"].notna()].reset_index(drop=True))

    # 3) 범위 선택 (2025)
    ims = interval_methods()
    e25 = (known["date"].dt.year == 2025).values
    sel_rows = []
    for name, fn in ims.items():
        lo, hi = fn(known, e25)
        g = known[e25].assign(lo=lo, hi=hi)
        sel_rows += report(g, name, "선택 2025")
        r = [x for x in sel_rows if x["방법"] == name and x["구간"] == "전체"][0]
        print(f"[범위 2025] {name}: 포함률 {r['포함률']:.1%}, 폭 {r['평균폭']:.1f}, 구간점수 {r['구간점수']:.1f}", flush=True)
    sel = pd.DataFrame(sel_rows)
    sel.round(4).to_csv(OUT_TAB / "B2_interval_selection_2025.csv", index=False, encoding="utf-8-sig")
    chosen = select(sel)
    print(f"\n선택 범위 방법: {chosen}\n", flush=True)

    # 4) 2026 시험 (한 번) — 선택 방법 + 비교용 단순 방법
    e26 = (known["date"] >= "2026-01-01").values
    show = list(dict.fromkeys([chosen, "오차분위수·고정·시간대", "분위수회귀"]))
    te_rows, bands = [], {}
    for name in show:
        lo, hi = ims[name](known, e26)
        g = known[e26].assign(lo=lo, hi=hi)
        bands[name] = g
        te_rows += report(g, name, "시험 2026")
    te = pd.DataFrame(te_rows)
    te["선택"] = te["방법"].eq(chosen)
    te.round(4).to_csv(OUT_TAB / "B3_test2026.csv", index=False, encoding="utf-8-sig")
    g = bands[chosen]
    pd.DataFrame([{"시간": h, **score(x["y"], x["lo"], x["hi"], x["pred"])} for h, x in g.groupby("hour")]).round(4).to_csv(
        OUT_TAB / "B3_test2026_by_hour.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"월": str(m), **score(x["y"], x["lo"], x["hi"], x["pred"])}
                  for m, x in g.groupby(g["date"].dt.to_period("M"))]).round(4).to_csv(
        OUT_TAB / "B3_test2026_by_month.csv", index=False, encoding="utf-8-sig")
    pt26 = point_metrics(g["y"], g["pred"], g["s_d1"])
    pt25 = point_metrics(known.loc[e25, "y"], known.loc[e25, "pred"], known.loc[e25, "s_d1"])

    # 5) 참고: 담당자 운영 주간 D+1(15시)과 같은 날 비교 + 같은 범위 폭을 붙였을 때 포함률
    w = weekly_d1().merge(g[["date", "hour", "y", "pred", "lo", "hi", "s_d1", "offday"]], on=["date", "hour"])
    ref = []
    if len(w):
        ref.append({"대상": "자체 모델", **point_metrics(w["y"], w["pred"], w["s_d1"]), **score(w["y"], w["lo"], w["hi"])})
        wl = w["pred_weekly"] + (w["lo"] - w["pred"])
        wh = w["pred_weekly"] + (w["hi"] - w["pred"])
        ref.append({"대상": "운영 주간 D+1 (같은 범위 폭 적용)", **point_metrics(w["y"], w["pred_weekly"], w["s_d1"]),
                    **score(w["y"], wl.clip(lower=0), wh)})
        pd.DataFrame(ref).assign(기간=f"{w.date.min().date()}~{w.date.max().date()}").round(4).to_csv(
            OUT_TAB / "B4_weekly_ref.csv", index=False, encoding="utf-8-sig")

    # 6) 저장: 예측·범위 (2025: 선택 구간 범위, 2026~: 시험 범위, 내일분 포함)
    lo25, hi25 = ims[chosen](known, e25)
    out = pd.concat([known[e25].assign(lo=lo25, hi=hi25), g])
    # 실적이 아직 없는 '내일' 행: 직전까지의 이력으로 범위 계산
    fut = d[d["y"].isna()].copy()
    if len(fut):
        fut = add_keys(fut.assign(y=np.nan))
        allk = pd.concat([known, fut]).reset_index(drop=True)
        mf = allk["y"].isna().values
        lo, hi = ims[chosen](allk.assign(err=allk["err"].fillna(0), rerr=allk["rerr"].fillna(0)), mf)
        out = pd.concat([out, allk[mf].assign(lo=lo, hi=hi)])
    keep = ["date", "hour", "y", "pred", "lo", "hi", "qlo", "qhi", "s_d1", "offday", "day_type"]
    out[keep].round(2).to_csv(OUT_DATA / "B_predictions_hourly.csv", index=False, encoding="utf-8-sig")

    info = {"점예측": {"학습시작": start[:4], "피처": fs, "목표": target, "재학습": "매월 1일",
                     "2025 선택": {k: round(float(v), 4) for k, v in pt25.items()},
                     "2026 시험": {k: round(float(v), 4) for k, v in pt26.items()}},
            "범위 방법": chosen, "시험 기간": f"2026-01-01 ~ {g.date.max().date()}",
            "주간 D+1 비교": ref and {r["대상"]: {k: round(float(r[k]), 4) for k in ("MAE", "rMAE", "포함률", "평균폭")} for r in ref}}
    (OUT_TAB / "B_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(te[["방법", "구간", "시간수", "포함률", "하한아래", "상한위", "평균폭", "구간점수", "MAE"]].round(3).to_string(index=False))
    print(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"완료 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
