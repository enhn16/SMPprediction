"""제주 일주일 2차 (21시 판단, D+2~D+7) — 육지 2차(R3, WEEK21.md) + 제주 보정. ROLES.md 4.2.

21시 정보 (ROLES.md 4.4): 1차 정보 + I+1 공표 SMP(육지·제주)·수요예측. 날씨는 1차와 같음.
후보 (2026 채점 전 고정, JEJU.md 결정 기록 2026-10-28)
  (a)  제주 1차 그대로 (육지 1차 + 0.25 × 제주 잔차 보정, H=2~7)                        — 기준선
  (b0) 육지 2차 그대로 (당긴 Chronos-2 small + 0.5 × 2차 잔차 보정, WZ2_choice)
  (j)  당긴 제주 Chronos-2 small + 육지 공변량 (1차 (I+1, H−1) 값 = 2차 (I, H), 입력이 SMP·달력뿐)
  (b)  육지 2차 + λ × 제주 1차 잔차 보정 (jzoo_resid/land_제주외부_l1, 다시 학습하지 않음)
  (c)  육지 2차 + λ × 새 잔차 보정 (제주 − 육지 2차, 1차 '제주외부' 피처 + 기준 파생)
  (d)  (c) + 내일 공표 제주 SMP: I+1 같은 시간·일평균·한낮 최저, I+1 제주−육지 차이(같은 시간), 기준 − I+1 제주 같은 시간
  잔차 학습: LightGBM L1 (jz_resid.LGB), H별 매월 1일 재학습, 실적 2023-01 ~ 재학습일 − H (1차와 같음)
  학습 기준: 2023~2024는 당긴 육지 Chronos-2 small(2차 잔차 보정 없음), 2025~ 육지 2차 최종 (1차 land_pred와 같은 방식)
선택 (1차·R3와 같음): λ ∈ {0.1, 0.25, 0.5, 0.75, 1}을 2025 상반기 MAE로 → 2025 하반기 MAE, 1위와 0.05원 이내면 단순한 쪽
  (a) < (b0) < (j) < (b) < (c) < (d) → λ 2025 전체로 재적합
범위 (지평별): 오차 분위수(gap = H−1, 365일 창 제외) · 보정 분위수 회귀(당긴 제주 Chronos 분위수, 갱신 90·180일) ·
  당긴 제주 Chronos 분위수 그대로 · 육지 2차 범위를 제주 예측으로 이동 — 2025 하반기 intervals.select
시험 (--test): 대상 2026-01-01 ~ 10-18, 제주 1차와 같은 (기준일, H, 날짜, 시간)끼리. 가정: 화요일 미공표(화요일 판단 = 1차 값) / 항상 공표
출력: data/processed/smp_forecast_app/jeju/jz2/ (잔차 예측), jeju/JW2_predictions_hourly.csv,
      outputs/tables/JW2_select_2025.csv, JW2_intervals_2025H2.csv, JW2_choice.json, (--test) JW2_test2026.csv
실행: python -m smp_forecast_app.src.jz2 [--fit] [--test]   (--fit: 잔차 보정 다시 학습)
"""
import json
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from smp_forecast_app.src.intervals import add_keys, methods, report, score, select
from smp_forecast_app.src.jz_common import JDIR, land_pred, load_pred, rolling_fit
from smp_forecast_app.src.jz_features import CAL, DEM, OM, SPREAD, WEATHER, load as load_feats
from smp_forecast_app.src.jz_resid import LGB, RES
from smp_forecast_app.src.jz_tab import table
from smp_forecast_app.src.model import cqr
from smp_forecast_app.src.wz_ens import mae

warnings.filterwarnings("ignore")
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
W2 = PROCESSED_DIR / "smp_forecast_app" / "wz2"
OUT = JDIR / "jz2"
HS2 = range(2, 8)
LAND_LAMBDA2 = 0.5          # 육지 2차 최종 = 당긴 Chronos-2 small + 0.5 × resid_b (WZ2_choice.json)
JEJU_LAMBDA1 = 0.25         # 제주 1차 (JWZ_choice.json)
LAMBDAS = (0.1, 0.25, 0.5, 0.75, 1.0)
ORDER = ["a", "b0", "j", "b", "c", "d"]
NAMES = {"a": "(a) 제주 1차 그대로", "b0": "(b0) 육지 2차 그대로", "j": "(j) 당긴 제주 Chronos-2 small+육지",
         "b": "(b) 육지 2차 + 제주 1차 잔차 보정", "c": "(c) 육지 2차 + 새 잔차 보정(제주외부)",
         "d": "(d) (c) + 내일 공표 제주 SMP"}
CD = ["b_pred", "b_spread", "b_vs_d1", "b_vs_same", "b_vs_ma28"]
P1 = ["p1_h", "p1_mean", "p1_midmin", "p1_jl_h", "b_vs_p1"]
COLS = {"c": CAL + DEM + SPREAD + WEATHER + OM + CD, "d": CAL + DEM + SPREAD + WEATHER + OM + CD + P1}


def shift(df):
    """1차 (기준일 I+1, H−1) 예측 → 2차 (기준일 I, H)."""
    d = df[df["H"] <= 6].copy()
    d["H"] = d["H"] + 1
    return d


def land2():
    """육지 2차 [date, hour, H, b_pred, b_lo, b_hi, L2_c2s] 2023~ (2025~ = 최종, 그 전 = 당긴 Chronos-2 small)."""
    c = pd.read_csv(W2 / "c2s.csv.gz", parse_dates=["date"])
    r = pd.read_csv(W2 / "resid_b.csv.gz", parse_dates=["date"])
    d = c.merge(r, on=["date", "hour", "H"], how="left")
    d["b_pred"] = (d["pred"] + LAND_LAMBDA2 * d["rhat"]).fillna(d["pred"])
    return d.rename(columns={"qlo": "b_lo", "qhi": "b_hi", "pred": "L2_c2s"})[["date", "hour", "H", "b_pred", "b_lo", "b_hi", "L2_c2s"]]


def jeju_chronos(name):
    d = pd.concat([load_pred(name, hist=True), load_pred(name)], ignore_index=True).drop_duplicates(["date", "hour", "H"], keep="last")
    return shift(d)


def rtable(H, b):
    """2차 잔차 학습표: 1차 피처(판단일 I, H) + 육지 2차 기준 + I+1 공표 제주 SMP(피처 표 H−1의 '전날' 열 = I+1)."""
    d = table(H).merge(b[b["H"] == H].drop(columns="H"), on=["date", "hour"])
    p = load_feats(H - 1)[["date", "hour", "s_d1", "s_d1_mean", "s_d1_midmin", "L_s_d1"]]
    p.columns = ["date", "hour", "p1_h", "p1_mean", "p1_midmin", "p1_land_h"]
    d = d.merge(p, on=["date", "hour"], how="left")
    d["p1_jl_h"] = d["p1_h"] - d["p1_land_h"]
    d["b_spread"] = d["b_hi"] - d["b_lo"]
    d["b_vs_d1"] = d["b_pred"] - d["s_d1"]
    d["b_vs_same"] = d["b_pred"] - d["s_same_h"]
    d["b_vs_ma28"] = d["b_pred"] - d["s_ma28"]
    d["b_vs_p1"] = d["b_pred"] - d["p1_h"]
    d["r"] = d["y"] - d["b_pred"]
    return d


def fit_predict(cols):
    def f(tr, te):
        tr = tr[tr["r"].notna()]
        return lgb.LGBMRegressor(**dict(LGB, objective="l1")).fit(tr[cols], tr["r"]).predict(te[cols])
    return f


def fit():
    OUT.mkdir(parents=True, exist_ok=True)
    t0, b = time.time(), land2()
    out = {k: [] for k in COLS}
    for H in HS2:
        d = rtable(H, b)
        for k in COLS:
            out[k].append(rolling_fit(d, H, "MS", fit_predict(COLS[k])).rename(columns={"pred": "rhat"}))
        print(f"  잔차 보정 H={H} ({time.time() - t0:.0f}초)", flush=True)
    for k, parts in out.items():
        pd.concat(parts).round(3).to_csv(OUT / f"resid_{k}.csv.gz", index=False, encoding="utf-8")


def frame():
    """후보 재료 P: [date, hour, H, y, offday, land2, a, j, rh_b, rh_c, rh_d, + 범위 재료] (대상 2025-01~)."""
    b = land2()
    P = b.rename(columns={"b_pred": "land2", "b_lo": "L2_lo", "b_hi": "L2_hi"})
    P = P[(P["date"] >= "2025-01-01") & P["H"].isin(HS2)]
    lp = land_pred()[["date", "hour", "H", "L_base"]]
    r1 = pd.read_csv(RES / "land_제주외부_l1.csv.gz", parse_dates=["date"]).rename(columns={"rhat": "rh_b"})
    P = P.merge(lp, on=["date", "hour", "H"], how="left").merge(r1, on=["date", "hour", "H"], how="left")
    P["a"] = P["L_base"] + JEJU_LAMBDA1 * P["rh_b"]
    j = jeju_chronos("jc2sx").rename(columns={"pred": "j", "qlo": "jx_lo", "qhi": "jx_hi"})
    js = jeju_chronos("jc2s").rename(columns={"pred": "js", "qlo": "js_lo", "qhi": "js_hi"})
    P = P.merge(j, on=["date", "hour", "H"], how="left").merge(js, on=["date", "hour", "H"], how="left")
    for k in ("c", "d"):
        P = P.merge(pd.read_csv(OUT / f"resid_{k}.csv.gz", parse_dates=["date"]).rename(columns={"rhat": f"rh_{k}"}),
                    on=["date", "hour", "H"], how="left")
    y = load_feats(1)[["date", "hour", "y"]]
    P = P.merge(y, on=["date", "hour"], how="left")
    cal = build_calendar(P["date"].min(), P["date"].max()).set_index("date")["is_offday"]
    P["offday"] = P["date"].map(cal).astype(int)
    P["base_date"] = P["date"] - pd.to_timedelta(P["H"], unit="D")
    return P.sort_values(["H", "date", "hour"]).reset_index(drop=True)


def candidates(P, fit_mask):
    """{코드: (예측, λ)} — 잔차 보정 후보의 λ는 fit_mask MAE로."""
    out = {"a": (P["a"], None), "b0": (P["land2"], None), "j": (P["j"], None)}
    for k in ("b", "c", "d"):
        F = P[fit_mask]
        lam = min(LAMBDAS, key=lambda l: mae(F["y"], F["land2"] + l * F[f"rh_{k}"]))
        out[k] = (P["land2"] + lam * P[f"rh_{k}"], lam)
    return out


def intervals(P, pred, H):
    """지평 H의 범위 후보 → (선택 이름, 성적 행, 전체 기간 lo·hi)."""
    d = P[P["H"] == H].assign(pred=pred[P["H"] == H]).copy()
    d["offday"] = d["offday"].astype(bool)
    k = add_keys(d.sort_values(["date", "hour"]).reset_index(drop=True))
    k = k.assign(err=k["err"].fillna(0), rerr=k["rerr"].fillna(0))
    ev = k["y"].notna().values & (k["date"] >= "2025-07-01").values & (k["date"] < "2026-01-01").values
    g = H - 1
    cand = {"오차분위수·" + n: f for n, f in methods(gap=g, clip0=False).items() if "최근1년" not in n and "365일" not in n}
    for q, lq in (("js", "당긴 제주 Chronos-2 small 분위수"), ("jx", "당긴 제주 Chronos-2 small+육지 분위수")):
        kq = k.assign(qlo=k[f"{q}_lo"], qhi=k[f"{q}_hi"])
        for w in (90, 180):
            cand[f"보정분위수회귀({lq})·갱신{w}일·시간대묶음×휴무일"] = (
                lambda kq, w: lambda dd, e: cqr(kq, e, window=w, key="k_band_off", gap=g, clip0=False))(kq, w)
        cand[f"{lq} 그대로"] = (lambda kq: lambda dd, e: (kq.loc[e, "qlo"].values, kq.loc[e, "qhi"].values))(kq)
    cand["육지 2차 범위 이동"] = lambda dd, e: ((dd["pred"] + dd["L2_lo"] - dd["land2"])[e].values,
                                          (dd["pred"] + dd["L2_hi"] - dd["land2"])[e].values)
    rows = []
    for n, f in cand.items():
        try:
            lo, hi = f(k, ev)
        except Exception as ex:  # noqa: BLE001
            print(f"  범위 {n} 실패: {ex}")
            continue
        for x in report(k[ev].assign(lo=lo, hi=hi), n, "선택 2025 하반기"):
            rows.append({**x, "H": H})
    sel = pd.DataFrame(rows)
    pick = select(sel)
    allm = (k["date"] >= "2025-07-01").values
    lo, hi = cand[pick](k, allm)
    return pick, rows, k[allm].assign(lo=lo, hi=hi)


def main(test=False):
    t0 = time.time()
    P = frame()
    h1 = (P["date"] < "2025-07-01")
    h2 = (P["date"] >= "2025-07-01") & (P["date"] < "2026-01-01")
    y25 = h1 | h2
    c1 = candidates(P, h1)
    full = pd.concat([c1[k][0] for k in ORDER], axis=1).notna().all(axis=1) & P["y"].notna()
    rows = []
    for k in ORDER:
        p, lam = c1[k]
        r = {"코드": k, "후보": NAMES[k], "λ(상반기)": lam, "2025 상반기": mae(P.loc[h1 & full, "y"], p[h1 & full]),
             "2025 하반기": mae(P.loc[h2 & full, "y"], p[h2 & full])}
        g = P[h2 & full]
        mid = g["offday"].astype(bool) & g["hour"].between(11, 15)
        r["하반기 휴무일 한낮"] = mae(g.loc[mid, "y"], p[h2 & full][mid])
        for H in HS2:
            m = h2 & full & (P["H"] == H)
            r[f"하반기 D+{H}"] = mae(P.loc[m, "y"], p[m])
        rows.append(r)
    st = pd.DataFrame(rows)
    best = st["2025 하반기"].min()
    pick = next(k for k in ORDER if st.loc[st["코드"] == k, "2025 하반기"].iloc[0] <= best + 0.05)
    st["선택"] = st["코드"] == pick
    TAB.mkdir(parents=True, exist_ok=True)
    st.round(4).to_csv(TAB / "JW2_select_2025.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(f"비교 행 상반기 {int((h1 & full).sum())} · 하반기 {int((h2 & full).sum())}")
    print(st.round(3).to_string(index=False))
    print(f"선택: {NAMES[pick]}", flush=True)

    cf = candidates(P, y25)
    pred, lam = cf[pick]
    P["pred"] = pred
    OUT.mkdir(parents=True, exist_ok=True)   # 운영 범위의 오차 이력 (항상 공표 가정 2차 예측, 2025-01~)
    P[["date", "hour", "H", "pred"]].dropna().round(3).to_csv(OUT / "JW2_hist.csv.gz", index=False, encoding="utf-8")
    irows, outs, ich = [], [], {}
    for H in HS2:
        ich[H], r, o = intervals(P, P["pred"], H)
        irows += r
        outs.append(o)
        s = pd.DataFrame(r)
        s = s[(s["방법"] == ich[H]) & (s["구간"] == "전체")].iloc[0]
        print(f"범위 D+{H}: {ich[H]} — 2025 하반기 포함률 {s['포함률']:.1%}, 폭 {s['평균폭']:.1f}, 구간점수 {s['구간점수']:.1f}", flush=True)
    pd.DataFrame(irows).round(4).to_csv(TAB / "JW2_intervals_2025H2.csv", index=False, encoding="utf-8-sig")
    W = pd.concat(outs)
    W["stage"] = "21h"
    W["pub"] = np.where(W["base_date"].dt.dayofweek == 1, 0, 1)   # 백테스트 가정: 화요일 판단 = 미공표
    one = pd.read_csv(JDIR / "JWZ_predictions_hourly.csv", parse_dates=["base_date", "date"])
    one = one[one["H"].isin(HS2)].rename(columns={"pred": "pred1", "lo": "lo1", "hi": "hi1"})
    W = W.merge(one[["base_date", "H", "date", "hour", "pred1", "lo1", "hi1"]], on=["base_date", "H", "date", "hour"], how="left")
    W["pred_2"], W["lo_2"], W["hi_2"] = W["pred"], W["lo"], W["hi"]
    tu = W["pub"] == 0
    W.loc[tu, "pred"], W.loc[tu, "lo"], W.loc[tu, "hi"] = W.loc[tu, "pred1"], W.loc[tu, "lo1"], W.loc[tu, "hi1"]
    keep = ["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday", "stage", "pub"]
    W[keep].assign(offday=W["offday"].astype(int)).round(2).to_csv(JDIR / "JW2_predictions_hourly.csv", index=False,
                                                                  encoding="utf-8-sig")
    info = {"선택": NAMES[pick], "후보 코드": pick, "잔차 보정 λ(2025 전체)": lam,
            "λ(2025 전체, 후보별)": {k: cf[k][1] for k in ORDER},
            "잔차 보정 피처": COLS.get(pick), "범위": {str(k): v for k, v in ich.items()}, "범위 하한": "0원 자르기 없음",
            "육지 2차": "WZ2_choice.json (당긴 Chronos-2 small + 0.5 × 2차 잔차 보정)",
            "선택 절차": "λ는 2025 상반기 → 2025 하반기 MAE 비교(0.05원 이내면 단순한 쪽: a < b0 < j < b < c < d) → λ 2025 전체로 재적합; "
                     "범위는 2025 하반기 포함률 75~85% 중 구간점수 최소",
            "잔차 학습": "LightGBM L1(jz_resid.LGB), H별 매월 1일 재학습, 실적 2023-01 ~ 재학습일 − H일",
            "21시 정보": "1차 정보 + I+1 공표 SMP(육지·제주)·수요예측. 운영은 공표가 없으면 1차 값(pub=0)",
            "시험": "2026-01-01 ~ 2026-10-18 한 번 채점 (제주 일주일 2차 1차 시험)"}
    (TAB / "JW2_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 (총 {time.time() - t0:.0f}초)")
    if test:
        score2026(W, P)


def score2026(W, P):
    """2026-01-01 ~ 10-18: 2차 vs 1차 같은 행끼리. 지난주 같은 시간 기준 rMAE."""
    from smp_forecast_app.src.jz_features import load as lf
    T = W[(W["date"] >= "2026-01-01") & (W["date"] <= "2026-10-18") & W["y"].notna() & W["pred1"].notna()].copy()
    nv = lf(1)[["date", "hour", "s_d7"]]   # H=1 표의 D-7 같은 시간 = 지난주 같은 시간 (rMAE 기준, 1차와 같음)
    T = T.merge(nv, on=["date", "hour"], how="left")
    rng = np.random.default_rng(42)
    days = T["date"].unique()
    rows = []
    segs = {"전체": T.index == T.index, "평일": ~T["offday"].astype(bool), "휴무일": T["offday"].astype(bool),
            "휴무일 한낮(11~15시)": T["offday"].astype(bool) & T["hour"].between(11, 15), "급락 시간(30원 미만)": T["y"] < 30}
    for H in ["전체"] + list(HS2):
        for sn, sm in segs.items():
            g = T[sm & ((T["H"] == H) if H != "전체" else True)]
            if g.empty:
                continue
            for nm, (p, lo, hi) in {"2차 (화요일 미공표)": ("pred", "lo", "hi"), "2차 (항상 공표)": ("pred_2", "lo_2", "hi_2"),
                                    "1차": ("pred1", "lo1", "hi1")}.items():
                r = score(g["y"], g[lo], g[hi], g[p])
                base = np.abs(g["y"] - g["s_d7"]).mean()
                rows.append({"H": H, "구간": sn, "예측": nm, "시간수": len(g), "MAE": r["MAE"], "rMAE": r["MAE"] / base,
                             "편향": float((g[p] - g["y"]).mean()),
                             **{c: r[c] for c in ("포함률", "하한아래", "상한위", "평균폭", "구간점수")}})
            if sn == "전체":
                e2 = (g["y"] - g["pred"]).abs().groupby(g["date"]).agg(["sum", "count"])
                e1 = (g["y"] - g["pred1"]).abs().groupby(g["date"]).agg(["sum", "count"])
                bs = []
                for _ in range(2000):
                    i = rng.integers(0, len(e2), len(e2))
                    bs.append(e2["sum"].values[i].sum() / e2["count"].values[i].sum() - e1["sum"].values[i].sum() / e1["count"].values[i].sum())
                rows[-3]["2차−1차 MAE"] = rows[-3]["MAE"] - rows[-1]["MAE"]
                rows[-3]["95%하한"], rows[-3]["95%상한"] = np.quantile(bs, [.025, .975])
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "JW2_test2026.csv", index=False, encoding="utf-8-sig")
    print(t[t["구간"] == "전체"].round(3).to_string(index=False))
    print(t[(t["H"] == "전체")].round(3).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--fit" in a:
        fit()
    main("--test" in a)
