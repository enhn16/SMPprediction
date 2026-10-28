"""일주일 예측 2차 (21시) — 후보 선택(2025) · 범위 선택 · 2026 한 번 채점. 설계: docs/WEEK21.md

점예측 후보 (2차 H = 2~7)
  a   : 1차 그대로 = 1차 Chronos-2 small + 0.5 × 1차 잔차 보정(외부만)   (WZ_predictions_hourly와 같은 값)
  b0  : 당긴 Chronos-2 small (보정 없음)
  b, c, d, dall : 당긴 Chronos-2 small + λ × 2차 잔차 보정 (wz2_backtest)
선택 (1차 wz_ens와 같은 절차, 2026 채점 전 고정)
  λ ∈ {0.25, 0.5, 0.75, 1.0}는 2025 상반기 MAE로 → 2025 하반기 MAE 비교 (모든 기준일, H=2~7)
  → 1위와 0.05원 이내면 단순한 쪽 (a < b0 < b < c < d < dall) → λ를 2025 전체로 다시 맞춤
범위 (지평별): 오차 이력 2025-01~, gap = H−1 (21시엔 I+1까지 실적 확보), 2025 하반기 포함률 75~85% 중 구간점수 최소
출력 (기본 실행 = 2025만): outputs/tables/WZ2_select_2025.csv, WZ2_intervals_2025H2.csv, WZ2_choice.json,
      data/processed/smp_forecast_app/wz2/WZ2_predictions_hourly.csv (대상 2025-07-01~, '화요일 미공표' 가정으로 pub)
      --test: 2026-01-01 ~ 10-18 한 번 채점 → WZ2_test2026.csv, WZ2_demand_value.csv
실행: python -m smp_forecast_app.src.wz2_select [--test]
"""
import json
import sys

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from smp_forecast_app.src.final_metrics import hourly_segments, metrics
from smp_forecast_app.src.intervals import add_keys, methods, report, select
from smp_forecast_app.src.model import cqr
from smp_forecast_app.src.wz2_backtest import HS2, W2, SETS2, chronos2
from smp_forecast_app.src.wz_common import OUT as WZOO, actual_table, load_pred
from smp_forecast_app.src.wz_resid import RES as RES1

DATA = PROCESSED_DIR / "smp_forecast_app"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
LAMBDAS = (0.25, 0.5, 0.75, 1.0)
ORDER = ["a", "b0"] + list(SETS2)                      # 단순한 순서
NAMES = {"a": "(a) 1차 그대로", "b0": "(b0) 당긴 Chronos만", "b": "(b) 당긴 Chronos + 1차 피처 보정",
         "c": "(c) + 내일 공표 SMP", "d": "(d) + 내일 수요예측", "dall": "(d+) + 1차 전체 피처"}
TEST = ("2026-01-01", "2026-10-18")
LAM1 = json.loads((TAB / "WZ_choice.json").read_text(encoding="utf-8"))["잔차 보정 λ"]


def mae(y, p):
    ok = y.notna() & p.notna()
    return float((y[ok] - p[ok]).abs().mean())


def load():
    """[date, hour, H, base_date, y, offday, c2s, qlo, qhi(Chronos-2 당김), qlo_s, qhi_s(Chronos-2 small 당김), a, rhat_*]."""
    ch = chronos2()
    P = ch["c2s"].rename(columns={"pred": "c2s", "qlo": "qlo_s", "qhi": "qhi_s"})
    P = P[P["date"] >= "2025-01-01"].merge(ch["c2"][["date", "hour", "H", "qlo", "qhi"]], on=["date", "hour", "H"], how="left")
    for k in SETS2:
        r = pd.read_csv(W2 / f"resid_{k}.csv.gz", parse_dates=["date"]).rename(columns={"rhat": f"rhat_{k}"})
        P = P.merge(r, on=["date", "hour", "H"], how="left")
    # (a) 1차: 기준일 I, 같은 H
    c1 = load_pred("c2s")[["date", "hour", "H", "pred", "qlo", "qhi"]].rename(columns={"pred": "c1", "qlo": "qlo_1s", "qhi": "qhi_1s"})
    r1 = pd.read_csv(RES1 / "외부만.csv.gz", parse_dates=["date"]).rename(columns={"rhat": "rhat1"})
    q1 = pd.read_csv(WZOO / "c2.csv.gz", parse_dates=["date"])[["date", "hour", "H", "qlo", "qhi"]].rename(columns={"qlo": "qlo_1", "qhi": "qhi_1"})
    P = P.merge(c1, on=["date", "hour", "H"], how="left").merge(r1, on=["date", "hour", "H"], how="left").merge(q1, on=["date", "hour", "H"], how="left")
    P["a"] = P["c1"] + LAM1 * P["rhat1"]
    P["b0"] = P["c2s"]
    bf = W2 / "resid1_backfill.csv.gz"     # 참고: 1차를 같은 2023 보강 자료로 다시 학습 (선택 후보 아님)
    if bf.exists():
        r1b = pd.read_csv(bf, parse_dates=["date"]).rename(columns={"rhat": "rhat1b"})
        P = P.merge(r1b, on=["date", "hour", "H"], how="left")
        P["a_bf"] = P["c1"] + LAM1 * P["rhat1b"]
    act = actual_table()[["date", "hour", "y"]]
    P = P.merge(act, on=["date", "hour"], how="left")
    cal = build_calendar(P["date"].min(), P["date"].max()).set_index("date")["is_offday"]
    P["offday"] = P["date"].map(cal).astype(bool)
    P["base_date"] = P["date"] - pd.to_timedelta(P["H"], unit="D")
    return P.sort_values(["H", "date", "hour"]).reset_index(drop=True)


def point(P, k, lam):
    return P[k] if k in ("a", "b0") else P["c2s"] + lam * P[f"rhat_{k}"]


def fit_lambda(P, k, mask):
    if k in ("a", "b0"):
        return None
    return min(LAMBDAS, key=lambda l: mae(P.loc[mask, "y"], point(P, k, l)[mask]))


def select_point(P):
    y25 = P["date"].dt.year == 2025
    h1, h2 = y25 & (P["date"] < "2025-07-01"), y25 & (P["date"] >= "2025-07-01")
    full = P[["a", "c2s"] + [f"rhat_{k}" for k in SETS2]].notna().all(axis=1) & P["y"].notna()
    rows = []
    for k in ORDER:
        lam = fit_lambda(P, k, h1 & full)
        p = point(P, k, lam)
        r = {"후보": k, "이름": NAMES[k], "λ(상반기)": lam, "2025 상반기": mae(P.loc[h1 & full, "y"], p[h1 & full]),
             "2025 하반기": mae(P.loc[h2 & full, "y"], p[h2 & full])}
        for H in HS2:
            m = h2 & full & (P["H"] == H)
            r[f"하반기 D+{H}"] = mae(P.loc[m, "y"], p[m])
        for seg, sm in hourly_segments(P[h2 & full]).items():
            r[f"하반기 {seg}"] = mae(P.loc[h2 & full, "y"][sm], p[h2 & full][sm])
        rows.append(r)
    t = pd.DataFrame(rows)
    best = t["2025 하반기"].min()
    pick = next(k for k in ORDER if t.set_index("후보").loc[k, "2025 하반기"] <= best + 0.05)
    if "a_bf" in P:   # 참고 행 (선택에 쓰지 않음)
        r = {"후보": "ref_a_bf", "이름": "참고: 1차 재학습(2023 보강 자료)", "2025 상반기": mae(P.loc[h1 & full, "y"], P.loc[h1 & full, "a_bf"]),
             "2025 하반기": mae(P.loc[h2 & full, "y"], P.loc[h2 & full, "a_bf"])}
        for H in HS2:
            m = h2 & full & (P["H"] == H)
            r[f"하반기 D+{H}"] = mae(P.loc[m, "y"], P.loc[m, "a_bf"])
        t = pd.concat([t, pd.DataFrame([r])], ignore_index=True)
    lam = fit_lambda(P, pick, y25 & full)
    t["선택"] = np.where(t["후보"] == pick, "✓", "")
    return t, pick, lam, {k: fit_lambda(P, k, y25 & full) for k in ORDER}


def interval_candidates(allk, H, lam_r):
    gap = H - 1
    cand = {}
    for k, f in methods(gap=gap).items():
        if "최근1년" in k or "365일" in k:
            continue
        cand["오차분위수·" + k] = f
    for tag, label in (("", "Chronos-2"), ("_s", "Chronos-2 small")):
        kq = allk.assign(qlo=allk[f"qlo{tag}"], qhi=allk[f"qhi{tag}"])
        cand[f"보정분위수회귀({label} 분위수)·갱신90일·시간대묶음×휴무일"] = (lambda kq: lambda d, e: cqr(kq, e, window=90, key="k_band_off", gap=gap))(kq)
        cand[f"{label} 분위수 그대로"] = (lambda kq: lambda d, e: (kq.loc[e, "qlo"].clip(lower=0).values, kq.loc[e, "qhi"].values))(kq)
        if lam_r is not None:
            ks = kq.assign(qlo=kq["qlo"] + lam_r, qhi=kq["qhi"] + lam_r)
            cand[f"{label} 분위수 + 잔차 이동"] = (lambda ks: lambda d, e: (ks.loc[e, "qlo"].clip(lower=0).values, ks.loc[e, "qhi"].values))(ks)
    return cand


def build_intervals(P, pick, lam):
    """지평별 범위 선택(2025 하반기) → 대상 2025-07-01~ 예측표."""
    P = P.assign(pred=point(P, pick, lam))
    irows, outs, choice = [], [], {}
    for H in HS2:
        d = P[(P["H"] == H) & P["pred"].notna()].copy()
        allk = add_keys(d.sort_values(["date", "hour"]).reset_index(drop=True))
        allk = allk.assign(err=allk["err"].fillna(0), rerr=allk["rerr"].fillna(0))
        lam_r = None if pick in ("a", "b0") else (lam * allk[f"rhat_{pick}"]).fillna(0).values
        has_y = allk["y"].notna().values
        ev = has_y & (allk["date"] >= "2025-07-01").values & (allk["date"] < "2026-01-01").values
        cand = interval_candidates(allk, H, lam_r)
        part = []
        for name, fn in cand.items():
            lo, hi = fn(allk, ev)
            r = report(allk[ev].assign(lo=lo, hi=hi), name, "선택 2025 하반기")
            for x in r:
                x["H"] = H
            part += r
        sel = pd.DataFrame(part)
        choice[H] = select(sel)
        irows += part
        allmask = (allk["date"] >= "2025-07-01").values
        lo, hi = cand[choice[H]](allk, allmask)
        outs.append(allk[allmask].assign(lo=np.clip(lo, 0, None), hi=hi))
        r = sel[(sel["방법"] == choice[H]) & (sel["구간"] == "전체")].iloc[0]
        r1 = sel[(sel["방법"] == "Chronos-2 분위수 그대로") & (sel["구간"] == "전체")].iloc[0]
        print(f"범위 D+{H}: {choice[H]} — 2025 하반기 포함률 {r['포함률']:.1%}, 폭 {r['평균폭']:.1f}, 구간점수 {r['구간점수']:.1f}"
              f" (1차 방법 Chronos-2 그대로: {r1['포함률']:.1%}, {r1['평균폭']:.1f}, {r1['구간점수']:.1f})", flush=True)
    return pd.concat(outs), pd.DataFrame(irows), choice


def first_stage():
    w = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["base_date", "date"])
    return w[w["H"] >= 2][["base_date", "H", "date", "hour", "pred", "lo", "hi"]].rename(
        columns={"pred": "pred1", "lo": "lo1", "hi": "hi1"})


def main():
    P = load()
    # 확인: (a) = 1차 운영 백테스트 값
    chk = P.merge(first_stage(), on=["base_date", "H", "date", "hour"])
    print(f"(a) 재현 확인: {len(chk)}행, 최대 차 {np.abs(chk['a'] - chk['pred1']).max():.3f}원", flush=True)
    t, pick, lam, lam_all = select_point(P)
    t.round(3).to_csv(TAB / "WZ2_select_2025.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t[["이름", "λ(상반기)", "2025 상반기", "2025 하반기"] + [f"하반기 D+{H}" for H in HS2] + ["하반기 휴무일", "선택"]].round(3).to_string(index=False))
    print(f"\n선택: {NAMES[pick]} (λ 2025 전체 = {lam})\n", flush=True)
    W, it, choice = build_intervals(P, pick, lam)
    it.round(4).to_csv(TAB / "WZ2_intervals_2025H2.csv", index=False, encoding="utf-8-sig")
    # 운영 형식 + '화요일 미공표' 가정 (화요일 기준일 = 1차 값)
    W = W.merge(first_stage(), on=["base_date", "H", "date", "hour"], how="left")
    tue = (W["base_date"].dt.dayofweek == 1) & W["pred1"].notna()
    W["pub"] = (~tue).astype(int)
    for c in ("pred", "lo", "hi"):
        W.loc[tue, c] = W.loc[tue, f"{c}1"]
    W["offday"], W["stage"] = W["offday"].astype(int), "21h"
    keep = ["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday", "stage", "pub"]
    W[keep].round(2).to_csv(W2 / "WZ2_predictions_hourly.csv", index=False, encoding="utf-8-sig")
    info = {"선택": NAMES[pick], "후보 코드": pick, "잔차 보정 λ": lam,
            "잔차 보정 피처": SETS2.get(pick), "범위": {str(k): v for k, v in choice.items()},
            "λ(2025 전체, 후보별)": {k: v for k, v in lam_all.items()},
            "Chronos": "1차 Chronos-2 small·Chronos-2를 기준일 +1, H +1로 당겨 씀 (입력이 SMP·달력뿐이라 같음, 재실행 확인 최대 차 0.001원)",
            "선택 절차": "λ는 2025 상반기 → 2025 하반기 MAE 비교(0.05원 이내면 단순한 쪽) → λ 2025 전체로 재적합; 범위는 2025 하반기 포함률 75~85% 중 구간점수 최소",
            "잔차 학습": "LightGBM L1(wz_resid.LGB), H별 매월 1일 재학습, 실적 2023-01 ~ 재학습일 − H일",
            "21시 정보": "1차 정보 + I+1 공표 SMP·하루전 수요예측. 운영은 공표가 없으면 1차 값(pub=0)",
            "시험": f"{TEST[0]} ~ {TEST[1]} 한 번 채점 (일주일 2차 예측 2차 시험: 2023 보강 자료로 재학습, 1차 시험 결과는 wz2/v1/)"}
    (TAB / "WZ2_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: info[k] for k in ("선택", "잔차 보정 λ", "범위")}, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- 2026 한 번 채점
def boot_diff(g, col_a, col_b, n=1000, seed=42):
    """날짜 단위 부트스트랩: MAE(b) − MAE(a)의 95% 구간."""
    e = g.assign(da=(g["y"] - g[col_a]).abs(), db=(g["y"] - g[col_b]).abs()).groupby("date")[["da", "db"]].agg(["sum", "count"])
    sa, sb, cnt = e[("da", "sum")].values, e[("db", "sum")].values, e[("da", "count")].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sa), (n, len(sa)))
    d = sb[idx].sum(1) / cnt[idx].sum(1) - sa[idx].sum(1) / cnt[idx].sum(1)
    return np.percentile(d, [2.5, 97.5])


def test():
    info = json.loads((TAB / "WZ2_choice.json").read_text(encoding="utf-8"))
    P = load()
    W = pd.read_csv(W2 / "WZ2_predictions_hourly.csv", parse_dates=["base_date", "date"])
    W = W.merge(first_stage(), on=["base_date", "H", "date", "hour"], how="left")
    pick, lam = info["후보 코드"], info["잔차 보정 λ"]
    raw = P.assign(pred_all=point(P, pick, lam))[["base_date", "H", "date", "hour", "pred_all"] + (["a_bf"] if "a_bf" in P else [])]
    W = W.merge(raw, on=["base_date", "H", "date", "hour"], how="left")
    n7 = load_pred("naive7")[["date", "hour", "H", "pred"]].rename(columns={"pred": "naive7"})
    W = W.merge(n7, on=["date", "hour", "H"], how="left")
    # 참고: 다음 날 15시 1차 (기준일 I+1, H−1) · 내일 공표 같은 시간
    w1 = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["date"])[["date", "hour", "H", "pred"]]
    W = W.merge(w1.assign(H=w1["H"] + 1).rename(columns={"pred": "next15"}), on=["date", "hour", "H"], how="left")
    from common.loaders import load_smp_actual
    s = load_smp_actual("육지")[["date", "hour", "smp"]]
    W["p1day"] = W["date"] - pd.to_timedelta(W["H"] - 1, unit="D")
    W = W.merge(s.rename(columns={"date": "p1day", "smp": "naive_p1"}), on=["p1day", "hour"], how="left")
    T = W[(W["date"] >= TEST[0]) & (W["date"] <= TEST[1]) & W["y"].notna() & W["pred1"].notna()].copy()
    tue = T["base_date"].dt.dayofweek == 1
    # 항상 공표 가정의 범위 = 화요일도 2차 범위: 범위는 파일에 1차로 대체돼 있으므로 다시 계산하지 않고 화요일 행만 따로 보고
    scen = {"화요일 미공표 (운영 파일)": (T, "pred", "lo", "hi"),
            "화요일 제외": (T[~tue], "pred", "lo", "hi"),
            "항상 공표 (점예측만)": (T, "pred_all", None, None),
            "화요일 기준일만 (공표 가정, 점예측)": (T[tue], "pred_all", None, None)}
    rows = []
    for sc, (g, pc, lc, hc) in scen.items():
        for seg, m in hourly_segments(g).items():
            for H in ["전체"] + list(HS2):
                x = g[m] if H == "전체" else g[m & (g["H"] == H)]
                if not len(x):
                    continue
                base = {"가정": sc, "구분": seg, "지평": H if H == "전체" else f"D+{H}"}
                m2 = metrics(x["y"], x[pc], x[lc] if lc else None, x[hc] if hc else None, x["naive7"])
                m1 = metrics(x["y"], x["pred1"], x["lo1"] if lc else None, x["hi1"] if hc else None, x["naive7"])
                r = {**base, **{f"2차 {k}": v for k, v in m2.items()}, **{f"1차 {k}": v for k, v in m1.items() if k != "표본수"}}
                r["MAE 차(2차−1차)"] = m2["MAE"] - m1["MAE"]
                if seg == "전체":
                    r["차 95%구간 하"], r["차 95%구간 상"] = boot_diff(x, "pred1", pc)
                    r["참고: 다음 날 15시 1차 MAE"] = mae(x["y"], x["next15"])
                    if "a_bf" in x:
                        r["참고: 1차 재학습(보강) MAE"] = mae(x["y"], x["a_bf"])
                    r["참고: 내일 공표 같은 시간 MAE"] = mae(x["y"], x["naive_p1"])
                rows.append(r)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(TAB / "WZ2_test2026.csv", index=False, encoding="utf-8-sig")
    show = out[(out["구분"] == "전체")][["가정", "지평", "2차 MAE", "1차 MAE", "MAE 차(2차−1차)", "차 95%구간 하", "차 95%구간 상",
                                     "2차 포함률", "1차 포함률", "2차 평균 폭", "1차 평균 폭", "참고: 다음 날 15시 1차 MAE"]]
    pd.set_option("display.width", 250)
    print(show.round(3).to_string(index=False))
    # 수요 정보의 가치: (d) − (c), 각자 λ(2025 전체)
    lam_all = info["λ(2025 전체, 후보별)"]
    Q = P.assign(pc=point(P, "c", lam_all["c"]), pd_=point(P, "d", lam_all["d"]))
    Q = Q[Q["y"].notna() & Q["pc"].notna() & Q["pd_"].notna()]
    drows = []
    for per, (a, b) in {"2025 하반기 (선택)": ("2025-07-01", "2025-12-31"), "2026 (시험)": TEST}.items():
        g0 = Q[(Q["date"] >= a) & (Q["date"] <= b)]
        for seg, m in hourly_segments(g0).items():
            for H in ["전체"] + list(HS2):
                x = g0[m] if H == "전체" else g0[m & (g0["H"] == H)]
                if not len(x):
                    continue
                lo_, hi_ = boot_diff(x, "pc", "pd_") if seg == "전체" else (np.nan, np.nan)
                drows.append({"기간": per, "구분": seg, "지평": H if H == "전체" else f"D+{H}", "시간수": len(x),
                              "(c) 내일 SMP MAE": mae(x["y"], x["pc"]), "(d) +수요예측 MAE": mae(x["y"], x["pd_"]),
                              "차 (d)−(c)": mae(x["y"], x["pd_"]) - mae(x["y"], x["pc"]), "95%구간 하": lo_, "95%구간 상": hi_})
    dv = pd.DataFrame(drows)
    dv.round(4).to_csv(TAB / "WZ2_demand_value.csv", index=False, encoding="utf-8-sig")
    print(dv[dv["구분"] == "전체"].round(3).to_string(index=False))


if __name__ == "__main__":
    test() if "--test" in sys.argv[1:] else main()
