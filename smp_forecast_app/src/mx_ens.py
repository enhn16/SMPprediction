"""한 달 예측 개선(mx) — 후보 비교·앙상블 선택·80% 범위·2026 한 번 채점 (지표별: 일평균·한낮·저녁·한낮 최저).

선택 절차 (팀 규칙 그대로, 2026 채점 전 고정)
  개별 모델은 걷기 전진(매월 재학습, 학습 대상일 2023-01~) 예측. 가중치·λ·순위는 **2024 대상일**로 맞추고
  **2025 대상일 MAE**로 후보 비교 → 1위와 0.05원 이내면 구성 모델이 적은 쪽 → 2024~2025로 다시 맞춤 → 2026 한 번 채점 ('한 달 4차 시험').
후보: 기준(same4·same8·same4×√연료정규화계절비) · 표 모델(lgb_r·lgb_a·lgb_d·et_r) · Chronos(hc2s·hc2·dc2)
      · Chronos + λ × 잔차 보정 · wz_ens.ens_candidates 앙상블(주차 묶음) · 기준 + λ × (후보 − 기준)
범위 (지표별, 2025로 선택, 포함률 75~85% 중 구간점수 최소): 오차 분위수(주차 × 휴무일; 고정 1년·갱신 90·180·365일; 가산·배수)
      + 보정 분위수(Chronos-2 일별 분위수 폭을 예측값에 옮기고 주차×휴무일 보정, 갱신 180·365일). 오차 이력은 대상일 ≤ 기준일만.
출력: outputs/tables/MX1_models.csv, MX2_ensembles_2025.csv, MX3_intervals_2025.csv, MX_choice.json,
      data/processed/smp_forecast_app/mx/MX_predictions_daily.csv (기준일 2024-01-01~; 2024는 예측만=범위 이력용, 범위는 2025~)
      --test: MX4_test2026.csv (현재 모델·기준과 같은 기준일·지평끼리 나란히, 차의 95% 구간)
실행: python -m smp_forecast_app.src.mx_ens [--test]
"""
import json
import sys

import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from smp_forecast_app.src.final_metrics import metrics
from smp_forecast_app.src.intervals import ALPHA, score, select
from smp_forecast_app.src.model_monthly import WEEK, interval
from smp_forecast_app.src.mx_features import MX, STATS, load
from smp_forecast_app.src.wz_ens import ens_candidates, mae

TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
DATA = PROCESSED_DIR / "smp_forecast_app"
LAMBDAS = (0.25, 0.5, 0.75, 1.0)
TEST = ("2026-01-01", "2026-10-18")
SN = {"mean": "일평균", "mid": "한낮 11~15시", "eve": "저녁 18~22시", "midmin": "한낮 최저"}
NAMES = {"same4": "기준: 같은 유형일 4일 평균", "same8": "기준: 같은 유형일 8일 평균", "anchor": "기준: same4 × √연료정규화 계절비",
         "lgb_r": "LightGBM(same4 비율)", "lgb_a": "LightGBM(계절 기준 비율)", "lgb_d": "LightGBM(same4 차이)", "et_r": "엑스트라트리(same4 비율)",
         "hc2s": "Chronos-2 small(시간별)", "hc2": "Chronos-2(시간별)", "dc2": "Chronos-2(일별 다변량)",
         "lstm": "LSTM", "gru": "GRU"}
FAMILY = {"표": ["lgb_r", "lgb_a", "lgb_d", "et_r"], "Chronos": ["hc2s", "hc2", "dc2"], "시계열 신경망": ["lstm", "gru"]}
# v2 (--v2): LSTM·GRU(mx_seq, 팀 규칙 학습) 후보 추가 + 'Chronos 평균·LSTM·GRU 평균 섞기' 유형. 결과는 *_v2 파일 (v1 = 4차 시험 결과 보존)
V2 = "--v2" in sys.argv[1:]
SUF, TRIAL = ("_v2", 5) if V2 else ("", 4)
BASES = ["same4", "same8", "anchor"]
INTV = {"고정1년·가산": (None, "add"), "고정1년·배수": (None, "mult"), "갱신90일·가산": (90, "add"), "갱신90일·배수": (90, "mult"),
        "갱신180일·가산": (180, "add"), "갱신180일·배수": (180, "mult"), "갱신365일·가산": (365, "add"), "갱신365일·배수": (365, "mult")}


def load_all():
    d = load()
    d = d[d["base_date"] >= "2024-01-01"].reset_index(drop=True)
    preds = {}
    for p in sorted(MX.glob("tab_*.csv.gz")):
        preds[p.name[4:-7]] = pd.read_csv(p, parse_dates=["base_date", "date"])
    for p in sorted(MX.glob("chr_*.csv.gz")):
        preds[p.name[4:-7]] = pd.read_csv(p, parse_dates=["base_date", "date"])
    if V2:
        for k in ("lstm", "gru"):
            preds[k] = pd.read_csv(MX / f"seq_{k}.csv.gz", parse_dates=["base_date", "date"])
    return d, preds


def stat_table(d, preds, s):
    P = d[["base_date", "H", "date", "day_type", "offday", f"y_{s}", f"{s}_same4", f"{s}_same8", f"{s}_seasN"]].rename(
        columns={f"y_{s}": "y", f"{s}_same4": "same4", f"{s}_same8": "same8"})
    P["anchor"] = P["same4"] * np.sqrt(P[f"{s}_seasN"].clip(0.6, 1.6).fillna(1))
    for m, f in preds.items():
        cols = ["base_date", "H", f"pred_{s}"] + ([f"qlo_{s}", f"qhi_{s}"] if f"qlo_{s}" in f else [])
        g = f[cols].rename(columns={f"pred_{s}": m, f"qlo_{s}": f"qlo_{m}", f"qhi_{s}": f"qhi_{m}"})
        P = P.merge(g, on=["base_date", "H"], how="left")
    P["W"] = WEEK(P["H"])
    return P


def candidates(P, fit, models, resid, bbest=None):
    """fit 마스크로 맞춘 후보 → {이름: (예측, 구성)}. bbest: 섞기 기준(재적합 때는 선택 때 기준을 그대로)."""
    out = {NAMES[b]: (P[b], [b]) for b in BASES}
    for m in models:
        out[NAMES[m]] = (P[m], [m])
    for tag in resid:
        col = f"resid_{tag}"
        lam = min(LAMBDAS, key=lambda l: mae(P.loc[fit, "y"], P.loc[fit, tag] + l * P.loc[fit, col]))
        out[f"{NAMES[tag]} + {lam} × 잔차 보정"] = (P[tag] + lam * P[col], [tag, col])
    PW = P.assign(H=P["W"])
    ens, rank = ens_candidates(PW, fit, models, family=FAMILY, hs=range(1, 5))
    out.update(ens)
    seq, chrs = [m for m in ("lstm", "gru") if m in models], [m for m in ("hc2s", "hc2", "dc2") if m in models]
    if seq and chrs:   # v2: Chronos 평균 × (1−λ) + LSTM·GRU 평균 × λ (λ는 fit 구간)
        pc, ps = P[chrs].mean(axis=1), P[seq].mean(axis=1)
        lam = min(LAMBDAS[:3], key=lambda l: mae(P.loc[fit, "y"], (1 - l) * pc[fit] + l * ps[fit]))
        out[f"Chronos 평균 × {1 - lam} + LSTM·GRU 평균 × {lam}"] = ((1 - lam) * pc + lam * ps, chrs + seq)
    bbest = bbest or min(BASES, key=lambda b: mae(P.loc[fit, "y"], P.loc[fit, b]))
    candidates.bbest = bbest
    for name, (p, ms) in list(out.items()):
        if ms == [bbest]:
            continue
        lam = min(LAMBDAS[:3], key=lambda l: mae(P.loc[fit, "y"], P.loc[fit, bbest] + l * (p[fit] - P.loc[fit, bbest])))
        out[f"{NAMES[bbest]} + {lam} × ({name} − 기준)"] = (P[bbest] + lam * (p - P[bbest]), sorted(set([bbest] + list(ms))))
    return out


def cqr_week(d, ev_mask, kind, qcol, window):
    """Chronos 일별 분위수 폭(qlo−중앙, qhi−중앙)을 예측값에 옮기고, 점수 max(lo−y, y−hi)의 80% 분위수로 주차×휴무일별 보정."""
    d = d.assign(k=WEEK(d["H"]).astype(str) + "_" + d["offday"].astype(str))
    d["lo0"] = d[kind] + (d[f"qlo_{qcol}"] - d[qcol])
    d["hi0"] = d[kind] + (d[f"qhi_{qcol}"] - d[qcol])
    d["cs"] = np.maximum(d["lo0"] - d["y"], d["y"] - d["hi0"])
    ev = d[ev_mask]
    hist = d[d["cs"].notna()]
    lo, hi = pd.Series(np.nan, index=ev.index), pd.Series(np.nan, index=ev.index)
    for i, part in ev.groupby("base_date"):
        rows = hist[(hist["date"] <= i) & (hist["date"] > i - pd.Timedelta(days=window))]
        q = rows.groupby("k")["cs"].quantile(1 - ALPHA)
        c = part["k"].map(q).fillna(rows["cs"].quantile(1 - ALPHA))
        lo[part.index] = np.minimum((part["lo0"] - c).clip(lower=0), part[kind])
        hi[part.index] = np.maximum(part["hi0"] + c, part[kind])
    return lo.values, hi.values


def interval_methods():
    m = {k: (lambda w, f: lambda d, e: interval(d.rename(columns={"y": "y_x"}), e, "x", "pred", w, f))(w, f) for k, (w, f) in INTV.items()}
    for w in (180, 365):
        m[f"보정분위수(Chronos-2 일별)·갱신{w}일"] = (lambda w: lambda d, e: cqr_week(d, e, "pred", "dc2", w))(w)
    return m


def main():
    d, preds = load_all()
    models = [m for m in NAMES if m in preds and m not in BASES]
    resid = [t for t in ("hc2s", "hc2", "dc2") if f"resid_{t}" in preds and t in preds]
    mrows, erows, irows, choice, outs, coefs = [], [], [], {}, None, {}
    for s in STATS:
        P = stat_table(d, preds, s)
        for t in resid:
            P = P.rename(columns={f"resid_{t}": f"resid_{t}"})
        y24, y25 = P["date"].dt.year == 2024, P["date"].dt.year == 2025
        full = P[models + BASES + [f"resid_{t}" for t in resid]].notna().all(axis=1) & P["y"].notna()
        for m in BASES + models:
            mrows.append({"지표": SN[s], "모델": NAMES[m], "2024": mae(P.loc[y24 & full, "y"], P.loc[y24 & full, m]),
                          "2025": mae(P.loc[y25 & full, "y"], P.loc[y25 & full, m]),
                          **{f"2025 {w}주차": mae(P.loc[y25 & full & (P["W"] == w), "y"], P.loc[y25 & full & (P["W"] == w), m]) for w in range(1, 5)}})
        cand = candidates(P, y24 & full, models, resid)
        bsel = candidates.bbest
        er = pd.DataFrame([{"지표": SN[s], "후보": n, "구성 수": len(ms), "2024(맞춤)": mae(P.loc[y24 & full, "y"], p[y24 & full]),
                            "2025(선택)": mae(P.loc[y25 & full, "y"], p[y25 & full]), "구성": ", ".join(NAMES.get(m, m) for m in ms)}
                           for n, (p, ms) in cand.items()]).sort_values("2025(선택)")
        top = er.iloc[0]["2025(선택)"]
        pick = er[er["2025(선택)"] <= top + 0.05].sort_values(["구성 수", "2025(선택)"]).iloc[0]["후보"]
        er["선택"] = np.where(er["후보"] == pick, "✓", "")
        erows.append(er)
        # 2024~2025로 다시 맞춤 (λ·가중치 재적합; 이름의 λ 값은 바뀔 수 있어 유형으로 찾음)
        refit = candidates(P, (y24 | y25) & full, models, resid, bbest=bsel)
        key = _same_kind(pick, refit)
        P["pred"] = refit[key][0]
        P["pred_sel"] = cand[pick][0]           # 2025 선택 때의 예측 (범위 선택용, 2024 맞춤)
        coefs[s] = linear_coef(P, refit[key][1], "pred")
        print(f"[{SN[s]}] 선택: {pick} → 재적합 {key} | 2025 {er.iloc[0]['2025(선택)']:.2f} (1위 {er.iloc[0]['후보']}), "
              f"same4 {mae(P.loc[y25 & full, 'y'], P.loc[y25 & full, 'same4']):.2f}", flush=True)
        # 범위 선택 (2025, 예측 = 2024로 맞춘 선택 후보)
        ms = interval_methods()
        Ds = P.assign(pred=P["pred_sel"])
        ev = (Ds["date"].dt.year == 2025).values & Ds["y"].notna().values & Ds["pred"].notna().values
        part = []
        for name, fn in ms.items():
            lo, hi = fn(Ds, ev)
            part.append({"지표": SN[s], "방법": name, "구간": "전체", **score(Ds.loc[ev, "y"], lo, hi, Ds.loc[ev, "pred"])})
        it = pd.DataFrame(part)
        ich = select(it)
        it["선택"] = np.where(it["방법"] == ich, "✓", "")
        irows.append(it)
        # 산출: 2025 = 선택 때 예측·범위, 2026~ = 재적합 예측 + 2025까지 오차로 만든 범위
        m24 = (Ds["date"].dt.year == 2024).values
        P.loc[m24, "out_pred"] = Ds.loc[m24, "pred"].values      # 2024 = 범위 오차 이력용 (범위 없음)
        m25 = (Ds["date"].dt.year == 2025).values & Ds["pred"].notna().values
        lo, hi = ms[ich](Ds, m25)
        P.loc[m25, "out_pred"], P.loc[m25, "lo"], P.loc[m25, "hi"] = Ds.loc[m25, "pred"].values, lo, hi
        D6 = P.copy()
        D6.loc[D6["date"] >= "2026-01-01", "pred"] = P.loc[P["date"] >= "2026-01-01", "pred"]
        D6.loc[D6["date"] < "2026-01-01", "pred"] = P.loc[P["date"] < "2026-01-01", "pred_sel"]
        m26 = (D6["date"] >= "2026-01-01").values & D6["pred"].notna().values
        lo, hi = ms[ich](D6, m26)
        P.loc[m26, "out_pred"], P.loc[m26, "lo"], P.loc[m26, "hi"] = D6.loc[m26, "pred"].values, lo, hi
        r = it[it["방법"] == ich].iloc[0]
        print(f"[{SN[s]}] 범위 {ich} — 2025 포함률 {r['포함률']:.1%}, 폭 {r['평균폭']:.1f}, 구간점수 {r['구간점수']:.1f}", flush=True)
        o = P[["base_date", "H", "date", "day_type", "offday", "y", "out_pred", "lo", "hi"]].rename(
            columns={"y": f"y_{s}", "out_pred": f"pred_{s}", "lo": f"lo_{s}", "hi": f"hi_{s}"})
        outs = o if outs is None else outs.merge(o.drop(columns=["date", "day_type", "offday"]), on=["base_date", "H"], how="outer")
        choice[s] = {"점예측(2025 선택)": pick, "재적합": key, "범위": ich, "구성": cand[pick][1]}
    pd.DataFrame(mrows).round(3).to_csv(TAB / f"MX1_models{SUF}.csv", index=False, encoding="utf-8-sig")
    pd.concat(erows).round(3).to_csv(TAB / f"MX2_ensembles_2025{SUF}.csv", index=False, encoding="utf-8-sig")
    pd.concat(irows).round(4).to_csv(TAB / f"MX3_intervals_2025{SUF}.csv", index=False, encoding="utf-8-sig")
    outs.sort_values(["base_date", "H"]).round(2).to_csv(MX / f"MX_predictions_daily{SUF}.csv", index=False, encoding="utf-8-sig")
    info = {"지표별": choice, "선택 절차": "개별 모델 걷기 전진(매월 재학습, 학습 대상일 2023-01~) → 가중치·λ는 2024 대상일로 → 2025 MAE 비교"
            "(0.05원 이내면 구성 적은 쪽) → 2024~2025로 재적합 → 2026 한 번 채점", "범위 선택": "2025 포함률 75~85% 중 구간점수 최소",
            "시험": f"{TEST[0]} ~ {TEST[1]} (한 달 예측 {TRIAL}차 시험)"}
    (TAB / f"MX_choice{SUF}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    (MX / f"mx_ens_coef{SUF}.json").write_text(json.dumps(coefs, ensure_ascii=False, indent=1), encoding="utf-8")


def linear_coef(P, comps, col):
    """선택 예측은 주차마다 구성 열의 선형식(평균·섞기·NNLS·Ridge 스태킹 모두) → 주차별 계수를 최소제곱으로 되찾아 운영에 씀.
    재현 오차(최대 절대 차)를 함께 저장."""
    out = {}
    for w in range(1, 5):
        m = (P["W"] == w) & P[comps + [col]].notna().all(axis=1)
        X = np.column_stack([P.loc[m, comps].values, np.ones(m.sum())])
        b, *_ = np.linalg.lstsq(X, P.loc[m, col].values, rcond=None)
        err = float(np.abs(X @ b - P.loc[m, col].values).max())
        out[str(w)] = {"계수": dict(zip(comps, b[:-1].round(8).tolist())), "절편": float(b[-1]), "재현 최대 차": round(err, 6)}
    return out


def _same_kind(pick, refit):
    """재적합 후보 중 선택 후보와 같은 유형 (λ 숫자만 다른 이름 허용)."""
    if pick in refit:
        return pick
    import re
    pat = re.sub(r"\d+(\.\d+)?", "#", pick)
    for k in refit:
        if re.sub(r"\d+(\.\d+)?", "#", k) == pat:
            return k
    raise KeyError(pick)


# ---------------------------------------------------------------- 2026 한 번 채점
def boot_diff(g, a, b, n=1000, seed=42):
    e = g.assign(da=(g["y"] - g[a]).abs(), db=(g["y"] - g[b]).abs()).groupby("date")[["da", "db"]].agg(["sum", "count"])
    sa, sb, cnt = e[("da", "sum")].values, e[("db", "sum")].values, e[("da", "count")].values
    idx = np.random.default_rng(seed).integers(0, len(sa), (n, len(sa)))
    return np.percentile(sb[idx].sum(1) / cnt[idx].sum(1) - sa[idx].sum(1) / cnt[idx].sum(1), [2.5, 97.5])


def test():
    X = pd.read_csv(MX / f"MX_predictions_daily{SUF}.csv", parse_dates=["base_date", "date"])
    Z = pd.read_csv(DATA / "MZ_predictions_daily.csv", parse_dates=["base_date", "date"])
    f = load()[["base_date", "H"] + [f"{s}_same4" for s in STATS]]
    rows = []
    for s in STATS:
        g = X[["base_date", "H", "date", f"y_{s}", f"pred_{s}", f"lo_{s}", f"hi_{s}"]].merge(
            Z[["base_date", "H", f"pred_{s}", f"lo_{s}", f"hi_{s}"]].rename(columns={f"pred_{s}": "p0", f"lo_{s}": "lo0", f"hi_{s}": "hi0"}),
            on=["base_date", "H"]).merge(f[["base_date", "H", f"{s}_same4"]], on=["base_date", "H"])
        g = g.rename(columns={f"y_{s}": "y", f"pred_{s}": "p", f"lo_{s}": "lo", f"hi_{s}": "hi", f"{s}_same4": "same4"})
        g = g[(g["date"] >= TEST[0]) & (g["date"] <= TEST[1]) & g["y"].notna() & g["p"].notna() & g["p0"].notna()]
        for w in ["전체", 1, 2, 3, 4]:
            x = g if w == "전체" else g[WEEK(g["H"]) == w]
            mn, m0 = metrics(x["y"], x["p"], x["lo"], x["hi"], x["same4"]), metrics(x["y"], x["p0"], x["lo0"], x["hi0"], x["same4"])
            r = {"지표": SN[s], "주차": w, **{f"새 {k}": v for k, v in mn.items()}, **{f"현재 {k}": v for k, v in m0.items() if k != "표본수"}}
            r["MAE 차(새−현재)"] = mn["MAE"] - m0["MAE"]
            r["MAE 차(새−기준)"] = mn["MAE"] - mn["단순 기준 MAE"]
            if w == "전체":
                r["차(새−현재) 95% 하"], r["차(새−현재) 95% 상"] = boot_diff(x, "p0", "p")
                r["차(새−기준) 95% 하"], r["차(새−기준) 95% 상"] = boot_diff(x, "same4", "p")
            rows.append(r)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / f"MX4_test2026{SUF}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t[["지표", "주차", "새 MAE", "현재 MAE", "새 단순 기준 MAE", "새 rMAE", "새 포함률", "현재 포함률", "새 평균 폭", "현재 평균 폭",
             "새 구간점수", "현재 구간점수"]].round(3).to_string(index=False))
    print(t[t["주차"] == "전체"][["지표", "MAE 차(새−현재)", "차(새−현재) 95% 하", "차(새−현재) 95% 상", "MAE 차(새−기준)",
                                 "차(새−기준) 95% 하", "차(새−기준) 95% 상"]].round(3).to_string(index=False))


if __name__ == "__main__":
    test() if "--test" in sys.argv[1:] else main()
