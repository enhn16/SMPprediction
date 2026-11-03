"""한 달 예측 mx3 — 시간별(7~34일 뒤 24칸) 후보 비교·선택·80% 범위. 지표 하나 = 시간별 MAE·포함률·구간점수.

선택 절차 (팀 규칙, 2026 채점 전 고정)
  개별 모델은 걷기 전진(매월 재학습, 학습 대상일 2023~). 가중치·λ·명절 계수는 **2024 대상일**로 맞추고 **2025 대상일 시간별 MAE**로 비교
  → 1위와 0.05원 이내면 구성 모델이 적은 쪽 → 안정성 검사(상·하반기 모두 더 단순한 후보보다 좋아야 함) → 2024~2025로 다시 맞춤 → 2026 한 번 채점 (mx3_ens --test, 한 달 7차 시험).
후보 (시간별)
  기준: h_same4(같은 유형일 4일 같은 시간), h_same8
  Chronos 시간별: hc2s, hc2, 평균 / 표: hlgb_r, hlgb_a, 평균 / hc2s + λ × 잔차 보정(hres_hc2s)
  수준 × 모양: 일평균 예측(Chronos 3종 평균, Chronos·LSTM·GRU 섞기 = v2 일평균 구성) × 하루 모양(최근 4일·28일·LightGBM·Chronos)
  앙상블: 주차별 NNLS(가중 평균), 주차 × 시간대별 NNLS
  명절·연휴 보정: 위 후보 × 달력 분류(단독 공휴일·연휴·명절) × 시간대별 계수 (재학습 달 전까지의 실적/예측 중앙값, 표본 적으면 1로 줄임)
범위 (2025 포함률 75~85% 중 구간점수 최소): 오차 분위수(칸 = 주차 × 휴무일 × 시간대 [× 명절]; 고정 1년·갱신 180·365일; 가산·배수·변동성 비례)
  + Chronos 시간별 분위수 보정. 오차 이력은 대상일 ≤ 기준일만.
출력: outputs/tables/MX3_S1_candidates_2025.csv, MX3_S2_intervals_2025.csv, MX3_S3_weak_2025.csv, MX3_choice.json,
      data/processed/smp_forecast_app/mx/MX3_predictions_hourly.csv.gz, MX3_predictions_daily.csv (4지표로 묶은 것, v2와 같은 열)
실행: python -m smp_forecast_app.src.mx3_ens → --daily (4지표 범위) → --weak (약한 구간, 2025) → --test (2026 한 번)
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from smp_forecast_app.src.final_metrics import metrics
from smp_forecast_app.src.intervals import ALPHA, BAND, Q_HI, Q_LO, score, select
from smp_forecast_app.src.mx3_data import load_hourly
from smp_forecast_app.src.mx_ens import TAB, TEST
from smp_forecast_app.src.mx_features import MX, STATS

LAMBDAS = (0.25, 0.5, 0.75, 1.0)
WEEK = lambda h: (np.asarray(h) - 7) // 7 + 1
BANDS = {"밤": 0, "오전": 1, "한낮": 2, "오후": 3, "저녁": 4}
HOL_MIN = 48             # 명절 계수: 분류 × 시간대 표본(시간 행)이 이보다 적으면 1 쪽으로 줄임
SN = {"mean": "일평균", "mid": "한낮 11~15시", "eve": "저녁 18~22시", "midmin": "한낮 최저"}


def mae(y, p):
    m = np.isfinite(np.asarray(y, float)) & np.isfinite(np.asarray(p, float))
    return float(np.abs(np.asarray(y, float)[m] - np.asarray(p, float)[m]).mean())


# ---------------------------------------------------------------- 자료
def load_table():
    d = load_hourly()
    keep = ["base_date", "H", "hour", "date", "y", "offday", "hol_cls", "mean_same4", "mean_seasN", "mean_l28", "h_same4", "h_same8",
            "h_shape4", "h_shape28"]
    d = d.loc[d["base_date"] >= "2024-01-01", keep].reset_index(drop=True)
    for tag in ("hc2s", "hc2"):
        if not (MX / f"mx3_chr_{tag}.csv.gz").exists():
            continue
        c = pd.read_csv(MX / f"mx3_chr_{tag}.csv.gz", parse_dates=["base_date"])
        d = d.merge(c.rename(columns={"q50": f"c_{tag}", "q10": f"q10_{tag}", "q90": f"q90_{tag}"}), on=["base_date", "H", "hour"], how="left")
    for n in ("hlgb_r", "hlgb_a", "hlgbx_r", "hlgbx_a", "hres_hc2s", "hresx_hc2s", "hshape", "hshapex", "hq10", "hq90"):
        p = MX / f"mx3_tab_{n}.csv.gz"
        if p.exists():
            d = d.merge(pd.read_csv(p, parse_dates=["base_date"]).rename(columns={"pred": n}), on=["base_date", "H", "hour"], how="left")
    # 일평균 수준 후보 (mx 일별 걷기 전진 예측)
    lv = None
    for f, col in [("chr_hc2s", "l_hc2s"), ("chr_hc2", "l_hc2"), ("chr_dc2", "l_dc2"), ("seq_lstm", "l_lstm"), ("seq_gru", "l_gru"),
                   ("tab_lgb_r", "l_lgb_r"), ("tab_lgb_a", "l_lgb_a"), ("tab_lgb_d", "l_lgb_d"), ("tab_et_r", "l_et_r")]:
        x = pd.read_csv(MX / f"{f}.csv.gz", parse_dates=["base_date"])[["base_date", "H", "pred_mean"]].rename(columns={"pred_mean": col})
        lv = x if lv is None else lv.merge(x, on=["base_date", "H"], how="outer")
    d = d.merge(lv, on=["base_date", "H"], how="left")
    d["W"] = WEEK(d["H"])
    d["band"] = d["hour"].map(BAND).map(BANDS)
    d["year"] = d["date"].dt.year
    return d


def day_norm(d, col):
    """하루 모양: col ÷ 같은 (기준일, H)의 24시간 평균."""
    return d[col] / d.groupby(["base_date", "H"])[col].transform("mean")


# ---------------------------------------------------------------- 후보
def candidates(d, fit, frozen=None):
    """fit 마스크로 λ·가중치를 맞춘 후보 → {이름: (예측, 구성)}. frozen: 재적합 때 수준 섞기 λ 등 고정할 값(없으면 fit으로 맞춤)."""
    y = d["y"]
    params = candidates.params = {}           # 운영 재현용 계수 (λ·수준 앙상블 가중치)
    out = {"기준: 같은 유형일 4일 같은 시간": (d["h_same4"], ["h_same4"]), "기준: 같은 유형일 8일 같은 시간": (d["h_same8"], ["h_same8"])}
    if "c_hc2s" in d:
        out["Chronos-2 small 시간별"] = (d["c_hc2s"], ["c_hc2s"])
    if "c_hc2" in d:
        out["Chronos-2 시간별"] = (d["c_hc2"], ["c_hc2"])
        out["Chronos 시간별 2종 평균"] = ((d["c_hc2s"] + d["c_hc2"]) / 2, ["c_hc2s", "c_hc2"])
    LG = {"hlgb_r": "LightGBM 시간별(same4 비율)", "hlgb_a": "LightGBM 시간별(계절 기준 비율)",
          "hlgbx_r": "LightGBM 시간별+추가 피처(same4 비율)", "hlgbx_a": "LightGBM 시간별+추가 피처(계절 기준 비율)"}
    for n, nm in LG.items():
        if n in d:
            out[nm] = (d[n], [n])
    for a, b, nm in (("hlgb_r", "hlgb_a", "LightGBM 시간별 2종 평균"), ("hlgbx_r", "hlgbx_a", "LightGBM 시간별+추가 피처 2종 평균")):
        if a in d and b in d:
            out[nm] = ((d[a] + d[b]) / 2, [a, b])
    for r, nm in (("hres_hc2s", "잔차 보정"), ("hresx_hc2s", "잔차 보정(추가 피처)")):
        if r in d:
            lam = min(LAMBDAS, key=lambda l: mae(y[fit], (d["c_hc2s"] + l * d[r])[fit]))
            params[f"lam_{r}"] = lam
            out[f"Chronos-2 small 시간별 + {lam} × {nm}"] = (d["c_hc2s"] + lam * d[r], ["c_hc2s", r])
    # 수준 × 모양
    lc = d[["l_hc2s", "l_hc2", "l_dc2"]].mean(axis=1)
    ls = d[["l_lstm", "l_gru"]].mean(axis=1)
    shapes = {"최근 4일 모양": d["h_shape4"], "최근 28일 모양": d["h_shape28"], "4·28일 모양 평균": (d["h_shape4"] + d["h_shape28"]) / 2}
    for n, nm in (("hlgb_r", "LightGBM 모양"), ("c_hc2s", "Chronos small 모양"), ("hshape", "모양 전용 LightGBM"),
                  ("hshapex", "모양 전용 LightGBM+추가 피처")):
        if n in d:
            shapes[nm] = day_norm(d, n)
    lam = (frozen or {}).get("level_lam")
    if lam is None:
        lam = min((0.0,) + LAMBDAS[:3], key=lambda l: mae(y[fit], (((1 - l) * lc + l * ls) * shapes["4·28일 모양 평균"])[fit]))
    candidates.level_lam = params["level_lam"] = lam
    lvl = (1 - lam) * lc + lam * ls
    for sn, sh in shapes.items():
        out[f"일평균(Chronos {1 - lam:g}·LSTM·GRU {lam:g}) × {sn}"] = (lvl * sh, ["l_hc2s", "l_hc2", "l_dc2", "l_lstm", "l_gru", sn])
    # 수준 앙상블(주차별 NNLS, 일평균 후보 = 일별 모델 + 시간별 모델의 하루 평균) × 모양 전용
    lv = {"l_chr": lc, "l_seq": ls, "l_tab": d[["l_lgb_r", "l_lgb_a", "l_lgb_d", "l_et_r"]].mean(axis=1)}
    for n in ("c_hc2s", "c_hc2", "hlgb_r", "hlgb_a", "hlgbx_r", "hlgbx_a"):
        if n in d:
            lv[f"l_{n}"] = d.groupby(["base_date", "H"])[n].transform("mean")
    L = pd.DataFrame(lv)
    ya = d.groupby(["base_date", "H"])["y"].transform("mean")
    lens = pd.Series(np.nan, index=d.index)
    okl = L.notna().all(axis=1)
    for _, g in d.groupby("W"):
        m = fit & okl & ya.notna() & d.index.isin(g.index)
        w = _nnls(L[m].values, ya[m].values)
        params.setdefault("level_w", {})[str(int(g["W"].iloc[0]))] = dict(zip(L.columns, w.round(8).tolist()))
        gi = g.index[okl[g.index]]
        lens[gi] = L.loc[gi].values @ w
    for sn in ("모양 전용 LightGBM", "모양 전용 LightGBM+추가 피처", "4·28일 모양 평균"):
        if sn in shapes:
            out[f"일평균 앙상블({len(lv)}개) × {sn}"] = (lens * shapes[sn], list(lv) + [sn])
    # 앙상블 (가중 평균, 음수 없음): 주차별 / 주차 × 시간대별
    pool = [k for k in out if not k.startswith("기준") and "2종 평균" not in k]     # 개별 후보만 (기준·단순 평균 제외)
    P = pd.DataFrame({k: out[k][0] for k in pool})
    comps = sorted(set(sum((out[k][1] for k in pool), [])))
    for keyname, keys in (("주차별", ["W"]), ("주차 × 시간대별", ["W", "band"])):
        pred = pd.Series(np.nan, index=d.index)
        ok = P.notna().all(axis=1)
        for _, g in d.groupby(keys):
            m = fit & ok & y.notna() & d.index.isin(g.index)
            w = _nnls(P[m].values, y[m].values)
            gi = g.index[ok[g.index]]
            pred[gi] = P.loc[gi].values @ w
        out[f"가중 평균({keyname}, {len(pool)}개)"] = (pred, comps)
    # 계열 대표만 (후보가 많으면 2024 맞춤 가중치가 2024에 쏠림 → 계열마다 하나씩)
    fam = [k for k in out if k == "LightGBM 시간별 2종 평균" or k.startswith("Chronos-2 small 시간별 + ") and k.endswith("× 잔차 보정")
           or k.endswith("× 모양 전용 LightGBM")]
    if len(fam) >= 3:
        F = pd.DataFrame({k: out[k][0] for k in fam})
        fc = sorted(set(sum((out[k][1] for k in fam), [])))
        out[f"계열 대표 {len(fam)}개 단순 평균"] = (F.mean(axis=1, skipna=False), fc + ["계열 평균"])
        pred = pd.Series(np.nan, index=d.index)
        ok = F.notna().all(axis=1)
        for _, g in d.groupby("W"):
            m = fit & ok & y.notna() & d.index.isin(g.index)
            gi = g.index[ok[g.index]]
            pred[gi] = F.loc[gi].values @ _nnls(F[m].values, y[m].values)
        out[f"계열 대표 {len(fam)}개 가중 평균(주차별)"] = (pred, fc + ["계열 가중"])
    return out


def stable(er, pick):
    """과적합 방지: 고른 후보가 2025 상·하반기 중 한쪽에서라도 '구성이 더 적은 후보 중 2025 1위'보다 나쁘면 그 후보로 내려감 (반복)."""
    e = er.set_index("후보")
    while True:
        simpler = e[e["구성 수"] < e.loc[pick, "구성 수"]]
        if not len(simpler):
            return pick
        alt = simpler["2025(선택)"].idxmin()
        if (e.loc[pick, "2025 상반기"] < e.loc[alt, "2025 상반기"]) and (e.loc[pick, "2025 하반기"] < e.loc[alt, "2025 하반기"]):
            return pick
        print(f"  안정성 검사: {pick} → {alt} (상·하반기 중 한쪽에서 더 단순한 후보보다 나쁨)")
        pick = alt


def _nnls(X, y):
    """합이 1이 되도록 정규화한 NNLS 가중치 (평균 쪽으로 정규화: 절편 없음)."""
    w, _ = nnls(X, y)
    return w / w.sum() if w.sum() > 0 else np.ones(X.shape[1]) / X.shape[1]


def holiday_adjust(d, p):
    """명절·연휴 보정: 분류(2 단독 공휴일, 3 연휴, 4 명절) × 시간대별 계수 = 중앙값(실적 ÷ 예측), 표본 적으면 1 쪽으로 줄임.
    걷기 전진: 기준일이 속한 달 1일 전에 실적이 나온 행(같은 후보의 예측, 기준일 2024~)만으로 계수를 만든다."""
    out = p.copy()
    q = d.assign(p=p, r=(d["y"] / p.clip(lower=1)).clip(0, 3))
    q = q[q["y"].notna() & q["p"].notna() & (q["hol_cls"] >= 2)]
    for a in pd.date_range(d["base_date"].min().replace(day=1), d["base_date"].max(), freq="MS"):
        h = q[q["date"] < a]
        te = (d["base_date"] >= a) & (d["base_date"] < a + pd.offsets.MonthBegin(1)) & (d["hol_cls"] >= 2)
        if te.sum() == 0 or len(h) == 0:
            continue
        g = h.groupby(["hol_cls", "band"])["r"]
        f = 1 + (g.median() - 1) * (g.size() / (g.size() + HOL_MIN))
        idx = pd.MultiIndex.from_frame(d.loc[te, ["hol_cls", "band"]])
        out[te] = p[te] * f.reindex(idx).fillna(1).values
    return out


# ---------------------------------------------------------------- 범위
def vol_scale(d):
    """기준일 직전 28일 일평균 변동 (판단 시각에 앎) ÷ 전체 중앙값 → 폭 배율."""
    from common.loaders import load_smp_actual
    s = load_smp_actual("육지").groupby("date")["smp"].mean()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max()))
    v = s.diff().abs().rolling(28, min_periods=20).mean()
    v = v / v[v.index < "2024-01-01"].median()
    return pd.Series(v.reindex(d["base_date"]).values, index=d.index).clip(0.5, 3)


def cell(d, key):
    """칸 번호 (열 값을 자리수로 이어 붙인 정수)."""
    k = np.zeros(len(d), dtype=np.int64)
    for c in key:
        k = k * 10 + d[c].astype(int).values
    return k


def err_interval(d, ev, window, form, key, vol=None):
    """오차 분위수 범위. window None = 평가 구간 시작 전 365일 고정, 정수 = 매 기준일 직전 window일(대상일 ≤ 기준일).
    form: add(원), mult(÷ 예측), vol(÷ 변동성 배율)."""
    if form == "add":
        sc = pd.Series(1.0, index=d.index)
    elif form == "mult":
        sc = d["pred"].clip(lower=20)
    else:
        sc = vol
    d = d.assign(err=(d["y"] - d["pred"]) / sc, sc=sc, k=cell(d, key))
    hist = d[d["err"].notna() & d["pred"].notna()]
    E = d[ev]
    lo, hi = pd.Series(np.nan, index=E.index), pd.Series(np.nan, index=E.index)
    groups = [(E["base_date"].min(), E)] if window is None else list(E.groupby("base_date"))
    for i, part in groups:
        w = 365 if window is None else window
        rows = hist[(hist["date"] <= i) & (hist["date"] > i - pd.Timedelta(days=w))]
        g = rows.groupby("k")["err"]
        qq = pd.DataFrame({"l": g.quantile(Q_LO), "h": g.quantile(Q_HI), "n": g.size()})
        allq = rows["err"].quantile([Q_LO, Q_HI]).values
        qq.loc[qq["n"] < 100, ["l", "h"]] = allq
        ql = part["k"].map(qq["l"]).fillna(allq[0]).values
        qh = part["k"].map(qq["h"]).fillna(allq[1]).values
        lo[part.index] = np.clip(part["pred"].values + ql * part["sc"].values, 0, None)
        hi[part.index] = part["pred"].values + qh * part["sc"].values
    return lo.values, hi.values


def cqr_interval(d, ev, window, key, src="chronos"):
    """분위수 폭을 예측값에 옮기고, max(lo−y, y−hi)의 80% 분위수로 칸별 보정 (보정 분위수).
    src: chronos = Chronos small 시간별 (q10−q50, q90−q50), lgb = LightGBM 분위수 회귀 (hq10−hlgb_r, hq90−hlgb_r)."""
    lo_w, hi_w = ((d["q10_hc2s"] - d["c_hc2s"]), (d["q90_hc2s"] - d["c_hc2s"])) if src == "chronos" else         ((d["hq10"] - d["hlgb_r"]).clip(upper=0), (d["hq90"] - d["hlgb_r"]).clip(lower=0))
    d = d.assign(lo0=d["pred"] + lo_w, hi0=d["pred"] + hi_w, k=cell(d, key))
    d["cs"] = np.maximum(d["lo0"] - d["y"], d["y"] - d["hi0"])
    hist = d[d["cs"].notna()]
    E = d[ev]
    lo, hi = pd.Series(np.nan, index=E.index), pd.Series(np.nan, index=E.index)
    for i, part in E.groupby("base_date"):
        rows = hist[(hist["date"] <= i) & (hist["date"] > i - pd.Timedelta(days=window))]
        q = rows.groupby("k")["cs"].quantile(1 - ALPHA)
        c = part["k"].map(q).fillna(rows["cs"].quantile(1 - ALPHA))
        lo[part.index] = np.minimum((part["lo0"] - c).clip(lower=0), part["pred"])
        hi[part.index] = np.maximum(part["hi0"] + c, part["pred"])
    return lo.values, hi.values


def interval_methods(d):
    vol = vol_scale(d)
    d["hol"] = (d["hol_cls"] >= 2).astype(int)
    K1, K2 = ["W", "offday", "band"], ["W", "offday", "band", "hol"]
    m = {}
    for kn, key in (("주차×휴무일×시간대", K1), ("주차×휴무일×시간대×명절", K2)):
        for wn, w in (("고정1년", None), ("갱신180일", 180), ("갱신365일", 365)):
            for fn in ("add", "mult", "vol"):
                m[f"오차분위수·{kn}·{wn}·{ {'add': '가산', 'mult': '배수', 'vol': '변동성비례'}[fn]}"] = \
                    (lambda w, fn, key: lambda D, e: err_interval(D, e, w, fn, key, vol))(w, fn, key)
        if "q10_hc2s" in d:
            m[f"보정분위수(Chronos small)·{kn}·갱신365일"] = (lambda key: lambda D, e: cqr_interval(D, e, 365, key))(key)
        if "hq10" in d and "hq90" in d:
            for w in (180, 365):
                m[f"보정분위수(LightGBM 분위수 회귀)·{kn}·갱신{w}일"] = (lambda key, w: lambda D, e: cqr_interval(D, e, w, key, "lgb"))(key, w)
    return m


# ---------------------------------------------------------------- 4지표로 묶기
def to_daily(h, cols=("y", "pred", "lo", "hi")):
    """시간별 → 일별 4지표. 범위: 일평균·한낮·저녁은 시간별 하한·상한 평균, 한낮 최저는 11~15시 하한·상한 최소."""
    g = h.groupby(["base_date", "H"])
    mid = h[h["hour"].between(11, 15)].groupby(["base_date", "H"])
    eve = h[h["hour"].between(18, 22)].groupby(["base_date", "H"])
    out = {}
    for c in cols:
        out[f"{c}_mean"] = g[c].mean()
        out[f"{c}_mid"] = mid[c].mean()
        out[f"{c}_eve"] = eve[c].mean()
        out[f"{c}_midmin"] = mid[c].min()
    o = pd.DataFrame(out).reset_index()
    if "y" in cols:
        n = g["y"].count().values
        for s in STATS:
            o.loc[n < 24, f"y_{s}"] = np.nan
    return o


# ---------------------------------------------------------------- 실행
def same_kind(pick, cands):
    if pick in cands:
        return pick
    pat = re.sub(r"\d+(\.\d+)?", "#", pick)
    for k in cands:
        if re.sub(r"\d+(\.\d+)?", "#", k) == pat:
            return k
    raise KeyError(pick)


def main():
    d = load_table()
    y24, y25 = d["year"] == 2024, d["year"] == 2025
    cand = candidates(d, y24)
    lam_sel = candidates.level_lam
    for k in list(cand):          # 명절 보정 변형 (후보마다)
        p, ms = cand[k]
        cand[f"{k} + 명절 보정"] = (holiday_adjust(d, p), ms + ["명절 보정"])
    full = pd.DataFrame({k: v[0] for k, v in cand.items()}).notna().all(axis=1) & d["y"].notna()
    rows = []
    for k, (p, ms) in cand.items():
        r = {"후보": k, "구성 수": len(ms), "2024(맞춤)": mae(d.loc[y24 & full, "y"], p[y24 & full]),
             "2025(선택)": mae(d.loc[y25 & full, "y"], p[y25 & full])}
        for hn, hm in (("상반기", d["date"].dt.month <= 6), ("하반기", d["date"].dt.month > 6)):
            r[f"2025 {hn}"] = mae(d.loc[y25 & full & hm, "y"], p[y25 & full & hm])
        for w in range(1, 5):
            m = y25 & full & (d["W"] == w)
            r[f"2025 {w}주차"] = mae(d.loc[m, "y"], p[m])
        for b, bi in BANDS.items():
            m = y25 & full & (d["band"] == bi)
            r[f"2025 {b}"] = mae(d.loc[m, "y"], p[m])
        m = y25 & full & (d["hol_cls"] >= 2)
        r["2025 명절·연휴·공휴일"] = mae(d.loc[m, "y"], p[m])
        r["구성"] = ", ".join(ms)
        rows.append(r)
    er = pd.DataFrame(rows).sort_values("2025(선택)")
    top = er.iloc[0]["2025(선택)"]
    pick = er[er["2025(선택)"] <= top + 0.05].sort_values(["구성 수", "2025(선택)"]).iloc[0]["후보"]
    pick = stable(er, pick)
    er["선택"] = np.where(er["후보"] == pick, "✓", "")
    er.round(3).to_csv(TAB / "MX3_S1_candidates_2025.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(er[["후보", "구성 수", "2024(맞춤)", "2025(선택)", "2025 1주차", "2025 4주차", "2025 한낮", "2025 명절·연휴·공휴일"]].round(2).to_string(index=False))
    print("선택:", pick, flush=True)

    # 범위 (2025, 예측 = 2024로 맞춘 선택 후보)
    D = d.assign(pred=cand[pick][0])
    ev = (y25 & D["y"].notna() & D["pred"].notna()).values
    ms = interval_methods(D)
    it = []
    for name, fn in ms.items():
        lo, hi = fn(D, ev)
        it.append({"방법": name, "구간": "전체", **score(D.loc[ev, "y"], lo, hi, D.loc[ev, "pred"])})
        print(f"  {name}: 포함률 {it[-1]['포함률']:.3f} 폭 {it[-1]['평균폭']:.1f} 점수 {it[-1]['구간점수']:.1f}", flush=True)
    it = pd.DataFrame(it)
    ich = select(it)
    it["선택"] = np.where(it["방법"] == ich, "✓", "")
    it.round(4).to_csv(TAB / "MX3_S2_intervals_2025.csv", index=False, encoding="utf-8-sig")
    print("범위 선택:", ich, flush=True)

    # 2024~2025로 다시 맞춤 → 2026 예측
    refit = candidates(d, (y24 | y25), frozen=None)
    for k in list(refit):
        p, msx = refit[k]
        refit[f"{k} + 명절 보정"] = (holiday_adjust(d, p), msx + ["명절 보정"])
    key = same_kind(pick, refit)
    D.loc[d["date"] >= "2026-01-01", "pred"] = refit[key][0][d["date"] >= "2026-01-01"]
    out = D[["base_date", "H", "hour", "date", "offday", "hol_cls", "y", "pred"]].copy()
    out["lo"], out["hi"] = np.nan, np.nan
    m25 = ev
    out.loc[m25, "lo"], out.loc[m25, "hi"] = ms[ich](D, m25)
    m26 = ((D["date"] >= "2026-01-01") & D["pred"].notna()).values
    out.loc[m26, "lo"], out.loc[m26, "hi"] = ms[ich](D, m26)
    out.round(2).to_csv(MX / "MX3_predictions_hourly.csv.gz", index=False, compression="gzip")
    dl = to_daily(out)
    dl["date"] = dl["base_date"] + pd.to_timedelta(dl["H"], unit="D")
    dl.round(2).to_csv(MX / "MX3_predictions_daily.csv", index=False, encoding="utf-8-sig")
    info = {"점예측(2025 선택)": pick, "재적합": key, "수준 섞기 λ(2024 맞춤)": lam_sel, "재적합 λ": candidates.level_lam, "범위": ich,
            "구성": cand[pick][1], "선택 절차": "개별 모델 걷기 전진(매월 재학습, 학습 대상일 2023~) → 가중치·λ·명절 계수는 2024 대상일로 → "
            "2025 시간별 MAE 비교(0.05원 이내면 구성 적은 쪽) → 2024~2025로 재적합 → 2026 한 번 채점",
            "범위 선택": "2025 시간별 포함률 75~85% 중 구간점수 최소", "시험": f"{TEST[0]} ~ {TEST[1]} (한 달 예측 7차 시험)"}
    (TAB / "MX3_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


def export_coef():
    """운영(mx3_ops)용 계수: 2024~2025로 다시 맞춘 후보(= 2026 예측에 쓴 것)의 λ·수준 앙상블 주차별 가중치 + 고른 후보의 계열 구성.
    재현 확인: 계수로 다시 계산한 2026 예측과 MX3_predictions_hourly의 최대 차."""
    d = load_table()
    ch = json.loads((TAB / "MX3_choice.json").read_text(encoding="utf-8"))
    cand = candidates(d, d["year"].isin([2024, 2025]))
    prm = dict(candidates.params)
    fam = [k for k in cand if k == "LightGBM 시간별 2종 평균" or k.startswith("Chronos-2 small 시간별 + ") and k.endswith("× 잔차 보정")
           or k.endswith("× 모양 전용 LightGBM")]
    prm["family"] = fam
    p = cand[ch["재적합"]][0]
    x = recompute(d, prm)
    h = pd.read_csv(MX / "MX3_predictions_hourly.csv.gz", parse_dates=["base_date"])
    m = d["date"] >= "2026-01-01"
    ref = d.loc[m, ["base_date", "H", "hour"]].assign(a=x[m].values).merge(h[["base_date", "H", "hour", "pred"]], on=["base_date", "H", "hour"])
    prm["재현 최대 차(2026, 원)"] = round(float((ref["a"] - ref["pred"]).abs().max()), 4)
    prm["후보식과 최대 차"] = round(float((x[m] - p[m]).abs().max()), 6)
    (MX / "mx3_coef.json").write_text(json.dumps(prm, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(prm, ensure_ascii=False, indent=1))


def recompute(d, prm):
    """계수로 최종 예측 다시 계산: 계열 대표 4개 단순 평균
    = 평균[ (hlgb_r+hlgb_a)/2,  c_hc2s + λ × hres_hc2s,  v2 수준 × 모양,  수준 앙상블 × 모양 ]  (모양 = hshape 하루 평균 1로 정규화)."""
    lam_l = prm["level_lam"]
    lc = d[["l_hc2s", "l_hc2", "l_dc2"]].mean(axis=1)
    ls = d[["l_lstm", "l_gru"]].mean(axis=1)
    sh = day_norm(d, "hshape")
    lv = {"l_chr": lc, "l_seq": ls, "l_tab": d[["l_lgb_r", "l_lgb_a", "l_lgb_d", "l_et_r"]].mean(axis=1)}
    for n in ("c_hc2s", "c_hc2", "hlgb_r", "hlgb_a", "hlgbx_r", "hlgbx_a"):
        if n in d and any(f"l_{n}" in w for w in prm["level_w"].values()):
            lv[f"l_{n}"] = d.groupby(["base_date", "H"])[n].transform("mean")
    lens = pd.Series(np.nan, index=d.index)
    for w, ww in prm["level_w"].items():
        m = d["W"] == int(w)
        lens[m] = sum(b * lv[c][m] for c, b in ww.items() if b != 0)
    f1 = (d["hlgb_r"] + d["hlgb_a"]) / 2
    f2 = d["c_hc2s"] + prm["lam_hres_hc2s"] * d["hres_hc2s"]
    f3 = ((1 - lam_l) * lc + lam_l * ls) * sh
    f4 = lens * sh
    return (f1 + f2 + f3 + f4) / 4


def daily_ranges():
    """4지표 범위 (팀 비교표용): 시간별 하한·상한을 묶으면 일평균 범위가 너무 넓어짐(2025 포함률 92%) →
    mx3 4지표 예측에 v2와 같은 후보(mx_ens.interval_methods: 오차 분위수 8종 + Chronos 일별 분위수 보정 2종)·같은 규칙
    (2025 포함률 75~85% 중 구간점수 최소)으로 지표별 범위를 다시 고름. MX3_predictions_daily.csv의 lo·hi를 바꾸고 MX3_S4에 기록."""
    from smp_forecast_app.src.mx_ens import interval_methods as daily_methods
    h = pd.read_csv(MX / "MX3_predictions_hourly.csv.gz", parse_dates=["base_date", "date"])
    dl = to_daily(h, cols=("y", "pred"))
    dl["date"] = dl["base_date"] + pd.to_timedelta(dl["H"], unit="D")
    dl = dl.merge(h.drop_duplicates(["base_date", "H"])[["base_date", "H", "offday"]], on=["base_date", "H"], how="left")
    c = pd.read_csv(MX / "chr_dc2.csv.gz", parse_dates=["base_date"])
    ms = daily_methods()
    rows, info = [], {}
    for s in STATS:
        D = dl[["base_date", "H", "date", "offday", f"y_{s}", f"pred_{s}"]].rename(columns={f"y_{s}": "y", f"pred_{s}": "pred"}).merge(
            c[["base_date", "H", f"pred_{s}", f"qlo_{s}", f"qhi_{s}"]].rename(
                columns={f"pred_{s}": "dc2", f"qlo_{s}": "qlo_dc2", f"qhi_{s}": "qhi_dc2"}), on=["base_date", "H"], how="left")
        D = D[D["pred"].notna()].sort_values(["base_date", "H"]).reset_index(drop=True)
        ev = ((D["date"].dt.year == 2025) & D["y"].notna()).values
        part = []
        for name, fn in ms.items():
            lo, hi = fn(D, ev)
            part.append({"지표": SN[s], "방법": name, "구간": "전체", **score(D.loc[ev, "y"], lo, hi, D.loc[ev, "pred"])})
        it = pd.DataFrame(part)
        ich = select(it)
        it["선택"] = np.where(it["방법"] == ich, "✓", "")
        rows.append(it)
        info[SN[s]] = ich
        D["lo"], D["hi"] = np.nan, np.nan
        m25 = (D["date"].dt.year == 2025).values
        D.loc[m25, "lo"], D.loc[m25, "hi"] = ms[ich](D, m25)
        m26 = (D["date"] >= "2026-01-01").values
        D.loc[m26, "lo"], D.loc[m26, "hi"] = ms[ich](D, m26)
        dl = dl.drop(columns=[f"lo_{s}", f"hi_{s}"], errors="ignore").merge(
            D[["base_date", "H", "lo", "hi"]].rename(columns={"lo": f"lo_{s}", "hi": f"hi_{s}"}), on=["base_date", "H"], how="left")
        r = it[it["방법"] == ich].iloc[0]
        print(f"[{SN[s]}] 4지표 범위 {ich} — 2025 포함률 {r['포함률']:.1%}, 폭 {r['평균폭']:.1f}, 구간점수 {r['구간점수']:.1f}", flush=True)
    pd.concat(rows).round(4).to_csv(TAB / "MX3_S4_daily_intervals_2025.csv", index=False, encoding="utf-8-sig")
    dl.round(2).to_csv(MX / "MX3_predictions_daily.csv", index=False, encoding="utf-8-sig")
    ch = json.loads((TAB / "MX3_choice.json").read_text(encoding="utf-8"))
    ch["4지표 범위(팀 비교표용)"] = info
    (TAB / "MX3_choice.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")


def daily_compare(dl, period):
    """4지표로 묶은 mx3 vs v2 (같은 기준일·지평). 반환 [지표, 행 마스크용 열, y, p3, lo3, hi3, p2, lo2, hi2, same4]."""
    from smp_forecast_app.src.mx_features import load
    v2 = pd.read_csv(MX / "MX_predictions_daily_v2.csv", parse_dates=["base_date", "date"])
    f = load()[["base_date", "H"] + [f"{s}_same4" for s in STATS]]
    out = []
    for s in STATS:
        g = dl[["base_date", "H", "date", f"y_{s}", f"pred_{s}", f"lo_{s}", f"hi_{s}"]].rename(
            columns={f"y_{s}": "y", f"pred_{s}": "p3", f"lo_{s}": "lo3", f"hi_{s}": "hi3"}).merge(
            v2[["base_date", "H", "y_" + s, f"pred_{s}", f"lo_{s}", f"hi_{s}", "offday", "day_type"]].rename(
                columns={"y_" + s: "y2", f"pred_{s}": "p2", f"lo_{s}": "lo2", f"hi_{s}": "hi2"}), on=["base_date", "H"]).merge(
            f[["base_date", "H", f"{s}_same4"]].rename(columns={f"{s}_same4": "same4"}), on=["base_date", "H"])
        g = g[(g["date"] >= period[0]) & (g["date"] <= period[1]) & g["y"].notna() & g["p3"].notna() & g["p2"].notna() & g["lo2"].notna()]
        out.append(g.assign(지표=SN[s], W=WEEK(g["H"]), month=g["date"].dt.month))
    return pd.concat(out, ignore_index=True)


def weak():
    """약한 구간 확인 (2025, 4지표 mx3 vs v2): 주차·월·휴무일·요일유형."""
    dl = pd.read_csv(MX / "MX3_predictions_daily.csv", parse_dates=["base_date", "date"])
    g = daily_compare(dl, ("2025-01-01", "2025-12-31"))
    rows = []
    for axis in ("전체", "W", "month", "offday", "day_type"):
        for k, x in (g.groupby(["지표"]) if axis == "전체" else g.groupby(["지표", axis])):
            k = k if isinstance(k, tuple) else (k,)
            m3, m2 = metrics(x["y"], x["p3"], x["lo3"], x["hi3"], x["same4"]), metrics(x["y"], x["p2"], x["lo2"], x["hi2"], x["same4"])
            rows.append({"지표": k[0], "축": axis, "값": k[1] if len(k) > 1 else "", "표본수": len(x), "mx3 MAE": m3["MAE"], "v2 MAE": m2["MAE"],
                         "same4 MAE": m3["단순 기준 MAE"], "mx3 포함률": m3["포함률"], "v2 포함률": m2["포함률"],
                         "mx3 구간점수": m3["구간점수"], "v2 구간점수": m2["구간점수"], "mx3 편향": m3["편향(공표−예측)"], "v2 편향": m2["편향(공표−예측)"]})
    t = pd.DataFrame(rows)
    t.round(3).to_csv(TAB / "MX3_S3_weak_2025.csv", index=False, encoding="utf-8-sig")
    return t


def hourly_scores(h, period, cols):
    """시간별 성적: cols = {이름: (pred, lo, hi)}. 기준 = h_same4."""
    x = h[(h["date"] >= period[0]) & (h["date"] <= period[1]) & h["y"].notna()]
    rows = []
    for seg, m in {"전체": slice(None), **{f"{w}주차": x["W"] == w for w in range(1, 5)},
                   **{b: x["band"] == i for b, i in BANDS.items()}, "평일": x["offday"] == 0, "휴무일": x["offday"] == 1,
                   "명절·연휴·공휴일": x["hol_cls"] >= 2}.items():
        z = x[m]
        for n, (p, lo, hi) in cols.items():
            r = metrics(z["y"], z[p], z[lo] if lo else None, z[hi] if hi else None, z["h_same4"])
            rows.append({"구간": seg, "예측": n, **r})
    return pd.DataFrame(rows)


def test():
    """2026-01-01 ~ 10-18 한 번 채점 (한 달 7차 시험): 시간별 + 4지표 (v2·단순 기준과 같은 기준일·지평, 차의 95% 구간)."""
    from smp_forecast_app.src.mx_ens import boot_diff
    h = pd.read_csv(MX / "MX3_predictions_hourly.csv.gz", parse_dates=["base_date", "date"])
    b = load_hourly()[["base_date", "H", "hour", "h_same4"]]
    h = h.merge(b, on=["base_date", "H", "hour"], how="left")
    h["W"], h["band"] = WEEK(h["H"]), h["hour"].map(BAND).map(BANDS)
    ht = hourly_scores(h, TEST, {"mx3": ("pred", "lo", "hi"), "단순 기준(같은 유형일 4일 같은 시간)": ("h_same4", None, None)})
    x = h[(h["date"] >= TEST[0]) & (h["date"] <= TEST[1]) & h["y"].notna()]
    lo_, hi_ = boot_diff(x, "h_same4", "pred")
    ht.round(4).to_csv(TAB / "MX3_T1_test2026_hourly.csv", index=False, encoding="utf-8-sig")
    dl = pd.read_csv(MX / "MX3_predictions_daily.csv", parse_dates=["base_date", "date"])
    g = daily_compare(dl, TEST)
    rows = []
    for s, x2 in g.groupby("지표", sort=False):
        for w in ["전체", 1, 2, 3, 4]:
            z = x2 if w == "전체" else x2[x2["W"] == w]
            m3, m2 = metrics(z["y"], z["p3"], z["lo3"], z["hi3"], z["same4"]), metrics(z["y"], z["p2"], z["lo2"], z["hi2"], z["same4"])
            r = {"지표": s, "주차": w, **{f"mx3 {k}": v for k, v in m3.items()}, **{f"v2 {k}": v for k, v in m2.items() if k != "표본수"}}
            if w == "전체":
                r["차(mx3−v2) 95% 하"], r["차(mx3−v2) 95% 상"] = boot_diff(z, "p2", "p3")
                r["차(mx3−기준) 95% 하"], r["차(mx3−기준) 95% 상"] = boot_diff(z, "same4", "p3")
            rows.append(r)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "MX3_T2_test2026_daily.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(f"시간별 차(mx3 − 기준) 95%: [{lo_:.2f}, {hi_:.2f}]")
    print(ht[ht["구간"].isin(["전체", "1주차", "4주차", "한낮", "저녁", "휴무일", "명절·연휴·공휴일"])][
        ["구간", "예측", "MAE", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수"]].round(3).to_string(index=False))
    print(t[t["주차"] == "전체"][["지표", "mx3 MAE", "v2 MAE", "mx3 단순 기준 MAE", "차(mx3−v2) 95% 하", "차(mx3−v2) 95% 상",
                                 "mx3 포함률", "v2 포함률", "mx3 구간점수", "v2 구간점수"]].round(3).to_string(index=False))


if __name__ == "__main__":
    if "--test" in sys.argv[1:]:
        test()
    elif "--coef" in sys.argv[1:]:
        export_coef()
    elif "--daily" in sys.argv[1:]:
        daily_ranges()
    elif "--weak" in sys.argv[1:]:
        print(weak().round(2).to_string(index=False))
    else:
        main()
