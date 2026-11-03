"""제주 일주일 1차(15시) — '제주가 육지와 다른 시간'을 맞추는 제주 조정 (R2, 2026-11-03, 사용자: "제주는 제주가 다른 시간을 맞춰야").

배경 (jz3): 육지 K3 그대로가 2025 하반기 9.48원. 제주 보정(J2)은 제주≠육지 시간(8%)에서 13.22 → 12.56원이지만
  같은 시간(92%)에서 조금 나빠져 전체 0.02원. 기존 제주 피처는 대부분 하루 요약값이라 시간별 차이를 잡기 어렵다.

새 시간별 피처 (모두 판단일 I = D−H 15시에 아는 값, 대상 (D, h)) — hourly_extra(H)
  차이 이력   dj_*: 제주−육지 SMP 차이 같은 시각 — I일, D−7일, 직전 같은 유형일, 7·28일 평균, 7·28일 제주>육지·제주<육지 비율
  계통 실적   ks_*: 5분 실적 같은 시각 I−1일 예비력(공급능력−수요)·순부하(수요−재생)·풍력·태양광, I−7 ~ I−1 평균 (I일은 14:25까지라 안 씀)
  수요예측    jd_*: 제주 ÷ 육지 하루전 수요예측 I일 같은 시각, 제주 수요예측 I일 같은 시각 ÷ 28일 평균
  Open-Meteo  om_*: 대상 시각 100m·10m 풍속, 일사, 운량 (4지점·모델 평균, H=1은 2일 전, H=2~6은 H+1일 전 실행분, H=7 없음, 2024~)
모델 (모두 점예측 = K3 + λ × 조정, LightGBM, 대상일의 달마다 재학습, 학습 2023-01 ~ 그달 1일 − (H−1)일 실적)
  R  잔차형  : 제주 실적 − K3            피처 = 기존 제주외부 + 새 피처, 목적 l1·l2
  D  차이형  : 제주 실적 − 육지 실적      같은 피처, 목적 l1·l2 (육지 오차가 섞이지 않게)
  Dw 차이형 l2 + 최근 가중(반감기 180일)  (2023~2024는 제주≠육지가 20~38%로 지금과 다른 시기)
  T  2단계형 : P(제주>육지)·P(제주<육지) 분류 + 각 경우의 차이 크기 회귀(l2)
         Te 기대값 = P↑·크기↑ + P↓·크기↓ ,  Tm 중앙값형 = P가 0.5 이상인 쪽 크기만 (MAE에 맞는 형태)
  참고: J1 K3 그대로, J2 기존 제주 보정(jz3 l2)
λ: 0, 0.1, 0.25, 0.5, 0.75, 1 중 2025 상반기 전체 MAE 최소
선택 (결과 보기 전 고정): 2025 하반기 전체 MAE ≤ 최소 + 0.05원인 후보 중 제주≠육지 시간 MAE 최소 (같으면 단순한 쪽)
출력: outputs/tables/JWZ7_select_2025.csv, data/processed/smp_forecast_app/jeju/jz4/{후보}.csv.gz
실행: python -m smp_forecast_app.src.jz4 [build] [select]
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import ROOT
from common.loaders import load_kpx_jeju_sukub, load_openmeteo_prev2_jeju, load_openmeteo_prevx_jeju, load_smp_actual, load_smp_forecast_api
from smp_forecast_app.src import jz3
from smp_forecast_app.src.final_metrics import hourly_segments
from smp_forecast_app.src.jz_common import HS, JDIR, TRAIN_START
from smp_forecast_app.src.jz_features import CAL, DEM, OM, SPREAD, WEATHER, load as load_feats
from smp_forecast_app.src.jz_resid import LGB

warnings.filterwarnings("ignore")

TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
OUT = JDIR / "jz4"
PRED_START = "2024-07-01"
H1, H2 = jz3.H1, jz3.H2
GRID = jz3.GRID
TIE = 0.05
EPS = 0.5          # 제주≠육지: |차이| ≥ 0.5원
NEW = None         # hourly_extra 열 (build 때 채움)
OLD = CAL + DEM + SPREAD + WEATHER + OM
LGBC = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, verbose=-1, n_jobs=3, random_state=42)


# ---------------------------------------------------------------- 시간별 새 피처
def _wide(s: pd.DataFrame, col: str) -> pd.DataFrame:
    return s.pivot(index="date", columns="hour", values=col).sort_index().asfreq("D")


def sources():
    J = load_smp_actual("제주")[["date", "hour", "smp"]]
    L = load_smp_actual("육지")[["date", "hour", "smp"]]
    sj, sl = _wide(J, "smp"), _wide(L, "smp")
    diff = sj - sl.reindex_like(sj)
    k = load_kpx_jeju_sukub()
    k["date"] = k["time"].dt.normalize()
    k["hour"] = k["time"].dt.hour + 1
    k["res"] = k["supply"] - k["demand"]
    k["net"] = k["demand"] - k["renew"]
    kh = k.groupby(["date", "hour"])[["res", "net", "wind", "solar"]].mean().reset_index()
    ks = {c: _wide(kh, c) for c in ("res", "net", "wind", "solar")}
    f = load_smp_forecast_api()
    jd = _wide(f[f["region"] == "제주"], "demand_fcst")
    ld = _wide(f[f["region"] == "육지"], "demand_fcst").reindex_like(jd)
    cal = build_calendar(sj.index.min(), sj.index.max() + pd.Timedelta(days=10)).set_index("date")["is_offday"]
    om2 = load_openmeteo_prev2_jeju()
    omx = load_openmeteo_prevx_jeju()
    return dict(diff=diff, ks=ks, jd=jd, ld=ld, cal=cal, om2=om2, omx=omx)


def _om(src, H):
    if H == 1:
        o = src["om2"]
    elif H <= 6:
        o = src["omx"][src["omx"]["k"] == H + 1]
    else:
        return None
    o = o.assign(date=o["time"].dt.normalize(), hour=o["time"].dt.hour + 1)
    o = o.groupby(["date", "hour"])[["ws100", "ws10", "rad", "cloud"]].mean().reset_index()
    return o.rename(columns={c: f"om_{c}_h" for c in ("ws100", "ws10", "rad", "cloud")})


def hourly_extra(H, src) -> pd.DataFrame:
    """대상 (date, hour) × 새 피처. 모든 값은 판단일 I = date − H 기준으로 이미 알려진 것."""
    diff = src["diff"]
    days = diff.index
    sh = lambda w, n: w.shift(n)                                     # 대상일 D 행에 D−n 값
    up, dn = (diff >= EPS).astype(float).where(diff.notna()), (diff <= -EPS).astype(float).where(diff.notna())
    feats = {
        "dj_i_h": sh(diff, H), "dj_d7_h": sh(diff, 7),
        "dj_m7_h": sh(diff.rolling(7, min_periods=4).mean(), H), "dj_m28_h": sh(diff.rolling(28, min_periods=14).mean(), H),
        "dj_up7_h": sh(up.rolling(7, min_periods=4).mean(), H), "dj_up28_h": sh(up.rolling(28, min_periods=14).mean(), H),
        "dj_dn7_h": sh(dn.rolling(7, min_periods=4).mean(), H), "dj_dn28_h": sh(dn.rolling(28, min_periods=14).mean(), H),
    }
    # 직전 같은 유형일(대상일과 같은 휴무 여부, I일 이전) 같은 시각 차이
    cal = src["cal"].reindex(days)
    off = cal.astype(bool)
    same = pd.DataFrame(np.nan, index=days, columns=diff.columns)
    for flag in (True, False):
        s = diff[off == flag].reindex(days).ffill()               # 그날까지의 마지막 같은 유형일 값
        tgt = (off == flag).values
        same.loc[tgt] = s.shift(H).loc[tgt].values                  # I일까지의 마지막 같은 유형일
    feats["dj_same_h"] = same
    for c, w in src["ks"].items():
        w = w.reindex(days)
        feats[f"ks_{c}_d1_h"] = sh(w, H + 1)
        feats[f"ks_{c}_m7_h"] = sh(w.rolling(7, min_periods=4).mean(), H + 1)
    jd, ld = src["jd"].reindex(days), src["ld"].reindex(days)
    feats["jd_ratio_i_h"] = sh(jd / ld, H)
    feats["jd_rel28_i_h"] = sh(jd / jd.rolling(28, min_periods=14).mean(), H)
    out = None
    for k, w in feats.items():
        s = w.stack(future_stack=True).rename(k)
        out = s.to_frame() if out is None else out.join(s, how="outer")
    out = out.reset_index().rename(columns={"level_0": "date", "level_1": "hour"})
    out.columns = ["date", "hour"] + list(feats)
    om = _om(src, H)
    if om is not None:
        out = out.merge(om, on=["date", "hour"], how="left")
    else:
        for c in ("ws100", "ws10", "rad", "cloud"):
            out[f"om_{c}_h"] = np.nan
    return out


# ---------------------------------------------------------------- 모델
def htable(H, b, src, f=None):
    """f: 제주 피처 표 H (없으면 저장 파일, 운영은 jz_features.build로 새로 만든 표)."""
    d = (load_feats(H) if f is None else f).merge(b[b["H"] == H].drop(columns="H"), on=["date", "hour"], how="inner")
    d = d.merge(hourly_extra(H, src), on=["date", "hour"], how="left")
    d["b_spread"] = d["b_hi"] - d["b_lo"]
    d["b_vs_d1"] = d["b_pred"] - d["s_d1"]
    d["b_vs_same"] = d["b_pred"] - d["s_same_h"]
    d["b_vs_ma28"] = d["b_pred"] - d["s_ma28"]
    d["r"] = d["y"] - d["b_pred"]
    d["dif"] = d["y"] - d["y_land"]
    return d


def cols_of(d):
    new = [c for c in d.columns if c.startswith(("dj_", "ks_", "jd_", "om_"))]
    return OLD + jz3.CD + new


def _w(tr, a):
    age = (a - tr["date"]).dt.days.values
    return 0.5 ** (age / 180)


def fit_month(tr, te, cols, a):
    """한 달치 후보별 조정값."""
    o = {}
    for obj in ("l1", "l2"):
        x = tr[tr["r"].notna()]
        o[f"R {obj}"] = lgb.LGBMRegressor(**dict(LGB, objective=obj)).fit(x[cols], x["r"]).predict(te[cols])
        x = tr[tr["dif"].notna()]
        o[f"D {obj}"] = lgb.LGBMRegressor(**dict(LGB, objective=obj)).fit(x[cols], x["dif"]).predict(te[cols])
    o["Dw l2"] = lgb.LGBMRegressor(**dict(LGB, objective="l2")).fit(x[cols], x["dif"], sample_weight=_w(x, a)).predict(te[cols])
    pu = lgb.LGBMClassifier(**LGBC).fit(x[cols], (x["dif"] >= EPS).astype(int)).predict_proba(te[cols])[:, 1]
    pd_ = lgb.LGBMClassifier(**LGBC).fit(x[cols], (x["dif"] <= -EPS).astype(int)).predict_proba(te[cols])[:, 1]
    xu, xd = x[x["dif"] >= EPS], x[x["dif"] <= -EPS]
    mu = lgb.LGBMRegressor(**dict(LGB, objective="l2")).fit(xu[cols], xu["dif"]).predict(te[cols])
    md = (lgb.LGBMRegressor(**dict(LGB, objective="l2", min_child_samples=50)).fit(xd[cols], xd["dif"]).predict(te[cols])
          if len(xd) >= 200 else np.full(len(te), xd["dif"].mean() if len(xd) else 0.0))
    o["Te"] = pu * mu + pd_ * md
    o["Tm"] = np.where(pu >= 0.5, mu, np.where(pd_ >= 0.5, md, 0.0))
    o["_pu"], o["_pd"] = pu, pd_
    return o


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src = sources()
    b = jz3.base_all()
    print(f"자료 준비 {time.time() - t0:.0f}초", flush=True)
    parts = []
    for H in HS:
        d = htable(H, b, src)
        cols = cols_of(d)
        for a in pd.date_range(PRED_START, d["date"].max(), freq="MS"):
            e = a + pd.offsets.MonthBegin(1)
            tr = d[(d["date"] >= TRAIN_START) & (d["date"] < a - pd.Timedelta(days=H - 1))]
            te = d[(d["date"] >= a) & (d["date"] < e)]
            if te.empty:
                continue
            o = fit_month(tr, te, cols, a)
            parts.append(te[["date", "hour"]].assign(H=H, **o))
        print(f"  H={H} 피처 {len(cols)}개 ({time.time() - t0:.0f}초)", flush=True)
    pd.concat(parts).round(3).to_csv(OUT / "adj.csv.gz", index=False, encoding="utf-8")
    pd.Series(cols).to_csv(OUT / "cols.csv", index=False)


# ---------------------------------------------------------------- 선택 (2025만)
def table():
    d = jz3.table()
    a = pd.read_csv(OUT / "adj.csv.gz", parse_dates=["date"])
    d = d.merge(a, on=["date", "hour", "H"], how="left")
    L = load_smp_actual("육지")[["date", "hour", "smp"]].rename(columns={"smp": "y_land"})
    d = d.merge(L, on=["date", "hour"], how="left")
    d["dif"] = d["y"] - d["y_land"]
    return d


CANDS = {"J1 K3 그대로": None, "J2 기존 제주 보정(l2)": "rhat_l2", "R 잔차형 l1": "R l1", "R 잔차형 l2": "R l2",
         "D 차이형 l1": "D l1", "D 차이형 l2": "D l2", "Dw 차이형 l2 최근 가중": "Dw l2", "Te 2단계 기대값": "Te", "Tm 2단계 중앙값형": "Tm"}


def points(d, fit_per):
    out = {}
    m = jz3.between(d, fit_per) & d["y"].notna().values
    for name, col in CANDS.items():
        if col is None:
            out[name] = (d["b_pred"].values, None)
            continue
        lam = min(GRID, key=lambda l: (jz3.mae(d["y"][m], (d["b_pred"] + l * d[col])[m]), l))
        out[name] = ((d["b_pred"] + lam * d[col]).values, lam)
    return out


def select():
    d = table()
    ok = d["y"].notna().values & d["Te"].notna().values & d["j0"].notna().values
    m = jz3.between(d, H2) & ok
    x = d[m]
    ne, up, dn = (x["dif"].abs() >= EPS).values, (x["dif"] >= EPS).values, (x["dif"] <= -EPS).values
    rows = []
    for i, (name, (p, lam)) in enumerate(points(d, H1).items()):
        pm = p[m]
        e = np.abs(x["y"].values - pm)
        r = {"후보": name, "단순 순서": i, "λ (2025 상반기)": lam,
             "2025 상반기 MAE": jz3.mae(d["y"][jz3.between(d, H1) & ok], p[jz3.between(d, H1) & ok]),
             "2025 하반기 MAE": e.mean(), "제주≠육지 MAE": e[ne].mean(), "제주=육지 MAE": e[~ne].mean(),
             "제주>육지 MAE": e[up].mean(), "제주<육지 MAE": e[dn].mean()}
        for seg, s in hourly_segments(x).items():
            if seg != "전체":
                r[seg] = e[s].mean()
        r["저녁 18~22시"] = e[x["hour"].between(18, 22).values].mean()
        for H in HS:
            r[f"D+{H}"] = e[(x["H"] == H).values].mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    # J0 지금 모델 (참고)
    e0 = np.abs(x["y"].values - x["j0"].values)
    t = pd.concat([t, pd.DataFrame([{"후보": "J0 지금 (참고)", "단순 순서": 99, "2025 하반기 MAE": e0.mean(), "제주≠육지 MAE": e0[ne].mean(),
                                      "제주=육지 MAE": e0[~ne].mean(), "제주>육지 MAE": e0[up].mean(), "제주<육지 MAE": e0[dn].mean()}])],
                  ignore_index=True)
    c = t[t["단순 순서"] < 99]
    best = c["2025 하반기 MAE"].min()
    pick = c[c["2025 하반기 MAE"] <= best + TIE].sort_values(["제주≠육지 MAE", "단순 순서"]).iloc[0]
    t["선택"] = np.where(t["후보"] == pick["후보"], "✓", "")
    t.round(4).to_csv(TAB / "JWZ7_select_2025.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    print(f"2025 하반기 {m.sum():,}시간, 제주≠육지 {ne.mean():.1%} (제주>육지 {up.mean():.1%}, 제주<육지 {dn.mean():.1%})")
    show = ["후보", "λ (2025 상반기)", "2025 상반기 MAE", "2025 하반기 MAE", "제주≠육지 MAE", "제주=육지 MAE", "제주>육지 MAE", "제주<육지 MAE",
            "휴무일 한낮(11~15시)", "저녁 18~22시", "선택"]
    print(t[show].round(3).to_string(index=False))
    print(t[["후보"] + [f"D+{H}" for H in HS]].round(3).to_string(index=False))
    # 2단계 분류 품질 (참고)
    from sklearn.metrics import average_precision_score, roc_auc_score
    for k, lab in (("_pu", up), ("_pd", dn)):
        if lab.sum():
            print(f"  {k}: PR-AUC {average_precision_score(lab, x[k]):.3f} (기저 {lab.mean():.3f}), ROC-AUC {roc_auc_score(lab, x[k]):.3f}")
    print("선택:", pick["후보"])


def _main():
    args = sys.argv[1:] or ["build", "select"]
    if "build" in args:
        build()
    if "select" in args:
        select()
    if "test" in args:
        test()


# ---------------------------------------------------------------- 범위 선택(2025 하반기) + 2026 한 번 채점
def test(pick="R 잔차형 l1"):
    """고른 후보로 범위(A1 대칭, 365 → 180 → 90일, 2025 하반기 포함률 75~85% 중 구간점수 최소·1% 이내 단순) → λ 2025 전체로 다시 맞춤
    → 2026-01-01 ~ 10-18 한 번 채점, 지금 모델(J0)·K3 그대로(J1)와 같은 시간끼리. 출력 JWZ7_range_2025H2.csv, JWZ7_test2026.csv,
    JWZ7_choice.json, jeju/JWZ7_predictions_hourly.csv (2025-01-01 ~, ROLES 4.2 형식)."""
    import json
    from smp_forecast_app.src.final_metrics import metrics
    d = table()
    col = CANDS[pick]
    lam_h1 = points(d, H1)[pick][1]
    p_sel = (d["b_pred"] + lam_h1 * d[col]).values
    ev = jz3.between(d, H2)
    ok = ev & d["y"].notna().values & d["j0"].notna().values
    rr = []
    for i, w in enumerate(jz3.WINDOWS):
        lo, hi = jz3.sym_range(d, p_sel, w, ev)
        m = ok & ~np.isnan(lo)
        rr.append({"범위": f"A1 대칭 · {w}일", "window": w, "단순 순서": i, **jz3.rrow(d["y"].values[m], lo[m], hi[m])})
    r = pd.DataFrame(rr)
    c = r[r["포함률"].between(0.75, 0.85)]
    c = c if len(c) else r.iloc[[(r["포함률"] - 0.8).abs().argmin()]]
    win = int(c[c["구간점수"] <= c["구간점수"].min() * (1 + jz3.TIE_RANGE)].sort_values("단순 순서").iloc[0]["window"])
    r["선택"] = np.where(r["window"] == win, "✓", "")
    r.round(4).to_csv(TAB / "JWZ7_range_2025H2.csv", index=False, encoding="utf-8-sig")
    print(r.round(3).to_string(index=False))

    lam = points(d, jz3.Y25)[pick][1]
    p = (d["b_pred"] + lam * d[col]).values
    ev = d["date"].values >= np.datetime64("2025-01-01")
    lo, hi = jz3.sym_range(d, p, win, ev)
    lo1, hi1 = jz3.sym_range(d, d["b_pred"].values, win, ev)       # 참고: K3 그대로 + 같은 범위
    w = d.assign(pred=p, lo=lo, hi=hi, p1=d["b_pred"], lo1=lo1, hi1=hi1)
    out = w[ev & ~np.isnan(lo)][["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday"]]
    out.assign(stage="15h").round(2).to_csv(JDIR / "JWZ7_predictions_hourly.csv", index=False, encoding="utf-8-sig")

    m = jz3.between(w, jz3.TEST) & w["y"].notna().values & w["j0"].notna().values & ~np.isnan(lo)
    x = w[m]
    x = x.assign(ne=(x["dif"].abs() >= EPS), up=(x["dif"] >= EPS))
    segs = list(hourly_segments(x).items()) + [("제주≠육지", x["ne"].values), ("제주>육지", x["up"].values),
                                              ("제주=육지", ~x["ne"].values)] + [(f"D+{H}", (x["H"] == H).values) for H in HS]
    rows = []
    for seg, s in segs:
        g = x[s]
        for mdl, (pc, lc, hc) in {f"새: K3 + {lam} × 제주 보정({pick})": ("pred", "lo", "hi"), "K3 그대로 (같은 범위)": ("p1", "lo1", "hi1"),
                                  "지금 (J0: 이전 육지 + 제주 보정)": ("j0", "j0_lo", "j0_hi")}.items():
            rows.append({"구간": seg, "모델": mdl, **metrics(g["y"], g[pc], g[lc], g[hc], naive=g["s_d7"])})
    t = pd.DataFrame(rows)
    rng = np.random.default_rng(42)
    e = x.assign(dn=np.abs(x["y"] - x["pred"]), d0=np.abs(x["y"] - x["j0"]), d1=np.abs(x["y"] - x["p1"])).groupby("date")[["dn", "d0", "d1"]].sum()
    n = x.groupby("date").size().values
    idx = [rng.integers(0, len(n), len(n)) for _ in range(2000)]
    ci = lambda a, b: [float(np.quantile([(e[a].values[i].sum() - e[b].values[i].sum()) / n[i].sum() for i in idx], q)) for q in (0.025, 0.975)]
    ch = {"점예측": f"K3 + λ × 제주 보정 ({pick})", "λ (2025 상반기, 선택)": lam_h1, "λ (2025 전체, 최종)": lam,
          "보정 모델": "LightGBM 잔차(제주 실적 − K3), 목적 l1 (jz_resid.LGB), 대상일의 달마다 재학습 (2023-01 ~ 그달 1일 − (H−1)일 실적)",
          "피처": pd.read_csv(OUT / "cols.csv").iloc[:, 0].tolist(),
          "범위": f"A1 대칭 · {win}일 (Chronos-2 0.1·0.9 + (점예측 − K3 Chronos), 칸 = 시간대묶음 × 휴무일 × 지평, 하한 0원 자르기 없음)",
          "window": win, "선택 규칙": "2025 하반기 전체 MAE ≤ 최소 + 0.05원 중 제주≠육지 시간 MAE 최소 (JWZ7_select_2025.csv)",
          "육지 입력": "K3 (wz3/ops/k3_parts.csv.gz) — 이전 육지 일주일 모델 불필요",
          "2026 MAE 차 95% (새 − 지금 J0)": ci("dn", "d0"), "2026 MAE 차 95% (새 − K3 그대로)": ci("dn", "d1")}
    t.round(4).to_csv(TAB / "JWZ7_test2026.csv", index=False, encoding="utf-8-sig")
    (TAB / "JWZ7_choice.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 320)
    print(t[["구간", "모델", "표본수", "MAE", "rMAE", "편향(공표−예측)", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수"]].round(3).to_string(index=False))
    print({k: v for k, v in ch.items() if "95%" in k or "λ" in k})


if __name__ == "__main__":
    _main()
