"""제주 일주일 1차(15시) — 육지 일주일이 K3로 바뀐 뒤 다시 고르기 (R2, 2026-11-03).

규칙 (2025 결과를 보기 전에 고정, JEJU.md 11절)
  기준 육지 = K3 (wz3/ops/k3_parts.csv.gz, 백테스트 + 운영 기록. 대상일 2024-01-02 ~). 2023 대상일(잔차 학습용)만
             이전 육지 예측(jz_common.land_pred의 L_base)으로 채운다. 범위·분위 폭은 Chronos-2 대형판 0.1·0.9 (K3와 같은 출처)
  후보 (단순한 순)
    J1  K3 그대로
    J2  K3 + λ × 제주 잔차 보정 (LightGBM, 피처 '제주외부' − b_vs_land, 목적 l1 → l2 순). 육지 이전 모델이 필요 없게
        jz_features.load(H) 표만 쓴다. 학습 2023-01 ~, 대상일의 달마다 재학습 (실적은 그달 1일 − (H−1)일까지)
    J0  지금 운영 (이전 육지 + 0.25 × 제주 보정, JWZ_predictions_hourly.csv) — 이전 육지 모델을 계속 돌려야 하므로 가장 무겁게 본다
  λ: 0, 0.1, 0.25, 0.5, 0.75, 1 중 2025 상반기 MAE 최소
  선택: 2025 하반기 MAE 최소, 0.05원 이내면 단순한 쪽 (J0과 같은 시간끼리)
  범위 (고른 점예측이 J1·J2일 때): A1 대칭 보정 — Chronos-2 0.1·0.9 + (점예측 − K3 Chronos), 칸(시간대묶음 × 휴무일)마다
        직전 window일 max(하한−실적, 실적−상한)의 80% 분위수만큼 넓힘, 하한 0원 자르기 없음(제주).
        window 365 → 180 → 90 (단순한 순 = 육지와 같은 365 먼저). 2025 하반기 포함률 75~85% 중 구간점수 최소, 1% 이내면 단순한 쪽
  최종: λ를 2025 전체로 다시 맞춤 → 2026-01-01 ~ 10-18 한 번 채점, 지금 모델과 같은 시간끼리
출력: outputs/tables/JWZ6_select_2025.csv, JWZ6_range_2025H2.csv, JWZ6_choice.json
      (test) JWZ6_test2026.csv, data/processed/smp_forecast_app/jeju/JWZ6_predictions_hourly.csv
실행: python -m smp_forecast_app.src.jz3 select | test
"""
import json
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from smp_forecast_app.src.final_metrics import hourly_segments, metrics
from smp_forecast_app.src.intervals import BAND
from smp_forecast_app.src.jz_common import HS, JDIR, TRAIN_START, land_pred
from smp_forecast_app.src.jz_features import CAL, DEM, OM, SPREAD, WEATHER, load as load_feats
from smp_forecast_app.src.jz_resid import LGB

warnings.filterwarnings("ignore")

W3 = PROCESSED_DIR / "smp_forecast_app" / "wz3"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
RES = JDIR / "jz3"
H1, H2, Y25, TEST = ("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"), ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-10-18")
PRED_START = "2024-01-01"          # 범위 이력(365일)을 위해 2024부터 표본 밖 잔차를 만든다
GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
TIE_MAE, TIE_RANGE = 0.05, 0.01
WINDOWS = (365, 180, 90)
CD = ["b_pred", "b_spread", "b_vs_d1", "b_vs_same", "b_vs_ma28"]
COLS = CAL + DEM + SPREAD + WEATHER + OM + CD
OBJ = ("l1", "l2")


# ---------------------------------------------------------------- 육지 K3
def k3_land() -> pd.DataFrame:
    """[date, hour, H, K_pred, K_c, K_qlo, K_qhi] — K3 점예측·기본 Chronos·Chronos-2 대형판 분위수 (같은 (날짜, H)는 마지막 실행)."""
    k = pd.read_csv(W3 / "ops" / "k3_parts.csv.gz", usecols=["date", "hour", "H", "pred", "c", "c2_qlo", "c2_qhi"],
                    parse_dates=["date"])
    k = k[k["pred"].notna()].drop_duplicates(["date", "hour", "H"], keep="last")
    return k.rename(columns={"pred": "K_pred", "c": "K_c", "c2_qlo": "K_qlo", "c2_qhi": "K_qhi"})


def c2_hist() -> pd.DataFrame:
    """Chronos-2 대형판 0.1·0.9 과거 예측 (2023~, K3 이력에 없는 날 채우기용)."""
    from smp_forecast_app.src.wz3_select import c2_quantiles
    return c2_quantiles().drop_duplicates(["date", "hour", "H"], keep="last").rename(columns={"qlo": "q2lo", "qhi": "q2hi"})


def base_all() -> pd.DataFrame:
    """기준 예측 b_pred(K3, 2023은 이전 육지) + 분위수 b_lo·b_hi (Chronos-2 대형판)."""
    k = k3_land()
    old = land_pred()[["date", "hour", "H", "L_base", "L_qlo", "L_qhi"]]
    d = old.merge(k, on=["date", "hour", "H"], how="outer").merge(c2_hist(), on=["date", "hour", "H"], how="left")
    d["b_pred"] = d["K_pred"].fillna(d["L_base"])
    d["b_lo"] = d["K_qlo"].fillna(d["q2lo"]).fillna(d["L_qlo"])
    d["b_hi"] = d["K_qhi"].fillna(d["q2hi"]).fillna(d["L_qhi"])
    d["b_c"] = d["K_c"].fillna(d["b_pred"])
    return d[["date", "hour", "H", "b_pred", "b_lo", "b_hi", "b_c", "K_pred"]]


def rtable(H, b):
    d = load_feats(H).merge(b[b["H"] == H].drop(columns="H"), on=["date", "hour"], how="inner")
    d["b_spread"] = d["b_hi"] - d["b_lo"]
    d["b_vs_d1"] = d["b_pred"] - d["s_d1"]
    d["b_vs_same"] = d["b_pred"] - d["s_same_h"]
    d["b_vs_ma28"] = d["b_pred"] - d["s_ma28"]
    d["r"] = d["y"] - d["b_pred"]
    return d


def resid(H, d, obj, start=PRED_START):
    """대상일의 달마다 재학습: 학습 = 2023-01 ~ (그달 1일 − (H−1)일) 실적, 예측 = 그달."""
    out = []
    for a in pd.date_range(start, d["date"].max(), freq="MS"):
        b = a + pd.offsets.MonthBegin(1)
        tr = d[(d["date"] >= TRAIN_START) & (d["date"] < a - pd.Timedelta(days=H - 1)) & d["r"].notna()]
        te = d[(d["date"] >= a) & (d["date"] < b)]
        if te.empty:
            continue
        m = lgb.LGBMRegressor(**dict(LGB, objective=obj)).fit(tr[COLS], tr["r"])
        out.append(te[["date", "hour"]].assign(H=H, rhat=m.predict(te[COLS])))
    return pd.concat(out)


def build_resid():
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b = base_all()
    parts = {o: [] for o in OBJ}
    for H in HS:
        d = rtable(H, b)
        for o in OBJ:
            parts[o].append(resid(H, d, o))
        print(f"  잔차 H={H} ({time.time() - t0:.0f}초)", flush=True)
    for o in OBJ:
        pd.concat(parts[o]).round(3).to_csv(RES / f"resid_{o}.csv.gz", index=False, encoding="utf-8")


def table() -> pd.DataFrame:
    """대상일 2024~ 제주 실적 + K3 기준 + 잔차(l1·l2) + 지금 모델(J0)."""
    b = base_all()
    b = b[b["K_pred"].notna()]
    y = pd.concat([load_feats(H)[["date", "hour", "y", "offday", "s_d7"]].assign(H=H) for H in HS])
    d = b.merge(y, on=["date", "hour", "H"], how="inner")
    for o in OBJ:
        r = pd.read_csv(RES / f"resid_{o}.csv.gz", parse_dates=["date"]).rename(columns={"rhat": f"rhat_{o}"})
        d = d.merge(r, on=["date", "hour", "H"], how="left")
    j0 = pd.read_csv(JDIR / "JWZ_predictions_hourly.csv", parse_dates=["date"])[["date", "hour", "H", "pred", "lo", "hi"]]
    d = d.merge(j0.rename(columns={"pred": "j0", "lo": "j0_lo", "hi": "j0_hi"}), on=["date", "hour", "H"], how="left")
    d["offday"] = d["offday"].astype(bool)
    d["base_date"] = d["date"] - pd.to_timedelta(d["H"], unit="D")
    return d[d["date"] >= PRED_START].sort_values(["H", "date", "hour"]).reset_index(drop=True)


def between(d, per):
    return d["date"].between(*per).values


def mae(y, p):
    return float(np.nanmean(np.abs(np.asarray(y) - np.asarray(p))))


def fit_lambda(d, col, per):
    m = between(d, per) & d["y"].notna().values & d[col].notna().values
    return min(GRID, key=lambda l: (mae(d["y"][m], (d["b_pred"] + l * d[col])[m]), l))


def points(d, fit_per):
    """후보 → (점예측, λ). 순서 = 단순한 순."""
    out = {"J1 K3 그대로": (d["b_pred"].values, None)}
    for o in OBJ:
        lam = fit_lambda(d, f"rhat_{o}", fit_per)
        out[f"J2 K3 + 제주 보정({o})"] = ((d["b_pred"] + lam * d[f"rhat_{o}"]).values, lam)
    out["J0 지금 (이전 육지 + 제주 보정)"] = (d["j0"].values, 0.25)
    return out


# ---------------------------------------------------------------- 범위 (A1, 하한 0원 자르기 없음)
def sym_range(d, pred, window, ev):
    """칸(시간대묶음×휴무일)·지평별, 대상일마다 직전 window일(대상일 < 그날 − (H−1)) 대칭 점수 80% 분위수로 넓힘."""
    lo, hi = np.full(len(d), np.nan), np.full(len(d), np.nan)
    shift = pred - d["b_c"].values
    ql, qh = d["b_lo"].values + shift, d["b_hi"].values + shift
    y = d["y"].values
    s = np.maximum(ql - y, y - qh)
    key = d["hour"].map(BAND).values + np.where(d["offday"].values, "_1", "_0") + "_" + d["H"].astype(str).values
    dates = d["date"].values.astype("datetime64[D]")
    Hs = d["H"].values
    for k in np.unique(key):
        ik = np.where(key == k)[0]
        H = int(Hs[ik[0]])
        dk, ok = dates[ik], ~np.isnan(s[ik])
        for day in np.unique(dk[ev[ik]]):
            tgt = ik[(dk == day) & ev[ik]]
            end = day - np.timedelta64(H - 1, "D")
            hist = ik[(dk < end) & (dk >= end - np.timedelta64(window, "D")) & ok]
            if len(hist) < 20:
                continue
            c = np.quantile(s[hist], 0.8)
            lo[tgt], hi[tgt] = ql[tgt] - c, qh[tgt] + c
    return np.minimum(lo, pred), np.maximum(hi, pred)


def rrow(y, lo, hi):
    r = metrics(y, y, lo, hi)
    return {k: r[k] for k in ("포함률", "하한 아래", "상한 위", "평균 폭", "구간점수")}


# ---------------------------------------------------------------- ① 선택 (2025만)
def select():
    t0 = time.time()
    if not (RES / "resid_l2.csv.gz").exists():
        build_resid()
    d = table()
    ok = d["y"].notna().values & d["j0"].notna().values & d["rhat_l1"].notna().values
    rows = []
    pts = points(d, H1)
    for i, (name, (p, lam)) in enumerate(pts.items()):
        r = {"후보": name, "단순 순서": i, "λ (2025 상반기)": lam}
        for lab, per in (("2025 상반기", H1), ("2025 하반기", H2)):
            m = between(d, per) & d["y"].notna().values & ~np.isnan(p)
            r[f"{lab} MAE"] = mae(d["y"][m], p[m])
        m = between(d, H2) & ok
        r["2025 하반기 MAE (J0과 같은 시간)"] = mae(d["y"][m], p[m])
        x = d[m]
        for seg, s in hourly_segments(x).items():
            r[f"하반기 {seg}"] = mae(x["y"][s], p[m][s])
        for H in HS:
            s = (x["H"] == H).values
            r[f"하반기 D+{H}"] = mae(x["y"][s], p[m][s])
        r["시간수"] = int(m.sum())
        rows.append(r)
    t = pd.DataFrame(rows)
    key = "2025 하반기 MAE (J0과 같은 시간)"
    best = t[key].min()
    pick = t[t[key] <= best + TIE_MAE].sort_values("단순 순서").iloc[0]
    t["선택"] = np.where(t["후보"] == pick["후보"], "✓", "")
    t.round(4).to_csv(TAB / "JWZ6_select_2025.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(t.drop(columns=[c for c in t.columns if c.startswith("하반기 D+")]).round(3).to_string(index=False))
    print(t[["후보"] + [f"하반기 D+{H}" for H in HS]].round(3).to_string(index=False))

    ch = {"점예측": pick["후보"], "λ (2025 상반기)": pick["λ (2025 상반기)"]}
    if not pick["후보"].startswith("J0"):
        p = pts[pick["후보"]][0]
        ev = between(d, H2)
        rr = []
        for i, w in enumerate(WINDOWS):
            lo, hi = sym_range(d, p, w, ev)
            m = ev & d["y"].notna().values & ~np.isnan(lo) & ok
            rr.append({"범위": f"A1 대칭 · {w}일", "단순 순서": i, **rrow(d["y"].values[m], lo[m], hi[m])})
        m = ev & ok
        rr.append({"범위": "J0 지금 범위 (참고)", "단순 순서": 9, **rrow(d["y"].values[m], d["j0_lo"].values[m], d["j0_hi"].values[m])})
        r = pd.DataFrame(rr)
        cand = r[r["범위"].str.startswith("A1") & r["포함률"].between(0.75, 0.85)]
        if len(cand):
            rp = cand[cand["구간점수"] <= cand["구간점수"].min() * (1 + TIE_RANGE)].sort_values("단순 순서").iloc[0]["범위"]
        else:   # 75~85%가 없으면 80%에 가장 가까운 것
            a1 = r[r["범위"].str.startswith("A1")]
            rp = a1.iloc[(a1["포함률"] - 0.8).abs().argmin()]["범위"]
        r["선택"] = np.where(r["범위"] == rp, "✓", "")
        r.round(4).to_csv(TAB / "JWZ6_range_2025H2.csv", index=False, encoding="utf-8-sig")
        print(r.round(3).to_string(index=False))
        ch["범위"] = rp
        ch["window"] = int(rp.split("·")[1].replace("일", "").strip())
    (TAB / "JWZ6_choice.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
    print("선택:", ch, f"({time.time() - t0:.0f}초)")


# ---------------------------------------------------------------- ② 2026 한 번 채점
def test():
    ch = json.loads((TAB / "JWZ6_choice.json").read_text(encoding="utf-8"))
    d = table()
    name = ch["점예측"]
    p, lam = points(d, Y25)[name]
    ch["λ (2025 전체, 최종)"] = lam
    ev = d["date"].values >= np.datetime64(Y25[0])
    lo, hi = sym_range(d, p, ch["window"], ev)
    w = d.assign(pred=p, lo=lo, hi=hi)
    out = w[ev & ~np.isnan(lo)][["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday"]]
    out.assign(stage="15h").round(2).to_csv(JDIR / "JWZ6_predictions_hourly.csv", index=False, encoding="utf-8-sig")

    m = between(w, TEST) & w["y"].notna().values & w["j0"].notna().values & ~np.isnan(lo)
    x = w[m]
    rows = []
    for seg, s in list(hourly_segments(x).items()) + [(f"D+{H}", (x["H"] == H).values) for H in HS]:
        g = x[s]
        for mdl, (pc, lc, hc) in {"새 (" + name + ")": ("pred", "lo", "hi"), "지금 (J0)": ("j0", "j0_lo", "j0_hi")}.items():
            r = metrics(g["y"], g[pc], g[lc], g[hc], naive=g["s_d7"])
            rows.append({"구간": seg, "모델": mdl, **r})
    t = pd.DataFrame(rows)
    # 날짜 부트스트랩 (전체 MAE 차이 95% 구간)
    rng = np.random.default_rng(42)
    e = x.assign(dn=np.abs(x["y"] - x["pred"]), d0=np.abs(x["y"] - x["j0"])).groupby("date")[["dn", "d0"]].agg(["sum", "count"])
    sn, s0, n = e[("dn", "sum")].values, e[("d0", "sum")].values, e[("dn", "count")].values
    bs = [(sn[i].sum() - s0[i].sum()) / n[i].sum() for i in (rng.integers(0, len(n), len(n)) for _ in range(2000))]
    ch["2026 MAE 차 (새 − 지금) 95%"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    t.round(4).to_csv(TAB / "JWZ6_test2026.csv", index=False, encoding="utf-8-sig")
    (TAB / "JWZ6_choice.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 300)
    print(t[["구간", "모델", "표본수", "MAE", "rMAE", "편향(공표−예측)", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수"]]
          .round(3).to_string(index=False))
    print("MAE 차 95%:", ch["2026 MAE 차 (새 − 지금) 95%"])


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1] if len(sys.argv) > 1 else "select"]()
