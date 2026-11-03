"""일주일 v3 (R3) ⑨ 팀원 일주일 코드 + 우리 외부 정보 잔차 (사후). 규칙은 WEEK_V3.md ⑨ (결과 보기 전 고정).

팀원 코드(reference/일주일예측_코드_20261102/weekly_forecast_code/task2_weekly.py, 읽기만)의 함수·순서를 그대로 쓰고
잔차 피처·LightGBM 설정만 바꾼다. Chronos 예측 = 팀원 코드 실행분 wz3/tm/chronos_raw.csv.
  select : T0·T1·T2를 팀원 규칙으로 맞춘 뒤 2025 대상일(K3와 같은 시간)로 비교 → WZ3_X9_tmx_2025.csv
           (2026 예측은 wz3/tm/T*_pred.csv로 저장만, 채점하지 않음)
  test   : 고른 하나(인자)를 2026-01-01 ~ 10-18 한 번 채점 → WZ3_X10_tmx_2026.csv
실행: python -m smp_forecast_app.src.wz3_tmx select | test T1
"""
import importlib.util
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.config import ROOT
from common.loaders import load_smp_actual
from smp_forecast_app.src import range_adjust
from smp_forecast_app.src.final_metrics import hourly_segments, metrics
from smp_forecast_app.src.wz3_select import DATA, FIT, TAB, TEST, W3, build_point
from smp_forecast_app.src.wz3_rangeB import A1, K3
from smp_forecast_app.src.wz3_select import c2_quantiles, make_range
from smp_forecast_app.src.wz_resid import LGB

TM_FILE = ROOT / "reference" / "일주일예측_코드_20261102" / "weekly_forecast_code" / "task2_weekly.py"
_spec = importlib.util.spec_from_file_location("tm_weekly", TM_FILE)
tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tm)

TMD = W3 / "tm"
EXO_ADD = ["prev_offday", "next_offday", "type_change", "is_myeongjeol", "fc_lng", "fc_coal", "cap_solar",
           "kpx_peak", "kpx_peak_rel", "kpx_peak_vs_d1", "rf_sky_pm", "rf_pop_pm", "rf_ta_pm", "rf_sky_am",
           "om_rad_mid", "om_cloud_mid"]
VARIANTS = {"T0": (tm.RES_FEATS, tm.RES_PARAMS), "T1": (tm.RES_FEATS + EXO_ADD, tm.RES_PARAMS),
            "T2": (tm.RES_FEATS + EXO_ADD, LGB)}
OUT1, OUT2 = TAB / "WZ3_X9_tmx_2025.csv", TAB / "WZ3_X10_tmx_2026.csv"


def rows_with_features():
    s, off = tm.load_series()
    fc = pd.read_csv(TMD / "chronos_raw.csv", encoding="utf-8-sig")
    rows = tm.build_rows(fc, s, off)
    rows = rows[(rows["k"] >= 1) & (rows["k"] <= tm.DAYS_AHEAD)].reset_index(drop=True)
    f = []
    for H in range(1, 8):
        x = pd.read_csv(W3 / "features" / f"features_h{H}.csv", usecols=["date", "hour"] + EXO_ADD)
        f.append(x.assign(k=H).rename(columns={"hour": "hour_n"}))
    rows = rows.merge(pd.concat(f), on=["date", "hour_n", "k"], how="left")
    return rows


def run_variant(rows, feats, params):
    """팀원 main()의 중심·범위 부분을 피처·설정만 바꿔 그대로 → (val, test, 설정 정보)."""
    fit = lambda r: lgb.LGBMRegressor(**params).fit(r.dropna(subset=["y"])[feats], r.dropna(subset=["y"]).eval("y - q50"))
    b24 = rows[(rows["origin"] < tm.VALID_FROM) & (rows["date"] < tm.VALID_FROM)].copy()
    val = rows[(rows["origin"] >= tm.VALID_FROM) & (rows["origin"] < tm.TEST_FROM)].copy()
    test = rows[rows["origin"] >= tm.TEST_FROM].copy()
    val["corr"] = fit(b24).predict(val[feats])
    b24["corr"] = np.nan
    for q in range(1, 5):
        mask = pd.to_datetime(b24["origin"]).dt.quarter == q
        b24.loc[mask, "corr"] = fit(b24[~mask]).predict(b24.loc[mask, feats])
    vok = val["y"].notna()
    mae_corr = (val.loc[vok, "y"] - val.loc[vok, "q50"] - val.loc[vok, "corr"]).abs().mean()
    mae_raw = (val.loc[vok, "y"] - val.loc[vok, "q50"]).abs().mean()
    use = mae_corr < mae_raw - 0.05
    test["corr"] = fit(pd.concat([b24, val[val["date"] < tm.TEST_FROM]])).predict(test[feats])
    for part in (b24, val, test):
        if not use:
            part["corr"] = 0.0
        part["p"] = part["q50"] + part["corr"]
        part["err"] = part["y"] - part["p"]
    b25 = pd.concat([b24, val[val["date"] < tm.TEST_FROM]])

    def methods(target, build_err, history):
        blo, bhi = tm.apply_q(target, tm.grouped_q(build_err.dropna(subset=["err"])))
        blo, bhi = np.nan_to_num(blo), np.nan_to_num(bhi)
        clo, chi = tm.rolling_bounds(target, history.dropna(subset=["err"]), blo, bhi)
        p = target["p"].values
        return {"A": (target["q10"].values + target["corr"].values, target["q90"].values + target["corr"].values),
                "B": (p + blo, p + bhi), "C": (p + clo, p + chi)}

    bv, bt = methods(val, b24, pd.concat([b24, val])), methods(test, b25, pd.concat([b25, test]))
    ok = val["y"].notna().values
    st = {k: tm.range_stats(val["y"].values[ok], lo[ok], hi[ok]) for k, (lo, hi) in bv.items()}
    inr = {k: v for k, v in st.items() if tm.TARGET[0] <= v["coverage"] <= tm.TARGET[1]}
    pick = min(inr, key=lambda k: inr[k]["interval_score"]) if inr else min(st, key=lambda k: abs(st[k]["coverage"] - 80))
    val = val.assign(lo=bv[pick][0], hi=bv[pick][1])
    test = test.assign(lo=bt[pick][0], hi=bt[pick][1])
    info = {"2025 Chronos 단독 MAE(팀원 기준)": mae_raw, "2025 보정 MAE(팀원 기준)": mae_corr, "보정 사용": bool(use),
            "범위": pick, **{f"2025 범위 {k} 포함률": v["coverage"] for k, v in st.items()}}
    return val, test, info


def k3_2025():
    d, pred, _ = build_point(K3, FIT)
    lo, hi = make_range(d, pred, A1, ("2025-01-01", "2025-12-31"), c2_quantiles())
    k = d.assign(pred_k3=pred, lo_k3=lo, hi_k3=hi)
    k = k[k["date"].between("2025-01-01", "2025-12-31")]
    return k[["date", "hour", "H", "y", "offday", "pred_k3", "lo_k3", "hi_k3"]]


def _std(x, tag):
    x = x.assign(date=pd.to_datetime(x["date"])).rename(columns={"hour_n": "hour", "k": "H"})
    return x[["date", "hour", "H", "p", "lo", "hi"]].rename(columns={"p": f"pred_{tag}", "lo": f"lo_{tag}", "hi": f"hi_{tag}"})


def _table(m, models, segs):
    out = []
    for seg, s in segs.items():
        g = m[s]
        for tag, lab in models.items():
            out.append({"구분": seg, "모델": lab, **metrics(g["y"], g[f"pred_{tag}"], g[f"lo_{tag}"], g[f"hi_{tag}"], g["lag7"])})
    return pd.DataFrame(out)


def select():
    rows = rows_with_features()
    print("외부 피처 빈칸 비율(2024~):", rows.loc[rows["origin"] >= "2024-01-01", EXO_ADD].isna().mean().round(3).to_dict())
    m = k3_2025()
    infos = []
    for name, (feats, params) in VARIANTS.items():
        val, test, info = run_variant(rows, feats, params)
        _std(test, name).to_csv(TMD / f"{name}_pred2026.csv", index=False, encoding="utf-8")
        m = m.merge(_std(val, name), on=["date", "hour", "H"])
        infos.append({"변형": name} | info)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}, flush=True)
    m = m[m["y"].notna()]
    act = load_smp_actual("육지")[["date", "hour", "smp"]]
    m = m.merge(act.assign(date=act["date"] + pd.Timedelta(days=7)).rename(columns={"smp": "lag7"}), on=["date", "hour"], how="left")
    m["offday"] = m["offday"].astype(bool)
    segs = hourly_segments(m) | {"D+1": (m["H"] == 1).values, "D+2~7": (m["H"] >= 2).values}
    t = _table(m, {"T0": "T0 팀원 그대로", "T1": "T1 +외부 정보", "T2": "T2 +외부 정보·우리 LGB", "k3": "K3 + A1"}, segs)
    t.round(4).to_csv(OUT1, index=False, encoding="utf-8-sig")
    pd.DataFrame(infos).round(4).to_csv(TAB / "WZ3_X9_tmx_settings.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 100)
    print(t[["구분", "모델", "표본수", "MAE", "RMSE", "편향(공표−예측)", "포함률", "평균 폭", "구간점수"]].round(3).to_string(index=False))


def test(name):
    p = pd.read_csv(TMD / f"{name}_pred2026.csv", parse_dates=["date"])
    k3 = pd.read_csv(W3 / "WZ3_predictions_hourly_K3.csv", parse_dates=["date"])[["date", "hour", "H", "pred", "lo", "hi"]]
    cur = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["base_date", "date"])
    cur = range_adjust.apply(cur[cur["y"].notna()].reset_index(drop=True), "육지")
    m = (cur.rename(columns={"pred": "pred_cur", "lo": "lo_cur", "hi": "hi_cur"})
         .merge(k3.rename(columns={"pred": "pred_k3", "lo": "lo_k3", "hi": "hi_k3"}), on=["date", "hour", "H"])
         .merge(p, on=["date", "hour", "H"]))
    m = m[m["date"].between(*TEST) & m["y"].notna()]
    act = load_smp_actual("육지")[["date", "hour", "smp"]]
    m = m.merge(act.assign(date=act["date"] + pd.Timedelta(days=7)).rename(columns={"smp": "lag7"}), on=["date", "hour"], how="left")
    m["offday"] = m["offday"].astype(bool)
    segs = hourly_segments(m) | {"D+1": (m["H"] == 1).values, "D+2~7": (m["H"] >= 2).values}
    t = _table(m, {"cur": "지금 모델", "k3": "K3", name: f"{name} 팀원 코드 + 외부 정보"}, segs)
    t.round(4).to_csv(OUT2, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(t[["구분", "모델", "표본수", "MAE", "RMSE", "포함률", "평균 폭", "구간점수"]].round(3).to_string(index=False))


if __name__ == "__main__":
    select() if sys.argv[1] == "select" else test(sys.argv[2])
