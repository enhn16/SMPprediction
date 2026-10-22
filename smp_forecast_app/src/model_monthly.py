"""한 달 전망 (D+7 ~ D+34, 4주) — 일별 모델 후보 비교 + 주별 비교 + 범위 + 급락 감지.

목표 4종(일평균·한낮·저녁·한낮최저) × 후보 모델. 매월 1일 재학습(대상일이 그 달 1일 이전인 행만 학습 = 실적 확정분).
선택 (2025, 2026 채점 전 고정)
  - 점예측: 목표별로 2025 일별 MAE 최소 후보
  - 주별: 일별 예측의 주 평균 vs 주별 전용 모델(LightGBM 비율 목표) → 2025 주별 MAE 비교
  - 범위: 오차 분위수형 (주차×휴무일, 고정 1년 / 매일 갱신 180·365일, 가산·배수형) → intervals.select 규칙
  - 급락: 한낮최저 예측 ≤ 문턱 → 급락 경보. 문턱 = 2024 예측에서 2024 급락 80%를 잡는 값
시험: 2026-01-01 ~ 실적 최신일 한 번 채점 (대상일 기준).
출력: outputs/tables/M1_point_selection_2025.csv, M2_weekly_compare_2025.csv, M3_interval_selection_2025.csv,
      M4_test2026.csv, M5_drop.csv, M_choice.json, data/processed/smp_forecast_app/M_predictions_daily.csv
실행: python -m smp_forecast_app.src.model_monthly
"""
import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from common.config import PROCESSED_DIR, ROOT, SEED
from smp_forecast_app.src.features_daily import FEATURES, STATS, load_daily
from smp_forecast_app.src.intervals import ALPHA, Q_HI, Q_LO, score, select

warnings.filterwarnings("ignore")
OUT_TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
OUT_DATA = PROCESSED_DIR / "smp_forecast_app"
LGB = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=60, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1)
WEEK = lambda h: (np.asarray(h) - 7) // 7 + 1   # 1주차 = 7~13일 뒤 … 4주차 = 28~34일 뒤


def months_of(a, b):
    return pd.date_range(a, b, freq="MS")


def _ridge_xy(tr, te, cols):
    X, Xt = tr[cols].copy(), te[cols].copy()
    med = X.median()
    X, Xt = X.fillna(med), Xt.fillna(med)
    for k in ("month", "dow", "daytype"):   # 범주는 원핫
        X = pd.concat([X, pd.get_dummies(tr[k], prefix=k, dtype=float)], axis=1)
        Xt = pd.concat([Xt, pd.get_dummies(te[k], prefix=k, dtype=float)], axis=1)
    Xt = Xt.reindex(columns=X.columns, fill_value=0)
    mu, sd = X.mean(), X.std().replace(0, 1)
    return (X - mu) / sd, (Xt - mu) / sd


def fit_predict(df, months, s, kind):
    """kind: lgb_raw, lgb_ratio, cat_ratio, ridge_ratio. 비율 목표 = y / (최근 28일 같은 지표 평균)."""
    y = df[f"y_{s}"]
    lvl = df[f"{s}_l28"].clip(lower=20)
    ratio = kind.endswith("ratio")
    yt = y / lvl if ratio else y
    pred = pd.Series(np.nan, index=df.index)
    for m in months:
        m0, m1 = m, m + pd.offsets.MonthBegin(1)
        tr = (df["date"] < m0) & y.notna() & yt.notna()
        te = (df["base_date"] >= m0) & (df["base_date"] < m1)
        if te.sum() == 0:
            continue
        a, b = df[tr], df[te]
        if kind.startswith("lgb"):
            p = lgb.LGBMRegressor(**LGB).fit(a[FEATURES], yt[tr]).predict(b[FEATURES])
        elif kind.startswith("cat"):
            p = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.06, loss_function="RMSE", random_seed=SEED,
                                  verbose=0, thread_count=-1).fit(a[FEATURES], yt[tr]).predict(b[FEATURES])
        else:
            X, Xt = _ridge_xy(a, b, FEATURES)
            p = Ridge(alpha=10.0).fit(X, yt[tr]).predict(Xt)
        pred[te] = p * (lvl[te].values if ratio else 1)
    return pred


def weekly_direct(df, months):
    """주별 전용 모델: (기준일, 주차) 행, 목표 = 주 평균 일평균 SMP / 최근 28일 평균. LightGBM."""
    df = df.assign(week=WEEK(df["H"]))
    agg = {c: "first" for c in FEATURES if c not in ("H", "dow", "doy_sin", "doy_cos", "offday", "daytype", "off_block_len",
                                                        "off_block_day", "is_myeongjeol", "is_labor_day", "prev_offday", "next_offday")}
    g = df.groupby(["base_date", "week"])
    w = g.agg({**agg, "offday": "sum", "is_myeongjeol": "sum", "doy_sin": "mean", "doy_cos": "mean", "date": "max",
               "y_mean": "mean"}).reset_index()
    w["n_days"] = g.size().values
    w["n_holiday_wd"] = g.apply(lambda x: ((x["daytype"] >= 2) & (x["dow"] < 5)).sum()).values
    feats = [c for c in w.columns if c not in ("base_date", "date", "y_mean")]
    w["end"] = w["date"]
    y, lvl = w["y_mean"], w["mean_l28"].clip(lower=20)
    pred = pd.Series(np.nan, index=w.index)
    for m in months:
        m0, m1 = m, m + pd.offsets.MonthBegin(1)
        tr = (w["end"] < m0) & y.notna() & (w["n_days"] == 7)
        te = (w["base_date"] >= m0) & (w["base_date"] < m1)
        if te.sum() == 0:
            continue
        mdl = lgb.LGBMRegressor(**dict(LGB, min_child_samples=30)).fit(w.loc[tr, feats], (y / lvl)[tr])
        pred[te] = mdl.predict(w.loc[te, feats]) * lvl[te].values
    return w.assign(pred_week=pred)


def interval(d, ev_mask, s, kind, window=None, form="add"):
    """오차 분위수 범위: 기준일 I의 범위에는 대상일 ≤ I인(실적 확정) 오차만. 칸 = 주차 × 휴무일."""
    d = d.assign(k=WEEK(d["H"]).astype(str) + "_" + d["offday"].astype(str))
    e = d[f"y_{s}"] - d[kind]
    d["err"] = e if form == "add" else e / d[kind].clip(lower=20)
    ev = d[ev_mask]
    lo, hi = pd.Series(np.nan, index=ev.index), pd.Series(np.nan, index=ev.index)
    hist = d[d["err"].notna()]

    def q(rows):
        g = rows.groupby("k")["err"]
        return g.quantile(Q_LO), g.quantile(Q_HI)

    def put(part, ql, qh):
        base = part[kind].clip(lower=20) if form == "mult" else 1.0
        lo[part.index] = (part[kind] + part["k"].map(ql) * base).clip(lower=0)
        hi[part.index] = part[kind] + part["k"].map(qh) * base

    if window is None:
        start = ev["base_date"].min()
        ql, qh = q(hist[(hist["date"] <= start) & (hist["date"] > start - pd.Timedelta(days=365))])
        put(ev, ql, qh)
    else:
        for i, part in ev.groupby("base_date"):
            ql, qh = q(hist[(hist["date"] <= i) & (hist["date"] > i - pd.Timedelta(days=window))])
            put(part, ql, qh)
    return lo.values, hi.values


INTERVALS = {"고정1년·가산": (None, "add"), "고정1년·배수": (None, "mult"), "갱신180일·가산": (180, "add"),
             "갱신180일·배수": (180, "mult"), "갱신365일·가산": (365, "add"), "갱신365일·배수": (365, "mult")}
KINDS = ["lgb_raw", "lgb_ratio", "cat_ratio", "ridge_ratio"]


def mae(y, p):
    ok = y.notna() & p.notna()
    return float((y[ok] - p[ok]).abs().mean())


def main():
    t0 = time.time()
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    df = load_daily()
    df = df[df["date"] >= "2022-03-01"].reset_index(drop=True)
    last_y = df.loc[df["y_mean"].notna(), "date"].max()
    months = months_of("2024-01-01", df["base_date"].max().strftime("%Y-%m-01"))
    print(f"행 {len(df):,}, 실적 최신 {last_y.date()}", flush=True)

    # 1) 후보 예측 (2024-01 ~ 최신 기준일)
    for s in STATS:
        df[f"p_{s}_same4"] = df[f"{s}_same4"]
        df[f"p_{s}_seas"] = df[f"{s}_same4"] * df["seas_smp"].clip(0.6, 1.6).fillna(1)
        for k in KINDS:
            df[f"p_{s}_{k}"] = fit_predict(df, months, s, k)
            print(f"[{s}] {k} 완료 ({time.time()-t0:.0f}s)", flush=True)
        df[f"p_{s}_ens"] = df[[f"p_{s}_lgb_ratio", f"p_{s}_cat_ratio"]].mean(axis=1)
        df[f"p_{s}_ens3"] = df[[f"p_{s}_lgb_ratio", f"p_{s}_cat_ratio", f"p_{s}_ridge_ratio"]].mean(axis=1)
    cands = ["same4", "seas"] + KINDS + ["ens", "ens3"]
    NAME = {"same4": "기준: 최근 같은 유형일 4일 평균", "seas": "기준: 같은 유형일 평균 × 작년 계절비",
            "lgb_raw": "LightGBM 원값", "lgb_ratio": "LightGBM 비율", "cat_ratio": "CatBoost 비율",
            "ridge_ratio": "Ridge 비율", "ens": "앙상블(LightGBM+CatBoost)", "ens3": "앙상블(LightGBM+CatBoost+Ridge)"}

    # 2) 점예측 선택 (2025 대상일)
    v = df[df["date"].dt.year == 2025]
    rows, chosen = [], {}
    for s in STATS:
        for c in cands:
            p = v[f"p_{s}_{c}"]
            r = {"목표": s, "후보": NAME[c], "코드": c, "일별 MAE": mae(v[f"y_{s}"], p)}
            for wk in range(1, 5):
                m = WEEK(v["H"]) == wk
                r[f"{wk}주차 MAE"] = mae(v.loc[m, f"y_{s}"], p[m])
            rows.append(r)
        sel = pd.DataFrame([r for r in rows if r["목표"] == s]).sort_values("일별 MAE")
        chosen[s] = sel.iloc[0]["코드"]
        print(f"[선택 2025] {s}: {NAME[chosen[s]]} MAE {sel.iloc[0]['일별 MAE']:.2f} (기준 same4 {sel[sel.코드=='same4']['일별 MAE'].iloc[0]:.2f})", flush=True)
    pd.DataFrame(rows).round(3).to_csv(OUT_TAB / "M1_point_selection_2025.csv", index=False, encoding="utf-8-sig")
    for s in STATS:
        df[f"pred_{s}"] = df[f"p_{s}_{chosen[s]}"]

    # 3) 주별 비교 (일평균, 2025): 일별 예측 주 평균 vs 주별 전용 모델
    df["week"] = WEEK(df["H"])
    wk_from_daily = df.groupby(["base_date", "week"]).agg(y=("y_mean", "mean"), p=("pred_mean", "mean"),
                                                          n=("y_mean", "count"), end=("date", "max")).reset_index()
    wd = weekly_direct(df, months)
    cmp = wk_from_daily.merge(wd[["base_date", "week", "pred_week"]], on=["base_date", "week"])
    cmp = cmp[cmp["n"] == 7]
    c25 = cmp[cmp["end"].dt.year == 2025]
    wrows = [{"주차": wk, "일별 예측의 주 평균 MAE": mae(g["y"], g["p"]), "주별 전용 모델 MAE": mae(g["y"], g["pred_week"]),
              "기준일 수": len(g)} for wk, g in c25.groupby("week")]
    wrows.append({"주차": "전체", "일별 예측의 주 평균 MAE": mae(c25["y"], c25["p"]), "주별 전용 모델 MAE": mae(c25["y"], c25["pred_week"]),
                  "기준일 수": len(c25)})
    wcmp = pd.DataFrame(wrows)
    wcmp.round(3).to_csv(OUT_TAB / "M2_weekly_compare_2025.csv", index=False, encoding="utf-8-sig")
    print(wcmp.round(2).to_string(index=False), flush=True)
    weekly_choice = "주별 전용 모델" if wrows[-1]["주별 전용 모델 MAE"] < wrows[-1]["일별 예측의 주 평균 MAE"] * 0.95 else "일별 예측의 주 평균"
    print(f"주별 값: {weekly_choice} (주별 전용이 5% 이상 나을 때만 채택)", flush=True)

    # 4) 범위 선택 (2025, 대상일 기준) — 목표별
    known = df[df["date"] <= last_y]
    e25 = (known["date"].dt.year == 2025).values
    irows, ichoice = [], {}
    for s in STATS:
        part = []
        for name, (wdw, form) in INTERVALS.items():
            lo, hi = interval(known, e25, s, f"pred_{s}", wdw, form)
            g = known[e25]
            r = {"목표": s, "방법": name, "구간": "전체", **score(g[f"y_{s}"], lo, hi, g[f"pred_{s}"])}
            irows.append(r); part.append(r)
        ichoice[s] = select(pd.DataFrame(part))
    pd.DataFrame(irows).round(4).to_csv(OUT_TAB / "M3_interval_selection_2025.csv", index=False, encoding="utf-8-sig")
    print("범위 선택:", ichoice, flush=True)

    # 5) 2026 시험 (한 번) + 저장용 범위 (2025·2026·미래)
    allmask = (df["date"] >= "2025-01-01").values
    for s in STATS:
        wdw, form = INTERVALS[ichoice[s]]
        lo, hi = interval(df, allmask, s, f"pred_{s}", wdw, form)
        df.loc[allmask, f"lo_{s}"], df.loc[allmask, f"hi_{s}"] = lo, hi
    te = df[(df["date"] >= "2026-01-01") & df["y_mean"].notna()]
    trows = []
    for s in STATS:
        for wk in ["전체", 1, 2, 3, 4]:
            g = te if wk == "전체" else te[te["week"] == wk]
            base_mae = mae(g[f"y_{s}"], g[f"{s}_same4"])
            r = {"목표": s, "주차": wk, "모델": NAME[chosen[s]], "범위": ichoice[s],
                 **score(g[f"y_{s}"], g[f"lo_{s}"], g[f"hi_{s}"], g[f"pred_{s}"]), "기준(same4) MAE": base_mae}
            r["기준 대비 개선"] = 1 - r["MAE"] / base_mae
            trows.append(r)
    test = pd.DataFrame(trows)
    test.round(4).to_csv(OUT_TAB / "M4_test2026.csv", index=False, encoding="utf-8-sig")
    # 주별 값 2026
    c26 = cmp[cmp["end"] >= "2026-01-01"]
    w26 = {"일별 예측의 주 평균 MAE": mae(c26["y"], c26["p"]), "주별 전용 모델 MAE": mae(c26["y"], c26["pred_week"]), "주 수": len(c26)}

    # 6) 한 달 앞 급락 감지 (한낮최저 < 30원) — 문턱은 2024 예측으로
    df["drop"] = (df["y_midmin"] < 30).astype(float).where(df["y_midmin"].notna())
    d24 = df[(df["date"].dt.year == 2024) & df["drop"].notna()]
    drows = []
    for wk in [1, 2, 3, 4]:
        a = d24[d24["week"] == wk]
        thr = np.quantile(a.loc[a["drop"] == 1, "pred_midmin"], 0.8) if a["drop"].sum() else np.nan
        for yr in (2025, 2026):
            g = df[(df["date"].dt.year == yr) & (df["week"] == wk) & df["drop"].notna()]
            # 기준일 × 대상일 행이라 같은 대상일이 7번(주차 안에서는 1번) 나오지 않도록 주차별로 집계
            al = g["pred_midmin"] <= thr
            y = g["drop"] == 1
            rule = g["offday"] == 1
            drows.append({"주차": wk, "연도": yr, "행(기준일×대상일)": len(g), "급락 행": int(y.sum()),
                          "문턱(원)": thr, "적중률": (al & y).sum() / max(y.sum(), 1), "헛경보": int((al & ~y).sum()),
                          "정밀도": (al & y).sum() / max(al.sum(), 1),
                          "ROC-AUC": roc_auc_score(y, -g["pred_midmin"]) if y.nunique() > 1 else np.nan,
                          "규칙(휴무일) 헛경보": int((rule & ~y).sum()), "규칙 적중률": (rule & y).sum() / max(y.sum(), 1)})
    drop = pd.DataFrame(drows)
    drop.round(4).to_csv(OUT_TAB / "M5_drop.csv", index=False, encoding="utf-8-sig")

    keep = ["base_date", "H", "date", "day_type", "offday"] + [f"{p}_{s}" for s in STATS for p in ("y", "pred", "lo", "hi")]
    out = df[df["date"] >= "2025-01-01"][keep].round(2)
    out.to_csv(OUT_DATA / "M_predictions_daily.csv", index=False, encoding="utf-8-sig")   # 2차가 덮어씀
    out.to_csv(OUT_DATA / "M1_predictions_daily.csv", index=False, encoding="utf-8-sig")  # 1차 보관
    info = {"점예측": {s: NAME[chosen[s]] for s in STATS}, "범위": ichoice, "주별 값": weekly_choice, "주별 2026": w26,
            "시험 기간": f"대상일 2026-01-01 ~ {last_y.date()}", "재학습": "매월 1일"}
    (OUT_TAB / "M_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(test[test["주차"] == "전체"][["목표", "모델", "범위", "포함률", "하한아래", "상한위", "평균폭", "MAE", "기준(same4) MAE", "기준 대비 개선"]].round(3).to_string(index=False))
    print(test[test["목표"] == "mean"][["주차", "포함률", "평균폭", "MAE", "기준 대비 개선"]].round(3).to_string(index=False))
    print(json.dumps(w26, ensure_ascii=False, default=float))
    print(drop.round(3).to_string(index=False))
    print(f"완료 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
