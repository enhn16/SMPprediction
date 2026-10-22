"""한 달 전망 2차 — 단순 기준(최근 같은 유형일 4일 평균)을 출발점으로 두고, 모델은 기준 대비 비율만 학습.

1차(model_monthly.py) 결과: 2025 선택에서 단순 기준이 기계학습 후보를 모두 이겼고, 2026 포함률 69%로 미달.
→ 2차 후보 (2025로 선택, 2026 성적은 '2차 시험'으로 표기)
  - 기준비율 LightGBM / CatBoost / 평균: 목표 = 실제 ÷ 기준, 예측 = 기준 × 예측 비율
  - 섞기: 기준 + λ × (기준비율 평균 − 기준), λ ∈ {0.25, 0.5, 0.75}
  - 1차 후보 중 2025 최우수(단순 기준, 저녁은 3종 앙상블)도 함께 비교
  범위: 1차 6종 + 갱신 90일(가산·배수)
선택 규칙: 점예측은 2025 일별 MAE 최소, 범위는 intervals.select.
출력: outputs/tables/M2v_point_selection_2025.csv, M2v_interval_selection_2025.csv, M2v_test2026.csv, M2v_choice.json,
      data/processed/smp_forecast_app/M_predictions_daily.csv (2차로 덮어씀, 1차는 M1_predictions_daily.csv로 보관)
실행: python -m smp_forecast_app.src.model_monthly_v2
"""
import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from common.config import PROCESSED_DIR, ROOT, SEED
from smp_forecast_app.src.features_daily import FEATURES, STATS, load_daily
from smp_forecast_app.src.intervals import score, select
from smp_forecast_app.src.model_monthly import INTERVALS, LGB, WEEK, interval, mae, months_of

warnings.filterwarnings("ignore")
OUT_TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
OUT_DATA = PROCESSED_DIR / "smp_forecast_app"
INTERVALS2 = {**INTERVALS, "갱신90일·가산": (90, "add"), "갱신90일·배수": (90, "mult")}
FEATS = FEATURES + ["base_ratio_l28"]


def fit_ratio(df, months, s, kind):
    base = df[f"{s}_same4"].clip(lower=10)
    yt = df[f"y_{s}"] / base
    pred = pd.Series(np.nan, index=df.index)
    for m in months:
        m0, m1 = m, m + pd.offsets.MonthBegin(1)
        tr = (df["date"] < m0) & yt.notna() & np.isfinite(yt)
        te = (df["base_date"] >= m0) & (df["base_date"] < m1)
        if te.sum() == 0:
            continue
        a, b = df.loc[tr, FEATS], df.loc[te, FEATS]
        if kind == "lgb":
            p = lgb.LGBMRegressor(**dict(LGB, objective="l1")).fit(a, yt[tr].clip(0, 3)).predict(b)
        else:
            p = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.06, loss_function="MAE", random_seed=SEED,
                                  verbose=0, thread_count=-1).fit(a, yt[tr].clip(0, 3)).predict(b)
        pred[te] = p * base[te].values
    return pred


def main():
    t0 = time.time()
    df = load_daily()
    df = df[df["date"] >= "2022-03-01"].reset_index(drop=True)
    last_y = df.loc[df["y_mean"].notna(), "date"].max()
    months = months_of("2024-01-01", df["base_date"].max().strftime("%Y-%m-01"))
    first = json.loads((OUT_TAB / "M_choice.json").read_text(encoding="utf-8"))
    old = pd.read_csv(OUT_DATA / "M1_predictions_daily.csv", parse_dates=["base_date", "date"])
    df = df.merge(old[["base_date", "H"] + [f"pred_{s}" for s in STATS]].rename(columns={f"pred_{s}": f"p1_{s}" for s in STATS}),
                  on=["base_date", "H"], how="left")

    rows, chosen = [], {}
    v = df["date"].dt.year == 2025
    for s in STATS:
        df["base_ratio_l28"] = df[f"{s}_same4"] / df[f"{s}_l28"].clip(lower=10)
        df[f"q_{s}_base"] = df[f"{s}_same4"]
        df[f"q_{s}_first"] = df[f"p1_{s}"]
        df[f"q_{s}_lgb"] = fit_ratio(df, months, s, "lgb")
        df[f"q_{s}_cat"] = fit_ratio(df, months, s, "cat")
        df[f"q_{s}_avg"] = df[[f"q_{s}_lgb", f"q_{s}_cat"]].mean(axis=1)
        for lam in (0.25, 0.5, 0.75):
            df[f"q_{s}_mix{lam}"] = df[f"q_{s}_base"] + lam * (df[f"q_{s}_avg"] - df[f"q_{s}_base"])
        print(f"[{s}] ({time.time()-t0:.0f}s)", flush=True)
        names = {"base": "기준: 최근 같은 유형일 4일 평균", "first": f"1차 선택 ({first['점예측'][s]})",
                 "lgb": "기준비율 LightGBM", "cat": "기준비율 CatBoost", "avg": "기준비율 평균(LightGBM+CatBoost)",
                 **{f"mix{l}": f"기준 + {l} × (기준비율 평균 − 기준)" for l in (0.25, 0.5, 0.75)}}
        part = []
        for c, nm in names.items():
            p = df.loc[v, f"q_{s}_{c}"]
            r = {"목표": s, "후보": nm, "코드": c, "일별 MAE": mae(df.loc[v, f"y_{s}"], p),
                 **{f"{w}주차 MAE": mae(df.loc[v & (WEEK(df["H"]) == w), f"y_{s}"], df.loc[v & (WEEK(df["H"]) == w), f"q_{s}_{c}"]) for w in range(1, 5)}}
            part.append(r)
        part = pd.DataFrame(part).sort_values("일별 MAE")
        chosen[s] = (part.iloc[0]["코드"], part.iloc[0]["후보"])
        rows.append(part)
        print(part[["후보", "일별 MAE", "1주차 MAE", "4주차 MAE"]].round(2).to_string(index=False), flush=True)
        df[f"pred_{s}"] = df[f"q_{s}_{chosen[s][0]}"]
    pd.concat(rows).round(3).to_csv(OUT_TAB / "M2v_point_selection_2025.csv", index=False, encoding="utf-8-sig")

    known = df[df["date"] <= last_y]
    e25 = (known["date"].dt.year == 2025).values
    irows, ich = [], {}
    for s in STATS:
        part = []
        for name, (w, form) in INTERVALS2.items():
            lo, hi = interval(known, e25, s, f"pred_{s}", w, form)
            r = {"목표": s, "방법": name, "구간": "전체", **score(known.loc[e25, f"y_{s}"], lo, hi, known.loc[e25, f"pred_{s}"])}
            part.append(r); irows.append(r)
        ich[s] = select(pd.DataFrame(part))
    pd.DataFrame(irows).round(4).to_csv(OUT_TAB / "M2v_interval_selection_2025.csv", index=False, encoding="utf-8-sig")
    print("범위 선택:", ich, flush=True)

    allm = (df["date"] >= "2025-01-01").values
    for s in STATS:
        w, form = INTERVALS2[ich[s]]
        lo, hi = interval(df, allm, s, f"pred_{s}", w, form)
        df.loc[allm, f"lo_{s}"], df.loc[allm, f"hi_{s}"] = lo, hi
    df["week"] = WEEK(df["H"])
    te = df[(df["date"] >= "2026-01-01") & df["y_mean"].notna()]
    trows = []
    for s in STATS:
        for wk in ["전체", 1, 2, 3, 4]:
            g = te if wk == "전체" else te[te["week"] == wk]
            b = mae(g[f"y_{s}"], g[f"{s}_same4"])
            r = {"목표": s, "주차": wk, "모델": chosen[s][1], "범위": ich[s],
                 **score(g[f"y_{s}"], g[f"lo_{s}"], g[f"hi_{s}"], g[f"pred_{s}"]), "기준(same4) MAE": b}
            r["기준 대비 개선"] = 1 - r["MAE"] / b
            trows.append(r)
    test = pd.DataFrame(trows)
    test.round(4).to_csv(OUT_TAB / "M2v_test2026.csv", index=False, encoding="utf-8-sig")
    keep = ["base_date", "H", "date", "day_type", "offday"] + [f"{p}_{s}" for s in STATS for p in ("y", "pred", "lo", "hi")]
    df[df["date"] >= "2025-01-01"][keep].round(2).to_csv(OUT_DATA / "M_predictions_daily.csv", index=False, encoding="utf-8-sig")
    info = {**first, "점예측": {s: chosen[s][1] for s in STATS}, "범위": ich, "차수": "2차 (1차 채점 후 2025로 다시 선택)",
            "1차 점예측": first["점예측"], "1차 범위": first["범위"]}
    (OUT_TAB / "M2v_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(test[test["주차"] == "전체"][["목표", "모델", "범위", "포함률", "하한아래", "상한위", "평균폭", "MAE", "기준(same4) MAE", "기준 대비 개선"]].round(3).to_string(index=False))
    print(test[test["목표"] == "mean"][["주차", "포함률", "평균폭", "MAE", "기준 대비 개선"]].round(3).to_string(index=False))
    print(f"완료 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
