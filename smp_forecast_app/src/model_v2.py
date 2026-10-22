"""2차 실험 — 시간별 D+1 (전날 15시): 점예측 모델 다양화 + 과제1 급락 판단 결합.

1차(model.py)는 2026을 이미 한 번 채점했으므로, 여기서 고른 구성의 2026 성적은 '2차 시험'으로 표기한다.
선택은 1차와 같이 2025 성적만으로 한다 (규칙은 아래에 고정).

후보
  점예측: LightGBM(1차와 같은 구성) / CatBoost / 평균 앙상블
  급락 결합 (과제1 drop_prob_육지.csv: 2025는 2024 학습 예측, 2026은 2024~2025 학습 예측 — 모두 표본 밖):
    - 범위만: 경보일 한낮(11~15시) 하한을 과제1 예상 한낮최저까지(주의) / 0원까지(강한 경보) 내림 — 예측값 유지(투트랙)
    - 예측값까지: 경보일 한낮 예측을 과제1 예상 한낮최저 쪽으로 w만큼 당김 (범위도 같이 이동), w ∈ {0.25, 0.5, 0.75, 1}
  범위: 1차 선택 방식(보정 분위수 회귀·고정1년·시간대묶음×휴무일) + 오차 분위수 2종
선택 규칙 (2025)
  1) 점예측: 전체 MAE 최소. 급락 결합(예측값)은 전체 MAE가 결합 전보다 나쁘지 않을 때만 (학생질의 Q1-4 조건)
  2) 범위: 포함률 75~85% 중 구간점수 최소 (intervals.select)
출력: outputs/tables/V1_selection_2025.csv, V2_test2026.csv, V_choice.json, data/processed/smp_forecast_app/V_predictions_hourly.csv
실행: python -m smp_forecast_app.src.model_v2
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from common.config import PROCESSED_DIR, ROOT, SEED
from smp_forecast_app.src.features import FEATURE_SETS, load_features
from smp_forecast_app.src.intervals import ALPHA, add_keys, methods, report, select
from smp_forecast_app.src.model import cqr, fit_predict, months_of, point_metrics

warnings.filterwarnings("ignore")
OUT_TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
OUT_DATA = PROCESSED_DIR / "smp_forecast_app"
DROP = ROOT / "drop_classifier" / "outputs" / "tables" / "drop_prob_육지.csv"
MID = list(range(11, 16))


def cat_predict(df, months, start, cols):
    pred = pd.Series(np.nan, index=df.index)
    for m in months:
        m0, m1 = pd.Timestamp(m), pd.Timestamp(m) + pd.offsets.MonthBegin(1)
        tr = (df["date"] >= start) & (df["date"] < m0) & df["y"].notna()
        te = (df["date"] >= m0) & (df["date"] < m1)
        if te.sum() == 0:
            continue
        mdl = CatBoostRegressor(iterations=800, depth=7, learning_rate=0.05, loss_function="RMSE", random_seed=SEED,
                                verbose=0, thread_count=-1).fit(df.loc[tr, cols], df.loc[tr, "y"])
        pred[te] = mdl.predict(df.loc[te, cols])
    return pred


def bands(known, mask):
    """범위 후보 3종 → {이름: (lo, hi)}"""
    m = methods()
    return {"보정분위수회귀·고정1년·시간대묶음×휴무일": cqr(known, mask, key="k_band_off"),
            "오차분위수·고정·시간대": m["고정·시간대"](known, mask),
            "오차분위수·갱신·시간대묶음×휴무일·56일": m["갱신·시간대묶음×휴무일·56일"](known, mask)}


def main():
    t0 = time.time()
    cfg = json.loads((OUT_TAB / "B_choice.json").read_text(encoding="utf-8"))["점예측"]
    df = load_features(1)
    cols = FEATURE_SETS[cfg["피처"]]
    start = cfg["학습시작"] + "-01-01"
    mall = months_of("2024-01-01", df["date"].max().strftime("%Y-%m-01"))
    df["p_lgb"] = fit_predict(df, mall, start, cols, cfg["목표"])
    print(f"LightGBM ({time.time()-t0:.0f}s)", flush=True)
    df["p_cat"] = cat_predict(df, mall, start, cols)
    print(f"CatBoost ({time.time()-t0:.0f}s)", flush=True)
    df["p_ens"] = df[["p_lgb", "p_cat"]].mean(axis=1)
    df["qlo"] = fit_predict(df, mall, start, cols, cfg["목표"], "quantile", ALPHA / 2)
    df["qhi"] = fit_predict(df, mall, start, cols, cfg["목표"], "quantile", 1 - ALPHA / 2)
    print(f"분위수 ({time.time()-t0:.0f}s)", flush=True)

    dp = pd.read_csv(DROP, parse_dates=["date"]).rename(columns={"예측_한낮최저SMP": "t1_min", "등급": "t1_grade"})
    d = df[df["date"] >= "2024-01-01"].merge(dp[["date", "t1_min", "t1_grade"]], on="date", how="left")
    d["t1_grade"] = d["t1_grade"].fillna("없음")
    d["offday"] = d["offday"].astype(bool)
    d["is_mid"] = d["hour"].isin(MID)
    alarm = d["is_mid"] & d["t1_grade"].isin(["주의", "강한 경보"])

    # 점예측 후보: 모델 3종 × 급락 결합(예측값) w
    pts = {}
    for mname, col in (("LightGBM", "p_lgb"), ("CatBoost", "p_cat"), ("앙상블", "p_ens")):
        pts[(mname, 0.0)] = d[col]
        for w in (0.25, 0.5, 0.75, 1.0):
            target = np.minimum(d[col], d["t1_min"])
            pts[(mname, w)] = d[col].where(~alarm, (1 - w) * d[col] + w * target)

    rows = []
    sel_masks = {"2025": d["date"].dt.year == 2025}
    for (mname, w), p in pts.items():
        g = d[sel_masks["2025"] & d["y"].notna()]
        pm = point_metrics(g["y"], p[g.index], g["s_d1"])
        dh = g["y"] < 30
        rows.append({"점예측": mname, "급락결합 w": w, "MAE": pm["MAE"], "rMAE": pm["rMAE"],
                     "휴무일 한낮 MAE": (g["y"] - p[g.index]).abs()[g["offday"] & g["is_mid"]].mean(),
                     "급락 시간 MAE": (g["y"] - p[g.index]).abs()[dh].mean()})
    psel = pd.DataFrame(rows)
    base_best = psel[psel["급락결합 w"] == 0].sort_values("MAE").iloc[0]
    ok = psel[(psel["점예측"] == base_best["점예측"]) & (psel["MAE"] <= base_best["MAE"] + 1e-9)]
    # 결합 전보다 전체 MAE가 나쁘지 않은 것 중 급락 시간 MAE 최소
    pick = ok.sort_values(["급락 시간 MAE", "MAE"]).iloc[0]
    print(psel.round(3).to_string(index=False), flush=True)
    print(f"점예측 선택: {pick['점예측']}, 급락결합 w={pick['급락결합 w']}", flush=True)
    # 범위는 결합 전 예측값(base)으로 계산한 뒤, 예측값을 당긴 만큼 이동한다
    COL = {"LightGBM": "p_lgb", "CatBoost": "p_cat", "앙상블": "p_ens"}[pick["점예측"]]
    d["pred"] = d[COL]
    d["pred_base"] = d[COL]   # 운영 범위 계산용 (결합 전)
    d["pred_final"] = pts[(pick["점예측"], pick["급락결합 w"])]
    d["shift"] = d["pred_final"] - d["pred"]
    d["alarm"] = alarm

    def candidates(k, mask):
        out = {}
        g = k[mask]
        sh, al = g["shift"].values, g["alarm"].values
        for bname, (lo, hi) in bands(k, mask).items():
            lo, hi = lo + sh, hi + sh
            out[bname] = (np.clip(lo, 0, None), hi)
            # 범위만 결합: 경보일 한낮 하한 내리기 (강한 경보 0원, 주의 과제1 예상 한낮최저)
            lo2 = np.where(al & (g["t1_grade"] == "강한 경보"), 0.0,
                           np.where(al, np.minimum(lo, g["t1_min"].fillna(np.inf)), lo))
            out[bname + " + 급락 하한"] = (np.clip(lo2, 0, None), hi)
        return out

    def scored(k, mask, lo, hi):
        return k[mask].assign(lo=lo, hi=hi, pred=k.loc[mask, "pred_final"])

    known = add_keys(d[d["y"].notna()].reset_index(drop=True))
    e25 = (known["date"].dt.year == 2025).values
    e26 = (known["date"] >= "2026-01-01").values

    srows = []
    c25 = candidates(known, e25)
    for name, (lo, hi) in c25.items():
        srows += report(scored(known, e25, lo, hi), name, "선택 2025")
    ssel = pd.DataFrame(srows)
    chosen = select(ssel)
    pd.concat([psel.assign(구분="점예측"), ssel.assign(구분="범위")]).round(4).to_csv(
        OUT_TAB / "V1_selection_2025.csv", index=False, encoding="utf-8-sig")
    print(ssel[ssel["구간"].isin(["전체", "급락 시간(30원 미만)"])][["방법", "구간", "포함률", "평균폭", "구간점수"]].round(3).to_string(index=False))
    print(f"범위 선택: {chosen}", flush=True)

    # 2차 시험 (2026)
    lo, hi = candidates(known, e26)[chosen]
    g = scored(known, e26, lo, hi)
    te = pd.DataFrame(report(g, chosen, "2차 시험 2026"))
    pt26 = point_metrics(g["y"], g["pred"], g["s_d1"])
    pt26_nojoin = point_metrics(g["y"], known.loc[e26, "pred"], g["s_d1"])
    te.round(4).to_csv(OUT_TAB / "V2_test2026.csv", index=False, encoding="utf-8-sig")

    out = pd.concat([scored(known, e25, *c25[chosen]), g])
    fut = d[d["y"].isna()]
    if len(fut):   # 아직 실적 없는 날: 직전까지의 이력으로
        allk = add_keys(pd.concat([d[d["y"].notna()], fut]).reset_index(drop=True))
        allk = allk.assign(err=allk["err"].fillna(0), rerr=allk["rerr"].fillna(0))
        mf = allk["y"].isna().values
        lo, hi = candidates(allk, mf)[chosen]
        out = pd.concat([out, scored(allk, mf, lo, hi)])
    keep = ["date", "hour", "y", "pred", "lo", "hi", "pred_base", "qlo", "qhi", "s_d1", "offday", "day_type", "t1_grade"]
    out[keep].round(2).to_csv(OUT_DATA / "V_predictions_hourly.csv", index=False, encoding="utf-8-sig")
    info = {"점예측": pick["점예측"], "급락결합 w": float(pick["급락결합 w"]), "범위 방법": chosen,
            "2차 시험 2026 점예측": {k: round(float(v), 4) for k, v in pt26.items()},
            "결합 전 2026 점예측": {k: round(float(v), 4) for k, v in pt26_nojoin.items()},
            "시험 기간": f"2026-01-01 ~ {g.date.max().date()}", "표기": "2026은 1차 채점 후 다시 본 2차 시험"}
    (OUT_TAB / "V_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(te[["방법", "구간", "시간수", "포함률", "하한아래", "상한위", "평균폭", "구간점수", "MAE"]].round(3).to_string(index=False))
    print(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"완료 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
