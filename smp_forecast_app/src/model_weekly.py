"""갈래 B 확장 — 주간(D+1~D+6) 시간별 예측 + 80% 범위 (판단 시각: 기준일 I의 15시, 대상일 D = I+H).

- 점예측 구성은 H=1(model.py)에서 2025 성적으로 고른 것(B_choice.json)을 그대로 쓴다 (지평별 추가 선택 없음).
- 피처는 features.build(lead=H): 판단일 I까지 확정된 값만 (SMP·수요예측은 I일분까지, 날씨는 H≤3 단기 육상예보·H≥4 중기예보).
- 재학습: 매월 1일, 단 실적은 m0-H일까지만. 범위 오차 이력도 D-H일까지만 (intervals gap=H).
- 범위 방법: 지평마다 2025 성적으로 선택(intervals.select), 2026-01-01 ~ 한 번 채점.
- H=1은 model.py 결과(B_predictions_hourly.csv)를 그대로 합친다.
출력: outputs/tables/W1_selection_2025.csv, W2_test2026.csv, W3_operator_weekly_ref.csv, W_choice.json,
      data/processed/smp_forecast_app/W_predictions_hourly.csv [base_date, H, date, hour, y, pred, lo, hi]
실행: python -m smp_forecast_app.src.model_weekly
"""
import json
import time

import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from smp_forecast_app.src.features import FEATURE_SETS, load_features
from smp_forecast_app.src.intervals import ALPHA, add_keys, report, select
from smp_forecast_app.src.model import fit_predict, interval_methods, months_of, point_metrics

OUT_TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
OUT_DATA = PROCESSED_DIR / "smp_forecast_app"
EDU = ROOT / "data" / "smp_edu_data"


def run_lead(H, cfg):
    t0 = time.time()
    df = load_features(H)
    cols = FEATURE_SETS[cfg["피처"]]
    start, target = cfg["학습시작"] + "-01-01", cfg["목표"]
    mall = months_of("2024-01-01", df["date"].max().strftime("%Y-%m-01"))
    df["pred"] = fit_predict(df, mall, start, cols, target, lead=H)
    df["qlo"] = fit_predict(df, mall, start, cols, target, "quantile", ALPHA / 2, lead=H)
    df["qhi"] = fit_predict(df, mall, start, cols, target, "quantile", 1 - ALPHA / 2, lead=H)
    d = df[df["date"] >= "2024-01-01"].copy()
    d["offday"] = d["offday"].astype(bool)
    known = add_keys(d[d["y"].notna()].reset_index(drop=True))
    ims = interval_methods(gap=H)

    e25 = (known["date"].dt.year == 2025).values
    sel_rows = []
    for name, fn in ims.items():
        lo, hi = fn(known, e25)
        sel_rows += report(known[e25].assign(lo=lo, hi=hi), name, "선택 2025")
    sel = pd.DataFrame(sel_rows).assign(H=H)
    chosen = select(sel)

    e26 = (known["date"] >= "2026-01-01").values
    lo, hi = ims[chosen](known, e26)
    g = known[e26].assign(lo=lo, hi=hi)
    te = pd.DataFrame(report(g, chosen, "시험 2026")).assign(H=H)
    pt = point_metrics(g["y"], g["pred"], g["s_d1"])
    pt25 = point_metrics(known.loc[e25, "y"], known.loc[e25, "pred"], known.loc[e25, "s_d1"])

    lo25, hi25 = ims[chosen](known, e25)
    out = pd.concat([known[e25].assign(lo=lo25, hi=hi25), g])
    fut = d[d["y"].isna()].copy()
    if len(fut):
        allk = pd.concat([known, add_keys(fut.assign(y=np.nan))]).reset_index(drop=True)
        mf = allk["y"].isna().values
        lo, hi = ims[chosen](allk.assign(err=allk["err"].fillna(0), rerr=allk["rerr"].fillna(0)), mf)
        out = pd.concat([out, allk[mf].assign(lo=lo, hi=hi)])
    out = out.assign(H=H, base_date=out["date"] - pd.Timedelta(days=H))
    r = te[te["구간"] == "전체"].iloc[0]
    print(f"[H={H}] 범위 {chosen}: 2026 포함률 {r['포함률']:.1%}, 폭 {r['평균폭']:.1f}원, 점예측 MAE {pt['MAE']:.2f} "
          f"(rMAE {pt['rMAE']:.3f}) — {time.time()-t0:.0f}s", flush=True)
    return sel, te, out, {"범위 방법": chosen, "2025 점예측": pt25, "2026 점예측": pt}


def operator_weekly() -> pd.DataFrame:
    """운영 주간 예측(15시) 전 지평 기록 (읽기 전용). 같은 기준일·지평이 여러 번이면 마지막."""
    import glob
    rows = [pd.read_csv(f, parse_dates=["기준일", "대상일"]) for f in sorted(glob.glob(str(EDU / "outputs/predict/SMP주간예측_*.csv")))]
    w = pd.concat(rows).drop_duplicates(["기준일", "지평", "대상일", "거래시간"], keep="last")
    w["H"] = w["지평"].str.replace("D+", "", regex=False).astype(int)
    return w.rename(columns={"기준일": "base_date", "대상일": "date", "거래시간": "hour", "예측": "pred_op"})[
        ["base_date", "H", "date", "hour", "pred_op"]]


def main():
    cfg = json.loads((OUT_TAB / "B_choice.json").read_text(encoding="utf-8"))["점예측"]
    b1 = pd.read_csv(OUT_DATA / "B_predictions_hourly.csv", parse_dates=["date"])
    b1 = b1.assign(H=1, base_date=b1["date"] - pd.Timedelta(days=1))
    sels, tes, outs, info = [], [], [b1], {}
    t1 = pd.read_csv(OUT_TAB / "B3_test2026.csv")
    tes.append(t1[t1["선택"]].drop(columns=["선택"]).assign(H=1))
    bc = json.loads((OUT_TAB / "B_choice.json").read_text(encoding="utf-8"))
    info[1] = {"범위 방법": bc["범위 방법"], "2026 점예측": bc["점예측"]["2026 시험"]}
    for H in range(2, 7):
        sel, te, out, inf = run_lead(H, cfg)
        sels.append(sel); tes.append(te); outs.append(out)
        info[H] = {"범위 방법": inf["범위 방법"], "2026 점예측": {k: round(float(v), 4) for k, v in inf["2026 점예측"].items()}}
    pd.concat(sels).round(4).to_csv(OUT_TAB / "W1_selection_2025.csv", index=False, encoding="utf-8-sig")
    te = pd.concat(tes)
    te.round(4).to_csv(OUT_TAB / "W2_test2026.csv", index=False, encoding="utf-8-sig")
    keep = ["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "qlo", "qhi", "offday", "day_type"]
    allp = pd.concat(outs)[keep].sort_values(["base_date", "H", "hour"])
    allp.round(2).to_csv(OUT_DATA / "W_predictions_hourly.csv", index=False, encoding="utf-8-sig")

    # 참고: 운영 주간 예측과 같은 (기준일, 지평, 시간) 비교 — 점예측 MAE, 같은 범위 폭을 붙였을 때 포함률
    op = operator_weekly().merge(allp.dropna(subset=["y"]), on=["base_date", "H", "date", "hour"])
    ref = []
    for H, g in op.groupby("H"):
        own_cov = ((g["y"] >= g["lo"]) & (g["y"] <= g["hi"])).mean()
        lo_op = (g["pred_op"] + g["lo"] - g["pred"]).clip(lower=0)
        hi_op = g["pred_op"] + g["hi"] - g["pred"]
        ref.append({"H": H, "시간수": len(g), "기준일 수": g["base_date"].nunique(),
                    "자체 MAE": (g["y"] - g["pred"]).abs().mean(), "운영 MAE": (g["y"] - g["pred_op"]).abs().mean(),
                    "자체 포함률": own_cov, "운영+같은 폭 포함률": ((g["y"] >= lo_op) & (g["y"] <= hi_op)).mean(),
                    "평균폭": (g["hi"] - g["lo"]).mean(), "기간": f"{g.date.min().date()}~{g.date.max().date()}"})
    ref = pd.DataFrame(ref)
    ref.round(4).to_csv(OUT_TAB / "W3_operator_weekly_ref.csv", index=False, encoding="utf-8-sig")
    (OUT_TAB / "W_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(te[te["구간"].isin(["전체", "휴무일 한낮(11~15시)"])][["H", "방법", "구간", "시간수", "포함률", "하한아래", "상한위", "평균폭", "MAE"]].round(3).to_string(index=False))
    print(ref.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
