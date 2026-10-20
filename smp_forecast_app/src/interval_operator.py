"""갈래 A — 담당자 시간별 모델 예측값 유지 + 오차 분위수 80% 범위.

예측값 (data/smp_edu_data, 읽기 전용)
  - OOF: data/processed/oof_predictions_smp_land.csv (2023-02-01 ~ 2025-12-31, 5개 연속 구간 교차검증)
  - 2026 홀드아웃: data/processed/model_predictions_smp_land.csv (2025년까지 학습, 2026-01-01 ~)
  ※ 담당자 시간별 모델은 D일 하루전 수요예측을 쓰는 23:30 산출 기준 → 15시 판단 예측이 아님 (참고 표기).
실적: data/raw SMP 엑셀 (공식 확정값).

절차 (평가 공정성)
  1) 선택: 2024-12-31까지의 오차로 만든 범위를 2025년에 채점 → intervals.select 규칙으로 한 방법 고정
  2) 시험: 2025-12-31까지의 오차로 만든 범위(갱신형은 매일 직전 오차 추가)를 2026-01-01 ~ 에 한 번 채점
  3) 참고: 운영 공표 예측 기록(predictions_smp_land.csv, 2026-07-30 ~)에 같은 방법을 적용해 채점
출력: outputs/tables/A1_selection_2025.csv, A2_test2026.csv, A2_test2026_by_hour.csv, A3_operational_ref.csv,
      data/processed/smp_forecast_app/A_intervals_hourly.csv
실행: python -m smp_forecast_app.src.interval_operator
"""
import json

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from common.loaders import load_smp_excel
from smp_forecast_app.src.intervals import add_keys, methods, report, score, segments, select

EDU = ROOT / "data" / "smp_edu_data"
OUT_TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
OUT_DATA = PROCESSED_DIR / "smp_forecast_app"


def actual_land() -> pd.DataFrame:
    return load_smp_excel("육지").rename(columns={"smp": "y"})


def offday_table(start, end) -> pd.DataFrame:
    cal = build_calendar(start, end)
    return cal[["date", "is_offday", "day_type"]].rename(columns={"is_offday": "offday"})


def load_operator_model() -> pd.DataFrame:
    oof = pd.read_csv(EDU / "data/processed/oof_predictions_smp_land.csv", parse_dates=["date"])
    ho = pd.read_csv(EDU / "data/processed/model_predictions_smp_land.csv", parse_dates=["date"])
    p = pd.concat([oof[["date", "hour_n", "pred"]], ho[["date", "hour_n", "pred"]]])
    p = p.rename(columns={"hour_n": "hour"}).drop_duplicates(["date", "hour"], keep="last")
    d = p.merge(actual_land(), on=["date", "hour"], how="inner").dropna(subset=["y", "pred"])
    d = d.merge(offday_table(d.date.min(), d.date.max()), on="date", how="left")
    return add_keys(d.sort_values(["date", "hour"]).reset_index(drop=True))


def run_period(d, ms, eval_mask, period):
    rows, bands = [], {}
    for name, fn in ms.items():
        lo, hi = fn(d, eval_mask)
        g = d[eval_mask].assign(lo=lo, hi=hi)
        rows += report(g, name, period)
        bands[name] = g
        r = rows[-len(segments(g))]
        print(f"[{period}] {name}: 포함률 {r['포함률']:.1%}, 폭 {r['평균폭']:.1f}원, 구간점수 {r['구간점수']:.1f}", flush=True)
    return pd.DataFrame(rows), bands


def main():
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    d = load_operator_model()
    print(f"담당자 모델 예측: {d.date.min().date()} ~ {d.date.max().date()}, {len(d):,}시간", flush=True)
    ms = methods()

    # 1) 선택 (2025)
    sel, _ = run_period(d, ms, (d["date"].dt.year == 2025).values, "선택 2025")
    sel.round(4).to_csv(OUT_TAB / "A1_selection_2025.csv", index=False, encoding="utf-8-sig")
    best = select(sel)
    print(f"\n선택된 방법: {best}\n", flush=True)

    # 2) 2026 시험 — 선택 방법 + 비교용(가장 단순한 1단계 방식 '고정·시간대') 함께 한 번 채점
    test_mask = (d["date"] >= "2026-01-01").values
    show = {k: ms[k] for k in dict.fromkeys([best, "고정·시간대", "고정·시간대×휴무일"])}
    te, bands = run_period(d, show, test_mask, "시험 2026")
    te["선택"] = te["방법"].eq(best)
    te.round(4).to_csv(OUT_TAB / "A2_test2026.csv", index=False, encoding="utf-8-sig")
    g = bands[best]
    byh = pd.DataFrame([{"시간": h, **score(x["y"], x["lo"], x["hi"], x["pred"])} for h, x in g.groupby("hour")])
    byh.round(4).to_csv(OUT_TAB / "A2_test2026_by_hour.csv", index=False, encoding="utf-8-sig")
    bym = pd.DataFrame([{"월": str(m), **score(x["y"], x["lo"], x["hi"], x["pred"])}
                        for m, x in g.groupby(g["date"].dt.to_period("M"))])
    bym.round(4).to_csv(OUT_TAB / "A2_test2026_by_month.csv", index=False, encoding="utf-8-sig")

    # 범위 데이터 저장 (2025 선택 구간 + 2026 시험 구간, 선택 방법)
    lo25, hi25 = ms[best](d, (d["date"].dt.year == 2025).values)
    a25 = d[d["date"].dt.year == 2025].assign(lo=lo25, hi=hi25)
    allb = pd.concat([a25, g])[["date", "hour", "y", "pred", "lo", "hi", "offday", "day_type"]]
    allb.round(2).to_csv(OUT_DATA / "A_intervals_hourly.csv", index=False, encoding="utf-8-sig")

    # 3) 참고: 운영 공표 예측 기록 (오차 이력은 담당자 모델 OOF·홀드아웃, 대상일 이전만)
    op = pd.read_csv(EDU / "outputs/predict/predictions_smp_land.csv", parse_dates=["date"])
    op = op.rename(columns={"hour_n": "hour", "예측": "pred_op"})[["date", "hour", "pred_op"]]
    op = op.merge(actual_land(), on=["date", "hour"]).dropna()
    hist = d[d["date"] < op["date"].min()]
    # 운영 예측일에는 해당일 이전의 모델 오차를 이력으로 쓰고, 대상 행의 예측값만 운영값으로 바꿈
    tgt = op.merge(offday_table(op.date.min(), op.date.max()), on="date").rename(columns={"pred_op": "pred"})
    tgt = add_keys(tgt)
    comb = pd.concat([d[d["date"] < op["date"].min()], d[(d["date"] >= op["date"].min())].iloc[0:0], tgt])
    # 갱신형이면 운영 기간 중 직전 오차도 운영 예측 오차로 쌓임 (실제 운영과 같은 방식)
    comb = comb.sort_values(["date", "hour"]).reset_index(drop=True)
    m_op = comb["date"].isin(tgt["date"]).values
    lo, hi = ms[best](comb, m_op)
    ref = comb[m_op].assign(lo=lo, hi=hi)
    a3 = pd.DataFrame(report(ref, best, f"운영 기록 {op.date.min().date()}~{op.date.max().date()}"))
    a3.round(4).to_csv(OUT_TAB / "A3_operational_ref.csv", index=False, encoding="utf-8-sig")
    print(a3[["구간", "시간수", "포함률", "하한아래", "상한위", "평균폭", "MAE"]].round(3).to_string(index=False))

    info = {"선택 방법": best, "예측값": "담당자 시간별 모델 (D일 수요예측 사용, 23:30 산출 기준)",
            "선택 기간": "2025 (2024까지 오차로 범위)", "시험 기간": f"2026-01-01 ~ {g.date.max().date()}",
            "이력 기간": f"{hist.date.min().date()} ~"}
    (OUT_TAB / "A_choice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(te[["방법", "구간", "시간수", "포함률", "하한아래", "상한위", "평균폭", "구간점수", "MAE"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
