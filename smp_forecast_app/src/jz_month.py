"""제주 한 달 참고 전망 (7~34일 뒤, 일별: 일평균·한낮·저녁·한낮 최저) — 육지 한 달 예측 + 최근 제주−육지 차이.

제주 가격은 평소 육지와 같아(같은 가격 시간 2026년 91%) 한 달 앞은 육지 전망에 '최근 차이'만 더한다 (인수 메모 제안).
점예측 후보 (지표별, 차이는 기준일 I 포함 이전 실적만)
  육지 그대로            : 육지 한 달 최종 예측 (MZ_predictions_daily.csv / 운영 기록 daily_log.csv)
  육지 + 최근 N일 차이   : N ∈ {7, 28, 56}
  육지 + 0.5 × 28일 차이 : 차이를 절반만
  육지 + 같은 유형일 8일 차이 : 대상일이 휴무일이면 최근 휴무일 8일, 평일이면 평일 8일의 차이
  기준: 제주 최근 같은 유형일 4일 평균 (rMAE 단순 기준)
범위 후보: 육지 범위를 제주 점예측으로 옮김 (폭 × 1, 1.25, 1.5, 1.75, 2), 하한 0원 자르기 없음
선택 (육지와 같은 규칙): 2025 하반기(대상일 2025-07 ~ 12) MAE, 0.05원 이내면 구성 단순한 쪽 / 범위는 intervals.select
  (육지 한 달 예측 기록이 기준일 2025-05-28부터라 상반기 맞춤 단계는 없음. 후보에 학습 모수가 없어 바로 비교)
출력: outputs/tables/JMZ1_compare_2025H2.csv, JMZ_choice.json, data/processed/smp_forecast_app/jeju/JMZ_predictions_daily.csv
      (--score2026) JMZ2_test2026.csv
실행: python -m smp_forecast_app.src.jz_month [--score2026]
"""
import json
import sys

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from common.loaders import load_smp_actual
from smp_forecast_app.src.intervals import score, select
from smp_forecast_app.src.jz_common import JDIR

TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
STATS = ["mean", "mid", "eve", "midmin"]
POINTS = {"육지 그대로": 0, "육지 + 최근 7일 차이": 1, "육지 + 최근 28일 차이": 1, "육지 + 최근 56일 차이": 1,
          "육지 + 0.5 × 28일 차이": 1, "육지 + 같은 유형일 8일 차이": 2}   # 값 = 구성 단순도 (작을수록 단순)
WIDTH = {"육지 범위 이동 ×1": 1.0, "육지 범위 이동 ×1.25": 1.25, "육지 범위 이동 ×1.5": 1.5, "육지 범위 이동 ×1.75": 1.75,
         "육지 범위 이동 ×2": 2.0}


def daily_stats(region):
    w = load_smp_actual(region).pivot_table(index="date", columns="hour", values="smp")
    return pd.DataFrame({"mean": w.mean(axis=1), "mid": w[[11, 12, 13, 14, 15]].mean(axis=1),
                         "eve": w[[18, 19, 20, 21, 22]].mean(axis=1), "midmin": w[[11, 12, 13, 14, 15]].min(axis=1)})


def base_values():
    """기준일 I별 (I 포함 이전) 제주−육지 차이·제주 같은 유형일 평균 → {이름: DataFrame(index=날짜, 열=지표)}, 휴무일 달력."""
    j, l = daily_stats("제주"), daily_stats("육지")
    full = pd.date_range(j.index.min(), j.index.max() + pd.Timedelta(days=40))
    j, l = j.reindex(full), l.reindex(full)
    off = build_calendar(full[0], full[-1]).set_index("date")["is_offday"].reindex(full).astype(bool)
    dif = j - l
    v = {f"d{n}": dif.rolling(n, min_periods=int(n * 0.7)).mean() for n in (7, 28, 56)}
    for o in (0, 1):
        m = off == bool(o)
        v[f"same8_{o}"] = dif[m].rolling(8, min_periods=6).mean().reindex(full).ffill()
        v[f"jsame4_{o}"] = j[m].rolling(4, min_periods=3).mean().reindex(full).ffill()
    last = dif.dropna(how="all").index.max()
    for k in v:   # 실적이 없는 날 이후는 비움 (기준일은 실적 마지막 날까지만 의미)
        v[k].loc[v[k].index > last] = np.nan
    return v, off, j


def apply(land: pd.DataFrame, point: dict, width: dict, v=None) -> pd.DataFrame:
    """육지 한 달 행 [base_date, H, date, day_type, offday, pred_*, lo_*, hi_*] → 제주 같은 형식 (지표별 point·width 이름)."""
    v = base_values()[0] if v is None else v
    out = land[["base_date", "H", "date", "day_type", "offday"]].copy()
    I, o = land["base_date"], land["offday"].astype(int)
    for s in STATS:
        d = cand_point(land, s, v)[point[s]]
        k = WIDTH[width[s]]
        out[f"pred_{s}"] = d
        out[f"lo_{s}"] = d + k * (land[f"lo_{s}"] - land[f"pred_{s}"]).values
        out[f"hi_{s}"] = d + k * (land[f"hi_{s}"] - land[f"pred_{s}"]).values
    return out


def cand_point(land, s, v):
    I, o = land["base_date"], land["offday"].astype(int).values
    p = land[f"pred_{s}"].values
    at = lambda key: v[key][s].reindex(I).values
    same8 = np.where(o == 1, at("same8_1"), at("same8_0"))
    return {"육지 그대로": p, "육지 + 최근 7일 차이": p + at("d7"), "육지 + 최근 28일 차이": p + at("d28"),
            "육지 + 최근 56일 차이": p + at("d56"), "육지 + 0.5 × 28일 차이": p + 0.5 * at("d28"),
            "육지 + 같은 유형일 8일 차이": p + same8}


def main(score2026=False):
    land = pd.read_csv(PROCESSED_DIR / "smp_forecast_app" / "MZ_predictions_daily.csv", parse_dates=["base_date", "date"])
    v, off, j = base_values()
    for s in STATS:
        land[f"y_{s}"] = j[s].reindex(land["date"]).values   # 제주 실적
    land["offday"] = off.reindex(land["date"]).astype(int).values
    ev = (land["date"] >= "2025-07-01") & (land["date"] < "2026-01-01") & land["y_mean"].notna()
    rows, choice_p, choice_w = [], {}, {}
    for s in STATS:
        cp = cand_point(land, s, v)
        o = land["offday"].values
        base = np.where(o == 1, v["jsame4_1"][s].reindex(land["base_date"]).values, v["jsame4_0"][s].reindex(land["base_date"]).values)
        y = land[f"y_{s}"].values
        mae = lambda p, m=ev.values: float(np.nanmean(np.abs(y[m] - p[m])))
        b = mae(base)
        rows.append({"지표": s, "구분": "점예측", "후보": "기준: 제주 같은 유형일 4일 평균", "MAE": b, "rMAE": 1.0})
        res = {k: mae(p) for k, p in cp.items()}
        for k, m in res.items():
            rows.append({"지표": s, "구분": "점예측", "후보": k, "MAE": m, "rMAE": m / b})
        best = min(res.values())
        pick = sorted([k for k in res if res[k] <= best + 0.05], key=lambda k: (POINTS[k], res[k]))[0]
        choice_p[s] = pick
        p = cp[pick]
        sel = []
        for wn, k in WIDTH.items():
            lo = p + k * (land[f"lo_{s}"] - land[f"pred_{s}"]).values
            hi = p + k * (land[f"hi_{s}"] - land[f"pred_{s}"]).values
            r = score(y[ev.values], lo[ev.values], hi[ev.values])
            sel.append({"구간": "전체", "방법": wn, **r})
            rows.append({"지표": s, "구분": "범위", "후보": wn, "MAE": None, "rMAE": None,
                         **{c: r[c] for c in ("포함률", "하한아래", "상한위", "평균폭", "구간점수")}})
        choice_w[s] = select(pd.DataFrame(sel))
    t = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(TAB / "JMZ1_compare_2025H2.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))

    out = apply(land, choice_p, choice_w, v)
    for s in STATS:
        out[f"y_{s}"] = land[f"y_{s}"]
    cols = ["base_date", "H", "date", "day_type", "offday"] + [f"y_{s}" for s in STATS] + \
           [f"{p}_{s}" for s in STATS for p in ("pred", "lo", "hi")]
    JDIR.mkdir(parents=True, exist_ok=True)
    out[cols].round(2).to_csv(JDIR / "JMZ_predictions_daily.csv", index=False, encoding="utf-8-sig")
    choice = {"지표별": {s: {"점예측": choice_p[s], "범위": choice_w[s]} for s in STATS},
              "선택 절차": "2025 하반기 MAE (0.05원 이내면 단순한 쪽), 범위는 포함률 75~85% 중 구간점수 최소",
              "범위 하한": "0원 자르기 없음", "성격": "참고 전망 (육지 한 달도 2026에 기준을 못 이김)",
              "시험": "2026-01-01 ~ (제주 한 달 1차 시험)"}
    (TAB / "JMZ_choice.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(choice, ensure_ascii=False, indent=1))

    if score2026:   # 사용자 확인 후 한 번
        te = (land["date"] >= "2026-01-01") & land["y_mean"].notna()
        rows = []
        for s in STATS:
            o = land["offday"].values
            base = np.where(o == 1, v["jsame4_1"][s].reindex(land["base_date"]).values, v["jsame4_0"][s].reindex(land["base_date"]).values)
            y, p = land.loc[te, f"y_{s}"].values, out.loc[te, f"pred_{s}"].values
            b = float(np.nanmean(np.abs(y - base[te.values])))
            lp = float(np.nanmean(np.abs(y - land.loc[te, f"pred_{s}"].values)))
            r = score(y, out.loc[te, f"lo_{s}"].values, out.loc[te, f"hi_{s}"].values, p)
            rows.append({"지표": s, "일수": int(te.sum()), "MAE": r["MAE"], "기준 MAE": b, "rMAE": r["MAE"] / b, "육지 그대로 MAE": lp,
                         **{c: r[c] for c in ("포함률", "하한아래", "상한위", "평균폭", "구간점수")}})
        pd.DataFrame(rows).round(4).to_csv(TAB / "JMZ2_test2026.csv", index=False, encoding="utf-8-sig")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main("--score2026" in sys.argv[1:])
