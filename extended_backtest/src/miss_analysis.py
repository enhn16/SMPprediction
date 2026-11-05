"""틀린 날 분석 (시험 2025-01 ~ 2026-10, 사후 분석).

원칙(2026-11-06): SMP는 전날 오후 거래소 전망으로 확정 → 실제 날씨는 가격 원인이 아니다.
  가격 원인 = 거래소 D일 하루전 수요예측(한낮, 28일 중앙값 대비)  ← SMP와 같은 시각 공표, 사후 분석용
  모델이 놓친 원인 = 우리가 쓴 15시 정보(D-1 11시 오후 하늘 예보, 전날 수요예측 비율)
1. 급락: 놓친 날(주의도 없음)·부분(주의만)·경보 헛경보 날 목록
2. 급락: 반기별 성적 vs 그 시기 모델이 학습한 급락 수 (학습 기간 영향)
3. 급등: 휴일·평일별 판정 분포와 휴일 급등일의 특징

실행: python -m extended_backtest.src.miss_analysis → outputs/tables/M1_drop_days.csv, M2_drop_by_half.csv, M3_surge_offday.csv
"""
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from extended_backtest.src.ablation import kpx_dday
from extended_backtest.src.data import offday_flags

TAB = ROOT / "extended_backtest" / "outputs" / "tables"
FEAT = PROCESSED_DIR / "extended_backtest"


def drop_days():
    g = pd.read_csv(TAB / "D1_drop_walk_graded.csv", parse_dates=["date"])
    f = pd.read_pickle(FEAT / "drop_features_육지.pkl").set_index("date")
    k = kpx_dday()
    g = g.join(f[["rf_sky_pm", "rf_clear_pm", "demand_prev_rel28", "day_type"]], on="date").join(
        k[["kpx_dem_mid_rel28_d"]], on="date")
    g["휴일"] = offday_flags(g["date"]).values
    g["구분"] = np.select([(g["label"] == 1) & (g["grade"] == "경보"), (g["label"] == 1) & (g["grade"] == "주의"),
                         (g["label"] == 1), (g["label"] == 0) & (g["grade"] == "경보"),
                         (g["label"] == 0) & (g["grade"] == "주의")],
                        ["성공", "부분", "실패", "경보 헛경보", "주의 헛경보"], "맞게 없음")
    # 사람이 읽을 이유 (같은 방향 근거만)
    def reason(r):
        price = "거래소 전망 한낮 수요 낮음" if r["kpx_dem_mid_rel28_d"] < 0.92 else "거래소 전망 한낮 수요 보통"
        sky = "우리 예보 맑음" if r["rf_sky_pm"] <= 2 else "우리 예보 흐림"
        return f"{price} ({r['kpx_dem_mid_rel28_d']:.2f}) / {sky} ({r['rf_sky_pm']:.1f})"
    g["원인 메모"] = g.apply(reason, axis=1)
    cols = ["date", "day_type", "휴일", "구분", "min_smp_11_15", "pred", "주의_예상최저이하", "경보_예상최저이하",
            "n_train_drops", "rf_sky_pm", "demand_prev_rel28", "kpx_dem_mid_rel28_d", "원인 메모"]
    out = g[g["구분"] != "맞게 없음"][cols].round(3)
    out.to_csv(TAB / "M1_drop_days.csv", index=False, encoding="utf-8-sig")

    g["반기"] = g["date"].dt.year.astype(str) + np.where(g["date"].dt.month <= 6, " 상반기", " 하반기")
    rows = []
    for h, x in g.groupby("반기"):
        ev = x[x["label"] == 1]
        rows.append({"반기": h, "학습한 급락일(시작~끝)": f"{x['n_train_drops'].min()}~{x['n_train_drops'].max()}",
                     "급락일": len(ev), "성공(경보)": int((ev["grade"] == "경보").sum()),
                     "부분(주의)": int((ev["grade"] == "주의").sum()), "실패": int((ev["grade"] == "없음").sum()),
                     "주의 헛경보": int(((x["label"] == 0) & (x["grade"] == "주의")).sum()),
                     "경보 헛경보": int(((x["label"] == 0) & (x["grade"] == "경보")).sum()),
                     "주의 문턱(원)": f"{x['주의_예상최저이하'].min():.0f}~{x['주의_예상최저이하'].max():.0f}",
                     "경보 문턱(원)": f"{x['경보_예상최저이하'].min():.0f}~{x['경보_예상최저이하'].max():.0f}",
                     "급락일 예상값 평균(원)": round(ev["pred"].mean(), 1) if len(ev) else None})
    hb = pd.DataFrame(rows)
    hb.to_csv(TAB / "M2_drop_by_half.csv", index=False, encoding="utf-8-sig")

    # 원인 메모 요약: 놓친·부분 급락일 중 '거래소 수요 낮음 & 우리 예보 흐림' 비율
    miss = g[(g["label"] == 1) & (g["grade"] != "경보")]
    summ = {"놓침+부분 급락일": len(miss),
            "그중 거래소 전망 수요 낮음(<0.92)": int((miss["kpx_dem_mid_rel28_d"] < 0.92).sum()),
            "그중 우리 예보 흐림(하늘≥2.5)": int((miss["rf_sky_pm"] >= 2.5).sum()),
            "거래소 낮음 & 우리 흐림": int(((miss["kpx_dem_mid_rel28_d"] < 0.92) & (miss["rf_sky_pm"] >= 2.5)).sum())}
    return out, hb, summ


def surge_offday():
    s = pd.read_csv(TAB / "S1_surge_walk_graded.csv", parse_dates=["date"])
    f = pd.read_pickle(FEAT / "surge_features_육지.pkl").set_index("date")
    iv = pd.read_csv(TAB / "E1_intervals_land_surge.csv", parse_dates=["date"])
    iv = iv[iv["date"].between("2025-01-01", "2026-10-31")]
    s["휴일"] = offday_flags(s["date"]).values
    rows = []
    for lab, m in (("휴일", s["휴일"]), ("평일", ~s["휴일"])):
        x = s[m]
        ev = x[x["y_new"] == 1]
        ivx = iv[iv["is_offday"] == (lab == "휴일")]
        rows.append({"구분": lab, "날": len(x), "급등일 비율": round(x["y_new"].mean(), 2),
                     "급등일 점수 평균 B": round(ev["score_B"].mean(), 3),
                     "급등 없는 날 점수 평균 B": round(x[x["y_new"] == 0]["score_B"].mean(), 3),
                     "급등 구간 시간 수 중앙값": float(ivx["n_hours"].median()),
                     "급등 구간 시작 시각 최빈": int(ivx["start_hour"].mode().iloc[0])})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "M3_surge_offday.csv", index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    out, hb, summ = drop_days()
    print(hb.to_string(index=False))
    print(summ)
    print(out[out["구분"].isin(["실패", "부분", "경보 헛경보"])].to_string(index=False))
    print(surge_offday().to_string(index=False))
