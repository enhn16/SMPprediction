"""모델·피처 탐색 3차: 새 자료 (전력거래소 발전원별 5분 실적, Open-Meteo 과거 예보 일사).

새 피처 (모두 D-1 15시 확보)
  KPX  : D-1 11:00~13:55 평균 — 원전·시장 태양광·양수(음수=펌핑)·가스·석탄·화력 합 출력(GW), 화력 비중
         (실시간 공개 자료라 15시 전 구간만 사용)
  OM   : D일 11~15시 일사 예보 합(Wh/m², 7개 도시 평균) — JMA·ECMWF·GFS의 2일 전 실행분(previous_day2), 세 모델 평균
         예상 태양광2 = 시장 태양광 설비(GW) × 일사/1000, 비중 = ÷ 주간전망 최대부하
검증·채택 규칙: explore.py와 동일 (2025 확장 창 2회, D-1 평일 날 채점, 15% 이상 개선 + 전체 비악화일 때만 교체)
실행: python -m drop_classifier.src.explore2
"""
import warnings

import numpy as np
import pandas as pd

from common.loaders import load_kpx_power_source, load_openmeteo_prev2
from drop_classifier.src.explore import CAT_C, CAT_R, CURRENT, EBM_R, PHYS, RIDGE, avg, cv_scores, data, rank_avg, score
from drop_classifier.src.fuel_level import BASE
from drop_classifier.src.supply_level import CURT, NUC, SOLAR
from drop_classifier.src.two_tier import OUT_TAB

warnings.filterwarnings("ignore")
KPX = ["k_nuc_d1", "k_solar_d1", "k_pump_d1", "k_thermal_d1", "k_thermal_share_d1"]
OM = ["om_rad", "om_cloud", "om_solar_exp", "om_solar_share"]


def kpx_features() -> pd.DataFrame:
    k = load_kpx_power_source()
    k = k[(k["ts"].dt.hour >= 11) & (k["ts"].dt.hour <= 13)]
    k["thermal"] = k[["gas", "coal_bit", "coal_dom", "oil"]].sum(axis=1, min_count=1)
    k["total"] = k[["solar_mkt", "pump", "hydro", "gas", "wind", "renew", "oil", "coal_dom", "coal_bit", "nuclear"]].sum(axis=1)
    g = k.groupby("date").agg(nuc=("nuclear", "mean"), solar=("solar_mkt", "mean"), pump=("pump", "mean"),
                              thermal=("thermal", "mean"), total=("total", "mean"))
    out = pd.DataFrame({"k_nuc_d1": g["nuc"] / 1000, "k_solar_d1": g["solar"] / 1000, "k_pump_d1": g["pump"] / 1000,
                        "k_thermal_d1": g["thermal"] / 1000, "k_thermal_share_d1": g["thermal"] / g["total"]})
    out.index = out.index + pd.Timedelta(days=1)   # D-1 값 → D일 피처
    return out.rename_axis("date").reset_index()


def om_features() -> pd.DataFrame:
    o = load_openmeteo_prev2()
    w = o.pivot_table(index="date", columns="model", values=["rad_mid", "cloud_mid"])
    return pd.DataFrame({"om_rad": w["rad_mid"].mean(axis=1), "om_cloud": w["cloud_mid"].mean(axis=1)}).reset_index()


def data2() -> pd.DataFrame:
    df = data().merge(kpx_features(), on="date", how="left").merge(om_features(), on="date", how="left")
    df["om_solar_exp"] = df["cap_solar_gw"] * df["om_rad"] / 1000
    df["om_solar_share"] = df["om_solar_exp"] / (df["kpx_peak"] / 1000)
    return df


FEATS = {
    "현재(기본+태양광설비)": BASE + SOLAR,
    "+KPX실적": BASE + SOLAR + KPX,
    "+일사예보": BASE + SOLAR + OM,
    "+KPX실적+일사예보": BASE + SOLAR + KPX + OM,
    "+KPX+일사+물리+원전": BASE + SOLAR + KPX + OM + PHYS + NUC,
}
MODELS = {
    "Ridge+EBM(현재)": CURRENT,
    "Ridge+EBM+CatBoost 회귀": avg(RIDGE, EBM_R, CAT_R),
    "순위평균(Ridge+EBM, CatBoost분류)": rank_avg(CURRENT, CAT_C),
}


def main():
    df = data2()
    cov = df[df["date"].between("2024-01-01", "2025-12-31")]
    print("결측 비율:", cov[KPX + OM].isna().mean().round(3).to_dict(), flush=True)
    rows = []
    for fname, cols in FEATS.items():
        for mname, fn in MODELS.items():
            p = cv_scores(df, cols, fn)
            sw, aw = score(p[p["d1_weekday"] == 1])
            sa, aa = score(p)
            rows.append({"피처": fname, "모델": mname, "D-1평일 선택점수": sw, "D-1평일 PR-AUC": aw,
                         "전체 선택점수": sa, "전체 PR-AUC": aa})
            print(f"{fname} / {mname}: D-1평일 {sw} (PR {aw:.3f}) | 전체 {sa} (PR {aa:.3f})", flush=True)
    res = pd.DataFrame(rows).sort_values(["D-1평일 선택점수", "D-1평일 PR-AUC"], ascending=[True, False])
    res.round(3).to_csv(OUT_TAB / "15_explore2_cv2025_육지.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n", res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
