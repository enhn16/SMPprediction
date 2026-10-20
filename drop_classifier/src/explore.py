"""모델·피처 탐색 2차 (육지, 판단 시각 D-1 15:00) — '모델이 실제로 필요한 날'에 집중.

운영 설계 (2026-10-19)
  D-1이 토·일이면 D일 SMP가 14시 전에 공표된다(공표경계 판정서 v1.2) → 공표값으로 급락 판정(모델 불필요).
  모델은 D-1이 월~금(공휴일 포함, 보수적)인 날만 판단한다.

검증 (2025년 이전 자료만, 확장 창 2회)
  F1: 2024-01 ~ 2024-12 학습 → 2025-01 ~ 06 예측
  F2: 2024-01 ~ 2025-06 학습 → 2025-07 ~ 12 예측
  두 예측을 이어 붙인 2025년 중 'D-1 평일' 날(급락 13일)로 채점.
  선택점수 = 적중 80%·90%를 내는 데 필요한 헛경보 합 (작을수록 좋음), 보조 = 전체 날 선택점수.

채택 규칙 (2026 채점 전에 고정)
  - 후보 중 D-1 평일 선택점수 최소를 고른다(같으면 PR-AUC).
  - 현 최종 모델보다 D-1 평일 선택점수가 15% 이상 작고, 전체 날 선택점수가 나빠지지 않을 때만 교체.
    (후보가 많아 2025년에 우연히 맞는 모델을 고를 위험을 줄이기 위한 여유)
  - 2026년은 채택 여부가 정해진 뒤 한 번만 채점.

새 피처 (모두 15시 확보)
  - 물리 피처: 예상 태양광 = 시장 태양광 설비(GW) × 정오 태양고도 sin × 하늘상태 보정(전날 11시 예보)
    태양광 비중 = 예상 태양광 / 주간전망 최대부하, 원전 비중 = 원전 가동가능 / 주간전망 최대부하,
    경직성 비중 = (원전 + 3 × 예상 태양광) / 최대부하 (자가용·PPA 포함 총 태양광 ≈ 시장분의 약 3배 가정)
실행: python -m drop_classifier.src.explore
"""
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score

from common.config import SEED
from drop_classifier.src.experiments import false_alarms_at
from drop_classifier.src.fair_select import fcst15
from drop_classifier.src.fuel_level import BASE
from drop_classifier.src.search import _imp, day_table
from drop_classifier.src.supply_level import CURT, NUC, SOLAR, supply_features
from drop_classifier.src.two_tier import OUT_TAB

warnings.filterwarnings("ignore")
FOLDS = [(("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-06-30")),
         (("2024-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"))]
PHYS = ["solar_exp", "solar_share", "nuc_share", "rigid_share"]
CLIP = (0, 150)


def add_physics(df: pd.DataFrame) -> pd.DataFrame:
    doy = df["date"].dt.dayofyear
    elev = 90 - 36 + 23.44 * np.sin(2 * np.pi * (doy - 81) / 365)   # 북위 36도 정오 태양고도
    cloud = ((df["rf_sky_pm"] - 1) / 3).clip(0, 1).fillna(0.5)
    rain = (df["rf_pop_pm"].fillna(30) / 100)
    df["solar_exp"] = df["cap_solar_gw"] * np.sin(np.radians(elev)) * (1 - 0.7 * cloud) * (1 - 0.3 * rain)
    peak = df["kpx_peak"] / 1000
    df["solar_share"] = df["solar_exp"] / peak
    df["nuc_share"] = df["nuc_avail_gw"] / peak
    df["rigid_share"] = (df["nuc_avail_gw"] + 3 * df["solar_exp"]) / peak
    return df


def reg(make):
    def f(tr, te, cols):
        return -make().fit(tr[cols], tr["min_smp_11_15"].clip(*CLIP)).predict(te[cols])
    return f


def cls(make, th=30):
    def f(tr, te, cols):
        return make().fit(tr[cols], (tr["min_smp_11_15"] < th).astype(int)).predict_proba(te[cols])[:, 1]
    return f


def rank_avg(*fns):
    def f(tr, te, cols):
        return np.mean([pd.Series(fn(tr, te, cols)).rank(pct=True).values for fn in fns], axis=0)
    return f


def avg(*fns):
    def f(tr, te, cols):
        return np.mean([fn(tr, te, cols) for fn in fns], axis=0)
    return f


RIDGE = reg(lambda: _imp(Ridge(alpha=3)))
EBM_R = reg(lambda: ExplainableBoostingRegressor(random_state=SEED, interactions=5, min_samples_leaf=5, max_bins=64, n_jobs=1))
LGB_R = reg(lambda: LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1))
LGB_Q = reg(lambda: LGBMRegressor(objective="quantile", alpha=0.2, n_estimators=300, learning_rate=0.03, num_leaves=15,
                                  min_child_samples=10, random_state=SEED, verbose=-1))
CAT_R = reg(lambda: CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05, random_seed=SEED, verbose=0))
LOGIT70 = cls(lambda: _imp(LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")), 70)
CAT_C = cls(lambda: CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, auto_class_weights="Balanced",
                                       random_seed=SEED, verbose=0), 50)
EBM_C = cls(lambda: ExplainableBoostingClassifier(random_state=SEED, interactions=5, min_samples_leaf=5, max_bins=64,
                                                  n_jobs=1), 50)
CURRENT = avg(RIDGE, EBM_R)

FEATS = {
    "현재(기본+태양광설비)": BASE + SOLAR,
    "+물리": BASE + SOLAR + NUC + PHYS,
    "+물리+출력제어": BASE + SOLAR + NUC + PHYS + CURT,
}
MODELS = {
    "Ridge+EBM(현재)": CURRENT,
    "EBM 회귀": EBM_R,
    "LightGBM 회귀": LGB_R,
    "LightGBM 20%분위": LGB_Q,
    "CatBoost 회귀": CAT_R,
    "로지스틱(<70원)": LOGIT70,
    "CatBoost 분류(<50원)": CAT_C,
    "Ridge+EBM+CatBoost 회귀": avg(RIDGE, EBM_R, CAT_R),
    "순위평균(Ridge+EBM, CatBoost분류)": rank_avg(CURRENT, CAT_C),
    "순위평균(Ridge+EBM, 로지스틱70)": rank_avg(CURRENT, LOGIT70),
}


def data() -> pd.DataFrame:
    df = day_table().merge(fcst15(), on="date", how="left")
    df = pd.concat([df.reset_index(drop=True), supply_features(df["date"])], axis=1)
    df["d1_weekday"] = ((df["date"] - pd.Timedelta(days=1)).dt.dayofweek < 5).astype(int)
    return add_physics(df)


def cv_scores(df, cols, fn):
    parts = []
    for (a, b), (c, d) in FOLDS:
        tr = df[df["date"].between(a, b)]
        te = df[df["date"].between(c, d)].copy()
        te["s"] = fn(tr, te, cols)
        parts.append(te[["date", "label", "d1_weekday", "s"]])
    return pd.concat(parts)


def score(p):
    y, s = p["label"].values, p["s"].values
    return false_alarms_at(y, s, 0.8)[0] + false_alarms_at(y, s, 0.9)[0], average_precision_score(y, s)


def main():
    df = data()
    rows = []
    for fname, cols in FEATS.items():
        for mname, fn in MODELS.items():
            p = cv_scores(df, cols, fn)
            w = p[p["d1_weekday"] == 1]
            sw, aw = score(w)
            sa, aa = score(p)
            rows.append({"피처": fname, "모델": mname, "D-1평일 선택점수": sw, "D-1평일 PR-AUC": aw,
                         "전체 선택점수": sa, "전체 PR-AUC": aa})
            print(f"{fname} / {mname}: D-1평일 {sw} (PR {aw:.3f}) | 전체 {sa} (PR {aa:.3f})", flush=True)
    res = pd.DataFrame(rows).sort_values(["D-1평일 선택점수", "D-1평일 PR-AUC"], ascending=[True, False])
    res.round(3).to_csv(OUT_TAB / "14_explore_cv2025_육지.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n", res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
