"""③-3 추가 실험 (육지).

A. 예측 시점 앞당기기: D일 급락을 D-h일 17시에 판단 (h = 1~7)
   - 사용 정보: D일 달력 + D-h일까지의 SMP 실적 + D-h일까지 공개된 수요예측으로 만든 수요 대리값
   - 날씨 예보와 D일 수요예측은 쓰지 않음 (주간 예측에서 쓸 수 있는 정보만 가정)
B. 학습 목표 바꾸기: 급락 여부(0/1) 대신 한낮 최저 SMP(연속값)나 더 느슨한 기준(<50, <70원)으로 학습해
   급락(<30원) 판정 점수로 사용
평가: 롤링 2회(2025, 2026 상반기) 예측을 합쳐 적중률 70/80/90%에서 필요한 헛경보 수

실행: python -m drop_classifier.src.experiments_more
"""
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT, SEED
from common.loaders import load_smp_forecast_api
from drop_classifier.src.experiments import BASE, false_alarms_at
from drop_classifier.src.models import FOLDS, Logit, make_models, split

warnings.filterwarnings("ignore")
REGION = "육지"
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
CAL_COLS = ["is_offday", "is_weekend_only", "is_holiday_single", "is_long_holiday", "off_block_len", "off_block_day",
            "is_myeongjeol", "is_labor_day", "month_sin", "month_cos"]


def score(y, p) -> dict:
    out = {}
    for t in (0.7, 0.8, 0.9):
        fa, _ = false_alarms_at(y, p, t)
        out[f"헛경보@{int(t * 100)}%"] = fa
    return out


def rolling(df, fit_predict) -> dict:
    ys, ps, ap = [], [], []
    for fold in ("롤링1", "롤링2"):
        tr, te = split(df, FOLDS[fold][0]), split(df, FOLDS[fold][1])
        p = fit_predict(tr, te)
        ys.append(te["label"].values)
        ps.append(p)
        ap.append(average_precision_score(te["label"], p))
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {"PR-AUC 평균": np.mean(ap), **score(y, p)}


# ---------- A. 예측 시점 ----------
def horizon_table() -> pd.DataFrame:
    label = pd.read_csv(PROCESSED_DIR / "drop_classifier" / f"daily_label_{REGION}.csv", parse_dates=["date"])
    label = label.sort_values("date").reset_index(drop=True)
    fc = load_smp_forecast_api()
    dem = fc[(fc["region"] == REGION) & fc["hour"].between(11, 15)].groupby("date")["demand_fcst"].mean()
    label["demand_fcst"] = label["date"].map(dem)
    cal = build_calendar(label["date"].min(), label["date"].max())
    label = label.drop(columns=[c for c in cal.columns if c != "date"], errors="ignore").merge(cal, on="date")
    label["is_offday"] = (label["day_type"] != "평일").astype(int)
    label["is_weekend_only"] = (label["day_type"] == "주말").astype(int)
    label["is_holiday_single"] = (label["day_type"] == "공휴일").astype(int)
    label["is_long_holiday"] = (label["day_type"] == "연휴").astype(int)
    label["is_myeongjeol"] = label["is_myeongjeol"].astype(int)
    label["is_labor_day"] = label["is_labor_day"].astype(int)
    m = label["date"].dt.month
    label["month_sin"], label["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    return label


def horizon_features(lab: pd.DataFrame, h: int) -> pd.DataFrame:
    """D일 행에, D-h일(판단일)까지 알 수 있는 값만 붙인다."""
    smp = lab["min_smp_11_15"].values
    off = lab["is_offday"].values
    dem = lab["demand_fcst"].values
    y = lab["label"].values
    rows = []
    for i in range(len(lab)):
        k = i - h  # 판단일 인덱스 (이날까지의 SMP·수요예측 공개분 사용)
        if k < 60:
            rows.append({})
            continue
        past = slice(k - 55, k + 1)
        same = off[past] == off[i]
        same_idx = np.arange(k - 55, k + 1)[same]
        d_same = dem[same_idx][-4:]
        rows.append({
            "prev_same_type_min_smp": smp[same_idx[-1]] if len(same_idx) else np.nan,
            "min_smp_last7": smp[k - 6:k + 1].min(),
            "drops_last14": y[k - 13:k + 1].sum(),
            "demand_proxy": np.median(d_same) if len(d_same) else np.nan,
            "demand_proxy_rel": (np.median(d_same) / np.median(dem[k - 27:k + 1])) if len(d_same) else np.nan,
        })
    f = pd.concat([lab, pd.DataFrame(rows, index=lab.index)], axis=1)
    return f[f["date"] >= "2024-01-01"].reset_index(drop=True)


def run_horizon() -> pd.DataFrame:
    lab = horizon_table()
    cols = CAL_COLS + ["prev_same_type_min_smp", "min_smp_last7", "drops_last14", "demand_proxy", "demand_proxy_rel"]
    rows = []
    for h in range(1, 8):
        df = horizon_features(lab, h)
        df["label"] = df["label"].astype(int)
        r = rolling(df, lambda tr, te: Logit().fit(tr[cols], tr["label"]).predict_proba(te[cols]))
        rows.append({"판단 시점": f"D-{h}", **r})
        print(f"D-{h}: {r}", flush=True)
    return pd.DataFrame(rows)


# ---------- B. 학습 목표 ----------
def run_targets() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "drop_classifier" / f"features_{REGION}.csv", parse_dates=["date"])
    df["label"] = df["label"].astype(int)
    cols = BASE
    mk = make_models()

    def logit_at(th):
        def f(tr, te):
            return Logit().fit(tr[cols], (tr["min_smp_11_15"] < th).astype(int)).predict_proba(te[cols])
        return f

    def multi(tr, te):
        return np.mean([logit_at(th)(tr, te) for th in (30, 50, 70)], axis=0)

    def ridge(tr, te):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3.0))
        m.fit(tr[cols], tr["min_smp_11_15"].clip(0, 150))
        return -m.predict(te[cols])

    def lgbm_reg(tr, te):
        m = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
        m.fit(tr[cols], tr["min_smp_11_15"].clip(0, 150))
        return -m.predict(te[cols])

    def lgbm_quantile(tr, te):
        m = LGBMRegressor(objective="quantile", alpha=0.1, n_estimators=300, learning_rate=0.03, num_leaves=7,
                          min_child_samples=10, random_state=SEED, verbose=-1)
        m.fit(tr[cols], tr["min_smp_11_15"].clip(0, 150))
        return -m.predict(te[cols])

    def ens_logit_multi(tr, te):
        a = Logit().fit(tr[cols], tr["label"]).predict_proba(te[cols])
        return (a + multi(tr, te)) / 2

    cands = {
        "기준: 로지스틱(급락<30 학습)": logit_at(30),
        "로지스틱(<50원으로 학습)": logit_at(50),
        "로지스틱(<70원으로 학습)": logit_at(70),
        "로지스틱 다중 기준(<30·<50·<70 평균)": multi,
        "기준 + 다중 기준 평균": ens_logit_multi,
        "Ridge 회귀(최저 SMP 예측)": ridge,
        "LightGBM 회귀(최저 SMP 예측)": lgbm_reg,
        "LightGBM 하위 10% 분위 회귀": lgbm_quantile,
        "EBM(급락<30)": lambda tr, te: mk["EBM"]().fit(tr[cols], tr["label"]).predict_proba(te[cols]),
    }
    rows = []
    for name, f in cands.items():
        r = rolling(df, f)
        rows.append({"후보": name, **r})
        print(f"{name}: {r}", flush=True)
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    print("[A. 예측 시점 앞당기기]")
    a = run_horizon().round(3)
    a.to_csv(OUT_TAB / f"03c_horizon_{REGION}.csv", index=False, encoding="utf-8-sig")
    print("\n[B. 학습 목표 바꾸기]")
    b = run_targets().round(3)
    b.to_csv(OUT_TAB / f"03c_targets_{REGION}.csv", index=False, encoding="utf-8-sig")
    print("\n", a.to_string(index=False), "\n\n", b.to_string(index=False))


if __name__ == "__main__":
    main()
