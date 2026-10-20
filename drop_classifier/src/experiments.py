"""③-2 개선 실험: 급락을 더 잘 맞히는 모델·요인이 있는가 (육지).

성공 기준(사용자): 과거 급락일 적중률 + 헛경보 수.
→ 롤링 평가(2025 + 2026 상반기, 급락 34일) 예측을 합쳐, 적중률 70/80/90%를 달성할 때 필요한 헛경보 수로 비교.
   같은 적중률에서 헛경보가 적을수록 좋다. 롤링 두 구간 PR-AUC도 함께 본다.
검증 기간(2026-07~)은 후보를 고를 때 보지 않는다.

실행: python -m drop_classifier.src.experiments
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common.config import PROCESSED_DIR, ROOT
from drop_classifier.src.features import CALENDAR, CALENDAR_X, DEMAND, DEMAND_X, FCST, FCST_X, LAG, LAG_X
from drop_classifier.src.models import FOLDS, Logit, make_models, split

warnings.filterwarnings("ignore")
REGION = "육지"
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
BASE = CALENDAR + DEMAND + LAG + FCST
ALL = BASE + DEMAND_X + LAG_X + FCST_X + CALENDAR_X


class LogitC(Logit):
    def __init__(self, C):
        super().__init__()
        self.m.set_params(logisticregression__C=C)


class Ensemble:
    """로지스틱·EBM·CatBoost 예측 확률 평균."""

    def __init__(self):
        mk = make_models()
        self.ms = [mk["로지스틱"](), mk["EBM"](), mk["CatBoost"]()]

    def fit(self, X, y):
        for m in self.ms:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


def candidates() -> dict:
    mk = make_models()
    return {
        "기준: 로지스틱(예보)": (mk["로지스틱"], BASE),
        "로지스틱 + 수요 모양": (mk["로지스틱"], BASE + DEMAND_X),
        "로지스틱 + 시장 흐름": (mk["로지스틱"], BASE + LAG_X),
        "로지스틱 + 지역 날씨": (mk["로지스틱"], BASE + FCST_X),
        "로지스틱 + 달력 보강": (mk["로지스틱"], BASE + CALENDAR_X),
        "로지스틱 + 전체 추가": (mk["로지스틱"], ALL),
        "로지스틱(C=0.1) + 전체 추가": (lambda: LogitC(0.1), ALL),
        "로지스틱(C=1) + 전체 추가": (lambda: LogitC(1.0), ALL),
        "EBM(예보)": (mk["EBM"], BASE),
        "EBM + 전체 추가": (mk["EBM"], ALL),
        "CatBoost(예보)": (mk["CatBoost"], BASE),
        "CatBoost + 전체 추가": (mk["CatBoost"], ALL),
        "앙상블(예보)": (lambda: Ensemble(), BASE),
        "앙상블 + 전체 추가": (lambda: Ensemble(), ALL),
    }


def false_alarms_at(y, p, target):
    """적중률 target 이상을 달성하는 가장 높은 임계값에서의 (헛경보 수, 적중 수)."""
    order = np.argsort(-p)
    ys = y[order]
    hits = np.cumsum(ys)
    need = int(np.ceil(target * y.sum()))
    k = int(np.argmax(hits >= need))  # 처음으로 목표에 닿는 순위
    # 같은 점수 동률은 함께 경보
    thr = p[order][k]
    pred = p >= thr
    return int((pred & (y == 0)).sum()), int((pred & (y == 1)).sum())


def evaluate(factory, cols, df):
    ys, ps, ap = [], [], {}
    for fold in ("롤링1", "롤링2"):
        tr, te = split(df, FOLDS[fold][0]), split(df, FOLDS[fold][1])
        p = factory().fit(tr[cols], tr["label"]).predict_proba(te[cols])
        ap[fold] = average_precision_score(te["label"], p)
        ys.append(te["label"].values)
        ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    out = {"PR-AUC 롤링1": ap["롤링1"], "PR-AUC 롤링2": ap["롤링2"], "PR-AUC 평균": np.mean(list(ap.values()))}
    for t in (0.7, 0.8, 0.9):
        fa, hit = false_alarms_at(y, p, t)
        out[f"헛경보@{int(t * 100)}%"] = fa
        out[f"적중@{int(t * 100)}%"] = hit
    return out


def main():
    df = pd.read_csv(PROCESSED_DIR / "drop_classifier" / f"features_{REGION}.csv", parse_dates=["date"])
    df["label"] = df["label"].astype(int)
    n_pos = int(df[df["date"].between("2025-01-01", "2026-06-30")]["label"].sum())
    rows = []
    for name, (factory, cols) in candidates().items():
        r = evaluate(factory, cols, df)
        rows.append({"후보": name, "피처 수": len(cols), **r})
        print(f"{name}: PR-AUC {r['PR-AUC 평균']:.3f}, 헛경보@70% {r['헛경보@70%']}, @80% {r['헛경보@80%']}, @90% {r['헛경보@90%']}",
              flush=True)
    res = pd.DataFrame(rows).round(3)
    res.to_csv(OUT_TAB / f"03b_experiments_{REGION}.csv", index=False, encoding="utf-8-sig")
    print(f"\n롤링 평가 급락일 {n_pos}일 (적중 70% = {int(np.ceil(0.7 * n_pos))}일, 80% = {int(np.ceil(0.8 * n_pos))}일, 90% = {int(np.ceil(0.9 * n_pos))}일)")
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
