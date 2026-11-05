"""제주 급락(SMP ≤ 0원) 후보 모델 — 매달 다시 학습한 '앞으로만' 예측 (2024-03 ~ 2026-10).

후보는 결과를 보기 전에 아래 SETS·MODELS·ENSEMBLES로 고정 (2026-11-06).
매달 m: 학습 = 2024-01-01 ~ m 직전 날(0원 레이블), 예측 = m의 모든 날.
  1단계(잉여일) = 2021-01-01 ~ min(m 직전, 2024-05-31)의 출력제어·0원 레이블, 실제 계통 모양(act_*)으로 로지스틱 학습
               → 전망 계통 모양(fc_*)을 넣어 s1(잉여 확률). 2024-06 이후 1단계는 고정.
고르기·문턱·채점은 jeju_select.py. 여기서는 예측만 저장.
저장: data/processed/extended_backtest/jeju_fwd_preds.csv [date, event, n_intervals, offday, s1, <후보>...]
실행(루트에서): python -m extended_backtest.src.jeju_models
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common.config import PROCESSED_DIR, SEED

warnings.filterwarnings("ignore")
DIR = PROCESSED_DIR / "extended_backtest"
FEAT, S5F, OUT = DIR / "jeju_features.csv", DIR / "jeju_s5_features.csv", DIR / "jeju_fwd_preds.csv"
FIRST, LAST = pd.Timestamp("2024-03-01"), pd.Timestamp("2026-10-01")
TRAIN0 = pd.Timestamp("2024-01-01")

SHAPE = ["net_min", "net_min_mid", "net_min_night", "sol_max", "wind_mean", "wind_night", "dem_min_mid", "dem_mean", "re_max"]
FC = [f"fc_{c}" for c in SHAPE] + ["fc_cloud"]
CAL = ["offday", "m_sin", "m_cos"]
REC = ["yd_smp_min", "yd_mid_min", "yd_zero", "zero_7", "zero_28", "yd_land_mid_min", "yd_mid_rel"]
YD = ["yd_net_min_mid", "yd_sol_max", "yd_wind_mean"]
S1_COLS = [f"act_{c}" for c in SHAPE] + CAL


def _s5_cols():
    return [c for c in pd.read_csv(S5F, nrows=1).columns if c != "date"]


def sets():
    s5 = _s5_cols()
    pr = FC + CAL + REC + YD + ["fuel_lng"]
    return {"P": FC + CAL,                       # 물리(전망 계통 모양) + 달력
            "PR": pr,                            # + 최근 가격·전날 오전 계통·연료비
            "PRS": pr + ["s1"],                  # + 1단계 잉여 확률 (2단계 설계)
            "S5": s5,                            # 지금 운영 제주 급락 피처 63개
            "S5P": s5 + FC + ["s1"]}             # 운영 피처 + 전망 계통 모양 + 잉여 확률


def pipe(m):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


def _lgb():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=15,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)


def _ebm():
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(interactions=0, max_bins=32, outer_bags=4, random_state=SEED, n_jobs=1)


MODELS = {
    "LR2": lambda: pipe(LogisticRegression(C=0.3, max_iter=2000)),
    "LR1": lambda: pipe(LogisticRegression(C=0.3, penalty="l1", solver="liblinear", max_iter=2000)),
    "LGB": lambda: make_pipeline(SimpleImputer(strategy="median"), _lgb()),
    "RF": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
        n_estimators=400, max_depth=5, min_samples_leaf=5, max_features="sqrt", random_state=SEED, n_jobs=1)),
    "ET": lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=5, max_features="sqrt", random_state=SEED, n_jobs=1)),
    "KNN": lambda: pipe(KNeighborsClassifier(n_neighbors=25, weights="distance")),
    "NB": lambda: pipe(GaussianNB()),
    "SVC": lambda: pipe(SVC(C=1.0, probability=True, random_state=SEED)),
    "EBM": lambda: make_pipeline(SimpleImputer(strategy="median"), _ebm()),
}
# (묶음, 모델) 단일 후보. EBM·SVC는 느려 PRS·S5P에만.
SINGLES = [(s, m) for s in ["P", "PR", "PRS", "S5", "S5P"] for m in ["LR2", "LR1", "LGB", "RF", "ET", "KNN", "NB"]] + \
          [(s, m) for s in ["PRS", "S5P"] for m in ["SVC", "EBM"]]
# 규칙·기준선
RULES = {"규칙:s1": "s1", "규칙:-fc_net_min": "fc_net_min", "규칙:-fc_net_min_mid": "fc_net_min_mid"}
# 사전 고정 앙상블 (확률 평균)
ENSEMBLES = {
    "E1 PRS LR2+LGB": ["PRS|LR2", "PRS|LGB"],
    "E2 PRS LR2+LGB+KNN": ["PRS|LR2", "PRS|LGB", "PRS|KNN"],
    "E3 PRS 트리3+LR2": ["PRS|LR2", "PRS|LGB", "PRS|RF", "PRS|ET"],
    "E4 PRS 전체": [f"PRS|{m}" for m in ["LR2", "LR1", "LGB", "RF", "ET", "KNN", "NB", "SVC", "EBM"]],
    "E5 PRS LR2 + S5 KNN(운영)": ["PRS|LR2", "S5|KNN"],
    "E6 PRS LGB + S5P LR2": ["PRS|LGB", "S5P|LR2"],
    "E7 S5P LR2+LGB+EBM": ["S5P|LR2", "S5P|LGB", "S5P|EBM"],
    "E8 PRS+S5P LR2·LGB·EBM": ["PRS|LR2", "PRS|LGB", "PRS|EBM", "S5P|LR2", "S5P|LGB", "S5P|EBM"],
    "E9 PRS LR2+RF+EBM": ["PRS|LR2", "PRS|RF", "PRS|EBM"],
}


def load_table() -> pd.DataFrame:
    d = pd.read_csv(FEAT, parse_dates=["date"])
    s5 = pd.read_csv(S5F, parse_dates=["date"])
    s5 = s5[[c for c in s5.columns if c == "date" or c not in d.columns]]
    return d.merge(s5, on="date", how="left")


def stage1(d: pd.DataFrame, m: pd.Timestamp) -> np.ndarray:
    """m 직전까지의 잉여일 레이블로 학습 → 모든 행의 전망 모양으로 s1."""
    tr = d[(d["date"] < min(m, pd.Timestamp("2024-06-01"))) & d["surplus"].notna()]
    mdl = pipe(LogisticRegression(C=0.3, max_iter=2000)).fit(tr[S1_COLS], tr["surplus"])
    X = d[[c.replace("act_", "fc_") if c.startswith("act_") else c for c in S1_COLS]]
    X.columns = S1_COLS
    return mdl.predict_proba(X)[:, 1]


def _fit_pred(name, make, Xtr, ytr, Xte):
    if ytr.sum() < 2:
        return name, np.full(len(Xte), ytr.mean())
    return name, make().fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def month_preds(d: pd.DataFrame, m: pd.Timestamp, S: dict) -> pd.DataFrame:
    d = d.copy()
    d["s1"] = stage1(d, m)
    nxt = m + pd.offsets.MonthBegin(1)
    tr = d[(d["date"] >= TRAIN0) & (d["date"] < m) & d["event"].notna()]
    te = d[(d["date"] >= m) & (d["date"] < nxt)]
    out = te[["date", "event", "n_intervals", "offday", "s1"]].copy()
    jobs = [delayed(_fit_pred)(f"{s}|{mn}", MODELS[mn], tr[S[s]], tr["event"], te[S[s]]) for s, mn in SINGLES]
    for name, p in Parallel(n_jobs=8)(jobs):
        out[name] = p
    # 규칙: 점수만 (값이 작을수록 위험한 열은 부호 반대). 확률 보정은 하지 않음
    for name, col in RULES.items():
        out[name] = te["s1"].values if col == "s1" else -te[col].values
    # 순부하 하나만 쓴 로지스틱 (기준선)
    _, out["기준선:LR(fc_net_min)"] = _fit_pred("b", MODELS["LR2"], tr[["fc_net_min"]], tr["event"], te[["fc_net_min"]])
    for name, mem in ENSEMBLES.items():
        out[name] = out[mem].mean(axis=1)
    return out


def main():
    d = load_table()
    S = sets()
    parts = []
    for m in pd.date_range(FIRST, LAST, freq="MS"):
        p = month_preds(d, m, S)
        parts.append(p)
        print(m.strftime("%Y-%m"), len(p), "사건일", int(p["event"].sum()), flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("저장", OUT, out.shape)


if __name__ == "__main__":
    main()
