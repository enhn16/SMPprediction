"""하루 전·일주일 모델 탐색 (육지).

공공데이터로 판단 시각에 알 수 있는 값만 사용 (D일 수요예측 제외).
후보: 급락 분류(로지스틱·EBM·CatBoost) + 한낮 최저 SMP 회귀(Ridge·LightGBM·EBM·CatBoost) 등
점수는 '클수록 급락 가능성 높음'으로 통일 (회귀는 예측 SMP에 −1).
선택 기준(사용자 성공 기준): 롤링 2회 예측을 합쳐 적중률 80%·90% 달성에 필요한 헛경보 수의 합 (작을수록 좋음).

실행: python -m drop_classifier.src.search
"""
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import PROCESSED_DIR, ROOT, SEED
from drop_classifier.src.experiments import false_alarms_at
from drop_classifier.src.features import CALENDAR, CALENDAR_X, DEMAND_PREV, FCST, FCST_X, LAG, LAG_X
from drop_classifier.src.features_weekly import W_CALENDAR, W_MID, W_STATE
from drop_classifier.src.models import FOLDS, Logit, split
from drop_classifier.src.outlook_features import OUTLOOK
from drop_classifier.src.outlook_features import attach as outlook_attach

warnings.filterwarnings("ignore")
REGION = "육지"
OUT_DATA = PROCESSED_DIR / "drop_classifier"
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
CLIP = (0, 150)

DAY_SETS = {
    "기본": CALENDAR + DEMAND_PREV + LAG + FCST,
    # 전력거래소 주간 전력수요전망 (게시판 등록 다음 날부터 사용, 2026-10-19 사용자 결정으로 하루 전 최종 모델에 채택)
    "기본+주간전망": CALENDAR + DEMAND_PREV + LAG + FCST + OUTLOOK,
    "기본+보강": CALENDAR + CALENDAR_X + DEMAND_PREV + LAG + LAG_X + FCST + FCST_X,
    "기본+보강+수요대리": CALENDAR + CALENDAR_X + DEMAND_PREV + LAG + LAG_X + FCST + FCST_X
                     + ["w_demand_proxy", "w_demand_proxy_rel"],
}


def _imp(m):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


def reg(model_fn):
    """최저 SMP 회귀 → 점수 = −예측값."""
    def f(tr, te, cols):
        m = model_fn().fit(tr[cols], tr["min_smp_11_15"].clip(*CLIP))
        return -m.predict(te[cols])
    return f


def cls(model_fn, th=30):
    def f(tr, te, cols):
        m = model_fn().fit(tr[cols], (tr["min_smp_11_15"] < th).astype(int))
        p = m.predict_proba(te[cols])
        return p[:, 1] if p.ndim == 2 else p
    return f


def avg_reg(*fns):
    def f(tr, te, cols):
        return np.mean([fn(tr, te, cols) for fn in fns], axis=0)
    return f


CANDIDATES = {
    "로지스틱(급락)": cls(Logit),
    "로지스틱(<70원)": cls(Logit, 70),
    "EBM 분류": cls(lambda: ExplainableBoostingClassifier(random_state=SEED, interactions=5, min_samples_leaf=5, max_bins=64, n_jobs=1)),
    "CatBoost 분류": cls(lambda: CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05,
                                                  auto_class_weights="Balanced", random_seed=SEED, verbose=0)),
    "Ridge 회귀(α=3)": reg(lambda: _imp(Ridge(alpha=3))),
    "Ridge 회귀(α=10)": reg(lambda: _imp(Ridge(alpha=10))),
    "LightGBM 회귀": reg(lambda: LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                               random_state=SEED, verbose=-1)),
    "EBM 회귀": reg(lambda: ExplainableBoostingRegressor(random_state=SEED, interactions=5, min_samples_leaf=5, max_bins=64, n_jobs=1)),
    "CatBoost 회귀": reg(lambda: CatBoostRegressor(iterations=400, depth=4, learning_rate=0.05, random_seed=SEED, verbose=0)),
}
CANDIDATES["Ridge+EBM 회귀 평균"] = avg_reg(CANDIDATES["Ridge 회귀(α=3)"], CANDIDATES["EBM 회귀"])
INTERPRETABLE = {"로지스틱(급락)", "로지스틱(<70원)", "EBM 분류", "Ridge 회귀(α=3)", "Ridge 회귀(α=10)", "EBM 회귀",
                 "Ridge+EBM 회귀 평균"}


def evaluate(df: pd.DataFrame, cols: list[str], fn) -> dict:
    ys, ps, out = [], [], {}
    for fold in ("롤링1", "롤링2"):
        tr, te = split(df, FOLDS[fold][0]), split(df, FOLDS[fold][1])
        p = fn(tr, te, cols)
        ys.append(te["label"].values)
        ps.append(p)
        out[f"PR-AUC {fold}"] = average_precision_score(te["label"], p)
        out[f"헛경보@90% {fold}"] = false_alarms_at(te["label"].values, p, 0.9)[0]
    y, p = np.concatenate(ys), np.concatenate(ps)
    for t in (0.7, 0.8, 0.9):
        out[f"헛경보@{int(t * 100)}%"] = false_alarms_at(y, p, t)[0]
    out["PR-AUC 평균"] = (out["PR-AUC 롤링1"] + out["PR-AUC 롤링2"]) / 2
    out["선택점수"] = out["헛경보@80%"] + out["헛경보@90%"]
    return out


def pick(res: pd.DataFrame) -> pd.Series:
    """선택점수 최소. 해석 가능한 모델이 10% 이내면 해석 가능한 모델 우선."""
    res = res.sort_values(["선택점수", "PR-AUC 평균"], ascending=[True, False])
    best = res.iloc[0]
    interp = res[res["모델"].isin(INTERPRETABLE)].iloc[0]
    return interp if interp["선택점수"] <= best["선택점수"] * 1.10 else best


def day_table() -> pd.DataFrame:
    df = pd.read_csv(OUT_DATA / f"features_{REGION}.csv", parse_dates=["date"])
    w = pd.read_csv(OUT_DATA / f"features_weekly_{REGION}.csv", parse_dates=["date"])
    w1 = w[w["lead"] == 1][["date", "w_demand_proxy", "w_demand_proxy_rel"]]
    df = df.merge(w1, on="date", how="left")
    # 주간전망: 판단일(D-1) 전날까지 등록분 = 등록 다음 날부터 사용 (보수적)
    df = pd.concat([df, outlook_attach(df["date"], df["date"] - pd.Timedelta(days=1))], axis=1)
    df["label"] = df["label"].astype(int)
    return df


def run_day() -> tuple[pd.DataFrame, pd.Series]:
    df = day_table()
    rows = []
    for set_name, cols in DAY_SETS.items():
        for name, fn in CANDIDATES.items():
            r = evaluate(df, cols, fn)
            rows.append({"피처": set_name, "모델": name, **r})
            print(f"[하루 전] {set_name} / {name}: 헛경보@80% {r['헛경보@80%']}, @90% {r['헛경보@90%']}, "
                  f"PR-AUC {r['PR-AUC 평균']:.3f}", flush=True)
    res = pd.DataFrame(rows)
    return res, pick(res)


def run_week() -> tuple[pd.DataFrame, pd.DataFrame]:
    w = pd.read_csv(OUT_DATA / f"features_weekly_{REGION}.csv", parse_dates=["date"])
    w["label"] = w["label"].astype(int)
    rows, picks = [], []
    for h in range(2, 8):
        d = w[w["lead"] == h].reset_index(drop=True)
        cols = W_CALENDAR + W_STATE + (W_MID if h >= 5 else [])
        res_h = []
        for name, fn in CANDIDATES.items():
            r = evaluate(d, cols, fn)
            res_h.append({"며칠 뒤": h, "피처": "달력+상태" + ("+중기예보" if h >= 5 else ""), "모델": name, **r})
            print(f"[일주일] {h}일 뒤 / {name}: 헛경보@80% {r['헛경보@80%']}, @90% {r['헛경보@90%']}, "
                  f"PR-AUC {r['PR-AUC 평균']:.3f}", flush=True)
        res_h = pd.DataFrame(res_h)
        rows.append(res_h)
        picks.append(pick(res_h))
    return pd.concat(rows), pd.DataFrame(picks)


def main(part: str = "all"):
    """part: all | week (하루 전 결과가 이미 있으면 week만 다시 실행 가능)"""
    pd.set_option("display.width", 250)
    day_path = OUT_TAB / f"05_search_day_{REGION}.csv"
    if part == "week" and day_path.exists():
        day = pd.read_csv(day_path)
        day_pick = pick(day)
    else:
        day, day_pick = run_day()
        day.round(3).to_csv(day_path, index=False, encoding="utf-8-sig")
    week, week_pick = run_week()
    week.round(3).to_csv(OUT_TAB / f"05_search_week_{REGION}.csv", index=False, encoding="utf-8-sig")
    cols = ["피처", "모델", "헛경보@70%", "헛경보@80%", "헛경보@90%", "헛경보@90% 롤링1", "헛경보@90% 롤링2", "PR-AUC 평균", "선택점수"]
    print("\n[하루 전 상위 8]\n", day.sort_values("선택점수").head(8)[cols].round(3).to_string(index=False))
    print("\n하루 전 선택:", dict(day_pick[["피처", "모델", "헛경보@80%", "헛경보@90%", "PR-AUC 평균"]]))
    print("\n[일주일 리드별 선택]\n", week_pick[["며칠 뒤"] + cols].round(3).to_string(index=False))
    sel = pd.concat([day_pick.to_frame().T.assign(**{"며칠 뒤": 1}), week_pick])
    sel.round(3).to_csv(OUT_TAB / f"05_selected_{REGION}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])
