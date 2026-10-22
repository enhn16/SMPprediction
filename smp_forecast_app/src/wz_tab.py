"""일주일 예측 — 표 형식 모델 (며칠 뒤 H마다 따로 학습). 피처는 features.py '기본+전망·날씨'(15시 판단).

모델 (이름: 재학습 주기)
  lgb LightGBM(월) · lgbq LightGBM 분위수 0.1/0.9(월, 범위용) · xgb XGBoost(월) · ridge Ridge(월)
  cat CatBoost(분기) · rf 랜덤포레스트(분기) · et 엑스트라트리(분기) · ebm EBM(분기) · mlp MLP(분기)
  tabpfn TabPFN(분기, 최근 4천 행 — CPU에서 느려서 줄임)
  기준: naive7 지난주 같은 요일·시간(s_d7), same 최근 같은 유형일 같은 시간(s_same_h)
실행: python -m smp_forecast_app.src.wz_tab lgb xgb ridge ...   (여러 프로세스로 나눠 병렬 실행 가능)
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

from common.config import SEED
from smp_forecast_app.src.model import PARAMS
from smp_forecast_app.src.wz_common import COLS, HS, OUT, feats, rolling_fit, save

warnings.filterwarnings("ignore")


def _prep(tr, te, onehot=True):
    X, Xt = tr[COLS].copy(), te[COLS].copy()
    med = X.median()
    X, Xt = X.fillna(med), Xt.fillna(med)
    if onehot:
        for k in ("hour", "dow", "month", "daytype"):
            X = pd.concat([X, pd.get_dummies(tr[k], prefix=k, dtype=float)], axis=1)
            Xt = pd.concat([Xt, pd.get_dummies(te[k], prefix=k, dtype=float)], axis=1)
        Xt = Xt.reindex(columns=X.columns, fill_value=0)
    X, Xt = X.fillna(0), Xt.fillna(0)          # 학습 구간 전체가 빈 열(예: H≥4 기온 예보)은 0
    mu, sd = X.mean(), X.std().replace(0, 1).fillna(1)
    return (X - mu) / sd, (Xt - mu) / sd


def m_lgb(tr, te):
    import lightgbm as lgb
    return lgb.LGBMRegressor(**dict(PARAMS, objective="regression")).fit(tr[COLS], tr["y"]).predict(te[COLS])


def m_lgbq(tr, te):
    import lightgbm as lgb
    p = lgb.LGBMRegressor(**dict(PARAMS, objective="regression")).fit(tr[COLS], tr["y"]).predict(te[COLS])
    lo = lgb.LGBMRegressor(**dict(PARAMS, objective="quantile", alpha=0.1)).fit(tr[COLS], tr["y"]).predict(te[COLS])
    hi = lgb.LGBMRegressor(**dict(PARAMS, objective="quantile", alpha=0.9)).fit(tr[COLS], tr["y"]).predict(te[COLS])
    return p, lo, hi


def m_xgb(tr, te):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", random_state=SEED, n_jobs=-1).fit(tr[COLS], tr["y"]).predict(te[COLS])


def m_ridge(tr, te):
    from sklearn.linear_model import Ridge
    X, Xt = _prep(tr, te)
    return Ridge(alpha=10).fit(X, tr["y"]).predict(Xt)


def m_cat(tr, te):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=800, depth=7, learning_rate=0.05, random_seed=SEED, verbose=0,
                             thread_count=-1).fit(tr[COLS], tr["y"]).predict(te[COLS])


def m_rf(tr, te):
    from sklearn.ensemble import RandomForestRegressor
    X, Xt = _prep(tr, te, onehot=False)
    return RandomForestRegressor(300, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=-1).fit(X, tr["y"]).predict(Xt)


def m_et(tr, te):
    from sklearn.ensemble import ExtraTreesRegressor
    X, Xt = _prep(tr, te, onehot=False)
    return ExtraTreesRegressor(300, min_samples_leaf=5, max_features=0.5, random_state=SEED, n_jobs=-1).fit(X, tr["y"]).predict(Xt)


def m_ebm(tr, te):
    from interpret.glassbox import ExplainableBoostingRegressor
    X, Xt = _prep(tr, te, onehot=False)
    return ExplainableBoostingRegressor(random_state=SEED, interactions=10, max_bins=128, outer_bags=4,
                                        n_jobs=6).fit(X, tr["y"]).predict(Xt)


def m_mlp(tr, te):
    from sklearn.neural_network import MLPRegressor
    X, Xt = _prep(tr, te)
    mu, sd = tr["y"].mean(), tr["y"].std()
    m = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, learning_rate_init=1e-3, max_iter=200, early_stopping=True,
                     n_iter_no_change=10, random_state=SEED).fit(X, (tr["y"] - mu) / sd)
    return m.predict(Xt) * sd + mu


def m_tabpfn(tr, te):
    from tabpfn import TabPFNRegressor
    tr = tr.tail(4000)
    return TabPFNRegressor(device="cpu", random_state=SEED, ignore_pretraining_limits=True).fit(tr[COLS], tr["y"]).predict(te[COLS])


MODELS = {"lgb": (m_lgb, "MS"), "lgbq": (m_lgbq, "MS"), "xgb": (m_xgb, "MS"), "ridge": (m_ridge, "MS"),
          "cat": (m_cat, "QS"), "rf": (m_rf, "QS"), "et": (m_et, "QS"), "ebm": (m_ebm, "QS"), "mlp": (m_mlp, "QS"),
          "tabpfn": (m_tabpfn, "QS")}


def baselines():
    for name, col in (("naive7", "s_d7"), ("same", "s_same_h")):
        out = []
        for H in HS:
            df = feats(H)
            df = df[df["date"] >= "2025-01-01"]
            out.append(df[["date", "hour"]].assign(pred=df[col].values, H=H))
        save(name, pd.concat(out))
        print(f"{name} 저장", flush=True)


def main(names):
    for name in names:
        if name == "base":
            baselines()
            continue
        if (OUT / f"{name}.csv.gz").exists():
            print(f"{name}: 이미 있음 → 건너뜀", flush=True)
            continue
        fn, freq = MODELS[name]
        t0, out = time.time(), []
        for H in HS:
            out.append(rolling_fit(feats(H), H, freq, fn))
            print(f"{name} H={H} ({time.time()-t0:.0f}s)", flush=True)
        save(name, pd.concat(out))
        print(f"{name} 완료 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", "lgb", "lgbq", "xgb", "ridge"])
