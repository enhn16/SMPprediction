"""제주 일주일 예측 — 표 형식 모델 (H마다 따로 학습). 피처 = jz_features.FEATURES + 육지 일주일 예측(L_*).

모델 (이름: 재학습 주기) — 육지 wz_tab과 같은 설정
  jlgb LightGBM(월) · jlgbl1 LightGBM 절대오차(월) · jlgbq LightGBM 분위수 0.1/0.9(월, 범위용) · jxgb XGBoost(월)
  jridge Ridge(월) · jcat CatBoost(분기) · jrf 랜덤포레스트(분기) · jet 엑스트라트리(분기) · jebm EBM(분기)
  jmlp MLP(분기) · jtabpfn TabPFN(분기, 최근 4천 행)
  기준: jnaive7 지난주 같은 시간, jsame 최근 같은 유형일 같은 시간, land 육지 최종 예측 그대로, land_c2s 육지 Chronos-2 small
실행: python -m smp_forecast_app.src.jz_tab base jlgb jxgb ...
"""
import sys
import time
import warnings

import pandas as pd

from common.config import SEED
from smp_forecast_app.src.jz_common import HS, OUT, feats, rolling_fit, save
from smp_forecast_app.src.jz_features import FEATURES
from smp_forecast_app.src.model import PARAMS

warnings.filterwarnings("ignore")

COLS = FEATURES + ["L_c2s", "L_base", "L_spread"]
NJ = 3   # 프로세스당 스레드 (여러 모델을 병렬 프로세스로 돌릴 때 코어 과다 사용 방지)
LGBP = dict(PARAMS, n_jobs=NJ)


def table(H):
    f = feats(H)
    f["L_spread"] = f["L_qhi"] - f["L_qlo"]
    return f


def _prep(tr, te, onehot=True):
    X, Xt = tr[COLS].copy(), te[COLS].copy()
    med = X.median()
    X, Xt = X.fillna(med), Xt.fillna(med)
    if onehot:
        for k in ("hour", "dow", "month", "daytype"):
            X = pd.concat([X, pd.get_dummies(tr[k], prefix=k, dtype=float)], axis=1)
            Xt = pd.concat([Xt, pd.get_dummies(te[k], prefix=k, dtype=float)], axis=1)
        Xt = Xt.reindex(columns=X.columns, fill_value=0)
    X, Xt = X.fillna(0), Xt.fillna(0)
    mu, sd = X.mean(), X.std().replace(0, 1).fillna(1)
    return (X - mu) / sd, (Xt - mu) / sd


def m_lgb(tr, te, obj="regression"):
    import lightgbm as lgb
    return lgb.LGBMRegressor(**dict(LGBP, objective=obj)).fit(tr[COLS], tr["y"]).predict(te[COLS])


def m_lgbq(tr, te):
    import lightgbm as lgb
    p = m_lgb(tr, te)
    lo = lgb.LGBMRegressor(**dict(LGBP, objective="quantile", alpha=0.1)).fit(tr[COLS], tr["y"]).predict(te[COLS])
    hi = lgb.LGBMRegressor(**dict(LGBP, objective="quantile", alpha=0.9)).fit(tr[COLS], tr["y"]).predict(te[COLS])
    return p, lo, hi


def m_xgb(tr, te):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", random_state=SEED, n_jobs=NJ).fit(tr[COLS], tr["y"]).predict(te[COLS])


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
                                        n_jobs=NJ).fit(X, tr["y"]).predict(Xt)


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


MODELS = {"jlgb": (m_lgb, "MS"), "jlgbl1": (lambda tr, te: m_lgb(tr, te, "l1"), "MS"), "jlgbq": (m_lgbq, "MS"),
          "jxgb": (m_xgb, "MS"), "jridge": (m_ridge, "MS"), "jcat": (m_cat, "QS"), "jrf": (m_rf, "QS"),
          "jet": (m_et, "QS"), "jebm": (m_ebm, "QS"), "jmlp": (m_mlp, "QS"), "jtabpfn": (m_tabpfn, "QS")}


def baselines():
    cols = {"jnaive7": "s_d7", "jsame": "s_same_h", "land": "L_final", "land_c2s": "L_c2s"}
    out = {k: [] for k in cols}
    for H in HS:
        df = table(H)
        df = df[df["date"] >= "2025-01-01"]
        for k, c in cols.items():
            out[k].append(df[["date", "hour"]].assign(pred=df[c].values, H=H))
    for k, parts in out.items():
        save(k, pd.concat(parts).dropna(subset=["pred"]))
        print(f"{k} 저장", flush=True)


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
            out.append(rolling_fit(table(H), H, freq, fn))
            print(f"{name} H={H} ({time.time()-t0:.0f}s)", flush=True)
        save(name, pd.concat(out))
        print(f"{name} 완료 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", "jlgb", "jlgbl1", "jlgbq", "jxgb", "jridge"])
