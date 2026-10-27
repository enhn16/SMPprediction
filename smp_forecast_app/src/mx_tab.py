"""한 달 예측 개선(mx) — 표 형식 모델 걷기 전진 백테스트 (기준일 2024-01-01 ~, 매월 1일 재학습).

학습: 대상일 2023-01-01 이후(팀 규칙) 중 재학습일 이전에 실적이 확정된 행 (date < 재학습일).
목표는 '기준값 대비 비율' (수준 변화에 강하게): 예측 = 기준값 × 비율
  lgb_r : LightGBM(L1), 기준값 = same4 (최근 같은 유형일 4일 평균)
  lgb_a : LightGBM(L1), 기준값 = same4 × √(연료 정규화 계절비)
  et_r  : 엑스트라트리, 기준값 = same4
  lgb_d : LightGBM(L1), 원값 차이 목표 y − same4 (비율 대신)
  *_resid_{tag} : Chronos(tag) 잔차 y − Chronos 를 같은 피처 + Chronos 파생으로 (최종 = Chronos + λ × 잔차, λ는 mx_ens)
2024 예측은 2025 선택용 가중치·λ를 맞추는 데, 2025는 선택, 2026은 시험에만 쓴다.
출력: data/processed/smp_forecast_app/mx/tab_{name}.csv.gz [base_date, H, date, pred_{지표}]
실행: python -m smp_forecast_app.src.mx_tab lgb_r lgb_a et_r lgb_d resid_hc2s resid_dc2 ...   (이미 있으면 건너뜀)
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

from common.config import SEED
from smp_forecast_app.src.mx_features import FEATURES, MX, STATS, load

warnings.filterwarnings("ignore")
START, TRAIN_FROM = "2024-01-01", "2023-01-01"
LGB = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=80, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.7, objective="l1", random_state=SEED, verbose=-1)


def anchor(d, s, kind):
    base = d[f"{s}_same4"].clip(lower=10)
    if kind == "a":
        base = base * np.sqrt(d[f"{s}_seasN"].clip(0.6, 1.6).fillna(1))
    return base


def model(name):
    if name.startswith("et"):
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(300, min_samples_leaf=20, max_features=0.5, random_state=SEED, n_jobs=6)
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**LGB)


def periods(d):
    return list(pd.date_range(START, d["base_date"].max(), freq="MS")) + [pd.Timestamp("2100-01-01")]


def chronos_cols(d, tag):
    c = pd.read_csv(MX / f"chr_{tag}.csv.gz", parse_dates=["base_date", "date"])
    c = c[["base_date", "H"] + [f"pred_{s}" for s in STATS]].rename(columns={f"pred_{s}": f"chr_{s}" for s in STATS})
    return d.merge(c, on=["base_date", "H"], how="left")


def spec(d, s, name):
    """모델 name·지표 s의 (피처 열, 기준값, 방식). 방식: ratio → 기준 × 예측, diff → 기준 + 예측, resid → 예측(잔차) 그대로.
    resid_*는 d에 chr_{s} 열이 있어야 함."""
    cols = list(FEATURES)
    if name.startswith("resid_"):
        d[f"c_ratio_{s}"] = d[f"chr_{s}"] / d[f"{s}_same4"].clip(lower=10)
        return cols + [f"chr_{s}", f"c_ratio_{s}"], d[f"chr_{s}"], "resid"
    if name == "lgb_d":
        return cols, d[f"{s}_same4"], "diff"
    return cols, anchor(d, s, "a" if name == "lgb_a" else "r"), "ratio"


def target(y, base, mode):
    return (y / base).clip(0, 3) if mode == "ratio" else y - base


def finish(p, base, mode):
    return base * p if mode == "ratio" else (base + p if mode == "diff" else p)


def run(name):
    d = load()
    resid = name.startswith("resid_")
    if resid:
        d = chronos_cols(d, name.split("_", 1)[1])
    out = d[["base_date", "H", "date"]].copy()
    ps = periods(d)
    for s in STATS:
        y = d[f"y_{s}"]
        cols, base, mode = spec(d, s, name)
        yt = target(y, base, mode)
        pred = pd.Series(np.nan, index=d.index)
        X = d[cols]
        if name.startswith("et"):
            X = X.fillna(X[d["date"] < START].median())
        for a, b in zip(ps[:-1], ps[1:]):
            tr = (d["date"] >= TRAIN_FROM) & (d["date"] < a) & yt.notna() & base.notna()
            te = (d["base_date"] >= a) & (d["base_date"] < b)
            if te.sum() == 0:
                continue
            p = model(name).fit(X[tr], yt[tr]).predict(X[te])
            pred[te] = p
        # resid_*는 잔차 예측(rhat)만 저장 → 최종 = Chronos + λ × rhat (mx_ens)
        out[f"pred_{s}"] = finish(pred, base, mode)
        print(f"{name} {s} 완료", flush=True)
    keep = out[out["base_date"] >= START]
    keep.round(3).to_csv(MX / f"tab_{name}.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        if (MX / f"tab_{n}.csv.gz").exists():
            print(f"{n}: 이미 있음 → 건너뜀")
            continue
        t0 = time.time()
        run(n)
        print(f"{n} ({time.time()-t0:.0f}s)", flush=True)
