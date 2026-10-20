"""하루 전 모델 앙상블 시험 (육지, 내일 수요예측 없음 + 주간전망 = 현재 최종 조건).

섞는 방법: 각 모델 점수(클수록 급락 위험)를 '학습 기간 예측'의 평균·표준편차로 표준화한 뒤 가중 평균.
→ 회귀(예측 최저 SMP)와 분류(확률)처럼 척도가 다른 모델도 섞을 수 있고, 시험 기간 정보는 쓰지 않는다.
평가: 롤링 2회 합산 헛경보@80·90%(구간별 @90% 포함), 검증 기간 적중·헛경보(롤링에서 정한 80%·90% 기준), 9/25·10/5.

실행: python -m drop_classifier.src.experiments_ensemble
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

from common.config import ROOT, SEED
from drop_classifier.src.experiments import false_alarms_at
from drop_classifier.src.final import threshold_for
from drop_classifier.src.models import FOLDS, split
from drop_classifier.src.search import CANDIDATES, DAY_SETS, day_table, reg

warnings.filterwarnings("ignore")
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
COLS = DAY_SETS["기본+주간전망"]

BASE = {
    "Ridge": CANDIDATES["Ridge 회귀(α=3)"],
    "의사결정나무": reg(lambda: make_pipeline(SimpleImputer(strategy="median"),
                                         DecisionTreeRegressor(max_depth=4, min_samples_leaf=15, random_state=SEED))),
    "랜덤포레스트": reg(lambda: make_pipeline(SimpleImputer(strategy="median"),
                                         RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                                               random_state=SEED, n_jobs=1))),
    "LightGBM": CANDIDATES["LightGBM 회귀"],
    "CatBoost": CANDIDATES["CatBoost 회귀"],
    "EBM": CANDIDATES["EBM 회귀"],
    "로지스틱": CANDIDATES["로지스틱(급락)"],
}
ENSEMBLES = {
    "Ridge 단독 (현재)": {"Ridge": 1},
    "의사결정나무 단독": {"의사결정나무": 1},
    "랜덤포레스트 단독": {"랜덤포레스트": 1},
    "Ridge + 의사결정나무": {"Ridge": 1, "의사결정나무": 1},
    "Ridge + 랜덤포레스트": {"Ridge": 1, "랜덤포레스트": 1},
    "Ridge + LightGBM": {"Ridge": 1, "LightGBM": 1},
    "Ridge + CatBoost": {"Ridge": 1, "CatBoost": 1},
    "Ridge + EBM": {"Ridge": 1, "EBM": 1},
    "Ridge + 로지스틱": {"Ridge": 1, "로지스틱": 1},
    "Ridge 0.7 + 랜덤포레스트 0.3": {"Ridge": 0.7, "랜덤포레스트": 0.3},
    "Ridge 0.7 + CatBoost 0.3": {"Ridge": 0.7, "CatBoost": 0.3},
    "Ridge + 랜덤포레스트 + 로지스틱": {"Ridge": 1, "랜덤포레스트": 1, "로지스틱": 1},
    "Ridge + 나무 3종(RF·LGBM·CatBoost)": {"Ridge": 1, "랜덤포레스트": 1, "LightGBM": 1, "CatBoost": 1},
}


def member_scores(df):
    """모델별·구간별 (학습 예측, 시험 예측)을 한 번씩만 계산해 둔다."""
    out = {}
    for name, fn in BASE.items():
        for fold, (tr_rng, te_rng) in FOLDS.items():
            tr, te = split(df, tr_rng), split(df, te_rng)
            both = pd.concat([tr, te])
            p = fn(tr, both, COLS)  # 학습 구간으로만 학습, 학습·시험 행 모두 예측
            out[(name, fold)] = (p[:len(tr)], p[len(tr):])
        print(f"  {name} 완료", flush=True)
    return out


def combine(scores, weights, fold):
    z = 0
    for name, w in weights.items():
        ptr, pte = scores[(name, fold)]
        z = z + w * (pte - ptr.mean()) / (ptr.std() + 1e-9)
    return z / sum(weights.values())


def main():
    df = day_table()
    scores = member_scores(df)
    rows = []
    for ens, w in ENSEMBLES.items():
        parts = {f: (split(df, FOLDS[f][1]), combine(scores, w, f)) for f in FOLDS}
        y = np.concatenate([parts[f][0]["label"].values for f in ("롤링1", "롤링2")])
        p = np.concatenate([parts[f][1] for f in ("롤링1", "롤링2")])
        te, pv = parts["검증"]
        r = {"조합": ens}
        for t in (0.8, 0.9):
            r[f"헛경보@{int(t*100)}%"] = false_alarms_at(y, p, t)[0]
        for f in ("롤링1", "롤링2"):
            r[f"@90% {f}"] = false_alarms_at(parts[f][0]["label"].values, parts[f][1], 0.9)[0]
        for t in (0.8, 0.9):
            a = pv >= threshold_for(y, p, t)
            r[f"검증{int(t*100)} 적중"] = int((a & (te["label"].values == 1)).sum())
            r[f"검증{int(t*100)} 헛경보"] = int((a & (te["label"].values == 0)).sum())
        a80 = pv >= threshold_for(y, p, 0.8)
        for d, k in (("2026-09-25", "9/25"), ("2026-10-05", "10/5")):
            i = list(te["date"]).index(pd.Timestamp(d))
            r[k] = ("경보" if a80[i] else "없음")
        rows.append(r)
        print(r, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_TAB / "07_ensemble_육지.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
