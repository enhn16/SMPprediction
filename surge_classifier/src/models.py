"""③ 급등 경보 모델 선택·채점·4.1 형식 출력 (육지) — 규칙은 PLAN.md 결정 기록 2026-10-26에 사전 고정.

팀 규칙: 데이터 12종, 2023~2026. 급등 = 그날 최고 SMP ≥ 직전 28일 일 최고 중앙값 × 1.25 (전체 날짜)
학습 행 = 2023-01-01부터 전체 날. 2023 결측(날씨 예보·주간전망 2023-12~, Open-Meteo 2024-01~, 연료비 2023-03~)
  → 학습 구간 중앙값 + has_fcst 표시 (LightGBM·EBM은 빈칸 그대로)
후보: 피처 4묶음 × 모델 7개 + 앙상블 4개 (급락 담당 team_select.py와 같은 구성, 회귀 목표 = log(최고/기준))
      + 학습 없는 순부하 지수 (INDEX, 2024·2025 두 해 채점)
선택: 2023~2024→2025, 2024→2025 두 번 채점. 점수 = +20·25·30% 세 기준 PR-AUC 평균(급등일이 7일뿐이라 흔들림 완화),
      나쁜 쪽으로 순위. 1위의 5% 이내면 단순한 쪽 → 평가한 해들의 헛경보@재현율80% 평균 → 점수
문턱: 2023~2024→2025 예측에서 주의 = 급등 90%, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨 (없으면 급등 50%)
확률: 2025 검증 점수에 로지스틱 보정 (Platt)
최종: 2023~2025 재학습 → 2026-01-01 ~ 10-18 한 번 채점 (team_scorecard.py와 같은 지표)
2차(21h): 판단일이 화요일이 아니면 내일 공표 SMP로 확정(최고 ÷ 기준 ≥ 1.25 → 경보). 화요일 판단(수요일 대상)은 미공표로 보고
      2차 모델(1차 피처 + 내일 수요예측) — 같은 절차로 2025 전체 날(미공표로 가정)에서 선택.
출력 (outputs/tables): 03_select_육지_{stage}.csv, surge_육지_{stage}{SUFFIX}.csv, surge_choice_육지_{stage}{SUFFIX}.json,
      surge_score_육지_{stage}{SUFFIX}.csv — 2026-10-28 v3(팀 합의 모델) 교체 후 기록용 이름 (15h = _v1_index, 21h = _v1)
실행: python -m surge_classifier.src.models [15h|21h]   (기본 둘 다)
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import SEED
from smp_forecast_app.src.team_scorecard import drop_stats
from surge_classifier.src.features import (CALENDAR, DEMAND_D, DEMAND_PREV, OPENMETEO, OUTLOOK, SMP_HIST, SUPPLY,
                                           WEATHER, build)
from surge_classifier.src.label_days import OUT_TAB, RATIO

warnings.filterwarnings("ignore")
REGION = "육지"
TEST = ("2026-01-01", "2026-10-18")
TRAIN_SEL = {"2023~2024": ("2023-01-01", "2024-12-31"), "2024": ("2024-01-01", "2024-12-31")}
VALID = ("2025-01-01", "2025-12-31")
TRAIN_FIN = ("2023-01-01", "2025-12-31")
TIE = 0.03
RECALL_WARN = 0.9
STRONG_PREC, STRONG_MIN_TP, STRONG_FALLBACK = 0.5, 3, 0.5
LABELS = ["label_20", "label_25", "label_30"]
F0 = CALENDAR + SMP_HIST + DEMAND_PREV + ["has_fcst"]
CORE = ["kpx_peak", "kpx_peak_rel", "wk_sky_pm", "wk_cloud_mid", "sky_pm", "cloud_mid", "cdd", "hdd", "ta_pm", "ta_am",
        "is_offday", "dem_max_prev_rel", "has_fcst"]
FEATS = {"F0 달력·SMP·수요": F0,
         "F1 +날씨·주간전망": F0 + WEATHER + OUTLOOK,
         "F2 +Open-Meteo·설비·연료비": F0 + WEATHER + OUTLOOK + OPENMETEO + SUPPLY,
         "F3 순부하 핵심": CORE}
SUFFIX = {"15h": "_v1_index", "21h": "_v1"}   # 2026-10-28 v3 교체 후 기록용 파일 이름
TIE_REL = 0.05   # 1위의 5% 이내면 동점권 (급락의 0.03 ≈ PR-AUC 0.6의 5%)

# 학습 없는 순부하 지수 (2026-10-26 2025 검증에서 학습 모델이 모두 기준선에 못 미쳐 추가).
# z = (값 − 평균) / 표준편차, 통계는 Z_STATS 구간에서 한 번 계산해 고정. 빈칸 z = 0 (평균).
Z_STATS = ("2023-12-01", "2024-12-31")
INDEX = {   # 이름: (z로 더할 피처, 평일만 여부) — 평일만이면 휴무일 점수 = 가장 낮게
    "지수: 평일×한낮 운량": (["cloud_mid"], True),
    "지수: 평일×오후 흐림": (["sky_pm"], True),
    "지수: 평일×(운량+흐림+주간전망)": (["cloud_mid", "sky_pm", "kpx_peak"], True),
    "지수: 평일 운량+주간전망": (["wk_cloud_mid", "kpx_peak"], False),
}
INDEX_21H = {"지수: 평일×(운량+흐림+내일 수요예측)": (["cloud_mid", "sky_pm", "dem_max_d"], True)}


def z_stats(df) -> dict:
    s = split(df, Z_STATS)
    return {c: (float(s[c].mean()), float(s[c].std())) for c in
            {c for v in (INDEX | INDEX_21H).values() for c in v[0]}}


def index_score(d, name, stats) -> np.ndarray:
    cols, weekday_only = (INDEX | INDEX_21H)[name]
    z = sum(((d[c] - stats[c][0]) / stats[c][1]).fillna(0) for c in cols)
    if weekday_only:
        z = np.where(d["is_offday"] == 1, -10.0, z)
    return np.asarray(z, dtype=float)
UNPUBLISHED_WEEKDAY = 1   # 화요일 판단 → 수요일 대상은 미공표로 처리 (ROLES.md 4.4)


def data() -> pd.DataFrame:
    df = build(REGION)
    df["has_fcst"] = df["sky_pm"].notna().astype(int)
    df["target"] = np.log(df["ratio"])
    return df


# ---------------------------------------------------------------- 후보 (점수: 클수록 위험)
def imp(m):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


def reg(make):
    def f(tr, te, cols):
        tr = tr.dropna(subset=["target"])
        return make().fit(tr[cols], tr["target"]).predict(te[cols])
    return f


def cls(make):
    def f(tr, te, cols):
        tr = tr.dropna(subset=["label"])
        return make().fit(tr[cols], tr["label"].astype(int)).predict_proba(te[cols])[:, 1]
    return f


BASES = {
    "Ridge": reg(lambda: imp(Ridge(alpha=3))),
    "EBM(회귀)": reg(lambda: ExplainableBoostingRegressor(random_state=SEED, interactions=5, min_samples_leaf=5,
                                                         max_bins=64, n_jobs=1)),
    "랜덤포레스트(회귀)": reg(lambda: imp(RandomForestRegressor(500, min_samples_leaf=10, max_features="sqrt",
                                                          random_state=SEED, n_jobs=4))),
    "LightGBM(회귀)": reg(lambda: LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
                                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
                                               random_state=SEED, verbose=-1)),
    "로지스틱": cls(lambda: imp(LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))),
    "랜덤포레스트(분류)": cls(lambda: imp(RandomForestClassifier(500, min_samples_leaf=5, max_features="sqrt",
                                                           class_weight="balanced_subsample", random_state=SEED, n_jobs=4))),
    "LightGBM(분류)": cls(lambda: LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=25,
                                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
                                                class_weight="balanced", random_state=SEED, verbose=-1)),
}
# 앙상블: 같은 종류끼리 같은 가중치 평균 (회귀 = 예상 log 비율, 분류 = 확률)
ENSEMBLES = {
    "Ridge+EBM": ["Ridge", "EBM(회귀)"],
    "Ridge+EBM+랜덤포레스트": ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)"],
    "Ridge+랜덤포레스트+LightGBM": ["Ridge", "랜덤포레스트(회귀)", "LightGBM(회귀)"],
    "로지스틱+랜덤포레스트+LightGBM (분류)": ["로지스틱", "랜덤포레스트(분류)", "LightGBM(분류)"],
}
TIER = {"Ridge": 1, "로지스틱": 1, "EBM(회귀)": 2, "랜덤포레스트(회귀)": 2, "LightGBM(회귀)": 2, "랜덤포레스트(분류)": 2,
        "LightGBM(분류)": 2, **{k: len(v) + 1 for k, v in ENSEMBLES.items()}}
IS_REG = {k: "분류" not in k and k != "로지스틱" for k in BASES} | {k: "분류" not in k for k in ENSEMBLES}


def predict_all(tr, te, cols, only=None) -> dict:
    need = set(BASES) if only is None else set(ENSEMBLES.get(only, [only]))
    p = {k: f(tr, te, cols) for k, f in BASES.items() if k in need}
    for k, ms in ENSEMBLES.items():
        if all(m in p for m in ms):
            p[k] = np.mean([p[m] for m in ms], axis=0)
    return p


def split(df, rng):
    return df[df["date"].between(*rng)]


# ---------------------------------------------------------------- 문턱·채점
def threshold_for(y, s, target):
    """급등의 target 비율 이상을 잡는 가장 엄격한(높은) 점수 문턱."""
    for t in np.sort(np.unique(s))[::-1]:
        if ((s >= t) & (y == 1)).sum() >= np.ceil(target * y.sum()):
            return float(t)
    return float(s.min())


def strong_threshold(y, s):
    """정밀도 50% 이상·적중 3일 이상인 가장 느슨한 점수 문턱. 없으면 급등 50%를 잡는 문턱."""
    for t in np.sort(np.unique(s)):
        a = s >= t
        tp = int((a & (y == 1)).sum())
        if tp >= STRONG_MIN_TP and tp / a.sum() >= STRONG_PREC:
            return float(t), "정밀도 50%·적중 3일"
    return threshold_for(y, s, STRONG_FALLBACK), "대체: 급등 50%"


def sel_score(d, s):
    """+20·25·30% 세 기준 PR-AUC 평균."""
    return float(np.mean([average_precision_score(d[k].astype(int), s) for k in LABELS]))


def boot(y, s, n=2000):
    rng, bs = np.random.default_rng(SEED), []
    while len(bs) < n:
        i = rng.integers(0, len(y), len(y))
        if 3 <= y[i].sum() < len(i):
            bs.append(drop_stats(y[i], s[i]))
    return pd.DataFrame(bs)


def tier_row(y, a):
    hit, fa = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
    return {"적중": hit, "놓침": int(y.sum()) - hit, "헛경보": fa, "재현율": hit / max(y.sum(), 1),
            "정밀도": hit / max(hit + fa, 1), "헛경보율": fa / max(int((y == 0).sum()), 1)}


def score_rows(name, y, s, prob, warn, strong, n_boot=2000):
    """team_scorecard.py와 같은 지표. 급등일이 3일 미만이면 순위 지표·구간은 빈칸."""
    base = {"구간": name, "일수": len(y), "급등일": int(y.sum())}
    if 0 < y.sum() < len(y):
        base.update({**drop_stats(y, s), "Brier": brier_score_loss(y, prob)})
        # 급등이 드물어 PR-AUC 절대값이 낮음 → 무작위(= 급등 비율) 대비 배수를 함께 (2026-10-27 팀 제안)
        base["무작위 PR-AUC"] = y.mean()
        base["PR-AUC 배수"] = base["PR-AUC"] / y.mean()
        if y.sum() >= 3:
            bs = boot(y, s, n_boot)
            for k in ("PR-AUC", "헛경보@재현율80%", "헛경보@재현율90%"):
                base[f"{k} 95%하한"], base[f"{k} 95%상한"] = bs[k].quantile(.025), bs[k].quantile(.975)
            base["PR-AUC 배수 95%하한"] = base["PR-AUC 95%하한"] / y.mean()
            base["PR-AUC 배수 95%상한"] = base["PR-AUC 95%상한"] / y.mean()
    return [{**base, "경보": "주의 이상", **tier_row(y, s >= warn)},
            {**base, "경보": "경보", **tier_row(y, s >= strong)}]


# ---------------------------------------------------------------- 선택
def select(df, feats, extra=(), indices=()) -> tuple[pd.DataFrame, dict]:
    """학습 모델 = 2023~2024·2024 학습 → 2025 두 번 채점, 나쁜 쪽. 학습 없는 지수 = 2024·2025 두 해 채점, 나쁜 쪽."""
    va = split(df, VALID)
    rows, val_pred = [], {"지수": {}}
    t0 = time.time()
    stats = z_stats(df)
    y = va["label"].astype(int).values
    for name in indices:
        d24 = split(df, ("2024-01-01", "2024-12-31"))
        sc = {"2024": sel_score(d24, index_score(d24, name, stats)), "2025": sel_score(va, index_score(va, name, stats))}
        sv = index_score(va, name, stats)
        val_pred["지수"][name] = sv
        st = drop_stats(y, sv)
        rows.append({"피처": "지수", "모델": name, "단계": 0, "평가": "2024·2025 (학습 없음)",
                     "점수 A": sc["2024"], "점수 B": sc["2025"], "점수 나쁜쪽": min(sc.values()),
                     "PR-AUC(+25%)": st["PR-AUC"], "ROC-AUC": st["ROC-AUC"],
                     "헛경보@80%": st["헛경보@재현율80%"], "헛경보@90%": st["헛경보@재현율90%"],
                     "헛경보@80% 2024": drop_stats(d24["label"].astype(int).values,
                                                index_score(d24, name, stats))["헛경보@재현율80%"]})
    for fname, cols in feats.items():
        cols = cols + list(extra)
        res = {w: predict_all(split(df, rng), va, cols) for w, rng in TRAIN_SEL.items()}
        val_pred[fname] = res["2023~2024"]
        print(f"  {fname}: {time.time() - t0:.0f}초", flush=True)
        for m in res["2023~2024"]:
            sc = {w: sel_score(va, res[w][m]) for w in TRAIN_SEL}
            st = drop_stats(y, res["2023~2024"][m])
            rows.append({"피처": fname, "모델": m, "단계": TIER[m], "평가": "2025 (2023~2024·2024 학습)",
                         "점수 A": sc["2023~2024"], "점수 B": sc["2024"],
                         "점수 나쁜쪽": min(sc.values()), "PR-AUC(+25%)": st["PR-AUC"], "ROC-AUC": st["ROC-AUC"],
                         "헛경보@80%": st["헛경보@재현율80%"], "헛경보@90%": st["헛경보@재현율90%"]})
    sel = pd.DataFrame(rows).sort_values("점수 나쁜쪽", ascending=False).reset_index(drop=True)
    best = sel["점수 나쁜쪽"].iloc[0]
    sel["동점권"] = sel["점수 나쁜쪽"] >= best * (1 - TIE_REL)
    # 헛경보@80% 평균: 평가한 해 모두 (지수 = 2024·2025, 모델 = 2025)
    sel["헛경보@80% 평균"] = sel[["헛경보@80%", "헛경보@80% 2024"]].mean(axis=1) if "헛경보@80% 2024" in sel else sel["헛경보@80%"]
    pick = sel[sel["동점권"]].sort_values(["단계", "헛경보@80% 평균", "점수 나쁜쪽"], ascending=[True, True, False]).iloc[0]
    sel["선택"] = (sel["피처"] == pick["피처"]) & (sel["모델"] == pick["모델"])
    return sel, val_pred


def baselines(df) -> pd.DataFrame:
    """단순 규칙 기준선 (2025 검증)."""
    va, tr = split(df, VALID), split(df, TRAIN_SEL["2023~2024"])
    clim = tr.groupby([tr["date"].dt.month, "is_offday"])["label"].mean()
    rules = {"평일": 1 - va["is_offday"],
             "평일 × 흐림 예보": va["wk_sky_pm"].fillna(0),
             "지속성 (직전 7일 최고 ÷ 기준)": va["max7_ratio"],
             "기후값 (월 × 휴무일 급등 비율)": [clim.get((d.month, o), 0) for d, o in zip(va["date"], va["is_offday"])]}
    y = va["label"].astype(int).values
    rows = []
    for k, s in rules.items():
        s = np.asarray(s, dtype=float)
        rows.append({"기준선": k, "점수(+20·25·30% PR-AUC 평균)": sel_score(va, s), **drop_stats(y, s)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 실행
def run(stage: str, df: pd.DataFrame | None = None):
    t0 = time.time()
    df = data() if df is None else df
    extra = DEMAND_D if stage == "21h" else ()
    print(f"[{stage}] 피처 {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}", flush=True)

    # 1) 선택 (2025만)
    indices = list(INDEX) + (list(INDEX_21H) if stage == "21h" else [])
    sel, val_pred = select(df, FEATS, extra, indices)
    sel.round(4).to_csv(OUT_TAB / f"03_select_{REGION}_{stage}.csv", index=False, encoding="utf-8-sig")
    pick = sel[sel["선택"]].iloc[0]
    fname, mname = pick["피처"], pick["모델"]
    is_index = fname == "지수"
    stats = z_stats(df)
    cols = (INDEX | INDEX_21H)[mname][0] + ["is_offday"] if is_index else FEATS[fname] + list(extra)
    pd.set_option("display.width", 250)
    print(sel.head(15).round(3).to_string(index=False))
    print(f"\n선택: {fname} / {mname} (나쁜쪽 {pick['점수 나쁜쪽']:.3f}, 1위 {sel['점수 나쁜쪽'].iloc[0]:.3f})")
    bl = baselines(df)
    if stage == "15h":
        bl.round(4).to_csv(OUT_TAB / f"03_baselines_{REGION}.csv", index=False, encoding="utf-8-sig")
        print(bl.round(3).to_string(index=False))

    # 2) 문턱·확률 (2025 검증 예측)
    va = split(df, VALID)
    yv, sv = va["label"].astype(int).values, val_pred[fname][mname]
    warn = threshold_for(yv, sv, RECALL_WARN)
    strong, strong_rule = strong_threshold(yv, sv)
    cal = LogisticRegression(C=1e6).fit(sv.reshape(-1, 1), yv)
    to_prob = lambda s: cal.predict_proba(np.asarray(s, dtype=float).reshape(-1, 1))[:, 1]
    if os.environ.get("SURGE_NO_TEST") == "1":   # 선택·문턱만 확인 (2026 채점 안 함)
        print(f"[{stage}] 문턱(내부 점수): 주의 {warn:.4f}, 경보 {strong:.4f} ({strong_rule}) — 2026 채점 생략")
        return

    # 3) 2023~2025 재학습 → 2026 한 번 채점
    te = split(df, ("2026-01-01", "2099-12-31"))
    if is_index:   # 학습 없음: 같은 식·같은 z 통계 그대로
        st_ = index_score(te, mname, stats)
    else:
        st_ = predict_all(split(df, TRAIN_FIN), te, cols, only=mname)[mname]
    both = pd.concat([va.assign(s=sv), te.assign(s=st_)], ignore_index=True)
    both["base_date"] = both["date"] - pd.Timedelta(days=1)
    both["prob"] = to_prob(both["s"])
    both["grade"] = np.where(both["s"] >= strong, "경보", np.where(both["s"] >= warn, "주의", "없음"))
    # score 열: 지수 = 지수값, 회귀 = 예상 최고 ÷ 기준, 분류 = 빈칸
    both["mscore"] = both["s"] if is_index else (np.exp(both["s"]) if IS_REG[mname] else np.nan)

    if stage == "21h":
        # 공표일 = 화요일 판단이 아닌 날 → 공표 최고가로 확정 (score = 실제 최고 ÷ 기준)
        pub = (both["base_date"].dt.dayofweek != UNPUBLISHED_WEEKDAY) & both["ratio"].notna()
        both["published"] = pub
        both.loc[pub, "prob"] = (both.loc[pub, "ratio"] >= RATIO).astype(float)
        both.loc[pub, "grade"] = np.where(both.loc[pub, "ratio"] >= RATIO, "경보", "없음")
        both.loc[pub, "mscore"] = both.loc[pub, "ratio"]
        both["s_eval"] = np.where(pub, np.where(both["ratio"] >= RATIO, np.inf, -np.inf), both["s"])
    else:
        both["s_eval"] = both["s"]

    # 성적표
    def sub(mask, name, s_col="s_eval"):
        g = both[mask & both["label"].notna()]
        y, s = g["label"].astype(int).values, g[s_col].values
        s = np.where(np.isinf(s), np.sign(s) * 1e6, s)
        return score_rows(name, y, s, g["prob"].values, warn if s_col == "s" else min(warn, 1e5),
                          strong if s_col == "s" else min(strong, 1e5))
    in_va = both["date"].between(*VALID)
    in_te = both["date"].between(*TEST)
    if stage == "15h":
        rows = sub(in_va, "2025 검증") + sub(in_te, f"2026 시험 {TEST[0]} ~ {TEST[1]}")
    else:
        # 21시: 공표된 날은 내일 SMP(= 레이블을 만든 값)를 읽어 확정 → 예측이 아니라 확인이라 항상 맞음.
        # 예측 성능은 '미공표일 예측분'과 '2차 모델을 전체 날에 쓴 경우'로만 본다.
        wed = both["base_date"].dt.dayofweek == UNPUBLISHED_WEEKDAY
        rows = sub(in_va & wed, "2025 검증 · 예측분 (미공표 = 화요일 판단)", "s") + \
            sub(in_te & wed, f"2026 시험 {TEST[0]} ~ {TEST[1]} · 예측분 (미공표 = 화요일 판단)", "s") + \
            sub(in_va, "2025 검증 · 2차 모델을 전체 날에 쓴 경우 (예측 성능)", "s") + \
            sub(in_te, f"2026 시험 {TEST[0]} ~ {TEST[1]} · 2차 모델을 전체 날에 쓴 경우 (예측 성능)", "s") + \
            sub(in_va & ~wed, "2025 검증 · 공표 확인분 (예측 아님, 공표 SMP로 확정)") + \
            sub(in_te & ~wed, f"2026 시험 {TEST[0]} ~ {TEST[1]} · 공표 확인분 (예측 아님, 공표 SMP로 확정)") + \
            sub(in_va, "2025 검증 · 운영 전체 (확인 + 예측 섞임, 예측 성능 아님)") + \
            sub(in_te, f"2026 시험 {TEST[0]} ~ {TEST[1]} · 운영 전체 (확인 + 예측 섞임, 예측 성능 아님)")
    score = pd.DataFrame(rows)
    score.insert(0, "모델", f"{mname} ({fname})")
    score.round(4).to_csv(OUT_TAB / f"surge_score_{REGION}_{stage}{SUFFIX[stage]}.csv", index=False, encoding="utf-8-sig")

    # 4) 4.1 형식 일별 파일 (2025 = 2023~2024 학습 예측, 2026~ = 2023~2025 학습 예측)
    out = pd.DataFrame({"base_date": both["base_date"].dt.strftime("%Y-%m-%d"),
                        "date": both["date"].dt.strftime("%Y-%m-%d"), "stage": stage, "grade": both["grade"],
                        "prob": both["prob"].round(4), "score": both["mscore"].round(4),
                        "label": both["label"].map(lambda v: "" if pd.isna(v) else str(int(v)))})
    out.to_csv(OUT_TAB / f"surge_{REGION}_{stage}{SUFFIX[stage]}.csv", index=False, encoding="utf-8-sig")
    unit = (lambda s: round(float(np.exp(s)), 4)) if (not is_index and IS_REG[mname]) else (lambda s: round(s, 4))
    if is_index:
        icols, wk_only = (INDEX | INDEX_21H)[mname]
        how = {"점수": f"{' + '.join(f'z({c})' for c in icols)}{', 휴무일 = -10 (평일만)' if wk_only else ''}. "
                       "z = (값 − 평균) / 표준편차, 빈칸 z = 0",
               "z 통계": {c: {"평균": stats[c][0], "표준편차": stats[c][1]} for c in icols},
               "z 통계 구간": f"{Z_STATS[0]} ~ {Z_STATS[1]} (고정)",
               "학습": {"선택": "학습 없음 — 2024·2025 두 해 채점, 나쁜 쪽", "최종": "학습 없음 (같은 식)"}}
    else:
        how = {"점수": "예상 그날 최고 SMP ÷ 직전 28일 기준값 (클수록 위험)" if IS_REG[mname] else "급등 확률 점수, 클수록 위험",
               "학습": {"선택": "2023~2024 → 2025 (2024 → 2025 함께, 나쁜 쪽)", "최종": f"{TRAIN_FIN[0]} ~ {TRAIN_FIN[1]}"}}
    choice = {"stage": "15h (전날 15시)" if stage == "15h" else "21h (전날 21시)", "모델": mname, "피처 묶음": fname,
              "피처": cols, **how,
              "문턱": {"주의": unit(warn), "경보": unit(strong)},
              "문턱 (모델 내부 점수)": {"주의": warn, "경보": strong},
              "확률식": {"형태": "1/(1+exp(-(a·s+b))), s = 모델 내부 점수 (2025 검증에 맞춤)",
                      "a": float(cal.coef_[0][0]), "b": float(cal.intercept_[0])},
              "선택 절차": "PLAN.md 결정 기록 2026-10-26 (+20·25·30% PR-AUC 평균의 나쁜 쪽, 1위 5% 이내면 단순한 쪽 → 평가한 해들의 헛경보@재현율80% 평균)",
              "문턱 규칙": f"주의 = 2025 급등 90%, 경보 = {strong_rule}",
              "2023 결측": "날씨 예보·주간전망·Open-Meteo·연료비 = 학습 구간 중앙값 + has_fcst (LightGBM·EBM은 빈칸 그대로)"}
    if stage == "21h":
        choice["주의"] = ("공표된 날의 21시 판정은 예측이 아니라 확인이다. SMP는 하루 전 확정 가격이라 공표값 = 실제값 = 레이블의 원천이므로 "
                        "공표 확인분은 항상 맞는다. 예측 성능은 성적표의 '예측분'·'2차 모델을 전체 날에 쓴 경우' 행으로만 본다.")
        choice["2차 규칙"] = ("판단일이 화요일이 아니면 내일 공표 SMP로 확정 (최고 ÷ 기준 ≥ 1.25 → 경보, prob 1/0, score = 실제 비율). "
                          "화요일 판단(수요일 대상)은 미공표로 보고 2차 모델(1차 피처 + 내일 수요예측). "
                          "운영: 공표 SMP 있으면 확정 → 없고 내일 수요예측 있으면 2차 모델 → 둘 다 없으면 1차 모델 값")
    (OUT_TAB / f"surge_choice_{REGION}_{stage}{SUFFIX[stage]}.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1),
                                                                 encoding="utf-8")
    show = ["구간", "경보", "일수", "급등일", "PR-AUC", "무작위 PR-AUC", "PR-AUC 배수", "PR-AUC 배수 95%하한",
            "PR-AUC 배수 95%상한", "ROC-AUC", "헛경보@재현율80%",
            "헛경보@재현율90%", "Brier", "적중", "놓침", "헛경보", "재현율", "정밀도"]
    print(f"문턱: 주의 {unit(warn)}, 경보 {unit(strong)} ({strong_rule})")
    print(score[[c for c in show if c in score]].round(3).to_string(index=False))
    m26 = both[in_te & (both["label"] == 1)]
    print("2026 급등일 판정:", dict(zip(m26["date"].dt.strftime("%m-%d"), m26["grade"])))
    print(f"[{stage}] 총 {time.time() - t0:.0f}초")


def main(stage: str | None = None):
    df = data()
    for s in ([stage] if stage else ["15h", "21h"]):
        run(s, df)


if __name__ == "__main__":
    main(*sys.argv[1:2])
