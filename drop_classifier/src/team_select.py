"""팀 공통 규칙으로 1차(15시) 급락 모델 재선택 (육지) — 규칙은 PLAN.md 결정 기록 2026-10-26에 사전 고정.

팀 규칙: 데이터 12종, 2023~2026, 급락 = 휴일·주말만 (11~15시 최저 30원 미만)
학습 행 = 2023-01-01부터 전체 날(평일 포함), 선택·문턱·채점 = 휴일·주말만
2023 결측: 날씨 예보·주간전망(2023-12~)·연료비(2023-03~) → 학습 구간 중앙값 + has_fcst 표시 (부스팅은 빈칸 그대로)
후보: 피처 2묶음 × 모델 11개 (규제 강하게 고정, 신경망·Chronos·Open-Meteo 제외)
선택: 2023~2024→2025, 2024→2025 두 번 채점, 휴일·주말 PR-AUC 나쁜 쪽으로 순위.
      1위와 0.03 이내면 단순한 쪽 → 헛경보@재현율80% → PR-AUC
문턱: 2023~2024→2025 휴일·주말 예측에서 주의 = 급락 90%, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨
최종: 2023~2025 재학습 → 2026-01-01 ~ 10-18 휴일·주말 한 번 채점 (team_scorecard.py와 같은 지표)
출력 (outputs/tables): 18_team_select_육지.csv, 18_team_select_{daily,choice,score}_육지.* (비교용 — 공식 결과는 team_final.py)
실행: python -m drop_classifier.src.team_select
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import SEED
from drop_classifier.src.fair_select import fcst15
from drop_classifier.src.features import build
from drop_classifier.src.final import threshold_for
from drop_classifier.src.fuel_level import BASE, FUEL, fuel_features
from drop_classifier.src.outlook_features import attach as outlook_attach
from drop_classifier.src.supply_level import SOLAR, supply_features
from drop_classifier.src.two_tier import OUT_TAB, strong_threshold
from smp_forecast_app.src.team_scorecard import drop_stats

warnings.filterwarnings("ignore")
REGION = "육지"
START = "2023-01-01"
TEST = ("2026-01-01", "2026-10-18")
TRAIN_SEL = {"2023~2024": ("2023-01-01", "2024-12-31"), "2024": ("2024-01-01", "2024-12-31")}
VALID = ("2025-01-01", "2025-12-31")
TRAIN_FIN = ("2023-01-01", "2025-12-31")
CLIP = (0, 150)
TIE = 0.03
RECALL_WARN = 0.9
F0 = BASE + SOLAR + ["has_fcst"]
FEATS = {"F0 현 피처": F0, "F1 현 피처+연료비": F0 + FUEL}
BASELINE = {"PR-AUC": 0.6207, "헛경보@재현율80%": 20, "헛경보@재현율90%": 28}   # T2_team_drop.csv 현 모델


def data23() -> pd.DataFrame:
    df = build(START)
    df = pd.concat([df.reset_index(drop=True), outlook_attach(df["date"], df["date"] - pd.Timedelta(days=1))], axis=1)
    df = df.merge(fcst15(), on="date", how="left")
    df = pd.concat([df, supply_features(df["date"])[SOLAR], fuel_features(df["date"])], axis=1)
    df["has_fcst"] = df["rf_sky_pm"].notna().astype(int)
    df["label"] = df["label"].astype(int)
    df["offday"] = df["day_type"] != "평일"
    return df


# ---------------------------------------------------------------- 후보 (점수: 클수록 위험)
def imp(m):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


def reg(make):
    def f(tr, te, cols):
        return -make().fit(tr[cols], tr["min_smp_11_15"].clip(*CLIP)).predict(te[cols])
    return f


def cls(make):
    def f(tr, te, cols):
        return make().fit(tr[cols], tr["label"]).predict_proba(te[cols])[:, 1]
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
# 앙상블: 같은 종류끼리 같은 가중치 평균 (회귀 = 예상 원, 분류 = 확률). 순위 평균은 운영에서 재현이 안 돼 쓰지 않음
ENSEMBLES = {
    "Ridge+EBM": ["Ridge", "EBM(회귀)"],
    "Ridge+EBM+랜덤포레스트 (현 구성)": ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)"],
    "Ridge+랜덤포레스트+LightGBM": ["Ridge", "랜덤포레스트(회귀)", "LightGBM(회귀)"],
    "로지스틱+랜덤포레스트+LightGBM (분류)": ["로지스틱", "랜덤포레스트(분류)", "LightGBM(분류)"],
}
TIER = {"Ridge": 1, "로지스틱": 1, "EBM(회귀)": 2, "랜덤포레스트(회귀)": 2, "LightGBM(회귀)": 2, "랜덤포레스트(분류)": 2,
        "LightGBM(분류)": 2, **{k: len(v) + 1 for k, v in ENSEMBLES.items()}}   # 2개 결합 3, 3개 결합 4
IS_REG = {k: "분류" not in k and k != "로지스틱" for k in BASES} | {k: k != "로지스틱+랜덤포레스트+LightGBM (분류)" for k in ENSEMBLES}


def predict_all(tr, te, cols) -> dict:
    p = {k: f(tr, te, cols) for k, f in BASES.items()}
    for k, ms in ENSEMBLES.items():
        p[k] = np.mean([p[m] for m in ms], axis=0)
    return p


def split(df, rng):
    return df[df["date"].between(*rng)]


# ---------------------------------------------------------------- 채점
def boot(y, s, n=2000):
    rng, bs = np.random.default_rng(SEED), []
    while len(bs) < n:
        i = rng.integers(0, len(y), len(y))
        if 3 <= y[i].sum() < len(i):
            bs.append(drop_stats(y[i], s[i]))
    return pd.DataFrame(bs)


def tier_row(y, a):
    hit, fa = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
    neg = int((y == 0).sum())
    return {"적중": hit, "놓침": int(y.sum()) - hit, "헛경보": fa, "재현율": hit / y.sum(),
            "정밀도": hit / max(hit + fa, 1), "헛경보율": fa / max(neg, 1)}


def score_rows(name, y, s, prob, warn, strong):
    st, bs = drop_stats(y, s), boot(y, s)
    base = {"구간": name, "일수": len(y), "급락일": int(y.sum()), **st, "Brier": brier_score_loss(y, prob)}
    for k in ("PR-AUC", "헛경보@재현율80%", "헛경보@재현율90%"):
        base[f"{k} 95%하한"], base[f"{k} 95%상한"] = bs[k].quantile(.025), bs[k].quantile(.975)
    return [{**base, "경보": "주의 이상", **tier_row(y, s >= warn)},
            {**base, "경보": "경보", **tier_row(y, s >= strong)}]


def main():
    t0 = time.time()
    df = data23()
    print(f"피처 {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d} {len(df)}일, "
          f"예보 있음 {df['has_fcst'].mean():.1%}", flush=True)
    va = split(df, VALID)
    vo = va["offday"].values
    yv = va.loc[vo, "label"].values

    # 1) 선택 (2025만)
    rows, val_pred = [], {}
    for fname, cols in FEATS.items():
        res = {}
        for wname, rng in TRAIN_SEL.items():
            res[wname] = predict_all(split(df, rng), va, cols)
            print(f"  {fname} / 학습 {wname}: {time.time() - t0:.0f}초", flush=True)
        val_pred[fname] = res["2023~2024"]
        for m in res["2023~2024"]:
            st = {w: drop_stats(yv, res[w][m][vo]) for w in TRAIN_SEL}
            rows.append({"피처": fname, "모델": m, "단계": TIER[m],
                         "PR-AUC 2023~2024학습": st["2023~2024"]["PR-AUC"], "PR-AUC 2024학습": st["2024"]["PR-AUC"],
                         "PR-AUC 나쁜쪽": min(st[w]["PR-AUC"] for w in TRAIN_SEL),
                         "헛경보@80%": st["2023~2024"]["헛경보@재현율80%"], "헛경보@90%": st["2023~2024"]["헛경보@재현율90%"],
                         "헛경보@80% 2024학습": st["2024"]["헛경보@재현율80%"],
                         "ROC-AUC": st["2023~2024"]["ROC-AUC"]})
    sel = pd.DataFrame(rows).sort_values("PR-AUC 나쁜쪽", ascending=False).reset_index(drop=True)
    best = sel["PR-AUC 나쁜쪽"].iloc[0]
    sel["동점권"] = sel["PR-AUC 나쁜쪽"] >= best - TIE
    pick = sel[sel["동점권"]].sort_values(["단계", "헛경보@80%", "PR-AUC 나쁜쪽"], ascending=[True, True, False]).iloc[0]
    sel["선택"] = (sel["피처"] == pick["피처"]) & (sel["모델"] == pick["모델"])
    sel.round(4).to_csv(OUT_TAB / f"18_team_select_{REGION}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(sel.round(3).to_string(index=False))
    fname, mname, cols = pick["피처"], pick["모델"], FEATS[pick["피처"]]
    print(f"\n선택: {fname} / {mname} (나쁜쪽 PR-AUC {pick['PR-AUC 나쁜쪽']:.3f}, 1위 {best:.3f})", flush=True)

    # 2) 문턱·확률 (2025 검증 예측, 휴일·주말)
    sv = val_pred[fname][mname]
    warn, strong = threshold_for(yv, sv[vo], RECALL_WARN), strong_threshold(yv, sv[vo])
    cal = LogisticRegression(C=1e6).fit(sv[vo].reshape(-1, 1), yv)
    to_prob = lambda s: cal.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]
    unit = (lambda s: round(-s, 1)) if IS_REG[mname] else (lambda s: round(s, 4))
    print(f"문턱: 주의 {unit(warn)}, 경보 {unit(strong)} ({'예상 한낮 최저 원' if IS_REG[mname] else '확률 점수'})")

    # 3) 2026 한 번 채점
    te = split(df, ("2026-01-01", "2099-12-31"))
    p_te = predict_all(split(df, TRAIN_FIN), te, cols)[mname]
    to = te["offday"].values & te["date"].between(*TEST).values
    yt = te.loc[to, "label"].values
    score = pd.DataFrame(score_rows("2025 검증", yv, sv[vo], to_prob(sv[vo]), warn, strong)
                         + score_rows(f"2026 시험 {TEST[0]} ~ {TEST[1]}", yt, p_te[to], to_prob(p_te[to]), warn, strong))
    score.insert(0, "모델", f"{mname} ({fname})")
    score.round(4).to_csv(OUT_TAB / f"18_team_select_score_{REGION}.csv", index=False, encoding="utf-8-sig")

    # 4) 4.1 형식 일별 파일 (2025 = 2023~2024 학습 예측, 2026~ = 2023~2025 학습 예측)
    both = pd.concat([va.assign(s=sv), te.assign(s=p_te)])
    out = pd.DataFrame({"base_date": (both["date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
                        "date": both["date"].dt.strftime("%Y-%m-%d"), "stage": "15h",
                        "grade": np.where(both["s"] >= strong, "경보", np.where(both["s"] >= warn, "주의", "없음")),
                        "prob": to_prob(both["s"]).round(4),
                        "score": (-both["s"]).round(1) if IS_REG[mname] else "",
                        "label": np.where(both["offday"], both["label"].astype(str), "")})
    out.to_csv(OUT_TAB / f"18_team_select_daily_{REGION}.csv", index=False, encoding="utf-8-sig")
    choice = {"stage": "15h (전날 15시)", "모델": mname, "피처 묶음": fname, "피처": cols,
              "점수": "예상 한낮(11~15시) 최저 SMP(원), 낮을수록 위험" if IS_REG[mname] else "급락 확률 점수, 클수록 위험",
              "문턱": {"주의": unit(warn), "경보": unit(strong)},
              "확률식": {"형태": "1/(1+exp(-(a·s+b))), s = 모델 위험 점수(회귀는 −예상 원)",
                      "a": float(cal.coef_[0][0]), "b": float(cal.intercept_[0])},
              "학습": {"선택": "2023~2024 → 2025 (2024 → 2025 함께, 나쁜 쪽)", "최종": "2023-01-01 ~ 2025-12-31"},
              "선택 절차": "PLAN.md 결정 기록 2026-10-26 (휴일·주말 PR-AUC, 0.03 이내면 단순한 쪽)",
              "문턱 규칙": "주의 = 2025 휴일·주말 급락 90%, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨",
              "2023 결측": "날씨 예보·주간전망·연료비 = 학습 구간 중앙값 + has_fcst"}
    (OUT_TAB / f"18_team_select_choice_{REGION}.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")

    show = ["구간", "경보", "일수", "급락일", "PR-AUC", "PR-AUC 95%하한", "PR-AUC 95%상한", "ROC-AUC", "헛경보@재현율80%",
            "헛경보@재현율90%", "Brier", "적중", "놓침", "헛경보", "재현율", "정밀도"]
    print(score[show].round(3).to_string(index=False))
    print("기준선(현 모델 2026):", BASELINE)
    miss = te[to & (p_te < warn)]
    print("2026 놓침:", miss.loc[miss["label"] == 1, "date"].dt.strftime("%m-%d").tolist())
    print(f"총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
