"""제주 다음 날 급락 판별 (판단 시각 D-1 15:00) — 과제1 육지 최종 모델과 같은 절차, 제주 자료로 새로 비교.

급락 = 제주 11~15시 최저 SMP < 30원 (07_제주_레이블링.md), 학습 시작 2024-01-01.
피처 = smp_forecast_app.src.jz_features H=1 표의 하루 단위 값 + 하루 단위 파생 (아래 SETS)
  S1 육지형      : 달력 + 제주 SMP 지연값·최근 급락 + 제주 수요예측 + 제주 구역예보 (육지 최종 모델 피처의 제주판)
  S2 +풍속·일사  : S1 + Open-Meteo 제주 일사·운량·10m/100m 풍속 (D-2 실행분)
  S3 +육지       : S2 + 육지 SMP 지연값·육지 날씨·주간전망·연료비·설비 + 육지 일주일 예측의 내일 한낮 최저
  S4 +제주예측   : S3 + 제주 Chronos-2 small(육지 공변량)의 내일 한낮 최저·하위 분위
모델: 회귀(내일 한낮 최저, 0~150원으로 자름: 점수 = −예측) Ridge·EBM·랜덤포레스트·엑스트라트리·LightGBM·CatBoost,
      분류(급락 확률) 로지스틱·LightGBM·랜덤포레스트·EBM, 평균 조합(육지 최종 = Ridge+EBM+RF 등)
선택 (2026 채점 전 고정, 육지와 같은 규칙)
  검증 F1 = 2024 학습 → 2025 상반기, F2 = 2024~2025 상반기 학습 → 2025 하반기 (이어 붙여 채점)
  순위 = 급락 90%를 잡을 때 헛경보 수 ↑ → 100% 잡을 때 헛경보 수 ↑ → PR-AUC ↓
  상위 후보는 '매월 재학습'도 같은 규칙으로 비교
문턱: 2024 학습 → 2025 예측에서 주의 = 2025 급락 90%를 잡는 값, 강한 경보 = 정밀도 50%·적중 3일 이상인 가장 느슨한 값
확률: 2025 표본 밖 점수로 로지스틱 보정 (회귀 모델이면 예상 한낮 최저 원 단위)
최종: 2024~2025 학습 → 2026-01-01 ~ 예측. --score2026 일 때만 2026 채점 (사용자 확인 후 한 번)
J_zoo_pick.json(jeju_zoo 대규모 비교 결과)이 있으면 1차 후보 비교는 건너뛰고 그 선택으로 문턱·확률·결과 파일을 만든다.
출력: drop_classifier/outputs/tables/J1_candidates_2025.csv, J2_monthly_2025.csv, J_choice.json, drop_prob_제주.csv,
      (--score2026) J3_test2026.csv
      smp_forecast_app/outputs/tables/JD_choice.json, data/processed/smp_forecast_app/jeju/drop_badge_제주.csv
실행: python -m drop_classifier.src.jeju_drop [--score2026]
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from common.config import ROOT, SEED
from drop_classifier.src.experiments import false_alarms_at
from drop_classifier.src.final import threshold_for
from drop_classifier.src.two_tier import strong_threshold
from smp_forecast_app.src.jz_common import JDIR, land_pred, load_pred
from smp_forecast_app.src.jz_features import load as load_feats

warnings.filterwarnings("ignore")
TAB = ROOT / "drop_classifier" / "outputs" / "tables"
TAB2 = ROOT / "smp_forecast_app" / "outputs" / "tables"
MID = [11, 12, 13, 14, 15]
TRAIN_START = "2024-01-01"
RECALL = 0.9

CAL = ["offday", "daytype", "dow", "off_block_len", "off_block_day", "is_myeongjeol", "is_labor_day", "prev_offday",
       "next_offday", "month_sin", "month_cos", "trend_years"]
LAG = ["s_d1_midmin", "s_same_midmin", "s_d1_min", "s_d1_mean", "s_ma28", "j_min7_mid", "j_drop14", "j_drop28", "j_only28",
       "j_eq7", "j_eq28"]
DEMAND = ["d_d1_mid", "d_rel28", "d_d1_mean"]
FCST = ["jf_sky_pm", "jf_sky_am", "jf_pop_pm", "jf_rain_pm", "jf_ta_pm", "jf_north_pm"]
OM = ["jo_rad_mid", "jo_cloud_mid", "jo_ws10_day", "jo_ws100_day", "jo_ws100_mid", "jo_ws100_night"]
LAND = ["L_s_d1_midmin", "L_s_same_h12", "L_rf_sky_pm", "L_om_rad_mid", "kpx_peak", "kpx_peak_rel", "fc_lng", "fc_coal",
        "cap_solar", "L_pred_midmin", "L_c2s_midmin"]
JPRED = ["J_c2sx_midmin", "J_c2sx_qlo_midmin"]
SETS = {"S1 육지형": CAL + LAG + DEMAND + FCST, "S2 +풍속·일사": CAL + LAG + DEMAND + FCST + OM,
        "S3 +육지": CAL + LAG + DEMAND + FCST + OM + LAND, "S4 +제주예측": CAL + LAG + DEMAND + FCST + OM + LAND + JPRED}


def day_table(f: pd.DataFrame | None = None, jc2sx: pd.DataFrame | None = None) -> pd.DataFrame:
    """대상일 D 한 행: 피처(판단일 D-1 15시 기준) + 한낮 최저(min11_15)·급락 레이블.
    f: 제주 피처 표 H=1 (없으면 저장 파일), jc2sx: 제주 Chronos 예측 [date, hour, H, pred, qlo] (없으면 백테스트 파일) — 운영용."""
    f = load_feats(1) if f is None else f
    d = f[f["hour"] == 12].drop(columns=["hour"]).copy().reset_index(drop=True)
    d = d.rename(columns={"L_s_same_h": "L_s_same_h12"})
    y = f[f["hour"].isin(MID)].groupby("date")["y"].agg(["min", "count"])
    d["min11_15"] = d["date"].map(y["min"].where(y["count"] == 5))
    d["label"] = (d["min11_15"] < 30).astype(float).where(d["min11_15"].notna())
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * d["month"] / 12), np.cos(2 * np.pi * d["month"] / 12)
    d["trend_years"] = (d["date"] - pd.Timestamp("2024-01-01")).dt.days / 365.25
    d = d.sort_values("date").reset_index(drop=True)
    # 최근 14일 제주 급락 수 (D-1까지), 수요예측 상대값
    lab = d.set_index("date")["label"]
    d["j_drop14"] = d["date"].map(lab.rolling(14, min_periods=10).sum().shift(1))
    dm = d.set_index("date")["d_d1_mid"]
    d["d_rel28"] = d["d_d1_mid"] / d["date"].map(dm.rolling(28, min_periods=20).median())
    # 육지 일주일 예측(H=1)의 내일 한낮 최저
    lp = land_pred()
    lp = lp[(lp["H"] == 1) & lp["hour"].isin(MID)].groupby("date")[["L_base", "L_c2s"]].min()
    d["L_pred_midmin"], d["L_c2s_midmin"] = d["date"].map(lp["L_base"]), d["date"].map(lp["L_c2s"])
    # 제주 Chronos-2 small + 육지 공변량 (H=1, 2023~2024 과거 예측 + 2025~)
    try:
        c = pd.concat([load_pred("jc2sx", hist=True), load_pred("jc2sx") if jc2sx is None else jc2sx])
        c = c.drop_duplicates(["date", "hour", "H"], keep="last")
        c = c[(c["H"] == 1) & c["hour"].isin(MID)].groupby("date")[["pred", "qlo"]].min()
        d["J_c2sx_midmin"], d["J_c2sx_qlo_midmin"] = d["date"].map(c["pred"]), d["date"].map(c["qlo"])
    except FileNotFoundError:
        d["J_c2sx_midmin"], d["J_c2sx_qlo_midmin"] = np.nan, np.nan
    return d[d["date"] >= TRAIN_START].reset_index(drop=True)


# ---------- 모델: f(tr, te, cols) → 위험 점수 (높을수록 위험) ----------
def _imp(m):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


def _yr(tr):
    return tr["min11_15"].clip(0, 150)


def r_ridge(tr, te, c):
    from sklearn.linear_model import Ridge
    return -_imp(Ridge(alpha=3)).fit(tr[c], _yr(tr)).predict(te[c])


def r_ebm(tr, te, c):
    from interpret.glassbox import ExplainableBoostingRegressor
    return -ExplainableBoostingRegressor(random_state=SEED, interactions=5, min_samples_leaf=5, max_bins=64,
                                         n_jobs=4).fit(tr[c], _yr(tr)).predict(te[c])


def r_rf(tr, te, c):
    from sklearn.ensemble import RandomForestRegressor
    return -_imp(RandomForestRegressor(500, min_samples_leaf=5, random_state=SEED, n_jobs=4)).fit(tr[c], _yr(tr)).predict(te[c])


def r_et(tr, te, c):
    from sklearn.ensemble import ExtraTreesRegressor
    return -_imp(ExtraTreesRegressor(500, min_samples_leaf=5, random_state=SEED, n_jobs=4)).fit(tr[c], _yr(tr)).predict(te[c])


def r_lgb(tr, te, c):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=15, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=4)
    return -m.fit(tr[c], _yr(tr)).predict(te[c])


def r_cat(tr, te, c):
    from catboost import CatBoostRegressor
    return -CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05, random_seed=SEED, verbose=0,
                              thread_count=4).fit(tr[c], _yr(tr)).predict(te[c])


def c_logit(tr, te, c):
    return _imp(LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")).fit(tr[c], tr["label"]).predict_proba(te[c])[:, 1]


def c_lgb(tr, te, c):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=4)
    return m.fit(tr[c], tr["label"]).predict_proba(te[c])[:, 1]


def c_rf(tr, te, c):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced_subsample", random_state=SEED, n_jobs=4)
    return _imp(m).fit(tr[c], tr["label"]).predict_proba(te[c])[:, 1]


def c_ebm(tr, te, c):
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(random_state=SEED, interactions=5, min_samples_leaf=5, max_bins=64,
                                         n_jobs=4).fit(tr[c], tr["label"]).predict_proba(te[c])[:, 1]


BASES = {"Ridge": r_ridge, "EBM(회귀)": r_ebm, "랜덤포레스트(회귀)": r_rf, "엑스트라트리(회귀)": r_et, "LightGBM(회귀)": r_lgb,
         "CatBoost(회귀)": r_cat, "로지스틱": c_logit, "LightGBM(분류)": c_lgb, "랜덤포레스트(분류)": c_rf, "EBM(분류)": c_ebm}
REG = [k for k in BASES if "회귀" in k or k == "Ridge"]
COMBOS = {"Ridge+EBM+RF (육지 최종 구성)": ["Ridge", "EBM(회귀)", "랜덤포레스트(회귀)"], "Ridge+EBM": ["Ridge", "EBM(회귀)"],
          "트리 회귀 평균(RF·ET·LGB·Cat)": ["랜덤포레스트(회귀)", "엑스트라트리(회귀)", "LightGBM(회귀)", "CatBoost(회귀)"],
          "회귀 전부 평균": REG}
RULES = {"전날 제주 한낮 최저": "s_d1_midmin", "최근 같은 유형일 제주 한낮 최저": "s_same_midmin",
         "육지 일주일 예측의 내일 한낮 최저": "L_pred_midmin", "제주 Chronos(육지 공변량) 내일 한낮 최저": "J_c2sx_midmin",
         "제주 Chronos 하위 10% 분위 한낮 최저": "J_c2sx_qlo_midmin"}
RANK_COMBOS = {"순위 평균(회귀 전부 + 분류 전부)": list(BASES), "순위 평균(분류 전부)": [k for k in BASES if k not in REG]}


def _rank(a):
    return pd.Series(a).rank(pct=True).values


def ranking(y, s):
    fa90, hit90 = false_alarms_at(y, s, RECALL)
    fa100, _ = false_alarms_at(y, s, 1.0)
    return {"헛경보@90%": fa90, "적중@90%": hit90, "헛경보@100%": fa100, "PR-AUC": average_precision_score(y, s),
            "ROC-AUC": roc_auc_score(y, s)}


def folds(d):
    """F1: 2024 → 2025 상반기, F2: 2024~2025 상반기 → 2025 하반기."""
    dd = d[d["label"].notna()]
    yield dd[dd["date"] < "2025-01-01"], dd[(dd["date"] >= "2025-01-01") & (dd["date"] < "2025-07-01")]
    yield dd[dd["date"] < "2025-07-01"], dd[(dd["date"] >= "2025-07-01") & (dd["date"] < "2026-01-01")]


def monthly(d):
    """매월 재학습: 2025 각 달을 (2024-01 ~ 그 달 전날) 학습으로 예측."""
    dd = d[d["label"].notna()]
    for m in pd.date_range("2025-01-01", "2025-12-01", freq="MS"):
        yield dd[dd["date"] < m], dd[(dd["date"] >= m) & (dd["date"] < m + pd.offsets.MonthBegin(1))]


def oos_scores(d, scheme, cols):
    """후보별 표본 밖 점수 {이름: 점수}, 정답 y, 날짜."""
    parts, ys, ds = {k: [] for k in list(BASES) + list(COMBOS) + list(RANK_COMBOS)}, [], []
    for tr, te in scheme(d):
        base = {k: f(tr, te, cols) for k, f in BASES.items()}
        for k, v in base.items():
            parts[k].append(v)
        for k, ms in COMBOS.items():
            parts[k].append(np.mean([base[m] for m in ms], axis=0))
        for k, ms in RANK_COMBOS.items():
            parts[k].append(np.mean([_rank(base[m]) for m in ms], axis=0))
        ys.append(te["label"].values)
        ds.append(te["date"].values)
    return {k: np.concatenate(v) for k, v in parts.items()}, np.concatenate(ys).astype(int), np.concatenate(ds)


def score_fn(name, cols):
    """후보 이름 → f(tr, te) 위험 점수. 1차 후보에 없는 이름은 대규모 비교(jeju_zoo) 후보."""
    if name not in BASES and name not in COMBOS and name not in RANK_COMBOS:
        from drop_classifier.src import jeju_zoo
        return jeju_zoo.score_fn(name, cols)
    if name in BASES:
        return lambda tr, te: BASES[name](tr, te, cols)
    if name in COMBOS:
        return lambda tr, te: np.mean([BASES[m](tr, te, cols) for m in COMBOS[name]], axis=0)
    return lambda tr, te: np.mean([_rank(BASES[m](tr, te, cols)) for m in RANK_COMBOS[name]], axis=0)


def is_reg(name):
    if name not in BASES and name not in COMBOS and name not in RANK_COMBOS:
        from drop_classifier.src import jeju_zoo
        return jeju_zoo.is_reg(name)
    return name in REG or name in COMBOS


def main(score2026=False):
    pick = TAB / "J_zoo_pick.json"
    if pick.exists():   # 대규모 비교(jeju_zoo) 결과가 있으면 그 선택으로 문턱·확률·결과 파일만 만든다
        z = json.loads(pick.read_text(encoding="utf-8"))
        return finalize(day_table(), z["피처"], z["피처 목록"], z["모델"], z["재학습"], score2026,
                        "jeju_zoo 대규모 비교 (J4_zoo_2025.csv): 2025 F1·F2 순위 → 상위 5개 고정/매월 비교")
    d = day_table()
    print(f"표: {d['date'].min().date()} ~ {d['date'].max().date()}, 급락일 2024 {int(d.loc[d.date.dt.year == 2024, 'label'].sum())}"
          f" · 2025 {int(d.loc[d.date.dt.year == 2025, 'label'].sum())}", flush=True)
    rows = []
    for sname, cols in SETS.items():
        sc, y, _ = oos_scores(d, folds, cols)
        for k, s in sc.items():
            rows.append({"피처": sname, "모델": k, "급락일": int(y.sum()), **ranking(y, s)})
        print(f"{sname} 완료", flush=True)
    # 학습 없는 단순 규칙 (같은 검증 날짜): 낮을수록 위험한 값의 음수를 점수로
    te_all = pd.concat([te for _, te in folds(d)])
    for name, col in RULES.items():
        s = -te_all[col].fillna(te_all[col].median()).values
        rows.append({"피처": "규칙(학습 없음)", "모델": name, "급락일": int(te_all["label"].sum()),
                     **ranking(te_all["label"].values.astype(int), s)})
    t = pd.DataFrame(rows).sort_values(["헛경보@90%", "헛경보@100%", "PR-AUC"], ascending=[True, True, False])
    TAB.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(TAB / "J1_candidates_2025.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(t.head(25).round(3).to_string(index=False), flush=True)

    # 상위 5개는 매월 재학습도 비교 (같은 순위 규칙, 2025 전체 표본 밖)
    mrows = []
    for _, r in t[t["피처"] != "규칙(학습 없음)"].head(5).iterrows():
        cols = SETS[r["피처"]]
        f = score_fn(r["모델"], cols)
        for sch_name, sch in (("고정(F1·F2)", folds), ("매월 재학습", monthly)):
            ss, yy = [], []
            for tr, te in sch(d):
                ss.append(f(tr, te))
                yy.append(te["label"].values)
            mrows.append({"피처": r["피처"], "모델": r["모델"], "방식": sch_name, **ranking(np.concatenate(yy).astype(int), np.concatenate(ss))})
    mt = pd.DataFrame(mrows)
    mt.round(4).to_csv(TAB / "J2_monthly_2025.csv", index=False, encoding="utf-8-sig")
    print(mt.round(3).to_string(index=False), flush=True)
    best = mt.sort_values(["헛경보@90%", "헛경보@100%", "PR-AUC"], ascending=[True, True, False]).iloc[0]
    finalize(d, best["피처"], SETS[best["피처"]], best["모델"], best["방식"], score2026,
             "F1(2024→2025상)·F2(2024~2025상→2025하) 순위 → 상위 5개 고정/매월 비교")


def finalize(d, pick_set, cols, pick_model, pick_scheme, score2026, how):
    """선택 모델 → 문턱·확률 보정·2026 예측·결과 파일 (--score2026이면 2026 채점)."""
    f = score_fn(pick_model, cols)
    print(f"\n선택: {pick_set} · {pick_model} · {pick_scheme}\n", flush=True)

    # 문턱·확률: 2025 표본 밖 점수 (고정이면 2024 학습 → 2025, 매월이면 매월 재학습 2025)
    lab = d[d["label"].notna()]
    if pick_scheme == "매월 재학습":
        s25, d25 = [], []
        for tr, te in monthly(d):
            s25.append(f(tr, te))
            d25.append(te)
        s25, d25 = np.concatenate(s25), pd.concat(d25)
    else:
        d25 = lab[lab["date"].dt.year == 2025]
        s25 = f(lab[lab["date"] < "2025-01-01"], d25)
    y25 = d25["label"].values.astype(int)
    warn, strong = threshold_for(y25, s25, RECALL), strong_threshold(y25, s25)
    lr = LogisticRegression(C=1e6).fit(s25.reshape(-1, 1), y25)

    # 2026 ~ 예측 (+ 내일 행): 고정 = 2024~2025 학습, 매월 = 매월 1일 재학습
    fut = d[d["date"] >= "2026-01-01"]
    if pick_scheme == "매월 재학습":
        s26 = pd.Series(np.nan, index=fut.index)
        for m in pd.date_range("2026-01-01", fut["date"].max(), freq="MS"):
            te = fut[(fut["date"] >= m) & (fut["date"] < m + pd.offsets.MonthBegin(1))]
            s26[te.index] = f(lab[lab["date"] < m], te)
        s26 = s26.values
    else:
        s26 = f(lab[lab["date"] < "2026-01-01"], fut)
    out = pd.concat([d25[["date", "min11_15", "label"]].assign(score=s25), fut[["date", "min11_15", "label"]].assign(score=s26)])
    out["계통"] = "제주"
    out["pred_min"] = -out["score"] if is_reg(pick_model) else np.nan
    out["grade"] = np.where(out["score"] >= (strong if strong is not None else np.inf), "강한 경보",
                            np.where(out["score"] >= warn, "주의", "없음"))
    out["prob"] = lr.predict_proba(out[["score"]].values)[:, 1]
    out[["date", "계통", "pred_min", "score", "grade", "prob"]].round(4).to_csv(TAB / "drop_prob_제주.csv", index=False, encoding="utf-8-sig")
    JDIR.mkdir(parents=True, exist_ok=True)
    out[["date", "grade", "pred_min", "prob", "label"]].round(4).to_csv(JDIR / "drop_badge_제주.csv", index=False, encoding="utf-8-sig")
    unit = "원(예상 한낮 최저, 낮을수록 위험)" if is_reg(pick_model) else "점수(높을수록 위험)"
    conv = (lambda s: round(-s, 2)) if is_reg(pick_model) else (lambda s: round(s, 4))
    choice = {"계통": "제주", "피처": pick_set, "피처 목록": cols, "모델": pick_model, "재학습": pick_scheme,
              "점수 단위": unit, "주의 문턱": conv(warn), "강한 경보 문턱": None if strong is None else conv(strong),
              "문턱 규칙": "주의 = 2025 급락 90%를 잡는 값, 강한 경보 = 정밀도 50%·적중 3일 이상인 가장 느슨한 값",
              "확률": "P(급락) = 1 / (1 + exp(-(a + b × 점수))), 점수 = −예상 한낮 최저(회귀) 또는 분류 점수",
              "a": round(float(lr.intercept_[0]), 4), "b": round(float(lr.coef_[0][0]), 5),
              "등급 문턱 확률": {"주의": round(float(lr.predict_proba([[warn]])[0, 1]), 3),
                            "강한 경보": None if strong is None else round(float(lr.predict_proba([[strong]])[0, 1]), 3)},
              "2025 표본 밖": ranking(y25, s25), "선택 절차": how,
              "시험": "2026-01-01 ~ (제주 1차 시험)"}
    choice["2025 표본 밖"] = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in choice["2025 표본 밖"].items()}
    for p in (TAB / "J_choice.json", TAB2 / "JD_choice.json"):
        p.write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(choice, ensure_ascii=False, indent=1))

    if score2026:   # 사용자 확인 후 한 번
        g = out[(out["date"] >= "2026-01-01") & out["label"].notna()]
        y, s = g["label"].values.astype(int), g["score"].values
        rows = [{"구분": "전체", "일수": len(g), "급락일": int(y.sum()), "ROC-AUC": roc_auc_score(y, s),
                 "PR-AUC": average_precision_score(y, s)}]
        for gr, thr in (("주의 이상", warn), ("강한 경보", strong)):
            if thr is None:
                continue
            a = s >= thr
            rows.append({"구분": gr, "적중": int((a & (y == 1)).sum()), "놓침": int((~a & (y == 1)).sum()), "헛경보": int((a & (y == 0)).sum())})
        pd.DataFrame(rows).round(4).to_csv(TAB / "J3_test2026.csv", index=False, encoding="utf-8-sig")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main("--score2026" in sys.argv[1:])
