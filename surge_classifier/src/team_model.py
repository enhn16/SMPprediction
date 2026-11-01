"""④ 팀 합의 급등 모델 (육지, 15시) — 부호 고정 점수 + 스플라인 로지스틱 + LightGBM 그루터기(깊이 1) 세 점수의 평균.

구성은 고정(2026-10-28 팀 합의), 세부만 고른다. 고르는 근거 = 2024~2025 앞으로만 검증:
  접기 2024: 2023 학습 → 2024 예측 (급등 19일) / 접기 2025: 2023~2024 학습 → 2025 예측 (7일) → 이어 붙여 730일·26일
선택 점수 = 이어 붙인 PR-AUC의 +20·25·30% 세 기준 평균 (점수는 학습 구간 분포 기준 백분위로 바꿔 두 접기를 같은 척도로).
  1위의 5% 이내면 목록 앞쪽(단순한 쪽). 1단계 = 모델별 설정, 2단계 = 평균 척도·가중.
문턱(2024~2025 이어 붙인 평균 점수): 주의 = 급등 80% 이상 잡는 가장 높은 선, 경보 = 정밀도 50%·적중 3일 이상 중 가장 느슨한 선,
  없으면 정밀도가 가장 높은 선(적중 3일 이상). 확률 = 같은 점수에 Platt 보정.
최종: 2023~2025 재학습 → 2026-01-01 ~ 10-18 한 번 채점 (+ 팀원 비교용 ~10-02).
피처: features_육지.csv(13종, 내일 수요예측 제외) + extra_features_육지.csv의 JMA 운량·일사(13종).
  공급능력·원자력·특보(13종 밖)는 1차 선택에서 기여가 없어 후보에서 뺌 (2026-10-28 사용자 결정 B).
  태양광 대리값 pv_gw = 시장참여 태양광 설비(GW) × 한낮 평균 일사(kW/m²) × 0.8 — 부호 고정에서 (−)로 넣으면
  z 같은 가중의 '순부하(최대부하 − 태양광)'가 됨 (2026-10-28 추가, 검증 결과를 본 뒤 추가한 후보).

실행: python -m surge_classifier.src.team_model select   → outputs/tables/04_team_select_육지.csv (2026 열지 않음)
      python -m surge_classifier.src.team_model test     → 확정 설정(CHOICE)으로 2026 채점, 04_team_test_육지.csv
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import SplineTransformer

from common.config import SEED
from surge_classifier.src.features import CALENDAR, DEMAND_PREV, OUTLOOK, SMP_HIST, WEATHER
from surge_classifier.src.label_days import OUT_DATA, OUT_TAB

warnings.filterwarnings("ignore")

FOLDS = {"2024": (("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")),
         "2025": (("2023-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31"))}
FINAL_TRAIN = ("2023-01-01", "2025-12-31")
TEST = ("2026-01-01", "2026-10-18")
TEST_TEAM = ("2026-01-01", "2026-10-02")      # 팀원 수치 기간 (급등 6일, 10/13 빠짐)
LABELS = ["label_20", "label_25", "label_30"]
TIE_REL = 0.05

JMA = ["cloud_mid_jma", "rad_mid_jma", "wk_cloud_mid_jma"]
POOL13 = CALENDAR + SMP_HIST + DEMAND_PREV + OUTLOOK + WEATHER + JMA + ["cap_solar_gw", "fc_lng", "lng_coal_ratio"]
EXT = ["sup_rr_d2", "sup_avail_d2", "sup_rel_d2", "nuc_gw", "nuc_rel", "gas_rel", "cold_n", "cold_warn_n",
       "heat_n", "heat_warn_n", "wk_heat_n"]

# ---------------------------------------------------------------- 후보 (목록 앞쪽 = 단순)
S4 = {"wk_sky_pm": 1, "wk_cloud_mid_jma": 1, "kpx_peak": 1, "is_offday": -1}
S6 = S4 | {"pop_pm": 1, "prev_ratio": 1}
S8 = S6 | {"base_trend": 1, "kpx_peak_rel": 1}
SF_CANDS = [{"vars": v, "weight": w, "name": f"{n}·{w}"}
            for w in ("같음", "학습") for n, v in (("변수4", S4), ("변수6", S6), ("변수8", S8),
                                                  ("변수6+태양광", S6 | {"pv_gw": -1}), ("변수8+태양광", S8 | {"pv_gw": -1}))]
SP_SPLINE = ["kpx_peak", "ta_am", "ta_pm", "wk_cloud_mid_jma", "sky_pm"]
SP_LINEAR = ["is_offday", "prev_ratio", "base_trend"]
SP_CANDS = [{"knots": k, "C": c, "cw": cw, "target": t, "extra": e,
             "name": f"매듭{k}·C{c}·{'balanced' if cw else '가중없음'}·{t}" + ("·+태양광" if e else "")}
            for e in (False, True) for t in ("label_25", "label_20") for k in (4, 3, 5) for c in (0.1, 0.03, 0.3)
            for cw in ("balanced", None)]
ST_CANDS = [{"n": n, "leaf": m, "target": t, "pool": p, "name": f"{n}그루·잎{m}·{t}·{p}"}
            for p in ("13종", "13종+태양광") for t in ("label_25", "label_20") for n in (300, 100, 600) for m in (30, 15, 60)]
SCALES = ["백분위", "z"]
WEIGHTS = [(1 / 3, 1 / 3, 1 / 3), (.5, .25, .25), (.25, .5, .25), (.25, .25, .5)]


def data() -> pd.DataFrame:
    f = pd.read_csv(OUT_DATA / "features_육지.csv", parse_dates=["date"]).set_index("date")
    e = pd.read_csv(OUT_DATA / "extra_features_육지.csv", parse_dates=["date"]).set_index("date")
    d = f.join(e)
    d["wk_heat_n"] = (1 - d["is_offday"]) * d["heat_n"]
    d["pv_gw"] = d["cap_solar_gw"] * d["rad_mid_jma"] / 4 / 1000 * 0.8
    return d


def span(d, rng):
    return d.loc[rng[0]:rng[1]]


# ---------------------------------------------------------------- 세 모델 (반환: 학습 구간 점수, 예측 점수 — 클수록 위험)
def fit_sign(tr, te, c):
    cols, sign = list(c["vars"]), np.array(list(c["vars"].values()), dtype=float)
    med = tr[cols].median()
    xt = tr[cols].fillna(med)
    mu, sd = xt.mean(), xt.std().replace(0, 1)
    zt = ((xt - mu) / sd).values * sign
    ze = ((te[cols].fillna(med) - mu) / sd).values * sign
    if c["weight"] == "같음":
        w = np.ones(len(cols))
    else:   # 부호를 맞춘 z에 음이 아닌 계수 로지스틱 (L2 1.0), 목표 +25%
        y = tr["label_25"].values

        def loss(p):
            z = zt @ p[:-1] + p[-1]
            return np.mean(np.logaddexp(0, z) - y * z) + 1.0 * np.sum(p[:-1] ** 2) / len(y)
        r = minimize(loss, np.r_[np.ones(len(cols)) * .1, -3], method="L-BFGS-B",
                     bounds=[(0, None)] * len(cols) + [(None, None)])
        w = r.x[:-1]
    return zt @ w / len(cols), ze @ w / len(cols)


def fit_spline(tr, te, c):
    lin = SP_LINEAR + (["pv_gw"] if c["extra"] else [])
    cols = SP_SPLINE + lin
    med = tr[cols].median()
    xt, xe = tr[cols].fillna(med), te[cols].fillna(med)
    sp = SplineTransformer(n_knots=c["knots"], degree=3, knots="quantile", extrapolation="constant").fit(xt[SP_SPLINE])
    mu, sd = xt[lin].mean(), xt[lin].std().replace(0, 1)

    def X(x):
        return np.hstack([sp.transform(x[SP_SPLINE]), ((x[lin] - mu) / sd).values])
    m = LogisticRegression(C=c["C"], class_weight=c["cw"], max_iter=5000).fit(X(xt), tr[c["target"]].values)
    return m.decision_function(X(xt)), m.decision_function(X(xe))


def fit_stump(tr, te, c):
    cols = POOL13 + (["pv_gw"] if c["pool"] == "13종+태양광" else [])
    m = LGBMClassifier(objective="binary", max_depth=1, num_leaves=2, learning_rate=0.03, n_estimators=c["n"],
                       min_child_samples=c["leaf"], colsample_bytree=0.8, random_state=SEED, verbose=-1,
                       deterministic=True, n_jobs=1)
    m.fit(tr[cols], tr[c["target"]].values)
    return m.predict(tr[cols], raw_score=True), m.predict(te[cols], raw_score=True)


FITS = {"부호 고정": (fit_sign, SF_CANDS), "스플라인": (fit_spline, SP_CANDS), "그루터기": (fit_stump, ST_CANDS)}


def to_scale(s_tr, s_te, scale):
    if scale == "백분위":
        return np.searchsorted(np.sort(s_tr), s_te, side="right") / len(s_tr)
    return (s_te - s_tr.mean()) / s_tr.std()


# ---------------------------------------------------------------- 평가
def sel_score(y3: pd.DataFrame, s) -> float:
    return float(np.mean([average_precision_score(y3[k].astype(int), s) for k in LABELS]))


def oof(d, fit, c, scale="백분위") -> pd.Series:
    """두 접기 예측을 이어 붙인 점수 (같은 척도)."""
    out = []
    for tr_rng, te_rng in FOLDS.values():
        tr, te = span(d, tr_rng).dropna(subset=["label_25"]), span(d, te_rng)
        s_tr, s_te = fit(tr, te, c)
        out.append(pd.Series(to_scale(s_tr, s_te, scale), te.index))
    return pd.concat(out)


def eval_row(d, s: pd.Series) -> dict:
    v = d.loc[s.index]
    r = {"선택 점수": sel_score(v, s), "PR-AUC(+25%)": average_precision_score(v["label_25"], s),
         "ROC-AUC(+25%)": roc_auc_score(v["label_25"], s)}
    for yr in FOLDS:
        m = v.index.year == int(yr)
        r[f"{yr} 점수"] = sel_score(v[m], s[m])
        r[f"{yr} PR-AUC(+25%)"] = average_precision_score(v["label_25"][m], s[m])
    y, a = v["label_25"].values, s.values
    t = threshold_for(y, a, .8)
    r["헛경보@재현율80%"] = int(((a >= t) & (y == 0)).sum())
    return r


def pick(rows: pd.DataFrame) -> int:
    """1위의 5% 이내 중 목록 앞쪽(단순한 쪽)."""
    best = rows["선택 점수"].max()
    return int(rows.index[rows["선택 점수"] >= best * (1 - TIE_REL)][0])


def threshold_for(y, s, target):
    for t in np.sort(np.unique(s))[::-1]:
        if ((s >= t) & (y == 1)).sum() >= np.ceil(target * y.sum()):
            return float(t)
    return float(s.min())


def strong_threshold(y, s, prec=.5, min_tp=3):
    best = None
    for t in np.sort(np.unique(s)):
        a = s >= t
        tp = int((a & (y == 1)).sum())
        if tp < min_tp:
            continue
        p = tp / a.sum()
        if p >= prec:
            return float(t), f"정밀도 {p:.1%}·적중 {tp}일 (50% 이상 중 가장 느슨)"
        if best is None or p > best[1] + 1e-12:
            best = (float(t), p, tp)
    return best[0], f"정밀도 50% 선 없음 → 정밀도 최고 {best[1]:.1%}·적중 {best[2]}일"


def ensemble(parts: dict, scale, w) -> pd.Series:
    return sum(wi * parts[k][scale] for wi, k in zip(w, FITS))


# ---------------------------------------------------------------- 선택 (2024~2025만)
def select():
    d = data()
    rows, chosen = [], {}
    for comp, (fit, cands) in FITS.items():
        rs = pd.DataFrame([{"단계": "1 모델별", "모델": comp, "설정": c["name"], **eval_row(d, oof(d, fit, c))}
                           for c in cands])
        i = pick(rs)
        rs["선택"] = rs.index == i
        chosen[comp] = cands[i]
        rows.append(rs)
        print(f"[{comp}] {len(cands)}개 중 선택: {cands[i]['name']}  점수 {rs.loc[i, '선택 점수']:.4f} "
              f"(1위 {rs['선택 점수'].max():.4f})", flush=True)
    parts = {k: {sc: oof(d, FITS[k][0], chosen[k], sc) for sc in SCALES} for k in FITS}
    rs = pd.DataFrame([{"단계": "2 평균", "모델": "평균", "설정": f"{sc}·가중{tuple(round(x, 2) for x in w)}",
                        **eval_row(d, ensemble(parts, sc, w))} for sc in SCALES for w in WEIGHTS])
    # 가중: 같음이 기본. 기울인 가중은 같음보다 5% 넘게 좋고 두 해 모두 나을 때만
    base = {sc: rs[rs["설정"] == f"{sc}·가중(0.33, 0.33, 0.33)"].iloc[0] for sc in SCALES}
    sc = max(SCALES, key=lambda s: base[s]["선택 점수"] + (1e-9 if s == "백분위" else 0))
    pickrow = base[sc]
    for _, r in rs[rs["설정"].str.startswith(sc)].iterrows():
        if (r["선택 점수"] > pickrow["선택 점수"] * (1 + TIE_REL) and r["2024 점수"] > base[sc]["2024 점수"]
                and r["2025 점수"] > base[sc]["2025 점수"]):
            pickrow = r
    rs["선택"] = rs["설정"] == pickrow["설정"]
    rows.append(rs)
    # 비교: 지금 지수식 (학습 없음)
    from surge_classifier.src.models import index_score, z_stats
    v = span(d, ("2024-01-01", "2025-12-31"))
    idx = pd.Series(index_score(v, "지수: 평일×(운량+흐림+주간전망)", z_stats(d.reset_index())), v.index)
    rows.append(pd.DataFrame([{"단계": "비교", "모델": "지금 지수식", "설정": "평일×(운량+흐림+주간전망)",
                               **eval_row(d, idx), "선택": False}]))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_TAB / "04_team_select_육지.csv", index=False, encoding="utf-8-sig")
    w = next(w for w in WEIGHTS if pickrow["설정"] == f"{sc}·가중{tuple(round(x, 2) for x in w)}")
    ens = ensemble(parts, sc, w)
    y = d.loc[ens.index, "label_25"].values
    warn = threshold_for(y, ens.values, .8)
    strong, why = strong_threshold(y, ens.values)
    # 문턱 옮김 확인: 2024만으로 정한 주의 문턱을 2025에 적용
    m24 = ens.index.year == 2024
    t24 = threshold_for(y[m24], ens.values[m24], .8)
    rec25 = ((ens.values[~m24] >= t24) & (y[~m24] == 1)).sum() / y[~m24].sum()
    print(f"\n평균: {pickrow['설정']}  점수 {pickrow['선택 점수']:.4f}, PR-AUC(+25%) {pickrow['PR-AUC(+25%)']:.4f}, "
          f"헛경보@80% {pickrow['헛경보@재현율80%']}")
    print(f"문턱: 주의 {warn:.4f}, 경보 {strong:.4f} ({why}); 2024 문턱 → 2025 재현율 {rec25:.0%}")
    print(out[out["선택"] | (out["단계"] != "1 모델별")][["단계", "모델", "설정", "선택 점수", "PR-AUC(+25%)",
                                                        "2024 PR-AUC(+25%)", "2025 PR-AUC(+25%)", "헛경보@재현율80%"]]
          .round(4).to_string())
    platt = LogisticRegression(C=1e6).fit(ens.values.reshape(-1, 1), y)
    choice = {"설정": {k: chosen[k]["name"] for k in FITS}, "척도": sc, "가중": list(w),
              "문턱": {"주의": warn, "경보": strong, "경보 근거": why, "주의 근거": "2024~2025 급등 26일 중 80% 이상"},
              "확률식": {"a": float(platt.coef_[0, 0]), "b": float(platt.intercept_[0])},
              "검증 2024~2025": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                               for k, v in pickrow.items() if k not in ("단계", "모델", "선택")},
              "2024 문턱 → 2025 재현율": float(rec25)}
    (OUT_TAB / "04_team_choice_육지.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")
    return choice


# ---------------------------------------------------------------- 최종: 2023~2025 재학습 → 2026 한 번
def final_scores(d, choice, until=None) -> pd.Series:
    """확정 설정으로 2023~2025 학습 → 2026-01-01 이후 평균 점수 (운영 함수와 같은 경로)."""
    tr = span(d, FINAL_TRAIN).dropna(subset=["label_25"])
    te = d.loc["2026-01-01":until]
    parts = {}
    for k, (fit, cands) in FITS.items():
        c = next(c for c in cands if c["name"] == choice["설정"][k])
        s_tr, s_te = fit(tr, te, c)
        parts[k] = {choice["척도"]: pd.Series(to_scale(s_tr, s_te, choice["척도"]), te.index)}
    return ensemble(parts, choice["척도"], choice["가중"])


def grade_of(s, choice):
    t = choice["문턱"]
    return np.where(s >= t["경보"], "경보", np.where(s >= t["주의"], "주의", "없음"))


def prob_of(s, choice):
    return 1 / (1 + np.exp(-(choice["확률식"]["a"] * np.asarray(s) + choice["확률식"]["b"])))


def test():
    from surge_classifier.src.models import score_rows
    d = data()
    choice = json.loads((OUT_TAB / "04_team_choice_육지.json").read_text(encoding="utf-8"))
    s = final_scores(d, choice)
    daily = pd.DataFrame({"date": s.index, "score": s.values, "prob": prob_of(s, choice), "grade": grade_of(s, choice),
                          "label": d.loc[s.index, "label_25"].values})
    daily.to_csv(OUT_TAB / "04_team_daily_육지.csv", index=False, encoding="utf-8-sig")
    t = choice["문턱"]
    old = pd.read_csv(OUT_TAB / "surge_육지_15h_v1_index.csv", parse_dates=["date"]).set_index("date")   # 지금 지수식 (v1)
    old_choice = json.loads((OUT_TAB / "surge_choice_육지_15h_v1_index.json").read_text(encoding="utf-8"))["문턱 (모델 내부 점수)"]
    rows = []
    for nm, rng in (("2026 시험 (공식, ~10-18)", TEST), ("2026 팀원 비교 기간 (~10-02)", TEST_TEAM)):
        v = daily.set_index("date").loc[rng[0]:rng[1]].dropna(subset=["label"])
        y = v["label"].astype(int).values
        for r in score_rows(nm, y, v["score"].values, v["prob"].values, t["주의"], t["경보"]):
            rows.append({"모델": "팀 합의 (세 점수 평균)", **r, "상위 10일 적중": int(y[np.argsort(-v["score"].values)[:10]].sum())})
        o = old.loc[rng[0]:rng[1]].dropna(subset=["label"])
        yo = o["label"].astype(int).values
        for r in score_rows(nm, yo, o["score"].values, o["prob"].values, old_choice["주의"], old_choice["경보"]):
            rows.append({"모델": "지금 지수식 (v1)", **r, "상위 10일 적중": int(yo[np.argsort(-o["score"].values)[:10]].sum())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TAB / "04_team_test_육지.csv", index=False, encoding="utf-8-sig")
    cols = ["모델", "구간", "일수", "급등일", "PR-AUC", "PR-AUC 배수", "ROC-AUC", "헛경보@재현율80%", "경보", "적중", "놓침",
            "헛경보", "재현율", "정밀도", "상위 10일 적중"]
    print(out[cols].round(3).to_string())
    print("\n2026 급등일:", daily[(daily["label"] == 1) & (daily["date"] <= TEST[1])][["date", "score", "grade"]]
          .assign(date=lambda x: x["date"].dt.strftime("%m-%d")).round(3).to_string(index=False))


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1] if len(sys.argv) > 1 else "select"]()
