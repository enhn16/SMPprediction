"""③ 분류 모델 비교 (육지).

분할 (시간 순서만 사용)
- 롤링1: 학습 2024 → 평가 2025
- 롤링2: 학습 2024~2025 → 평가 2026-01~06
- 검증  : 학습 2024-01~2026-06 → 평가 2026-07~최신일

절차
1. 피처 묶음 × 모델마다 롤링 2회 평가 (주 지표: PR-AUC)
2. 롤링 평가 예측(OOF)으로 확률 보정(Platt)과 판정 임계값(F1 최대)을 정함 → 검증 기간에는 손대지 않음
3. 전체 학습 구간으로 다시 학습해 검증 기간 평가
4. 모델 선택: 롤링 PR-AUC 평균 최고 모델. 해석 가능한 모델(로지스틱·EBM·결정나무)과 차이가 0.05 미만이면 해석 가능한 모델 우선

실행: python -m drop_classifier.src.models
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from common.config import PROCESSED_DIR, ROOT, SEED
from common.loaders import load_gen_by_fuel
from common.plot import setup_korean_font
from drop_classifier.src.features import ACTUAL, CALENDAR, DEMAND, DEMAND_PREV, FCST, LAG, ko

warnings.filterwarnings("ignore")
REGION = "육지"
OUT_DATA = PROCESSED_DIR / "drop_classifier"
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
OUT_FIG = ROOT / "drop_classifier" / "outputs" / "figures"

FOLDS = {
    "롤링1": (("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31")),
    "롤링2": (("2024-01-01", "2025-12-31"), ("2026-01-01", "2026-06-30")),
    "검증": (("2024-01-01", "2026-06-30"), ("2026-07-01", "2099-12-31")),
}
FEATURE_SETS = {
    "기본(날씨 없음)": CALENDAR + DEMAND + LAG,
    "예보": CALENDAR + DEMAND + LAG + FCST,
    "예보·수요예측 D-1": CALENDAR + DEMAND_PREV + LAG + FCST,
    "상한선(실측 날씨)": CALENDAR + DEMAND + LAG + ACTUAL,
}
MAIN_SET = "예보"
INTERPRETABLE = {"로지스틱", "EBM", "결정나무(깊이3)"}
CASES = ["2026-09-25", "2026-10-05"]

# dataviz 기본 팔레트 (범주 순서 고정)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
C_INK, C_INK2, C_MUTED = "#0b0b0b", "#52514e", "#b9b8b2"


# ---------- 모델 ----------
def _with_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """로지스틱용 상호작용: 휴무일 × (하늘상태 예보·실측 운량, 상대 수요)."""
    X = X.copy()
    for c in ("fc_sky_mean", "act_cloud", "demand_rel28", "demand_prev_rel28"):
        if c in X:
            X[f"offday_x_{c}"] = X["is_offday"] * X[c].fillna(X[c].median())
    return X


class Logit:
    def __init__(self):
        self.m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))

    def fit(self, X, y):
        self.cols = _with_interactions(X).columns
        self.m.fit(_with_interactions(X), y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(_with_interactions(X)[self.cols])[:, 1]


class Wrap:
    def __init__(self, m, impute=False):
        self.m = make_pipeline(SimpleImputer(strategy="median"), m) if impute else m

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


def make_models() -> dict:
    return {
        "로지스틱": lambda: Logit(),
        "결정나무(깊이3)": lambda: Wrap(DecisionTreeClassifier(max_depth=3, min_samples_leaf=10,
                                                         class_weight="balanced", random_state=SEED), impute=True),
        "EBM": lambda: Wrap(ExplainableBoostingClassifier(random_state=SEED, interactions=5, outer_bags=8,
                                                          min_samples_leaf=5, max_bins=64)),
        "CatBoost": lambda: Wrap(CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05,
                                                    auto_class_weights="Balanced", random_seed=SEED, verbose=0)),
        "LightGBM": lambda: Wrap(LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                                min_child_samples=10, subsample=0.8, subsample_freq=1,
                                                colsample_bytree=0.8, is_unbalance=True,
                                                random_state=SEED, verbose=-1)),
    }


# ---------- 단순 규칙 기준선 ----------
def rule_scores(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    clim = train.groupby([train["date"].dt.month, "is_offday"])["label"].mean()
    key = list(zip(test["date"].dt.month, test["is_offday"]))
    return {
        "규칙: 휴무일": test["is_offday"].astype(float).values,
        "규칙: 휴무일+맑음 예보": ((test["is_offday"] == 1) & (test["fc_clear_share"].fillna(0) >= 0.5)).astype(float).values,
        "규칙: 지속성": (test["prev_same_type_min_smp"] < 30).astype(float).values,
        "규칙: 기후값(월×휴무일)": np.array([clim.get(k, train["label"].mean()) for k in key]),
    }


# ---------- 평가 ----------
def split(df, rng):
    return df[df["date"].between(*rng)]


def metrics(y, p, pred=None) -> dict:
    out = {"급락일": int(y.sum()), "일수": len(y)}
    if y.sum() and y.sum() < len(y):
        out["PR-AUC"] = average_precision_score(y, p)
        out["ROC-AUC"] = roc_auc_score(y, p)
    if pred is not None:
        tp = int(((pred == 1) & (y == 1)).sum())
        out["재현율"] = tp / max(y.sum(), 1)
        out["정밀도"] = tp / max(pred.sum(), 1) if pred.sum() else np.nan
        out["예측 급락"] = int(pred.sum())
        out["적중"] = tp
    return out


class Platt:
    """롤링 OOF 예측으로 맞춘 확률 보정."""

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=5000).fit(self._z(p), y)
        return self

    @staticmethod
    def _z(p):
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def transform(self, p):
        return self.lr.predict_proba(self._z(p))[:, 1]


RECALL_TARGET = 0.7


def recall_threshold(y, p, target: float = RECALL_TARGET) -> float:
    """롤링 예측에서 재현율이 target 이상인 가장 높은 임계값 (놓침을 줄이는 운영용)."""
    for t in np.sort(np.unique(np.round(p, 4)))[::-1]:
        if ((p >= t) & (y == 1)).sum() / max(y.sum(), 1) >= target:
            return float(t)
    return float(np.min(p))


def best_threshold(y, p) -> float:
    grid = np.unique(np.round(p, 4))
    best, best_f1 = 0.5, -1
    for t in grid:
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        if pred.sum() == 0:
            continue
        f1 = 2 * tp / (pred.sum() + y.sum())
        if f1 > best_f1:
            best, best_f1 = t, f1
    return float(best)


def run(df: pd.DataFrame):
    rows, preds = [], {}
    models = make_models()
    for set_name, cols in FEATURE_SETS.items():
        for model_name, factory in list(models.items()) + [("규칙", None)]:
            oof_p, oof_y, fold_scores = {}, {}, {}
            for fold, (tr_rng, te_rng) in FOLDS.items():
                tr, te = split(df, tr_rng), split(df, te_rng)
                if factory is None:
                    if set_name != MAIN_SET:
                        break
                    scores = rule_scores(tr, te)
                else:
                    m = factory().fit(tr[cols], tr["label"])
                    scores = {model_name: m.predict_proba(te[cols])}
                for name, p in scores.items():
                    fold_scores.setdefault(name, {})[fold] = (te, p)
            for name, by_fold in fold_scores.items():
                is_rule = name.startswith("규칙")
                oof = [by_fold[f] for f in ("롤링1", "롤링2")]
                y_oof = np.concatenate([t["label"].values for t, _ in oof])
                p_oof = np.concatenate([p for _, p in oof])
                if is_rule:
                    # 0/1 규칙은 0.5, 확률을 내는 기후값 규칙은 롤링 예측으로 임계값 결정
                    cal = None
                    thr = best_threshold(y_oof, p_oof) if "기후값" in name else 0.5
                    thr_r = recall_threshold(y_oof, p_oof) if "기후값" in name else 0.5
                else:
                    cal = Platt().fit(p_oof, y_oof)
                    thr = best_threshold(y_oof, cal.transform(p_oof))
                    thr_r = recall_threshold(y_oof, cal.transform(p_oof))
                for fold, (te, p) in by_fold.items():
                    pc = p if is_rule else cal.transform(p)
                    pred = (pc >= thr).astype(int)
                    r = {"피처": set_name, "모델": name, "구간": fold, "임계값": thr,
                         **metrics(te["label"].values, pc, pred)}
                    pred_r = (pc >= thr_r).astype(int)
                    mr = metrics(te["label"].values, pc, pred_r)
                    r.update({"임계값(재현율우선)": thr_r, "재현율(R)": mr["재현율"], "정밀도(R)": mr["정밀도"],
                              "예측 급락(R)": mr["예측 급락"], "적중(R)": mr["적중"]})
                    if fold == "검증" and not is_rule:
                        r["Brier"] = brier_score_loss(te["label"], pc)
                    rows.append(r)
                    preds[(set_name, name, fold)] = (te, pc, pred, pred_r)
            print(f"  {set_name} / {model_name} 완료", flush=True)
    return pd.DataFrame(rows), preds


def choose(res: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    roll = res[(res["피처"] == MAIN_SET) & res["구간"].str.startswith("롤링") & ~res["모델"].str.startswith("규칙")]
    score = roll.groupby("모델")["PR-AUC"].mean().sort_values(ascending=False)
    best = score.index[0]
    best_interp = next(m for m in score.index if m in INTERPRETABLE)
    chosen = best_interp if (best not in INTERPRETABLE and score[best] - score[best_interp] < 0.05) else best
    return chosen, score


# ---------- 산출물 ----------
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.grid(True, axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_comparison(res: pd.DataFrame):
    roll = res[res["구간"].str.startswith("롤링")]
    tab = roll.groupby(["모델", "피처"])["PR-AUC"].mean().unstack()[list(FEATURE_SETS)]
    order = tab[MAIN_SET].sort_values().index
    tab = tab.loc[order]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(tab) + 1.6))
    h = 0.8 / len(FEATURE_SETS)
    for i, (s, c) in enumerate(zip(FEATURE_SETS, SERIES)):
        y = np.arange(len(tab)) + (i - 1.5) * h
        ax.barh(y, tab[s], height=h * 0.9, color=c, label=s, edgecolor="white", linewidth=1)
    for j, v in enumerate(tab[MAIN_SET]):
        if pd.notna(v):
            ax.text(v + 0.01, j + (list(FEATURE_SETS).index(MAIN_SET) - 1.5) * h, f"{v:.2f}",
                    va="center", fontsize=8, color=C_INK)
    ax.set_yticks(range(len(tab)), tab.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("PR-AUC (롤링 2회 평균, 높을수록 좋음)", color=C_INK2)
    ax.set_title(f"{REGION} 급락 분류: 모델 × 피처 묶음 비교 (숫자 = '{MAIN_SET}' 묶음)", loc="left", fontsize=11, color=C_INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"03_model_comparison_{REGION}.png", dpi=150)
    plt.close(fig)


def explain(df: pd.DataFrame):
    """최종 학습 구간으로 해석 모델을 다시 학습해 요인 기여를 저장."""
    cols = FEATURE_SETS[MAIN_SET]
    tr = split(df, FOLDS["검증"][0])
    out = {}
    tree = make_pipeline(SimpleImputer(strategy="median"),
                         DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, class_weight="balanced",
                                                random_state=SEED)).fit(tr[cols], tr["label"])
    rules = export_text(tree[-1], feature_names=cols, show_weights=True)
    (OUT_TAB / f"03_tree_rules_{REGION}.txt").write_text(rules, encoding="utf-8")
    out["tree"] = rules

    lg = Logit().fit(tr[cols], tr["label"])
    coef = pd.Series(lg.m[-1].coef_[0], index=lg.cols).sort_values(key=abs, ascending=False).round(3)
    coef.rename("표준화 계수").to_csv(OUT_TAB / f"03_logistic_coef_{REGION}.csv", encoding="utf-8-sig")
    out["coef"] = coef

    ebm = ExplainableBoostingClassifier(random_state=SEED, interactions=5, outer_bags=8,
                                        min_samples_leaf=5, max_bins=64).fit(tr[cols], tr["label"])
    g = ebm.explain_global().data()
    imp = pd.Series(g["scores"], index=g["names"]).sort_values(ascending=False)
    imp.rename("중요도(평균 |기여|)").to_csv(OUT_TAB / f"03_ebm_importance_{REGION}.csv", encoding="utf-8-sig")
    out["ebm"] = imp

    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(range(len(top)), top.values, color=SERIES[0], height=0.6)
    ax.set_yticks(range(len(top)), [ko(n) for n in top.index], fontsize=9)
    ax.set_xlabel("평균 |기여| (로짓)", color=C_INK2)
    ax.set_title(f"{REGION} EBM 요인 중요도 상위 12개 (학습 2024-01 ~ 2026-06, '{MAIN_SET}' 묶음)",
                 loc="left", fontsize=11, color=C_INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"03_ebm_importance_{REGION}.png", dpi=150)
    plt.close(fig)
    return out


def residuals(df: pd.DataFrame, preds: dict, chosen: str) -> pd.DataFrame:
    """선택 모델이 놓친 급락일(FN)·헛경보(FP)를 공급 측 정보와 함께 정리."""
    parts = []
    for fold in FOLDS:
        te, pc, pred, pred_r = preds[(MAIN_SET, chosen, fold)]
        parts.append(te.assign(prob=pc, pred=pred, pred_recall=pred_r, 구간=fold))
    allp = pd.concat(parts)
    allp["결과"] = np.select(
        [(allp["label"] == 1) & (allp["pred"] == 1), (allp["label"] == 1) & (allp["pred"] == 0),
         (allp["label"] == 0) & (allp["pred"] == 1)], ["적중", "놓침", "헛경보"], "정상")
    try:
        g = load_gen_by_fuel()
        g = g[g["hour"].between(11, 15)].groupby(["date", "fuel"])["mwh"].mean().unstack()
        sup = pd.DataFrame({"nuclear_mw": g.get("원자력"), "coal_mw": g.get("유연탄"), "lng_mw": g.get("LNG"),
                            "renew_mw": g.get("신재생·기타")})
        sup["nuclear_share"] = sup["nuclear_mw"] / g.sum(axis=1)
        allp = allp.merge(sup.reset_index(), on="date", how="left")
    except (FileNotFoundError, ValueError):
        pass
    allp["결과(재현율우선)"] = np.select(
        [(allp["label"] == 1) & (allp["pred_recall"] == 1), (allp["label"] == 1) & (allp["pred_recall"] == 0),
         (allp["label"] == 0) & (allp["pred_recall"] == 1)], ["적중", "놓침", "헛경보"], "정상")
    keep = ["date", "구간", "weekday_ko", "day_type", "holiday_name", "결과", "결과(재현율우선)", "prob", "label", "min_smp_11_15",
            "demand_fcst", "demand_rel28", "fc_clear_share", "fc_sky_mean", "act_cloud",
            "nuclear_mw", "nuclear_share", "coal_mw", "lng_mw"]
    allp = allp[[c for c in keep if c in allp.columns]]
    err = allp[allp["결과"].isin(["놓침", "헛경보"])].sort_values("date")
    err.round(3).to_csv(OUT_TAB / f"03_errors_{REGION}.csv", index=False, encoding="utf-8-sig")
    allp.round(4).to_csv(OUT_TAB / f"03_all_predictions_{REGION}.csv", index=False, encoding="utf-8-sig")
    return allp


def main():
    df = pd.read_csv(OUT_DATA / f"features_{REGION}.csv", parse_dates=["date"])
    df["label"] = df["label"].astype(int)
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    setup_korean_font()

    print("학습·평가 중...", flush=True)
    res, preds = run(df)
    res.round(4).to_csv(OUT_TAB / f"03_model_comparison_{REGION}.csv", index=False, encoding="utf-8-sig")
    chosen, score = choose(res)

    summary = (res[res["구간"].str.startswith("롤링")].groupby(["피처", "모델"])["PR-AUC"].mean().unstack(0)
               .reindex(columns=list(FEATURE_SETS)).round(3))
    valid = res[res["구간"] == "검증"].set_index(["피처", "모델"])[["PR-AUC", "재현율", "정밀도", "예측 급락", "적중",
                                                                 "재현율(R)", "정밀도(R)", "예측 급락(R)", "적중(R)", "Brier"]]
    summary.to_csv(OUT_TAB / f"03_rolling_summary_{REGION}.csv", encoding="utf-8-sig")
    plot_comparison(res)
    ex = explain(df)
    allp = residuals(df, preds, chosen)

    # ④용 검증 기간 예측 (선택 모델 + 규칙 기준선)
    te, pc, pred, pred_r = preds[(MAIN_SET, chosen, "검증")]
    vp = te[["date", "weekday_ko", "day_type", "holiday_name", "min_smp_11_15", "smp_12h", "label"]].copy()
    vp["급락확률"], vp["예측(F1)"], vp["예측(재현율우선)"] = pc.round(3), pred, pred_r
    for rule in ("규칙: 휴무일", "규칙: 휴무일+맑음 예보"):
        vp[rule] = preds[(MAIN_SET, rule, "검증")][2]
    vp.to_csv(OUT_TAB / f"03_valid_pred_{REGION}.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print(f"\n[롤링 PR-AUC 평균]\n{summary.to_string()}")
    print(f"\n'{MAIN_SET}' 묶음 순위: {score.round(3).to_dict()}")
    print(f"선택 모델: {chosen}")
    print(f"\n[검증 기간]\n{valid.round(3).to_string()}")
    print(f"\n[EBM 중요도 상위]\n{ex['ebm'].head(10).round(3).to_string()}")
    print(f"\n[결정나무 규칙]\n{ex['tree']}")
    print(f"\n[실패 사례 예측 — {chosen}]\n{vp[vp['date'].isin(pd.to_datetime(CASES))].to_string(index=False)}")
    err = allp[allp["결과"].isin(["놓침", "헛경보"])]
    print(f"\n[오류 {len(err)}건 중 놓친 급락]\n{err[err['결과'] == '놓침'].round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
