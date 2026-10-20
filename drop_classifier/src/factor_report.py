"""급락 요인 종합 분석 (육지, 최종 모델 기준).

1. 요인 묶음 제거 실험: 묶음을 하나씩 빼고 롤링 평가 → 성공 기준(헛경보@80·90%)이 얼마나 나빠지는가
2. 최종 하루 전 Ridge 계수: 표준편차 1 변화당 예측 한낮 최저 SMP 변화(원)
3. 기여 분해: 급락일·정상 휴무일·평일에서 각 요인이 예측을 평균 대비 몇 원 끌어내렸는가
4. EBM 회귀 모양: 주요 요인의 수준별 효과(비선형)
5. 휴무일 수요 × 하늘상태 예보 조합별 급락 비율
6. 일주일 모델: 리드(2~7일 전)별 Ridge 계수

실행: python -m drop_classifier.src.factor_report
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.config import PROCESSED_DIR, ROOT, SEED
from common.plot import setup_korean_font
from drop_classifier.src.features import CALENDAR, DEMAND_PREV, FCST, LAG, ko
from drop_classifier.src.features_weekly import FEATURE_KO_W, W_CALENDAR, W_MID, W_STATE
from drop_classifier.src.models import FOLDS, split
from drop_classifier.src.outlook_features import OUTLOOK
from drop_classifier.src.search import CANDIDATES, day_table, evaluate

warnings.filterwarnings("ignore")
REGION = "육지"
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
OUT_FIG = ROOT / "drop_classifier" / "outputs" / "figures"
GROUPS = {
    "달력(휴일·연휴·명절·계절)": CALENDAR,
    "전날까지 수요예측": DEMAND_PREV,
    "최근 SMP 흐름": LAG,
    "단기 날씨 예보": FCST,
    "주간 전력수요전망": OUTLOOK,
}
FINAL = CALENDAR + DEMAND_PREV + LAG + FCST + OUTLOOK
KO_EXTRA = {"kpx_peak": "주간전망 최대부하", "kpx_peak_rel": "주간전망 최대부하(그 주 최고 대비)"}
C_DROP, C_NORMAL, C_MUTED, C_INK, C_INK2 = "#eb6834", "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e"


def name(c):
    return KO_EXTRA.get(c) or FEATURE_KO_W.get(c) or ko(c)


def _style(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.grid(True, axis=grid, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def ablation(df):
    fn = CANDIDATES["Ridge 회귀(α=3)"]
    rows = [{"구성": "전체(최종 모델)", **evaluate(df, FINAL, fn)}]
    for g, cols in GROUPS.items():
        rows.append({"구성": f"− {g}", **evaluate(df, [c for c in FINAL if c not in cols], fn)})
    res = pd.DataFrame(rows)
    base = res.iloc[0]
    for k in ("헛경보@80%", "헛경보@90%"):
        res[f"{k} 증가"] = res[k] - base[k]
    return res


def fit_ridge(df, cols):
    tr = split(df, FOLDS["검증"][0])
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3))
    m.fit(tr[cols], tr["min_smp_11_15"].clip(0, 150))
    return m, tr


def coefficients(m, cols):
    c = pd.Series(m[-1].coef_, index=cols)
    out = pd.DataFrame({"요인": [name(x) for x in cols], "피처": cols, "계수(원/표준편차)": c.values})
    out["급락 방향"] = np.where(c.values < 0, "값이 클수록 급락 위험↑", "값이 작을수록 급락 위험↑")
    return out.reindex(out["계수(원/표준편차)"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def contributions(m, df, cols):
    """각 행의 (표준화 값 × 계수) = 평균 대비 예측 변화(원). 그룹 평균."""
    X = m[:-1].transform(df[cols])
    contrib = pd.DataFrame(X * m[-1].coef_, columns=[name(c) for c in cols], index=df.index)
    grp = np.where(df["label"] == 1, "급락일", np.where(df["day_type"] == "평일", "정상 평일", "정상 휴무일"))
    tab = contrib.groupby(grp).mean().T.round(1)
    return tab[["급락일", "정상 휴무일", "정상 평일"]]


def ebm_shapes(df, cols, top):
    tr = split(df, FOLDS["검증"][0])
    ebm = ExplainableBoostingRegressor(random_state=SEED, interactions=0, min_samples_leaf=5, max_bins=32, n_jobs=1)
    ebm.fit(tr[cols], tr["min_smp_11_15"].clip(0, 150))
    g = ebm.explain_global()
    names = g.data()["names"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.flat, top):
        d = g.data(names.index(c))
        edges, scores = np.asarray(d["names"], float), np.asarray(d["scores"], float)
        if len(edges) == len(scores) + 1:
            xs = np.repeat(edges, 2)[1:-1]
            ys = np.repeat(scores, 2)
        else:
            xs, ys = edges, scores
        ax.plot(xs, ys, color=C_NORMAL, linewidth=2)
        ax.axhline(0, color=C_MUTED, linewidth=1)
        ax.fill_between(xs, ys, 0, where=np.asarray(ys) < 0, color=C_DROP, alpha=0.25, step=None)
        ax.set_title(name(c), loc="left", fontsize=10, color=C_INK)
        ax.set_ylabel("예측 최저 SMP 변화(원)", color=C_INK2, fontsize=8.5)
        _style(ax, "y")
    fig.suptitle(f"{REGION} 주요 요인의 수준별 효과 (EBM, 아래로 = 급락 위험↑, 주황 = 평균보다 위험)", x=0.01, ha="left",
                 fontsize=12, color=C_INK)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"06_factor_shapes_{REGION}.png", dpi=150)
    plt.close(fig)


def interaction_table(df):
    off = df[df["is_offday"] == 1].copy()
    off["수요(28일 대비)"] = pd.cut(off["demand_prev_rel28"], [0, 0.80, 0.90, 9], labels=["0.80 미만", "0.80~0.90", "0.90 이상"])
    off["하늘상태 예보"] = pd.cut(off["fc_sky_mean"], [0, 1.8, 3.0, 4.1], labels=["맑음(≤1.8)", "구름(1.8~3)", "흐림(>3)"])
    t = off.pivot_table(index="수요(28일 대비)", columns="하늘상태 예보", values="label", aggfunc=["mean", "size"])
    rate, n = t["mean"], t["size"]
    out = rate.applymap(lambda v: f"{v * 100:.0f}%") + " (" + n.astype(int).astype(str) + "일)"
    return out, rate


def plot_ablation_coef(abl, coef):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    a = abl.iloc[1:].copy()
    y = np.arange(len(a))
    axes[0].barh(y - 0.2, a["헛경보@80% 증가"], height=0.38, color=C_NORMAL, label="적중 80% 기준")
    axes[0].barh(y + 0.2, a["헛경보@90% 증가"], height=0.38, color=C_DROP, label="적중 90% 기준")
    axes[0].set_yticks(y, [s.replace("− ", "") for s in a["구성"]], fontsize=9)
    axes[0].axvline(0, color=C_INK2, linewidth=1)
    axes[0].set_xlabel("빼면 늘어나는 헛경보(일, 롤링 평가)", color=C_INK2)
    axes[0].set_title("요인 묶음을 뺐을 때 성공 기준 악화", loc="left", fontsize=11, color=C_INK)
    axes[0].legend(frameon=False, fontsize=8.5)
    _style(axes[0])
    c = coef.head(14)[::-1]
    colors = [C_DROP if v < 0 else C_NORMAL for v in c["계수(원/표준편차)"]]
    axes[1].barh(range(len(c)), c["계수(원/표준편차)"], color=colors, height=0.6)
    axes[1].set_yticks(range(len(c)), c["요인"], fontsize=9)
    axes[1].axvline(0, color=C_INK2, linewidth=1)
    axes[1].set_xlabel("표준편차 1 증가 시 예측 한낮 최저 SMP 변화(원)", color=C_INK2)
    axes[1].set_title("최종 하루 전 모델 계수 (주황 = 값이 클수록 급락 위험↑)", loc="left", fontsize=11, color=C_INK)
    _style(axes[1])
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"06_factor_importance_{REGION}.png", dpi=150)
    plt.close(fig)


def weekly_coefs():
    w = pd.read_csv(PROCESSED_DIR / "drop_classifier" / f"features_weekly_{REGION}.csv", parse_dates=["date"])
    rows = {}
    for h in range(2, 8):
        d = w[w["lead"] == h]
        cols = W_CALENDAR + W_STATE + (W_MID if h >= 5 else [])
        m, _ = fit_ridge(d, cols)
        rows[f"{h}일 전"] = pd.Series(m[-1].coef_, index=[name(c) for c in cols])
    t = pd.DataFrame(rows).round(1)
    return t.reindex(t.abs().max(axis=1).sort_values(ascending=False).index)


def main():
    setup_korean_font()
    pd.set_option("display.width", 250)
    df = day_table()

    abl = ablation(df)
    abl.round(3).to_csv(OUT_TAB / f"06_factor_ablation_{REGION}.csv", index=False, encoding="utf-8-sig")
    m, tr = fit_ridge(df, FINAL)
    coef = coefficients(m, FINAL)
    coef.round(2).to_csv(OUT_TAB / f"06_factor_coef_{REGION}.csv", index=False, encoding="utf-8-sig")
    contrib = contributions(m, df[df["date"] >= "2024-01-01"], FINAL)
    contrib.to_csv(OUT_TAB / f"06_factor_contrib_{REGION}.csv", encoding="utf-8-sig")
    inter, _ = interaction_table(df)
    inter.to_csv(OUT_TAB / f"06_factor_interaction_{REGION}.csv", encoding="utf-8-sig")
    wk = weekly_coefs()
    wk.to_csv(OUT_TAB / f"06_factor_weekly_coef_{REGION}.csv", encoding="utf-8-sig")

    plot_ablation_coef(abl, coef)
    top = [c for c in coef["피처"] if c not in ("month_sin", "month_cos")][:6]
    ebm_shapes(df, FINAL, top)

    cols = ["구성", "헛경보@80%", "헛경보@90%", "헛경보@80% 증가", "헛경보@90% 증가", "PR-AUC 평균"]
    print(f"[1. 묶음 제거]\n{abl[cols].round(3).to_string(index=False)}")
    print(f"\n[2. 계수]\n{coef.round(1).to_string(index=False)}")
    print(f"\n[3. 기여 분해(원, 평균 대비)]\n{contrib.to_string()}")
    print(f"\n[5. 휴무일 수요 × 하늘상태 급락 비율]\n{inter.to_string()}")
    print(f"\n[6. 일주일 모델 계수]\n{wk.head(12).to_string()}")
    print("\nEBM 모양 그림 요인:", [name(c) for c in top])


if __name__ == "__main__":
    main()
