"""사후 분석 (참고, 모델 선택에 쓰지 않음): '유망해 보였지만 효과가 없었던' 자료·모델 확인.

틀: 매달 다시 학습(2020-01 ~ 전달 말) → 2024-01 ~ 2026-10 예측 (drop_walk·surge_walk와 같음). 문턱 없이 순위 지표로 비교.
  - 급락: 계산 시간 때문에 LightGBM 구성원(T2 회귀, 운영 설정)만 / 급등: 세 모델 평균, 새 레이블(y_new)로 학습·채점
변형
  기본            운영 입력 그대로
  2023년부터 학습  2020~2022 보강 자료를 뺐을 때
  − 날씨 예보      구역 예보(급락 rf_*, 급등 wk_sky_pm)·Open-Meteo(급등 wx_*, wk_cloud_mid_jma) 제외
  − 주간전망       kpx_peak(_rel) 제외
  − 연료비         (급락만) fc_* 제외
  + 거래소 D일 수요예측 [상한선]  D일 하루전 수요예측(SMP와 같은 시각 공표 → 15시엔 못 씀) = 거래소의 전날 전망
  + D일 실측 태양광 [상한선]      지역별 시간별 태양광 실적(시장 참여분, 2025-12까지) = 실제 날씨의 결과
지표: PR-AUC, ROC-AUC, 헛경보@재현율80% (날짜). 기간: 2024-01~2025-12(실측 태양광 있는 공통 기간) / 2024-01~2026-10.
구성원 비교(급락 LSTM·LightGBM·평균)는 drop_walk 결과 파일에서 계산.

실행: python -m extended_backtest.src.ablation → outputs/tables/A1_ablation.csv, A2_members.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common.config import PROCESSED_DIR, ROOT
from common.loaders import load_renewables
from drop_classifier.src.ens8 import COLS, LGBM
from drop_classifier.src.fuel_level import FUEL, LEVEL
from lightgbm import LGBMRegressor
from surge_classifier.src import team_model2 as T2
from surge_classifier.src import team_model2_fa as FA
from extended_backtest.src.drop_features import smp_forecast_all
from extended_backtest.src.surge_features import CFG

TAB = ROOT / "extended_backtest" / "outputs" / "tables"
FEAT = PROCESSED_DIR / "extended_backtest"
MONTHS = pd.period_range("2024-01", "2026-10", freq="M")
WINDOWS = {"2024-01~2025-12 (공통)": ("2024-01-01", "2025-12-31"), "2024-01~2026-10": ("2024-01-01", "2026-10-31")}


def fa_at(y, s, r=0.8):
    need = int(np.ceil(r * y.sum()))
    order = np.argsort(-s)
    tp = np.cumsum(y[order] == 1)
    k = int(np.argmax(tp >= need))
    return int((y[order][:k + 1] == 0).sum())


def kpx_dday() -> pd.DataFrame:
    """거래소 D일 하루전 수요예측: 한낮(11~15시) 평균·최대, 직전 28일 중앙값 대비."""
    fc = smp_forecast_all()
    f = fc[fc["region"] == "육지"]
    mid = f[f["hour"].between(11, 15)].groupby("date")["demand_fcst"].mean().asfreq("D")
    mx = f.groupby("date")["demand_fcst"].max().asfreq("D")
    return pd.DataFrame({"kpx_dem_mid_d": mid, "kpx_dem_mid_rel28_d": mid / mid.shift(1).rolling(28, min_periods=20).median(),
                         "kpx_dem_max_d": mx, "kpx_dem_max_rel28_d": mx / mx.shift(1).rolling(28, min_periods=20).median()})


def solar_actual() -> pd.DataFrame:
    r = load_renewables()
    r = r[r["region"] == "육지"]
    mid = r[r["hour"].between(11, 15)].groupby("date")["solar"].sum().asfreq("D")
    return pd.DataFrame({"solar_mid_d": mid, "solar_mid_rel28_d": mid / mid.shift(1).rolling(28, min_periods=20).median()})


def attach(df):
    return df.join(kpx_dday(), on="date").join(solar_actual(), on="date")


def walk(df, fit_pred, cols, label, start="2020-01-01"):
    out = []
    for m in MONTHS:
        tr = df[(df["date"] >= start) & (df["date"] < m.start_time) & df[label].notna()]
        te = df[(df["date"].dt.to_period("M") == m) & df[label].notna()]
        out.append(te[["date", label]].assign(score=fit_pred(tr, te, cols, label)))
    return pd.concat(out, ignore_index=True).rename(columns={label: "y"})


def drop_fit(tr, te, cols, label):
    m = LGBMRegressor(**{k: v for k, v in LGBM.items() if k != "class_weight"})
    return -m.fit(tr[cols], tr["min_smp_11_15"].clip(0, 150)).predict(te[cols])


def surge_fit(tr, te, cols, label):
    y = tr[label].astype(int)
    keep_signs = {k: v for k, v in T2.SIGNS.items() if k in cols}
    old = dict(T2.SIGNS)
    T2.SIGNS.clear(); T2.SIGNS.update(keep_signs)
    try:
        ps = [T2.MAKE[k]().fit(tr[cols], y).predict(te[cols]) for k in T2.KEYS]
    finally:
        T2.SIGNS.clear(); T2.SIGNS.update(old)
    return np.mean(ps, axis=0)


def metrics(res):
    rows = {}
    for wname, (a, b) in WINDOWS.items():
        x = res[res["date"].between(a, b)].dropna(subset=["score"])
        y, s = x["y"].values.astype(int), x["score"].values
        rows[wname] = {"사건일": int(y.sum()), "PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s),
                       "헛경보@재현율80%": fa_at(y, s)}
    return rows


def main():
    rows = []
    # 급락 (LightGBM 구성원)
    d = attach(pd.read_pickle(FEAT / "drop_features_육지.pkl"))
    base = COLS + FUEL + LEVEL
    rf = [c for c in base if c.startswith("rf_")]
    var = {"기본": (base, "2020-01-01"), "2023년부터 학습": (base, "2023-01-01"),
           "− 날씨 예보": ([c for c in base if c not in rf], "2020-01-01"),
           "− 주간전망": ([c for c in base if c not in ("kpx_peak", "kpx_peak_rel")], "2020-01-01"),
           "− 연료비": ([c for c in base if c not in FUEL], "2020-01-01"),
           "+ 거래소 D일 수요예측 [상한선]": (base + ["kpx_dem_mid_d", "kpx_dem_mid_rel28_d"], "2020-01-01"),
           "+ D일 실측 태양광 [상한선]": (base + ["solar_mid_d", "solar_mid_rel28_d"], "2020-01-01")}
    for name, (cols, start) in var.items():
        res = walk(d, drop_fit, cols, "label", start)
        for w, m in metrics(res).items():
            rows.append({"사건": "육지 급락 (LightGBM 구성원)", "변형": name, "기간": w, **m})
        print("급락", name, {w: round(m["PR-AUC"], 3) for w, m in metrics(res).items()}, flush=True)

    # 급등 (세 모델 평균, 새 레이블)
    s = attach(pd.read_pickle(FEAT / "surge_features_육지.pkl"))
    FA.configure(CFG)
    try:
        base = list(T2.FEATS)
        wx = [c for c in base if c.startswith("wx_")] + ["wk_sky_pm", "wk_cloud_mid_jma"]
        var = {"기본": (base, "2020-01-01"), "2023년부터 학습": (base, "2023-01-01"),
               "− 날씨 예보": ([c for c in base if c not in wx], "2020-01-01"),
               "− 주간전망": ([c for c in base if c != "kpx_peak"], "2020-01-01"),
               "+ 거래소 D일 수요예측 [상한선]": (base + ["kpx_dem_max_d", "kpx_dem_max_rel28_d"], "2020-01-01"),
               "+ D일 실측 태양광 [상한선]": (base + ["solar_mid_d", "solar_mid_rel28_d"], "2020-01-01")}
        for name, (cols, start) in var.items():
            res = walk(s, surge_fit, cols, "y_new", start)
            for w, m in metrics(res).items():
                rows.append({"사건": "육지 급등 (세 모델 평균, 새 레이블)", "변형": name, "기간": w, **m})
            print("급등", name, {w: round(m["PR-AUC"], 3) for w, m in metrics(res).items()}, flush=True)
    finally:
        FA.configure({})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "A1_ablation.csv", index=False, encoding="utf-8-sig")

    # 급락 구성원 비교 (drop_walk 결과)
    p = TAB / "D1_drop_walk_daily.csv"
    if p.exists():
        dd = pd.read_csv(p, parse_dates=["date"])
        mem = []
        global WINDOWS
        WINDOWS = {"2024 (연습)": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"),
                   "2026 (2차 시험)": ("2026-01-01", "2026-10-31")}
        for col, name in (("pred_lstm", "LSTM 혼자"), ("pred_lgbm", "LightGBM 혼자"), ("pred", "두 모델 평균 (운영)")):
            res = dd[["date", "label"]].rename(columns={"label": "y"}).assign(score=-dd[col])
            for w, m in metrics(res).items():
                mem.append({"모델": name, "기간": w, **m})
        pd.DataFrame(mem).to_csv(TAB / "A2_members.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
