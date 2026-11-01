"""변동비 순위(전력거래소 월별, 현업 공유 2026-10-30)가 가격 예측을 낫게 하는지 확인 — 정보 가치 실험 (운영 모델은 그대로).

방법 (채점 전 고정, 2026-11-01)
  지금 운영 중인 예측(일주일 = WZ_predictions_hourly, 한 달 = mx v2) 위에 '가격 계단 변화' 보정만 더한다:
    보정 = β × (V[대상월, 판단 시점에 알려진 값] − V[기준일 7일 전이 속한 월])
  V 후보 3개: A = LNG(직도입) 변동비 중앙값, B = LNG 변동비 중앙값, C = LNG 전체(직도입 포함) 출력가중평균
  공개 시점 가정 2개 (실제 공개일 미확인): S1 보수 = 그달 1일 공개, S2 관찰 = 전달 25일 공개 (10/30에 11월분 확인)
  β는 2025년 대상일로만 최소제곱(절편 없음) — 일주일은 2025-07~12(백테스트 시작), 한 달은 2025 전체, 지표별 따로.
  선택 = 2025 MAE가 가장 많이 준 V (0.05원 이내면 A). 2026-01-01 ~ 10-18 한 번 채점, 전체·월초 7일·그 밖 나눠 보고.
입력: data/raw/변동비_연료별_월별요약_202203-202611.csv (현업 공유, 13종 밖 — 실험용)
출력: outputs/tables/VC1_select_2025.csv, VC2_test2026.csv
실행: python -m smp_forecast_app.src.vc_check
"""
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, RAW_DIR, ROOT
DATA = PROCESSED_DIR / "smp_forecast_app"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"

T0, T1 = "2026-01-01", "2026-10-18"


def vc_table() -> pd.DataFrame:
    s = pd.read_csv(RAW_DIR / "변동비_연료별_월별요약_202203-202611.csv", encoding="utf-8-sig")
    s["월"] = pd.PeriodIndex(s["연월"], freq="M")
    g = lambda f, c: s[s["연료구분"] == f].set_index("월")[c]
    lng = s[s["연료구분"].isin(["LNG", "LNG(직도입)"])].copy()
    lng["w"] = lng["최대출력합_MW"] * lng["변동비_출력가중평균"]
    c = lng.groupby("월").apply(lambda x: x["w"].sum() / x["최대출력합_MW"].sum())
    return pd.DataFrame({"A": g("LNG(직도입)", "변동비_중앙값"), "B": g("LNG", "변동비_중앙값"), "C": c}).sort_index()


def delta(base: pd.Series, date: pd.Series, V: pd.Series, scen: str) -> np.ndarray:
    """대상일마다 V[알려진 대상월 값] − V[기준일 7일 전 월]. 대상월 값이 아직 공개 전이면 공개된 가장 최근 월 값."""
    tm, ref = date.dt.to_period("M"), (base - pd.Timedelta(days=7)).dt.to_period("M")
    # 기준일에 알려진 가장 늦은 월: S1 = 기준일의 월, S2 = 기준일이 25일 이상이면 다음 달까지
    known = base.dt.to_period("M") + ((base.dt.day >= 25).astype(int) if scen == "S2" else 0)
    use = pd.Series(np.minimum(tm.astype("int64"), known.astype("int64")), index=date.index)
    vmap = {p.ordinal: v for p, v in V.items()}
    v_t = use.map(vmap)
    v_r = ref.map(lambda p: vmap.get(p.ordinal))
    return (v_t - v_r).fillna(0).to_numpy()


def fit(err, d):
    m = np.abs(d) > 0
    return float((d[m] @ err[m]) / (d[m] @ d[m])) if m.sum() > 10 and (d[m] @ d[m]) > 0 else 0.0


def evaluate(name, df, ycol, pcol, V, scen):
    """df: base_date, date, y, pred → 2025 β 적합 + 2025·2026 MAE (보정 전/후)."""
    d = delta(df["base_date"], df["date"], V, scen)
    err = (df[ycol] - df[pcol]).to_numpy()
    sel = (df["date"].dt.year == 2025).to_numpy()
    test = df["date"].between(T0, T1).to_numpy()
    b = fit(err[sel], d[sel])
    adj = df[pcol].to_numpy() + b * d
    first7 = (df["date"].dt.day <= 7).to_numpy()
    mae = lambda m, p: float(np.mean(np.abs(df[ycol].to_numpy()[m] - p[m]))) if m.sum() else np.nan
    p0 = df[pcol].to_numpy()
    return {"예측": name, "β": round(b, 3),
            "2025 MAE 전": mae(sel, p0), "2025 MAE 후": mae(sel, adj),
            "2026 MAE 전": mae(test, p0), "2026 MAE 후": mae(test, adj),
            "2026 월초7일 전": mae(test & first7, p0), "2026 월초7일 후": mae(test & first7, adj),
            "2026 그 밖 전": mae(test & ~first7, p0), "2026 그 밖 후": mae(test & ~first7, adj),
            "보정된 비율": float((np.abs(d[test]) > 0).mean())}


def main():
    vc = vc_table()
    w = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["base_date", "date"]).dropna(subset=["y", "pred"])
    m = pd.read_csv(DATA / "mx" / "MX_predictions_daily_v2.csv", parse_dates=["base_date", "date"])
    sets = [("일주일 (시간별)", w, "y", "pred")] + [(f"한 달 · {n}", m.dropna(subset=[f"y_{k}", f"pred_{k}"]), f"y_{k}", f"pred_{k}")
                                                 for k, n in (("mean", "일평균"), ("mid", "한낮"), ("eve", "저녁"), ("midmin", "한낮 최저"))]
    rows = []
    for scen in ("S1", "S2"):
        for v in ("A", "B", "C"):
            for name, df, yc, pc in sets:
                rows.append({"공개 가정": scen, "V": v, **evaluate(name, df, yc, pc, vc[v], scen)})
    r = pd.DataFrame(rows)
    r["2025 개선"] = r["2025 MAE 전"] - r["2025 MAE 후"]
    # 선택: 예측별·공개 가정별로 2025 개선이 가장 큰 V (0.05원 이내면 A)
    pick = []
    for (scen, name), g in r.groupby(["공개 가정", "예측"], sort=False):
        best = g["2025 개선"].max()
        a = g[g["V"] == "A"].iloc[0]
        pick.append(a if best - a["2025 개선"] <= 0.05 else g.loc[g["2025 개선"].idxmax()])
    p = pd.DataFrame(pick)
    r.round(3).to_csv(TAB / "VC1_select_2025.csv", index=False, encoding="utf-8-sig")
    p.round(3).to_csv(TAB / "VC2_test2026.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(vc.tail(8).round(1).to_string())
    print(p[["공개 가정", "예측", "V", "β", "2025 MAE 전", "2025 MAE 후", "2026 MAE 전", "2026 MAE 후",
             "2026 월초7일 전", "2026 월초7일 후", "2026 그 밖 전", "2026 그 밖 후", "보정된 비율"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
