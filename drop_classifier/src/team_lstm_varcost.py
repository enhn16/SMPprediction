"""변동비 순위 자료 실험 (현업 공유 2026-10-30, 팀 13종 밖): 운영 구성(T2 회귀)에 피처만 더해 2025로만 판단.

결정: PLAN.md 결정 기록 2026-11-01 '변동비 순위 자료 실험 사전 고정'. 운영 파일·ops_drop.predict_team은 바꾸지 않는다.
자료: data/raw/변동비순위_발전기별_202203-202611.csv (연월, 연료구분, 최대출력_MW, 변동비_원kWh, 변동비순위, 누적출력_MW)
공개 가정: S1 = 그달 1일, S2 = 전달 25일. 대상일 D의 값 = 공개일 ≤ D-1(기준일)인 가장 최근 월.
후보: V1 계단(월별) / V2 위치(일별) / V3 = V1+V2 — 정의는 monthly_steps(), position().
채택(가정별): 2023~2024·2024 학습 → 2025 전체 날 헛경보@재현율80% 나쁜 쪽이 기준보다 3일 이상 적은 후보 중 가장 적은 것.
실행: python -m drop_classifier.src.team_lstm_varcost select → 24_varcost_select_2025_육지.csv, 24_varcost_pick.json (2026 안 봄)
      python -m drop_classifier.src.team_lstm_varcost final  → 채택 가정만 22_varcost_S?_team_lstm_{score,daily}_육지.csv
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from common.config import RAW_DIR
from drop_classifier.src.final import threshold_for
from drop_classifier.src.team_lstm import OUT_TAB, REGION, TR_SEL, VALID, combine, data, final, lgbm, lstm, split
from smp_forecast_app.src.team_scorecard import drop_stats

warnings.filterwarnings("ignore")
FILE = RAW_DIR / "변동비순위_발전기별_202203-202611.csv"
STEP = ["vc_cap_lt30_gw", "vc_step1_cost", "vc_coal_min", "vc_coal_med", "vc_lngd_min", "vc_lng_med"]
POS = ["vc_net_gw", "vc_net_margin_gw", "vc_net_step_cost"]
VARIANTS = {"V1 계단": STEP, "V2 위치": POS, "V3 계단+위치": STEP + POS}
CHEAP = 30                       # 급락 기준(원/kWh)과 같은 선: 이 아래 설비만으로 한낮 수요를 채우면 급락
SOLAR_CF = (0.3, 0.4)            # 한낮 태양광 이용률 추정 = 0.3 + 0.4 × 오후 맑음 비율 (사전 고정, 튜닝 안 함)
MIN_GAIN = 3


def merit() -> pd.DataFrame:
    """월별 변동비 순위표 정제: 2022-12 ~ 2025-02 대부분과 2026-07 파일(약 300대)에는 복합화력 가스터빈 단위 행(예: 포천복합GT#1)이
    변동비 0원으로 따로 있어 원자력 앞에 놓이고 누적출력을 약 31GW 부풀린다 → 변동비 0원인 LNG·유류 행 제외(수력·양수 0원은 유지),
    누적출력은 변동비 순으로 다시 더한다 (원문 순위는 일부 달에서 변동비 순과 다름). 2023-03 파일 없음 → 공개 가정대로 직전 달."""
    g = pd.read_csv(FILE, encoding="utf-8-sig")
    g["month"] = pd.to_datetime(g["연월"] + "-01")
    g = g[~((g["변동비_원kWh"] == 0) & g["연료구분"].isin(["LNG", "LNG(직도입)", "유류"]))]
    g = g.sort_values(["month", "변동비_원kWh", "변동비순위"])
    g["누적출력_MW"] = g.groupby("month")["최대출력_MW"].cumsum()
    return g


def monthly_steps(g: pd.DataFrame) -> pd.DataFrame:
    """월별 계단 값 (V1)."""
    rows = []
    for m, d in g.groupby("month"):
        c, f = d["변동비_원kWh"], d["연료구분"]
        rows.append({"month": m,
                     "vc_cap_lt30_gw": d.loc[c < CHEAP, "최대출력_MW"].sum() / 1000,
                     "vc_step1_cost": c[c >= CHEAP].min(),
                     "vc_coal_min": c[f == "석탄"].min(), "vc_coal_med": c[f == "석탄"].median(),
                     "vc_lngd_min": c[f == "LNG(직도입)"].min(), "vc_lng_med": c[f == "LNG"].median()})
    return pd.DataFrame(rows)


def usable_month(dates: pd.Series, months: pd.Series, assume: str) -> pd.Series:
    """대상일 D → 기준일 D-1에 공개된 가장 최근 월. S1 공개일 = 그달 1일, S2 = 전달 25일."""
    m = pd.DataFrame({"month": sorted(months.unique())})
    m["pub"] = m["month"] if assume == "S1" else m["month"] - pd.DateOffset(months=1) + pd.Timedelta(days=24)
    d = pd.DataFrame({"base": pd.to_datetime(dates.values) - pd.Timedelta(days=1), "i": range(len(dates))})
    out = pd.merge_asof(d.sort_values("base"), m.sort_values("pub"), left_on="base", right_on="pub")
    return out.sort_values("i")["month"].reset_index(drop=True)


def position(df: pd.DataFrame, g: pd.DataFrame, month: pd.Series) -> pd.DataFrame:
    """V2: 한낮 순수요 추정이 변동비 순위표에서 닿는 위치. 모두 기준일 15시에 아는 값(D-1 수요예측, D 주간전망, 설비, D-1 11시 예보)."""
    peak = df.set_index("date")["kpx_peak"]
    ratio = (peak / peak.shift(1, freq="D").reindex(peak.index)).reindex(df["date"]).values
    dem = df["demand_fcst_prev"].values * np.where(np.isfinite(ratio), ratio, 1.0) / 1000
    solar = df["cap_solar_gw"].values * (SOLAR_CF[0] + SOLAR_CF[1] * df["rf_clear_pm"].fillna(df["rf_clear_pm"].median()).values)
    net = dem - solar
    cap = {m: (d["누적출력_MW"].values / 1000, d["변동비_원kWh"].values) for m, d in g.groupby("month")}
    lt30 = {m: d.loc[d["변동비_원kWh"] < CHEAP, "최대출력_MW"].sum() / 1000 for m, d in g.groupby("month")}
    step = []
    for n, m in zip(net, month):
        cum, cost = cap[m]
        step.append(cost[min(np.searchsorted(cum, n), len(cost) - 1)] if np.isfinite(n) else np.nan)
    return pd.DataFrame({"vc_net_gw": net, "vc_net_margin_gw": net - month.map(lt30).values,
                         "vc_net_step_cost": step})


def add_varcost(df: pd.DataFrame, assume: str) -> pd.DataFrame:
    g = merit()
    month = usable_month(df["date"], g["month"], assume)
    st = monthly_steps(g).set_index("month").reindex(month.values).reset_index(drop=True)
    return pd.concat([df.reset_index(drop=True), st, position(df.reset_index(drop=True), g, month)], axis=1)


def cfg():
    return json.loads((OUT_TAB / "22_team_lstm_pick.json").read_text(encoding="utf-8"))["선택"]


def run(df, ex, c, y, off, va):
    r = {}
    for w, rng in TR_SEL.items():
        tr = split(df, rng)
        s = combine(lstm(tr, va, c["target"], c["seq"], extra=ex), lgbm(tr, va, c["target"], c["x"], extra=ex),
                    c["target"], c["comb"])
        a = drop_stats(y, s)
        r[f"헛경보@80% {w}학습"], r[f"헛경보@90% {w}학습"] = a["헛경보@재현율80%"], a["헛경보@재현율90%"]
        r[f"PR-AUC {w}학습"] = a["PR-AUC"]
        r[f"평일 헛경보@80% {w}학습"] = int(((s >= threshold_for(y, s, .8)) & ~off & (y == 0)).sum())
    r["헛경보@80% 나쁜쪽"] = max(r[f"헛경보@80% {w}학습"] for w in TR_SEL)
    r["PR-AUC 나쁜쪽"] = min(r[f"PR-AUC {w}학습"] for w in TR_SEL)
    return r


def select():
    t0 = time.time()
    c = cfg()
    base = data()
    base = base[base["date"] <= VALID[1]].reset_index(drop=True)        # 2026 행 제거 (열지 않음)
    va = split(base, VALID)
    y, off = va["label"].values, va["offday"].values
    rows = [{"가정": "-", "후보": "기준 (운영 구성)", "추가 열": "", "피처 수": 0, **run(base, [], c, y, off, va)}]
    print(f"  기준: {rows[0]['헛경보@80% 나쁜쪽']} ({time.time() - t0:.0f}초)", flush=True)
    for assume in ("S1", "S2"):
        df = add_varcost(base, assume)
        print(f"  {assume} 빈칸:", {k: int(df[k].isna().sum()) for k in STEP + POS}, flush=True)
        for name, ex in VARIANTS.items():
            rows.append({"가정": assume, "후보": name, "추가 열": ", ".join(ex), "피처 수": len(ex),
                         **run(df, ex, c, y, off, split(df, VALID))})
            print(f"  {assume} {name}: 헛경보@80% 나쁜쪽 {rows[-1]['헛경보@80% 나쁜쪽']}, "
                  f"PR-AUC 나쁜쪽 {rows[-1]['PR-AUC 나쁜쪽']:.3f} ({time.time() - t0:.0f}초)", flush=True)
    res = pd.DataFrame(rows)
    ref = res.loc[0, "헛경보@80% 나쁜쪽"]
    pick = {}
    for assume in ("S1", "S2"):
        ok = res[(res["가정"] == assume) & (res["헛경보@80% 나쁜쪽"] <= ref - MIN_GAIN)]
        if not ok.empty:
            pick[assume] = ok.sort_values(["헛경보@80% 나쁜쪽", "PR-AUC 나쁜쪽", "피처 수"],
                                          ascending=[True, False, True]).iloc[0]["후보"]
    res["채택"] = [("채택" if pick.get(a) == h else "") for a, h in zip(res["가정"], res["후보"])]
    res.round(4).to_csv(OUT_TAB / f"24_varcost_select_2025_{REGION}.csv", index=False, encoding="utf-8-sig")
    (OUT_TAB / "24_varcost_pick.json").write_text(json.dumps(
        {"채택": {a: {"후보": h, "추가 열": VARIANTS[h]} for a, h in pick.items()}, "기준 헛경보@80% 나쁜쪽": int(ref),
         "규칙": f"가정별로 기준보다 {MIN_GAIN}일 이상 적을 때만 채택 (PLAN.md 2026-11-01)"}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print("채택:", pick or "없음 (2026 열지 않음)")


def final_varcost():
    p = json.loads((OUT_TAB / "24_varcost_pick.json").read_text(encoding="utf-8"))["채택"]
    if not p:
        print("채택된 가정 없음 — 2026 채점하지 않음")
    for assume, v in p.items():
        final(extra=v["추가 열"], tag=f"_varcost_{assume}", df=add_varcost(data(), assume))


if __name__ == "__main__":
    if "select" in sys.argv:
        select()
    elif "final" in sys.argv:
        final_varcost()
