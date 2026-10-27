"""팀 공통 성적표 — 현재 모델(육지·제주)을 팀 규칙대로 같은 코드로 채점해 팀원 모델과 비교한다.

팀 공통 규칙 (2026-10-26, 루트 CLAUDE.md)
  시험 기간 2026-01-01 ~ 2026-10-18 (현재 모델 공식 성적 기간), 급락일 = 휴일·주말만 (11~15시 최저 30원 미만)
지표 (ROLES.md·참고 문서 '성적 지표 읽는 법'과 같은 정의)
  경보: PR-AUC(주 비교)·무작위 PR-AUC(= 급락 비율)·PR-AUC 배수·ROC-AUC·Brier, 헛경보@재현율80%(현업 기준)·90%, 실전(2025로 정한 문턱) 적중·놓침·헛경보·재현율·정밀도.
        PR-AUC·헛경보@재현율은 날짜 부트스트랩 2,000회 95% 구간 병기 (급락일이 적어 차이가 우연일 수 있음)
  가격: MAE·RMSE·MAPE(공표 20원 이상)·rMAE·편향·포함률·하한 아래·상한 위·평균 폭·구간점수
        단순 기준(rMAE): 일주일 = 지난주 같은 시간, 내일(D+1) = 전날 같은 시간, 한 달 = 기준일까지 같은 유형일 4일 평균 (두 계통 같은 정의)
입력 (읽기만): 육지 WZ_predictions_hourly.csv·mx/MX_predictions_daily_v2.csv·drop_classifier/outputs/tables/drop_육지_15h.csv(R1), 제주 jeju/의 같은 파일
출력: outputs/tables/T1_team_price.csv, T2_team_drop.csv, T_team.json
실행: python -m smp_forecast_app.src.team_scorecard
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT, SEED
from common.loaders import load_smp_actual
from smp_forecast_app.src.final_metrics import hourly_segments, metrics
from smp_forecast_app.src import range_adjust
from smp_forecast_app.src.jz_month import daily_stats

DATA = PROCESSED_DIR / "smp_forecast_app"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
T0, T1 = "2026-01-01", "2026-10-18"
STATS = {"mean": "일평균", "mid": "한낮 11~15시", "eve": "저녁 18~22시", "midmin": "한낮 최저"}
SRC = {"육지": {"week": DATA / "WZ_predictions_hourly.csv", "month": DATA / "mx" / "MX_predictions_daily_v2.csv", "drop": ROOT / "drop_classifier" / "outputs" / "tables" / "drop_육지_15h.csv",
               "model": {"drop": "Ridge+EBM+랜덤포레스트 (예상 한낮 최저)", "week": "Chronos-2 small + 0.5 × 잔차 보정",
                         "month": "mx v2: Chronos 3종·LSTM·GRU 조합 (저녁은 9개 모델 가중)"}},
       "제주": {"week": DATA / "jeju" / "JWZ_predictions_hourly.csv", "month": DATA / "jeju" / "JMZ_predictions_daily.csv",
               "drop": DATA / "jeju" / "drop_badge_제주.csv",
               "model": {"drop": "KNN 분류 (피처 63개)", "week": "육지 일주일 예측 + 0.25 × 제주 잔차 보정",
                         "month": "육지 한 달 예측 + 지표별 제주−육지 차이"}}}


def in_test(d):
    return (d >= T0) & (d <= T1)


# ---------------------------------------------------------------- 가격
def price_rows(region):
    act = load_smp_actual(region)[["date", "hour", "smp"]]
    lag = lambda k: act.assign(date=act["date"] + pd.Timedelta(days=k)).rename(columns={"smp": f"lag{k}"})
    w = pd.read_csv(SRC[region]["week"], parse_dates=["base_date", "date"])
    w = w[w["y"].notna() & in_test(w["date"])].merge(lag(7), on=["date", "hour"], how="left").merge(lag(1), on=["date", "hour"], how="left")
    w = range_adjust.apply(w.reset_index(drop=True), region)   # 운영·웹과 같은 범위 보정 (휴일·주말 한낮 하한, RA_choice.json)
    rows = []
    for prod, g, naive in (("일주일", w, "lag7"), ("일주일 D+1", w[w["H"] == 1], "lag1")):
        for seg, m in hourly_segments(g).items():
            rows.append({"계통": region, "예측": prod, "구분": seg, **metrics(g[m]["y"], g[m]["pred"], g[m]["lo"], g[m]["hi"], g[m][naive])})
        if prod == "일주일":
            for H, x in g.groupby("H"):
                rows.append({"계통": region, "예측": prod, "구분": f"D+{H}", **metrics(x["y"], x["pred"], x["lo"], x["hi"], x[naive])})

    # 한 달: 기준 = 기준일까지 같은 유형일 4일 평균
    ds = daily_stats(region)
    full = pd.date_range(ds.index.min(), ds.index.max())
    ds = ds.reindex(full)
    off = build_calendar(full[0], full[-1]).set_index("date")["is_offday"].reindex(full).astype(bool)
    same4 = {o: ds[off == o].rolling(4, min_periods=3).mean().reindex(full).ffill() for o in (False, True)}
    m = pd.read_csv(SRC[region]["month"], parse_dates=["base_date", "date"])
    m = m[in_test(m["date"])]
    for s, nm in STATS.items():
        x = m[m[f"y_{s}"].notna() & m[f"pred_{s}"].notna()]
        naive = np.where(x["offday"].astype(bool), same4[True][s].reindex(x["base_date"]).values, same4[False][s].reindex(x["base_date"]).values)
        rows.append({"계통": region, "예측": "한 달", "구분": nm, **metrics(x[f"y_{s}"], x[f"pred_{s}"], x[f"lo_{s}"], x[f"hi_{s}"], naive)})
        wk = ((x["H"] - 7) // 7 + 1).values   # 1주차 = 7~13일 뒤 … 4주차 = 28~34일 뒤
        for k in range(1, 5):
            i = wk == k
            rows.append({"계통": region, "예측": "한 달", "구분": f"{nm} · {k}주차",
                         **metrics(x[f"y_{s}"][i], x[f"pred_{s}"][i], x[f"lo_{s}"][i], x[f"hi_{s}"][i], naive[i])})
    return rows


# ---------------------------------------------------------------- 경보
def fa_at(y, s, r):
    """재현율 r을 처음 넘기는 문턱(점수 내림차순)에서 헛경보 수 — 시험 기간 안에서 문턱을 맞춘 값(순위 비교용)."""
    o = np.argsort(-s, kind="stable")
    k = np.searchsorted(np.cumsum(y[o]), int(np.ceil(r * y.sum())))
    return int(((s >= s[o][k]) & (y == 0)).sum())


def drop_stats(y, s):
    return {"PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s),
            "헛경보@재현율80%": fa_at(y, s, .8), "헛경보@재현율90%": fa_at(y, s, .9)}


def drop_rows(region, n_boot=2000):
    d = pd.read_csv(SRC[region]["drop"], parse_dates=["date"])
    d = d[in_test(d["date"])].dropna(subset=["label"])
    off = build_calendar(T0, T1).set_index("date")["is_offday"].astype(bool)
    d = d[d["date"].map(off).values]                                   # 팀 규칙: 휴일·주말만
    y = d["label"].astype(int).values
    s = (-d["score"] if region == "육지" else d["prob"]).values      # 위험 점수 (육지 = −예상 한낮 최저, 확률과 순서 같음)
    st = drop_stats(y, s)
    rng, bs = np.random.default_rng(SEED), []
    while len(bs) < n_boot:
        i = rng.integers(0, len(y), len(y))
        if 3 <= y[i].sum() < len(i):
            bs.append(drop_stats(y[i], s[i]))
    bs = pd.DataFrame(bs)
    row = {"계통": region, "모델": SRC[region]["model"]["drop"], "일수": len(y), "급락일": int(y.sum()), **st,
           "무작위 PR-AUC": y.mean(), "PR-AUC 배수": st["PR-AUC"] / y.mean(),   # 급락 비율이 계통마다 달라 배수로도 비교 (R4 제안)
           "Brier": brier_score_loss(y, d["prob"])}
    bs["PR-AUC 배수"] = bs["PR-AUC"] / y.mean()
    for k in ("PR-AUC", "PR-AUC 배수", "헛경보@재현율80%", "헛경보@재현율90%"):
        row[f"{k} 95%하한"], row[f"{k} 95%상한"] = bs[k].quantile(.025), bs[k].quantile(.975)
    for nm, a in (("주의 이상", (d["grade"] != "없음").values), ("경보", d["grade"].isin(["강한 경보", "경보"]).values)):
        hit, fa = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
        row.update({f"{nm} 적중": hit, f"{nm} 놓침": int(y.sum()) - hit, f"{nm} 헛경보": fa,
                    f"{nm} 재현율": hit / y.sum(), f"{nm} 정밀도": hit / max(hit + fa, 1)})
    return row


def main():
    price = pd.DataFrame([r for reg in SRC for r in price_rows(reg)])
    drop = pd.DataFrame([drop_rows(reg) for reg in SRC])
    price.round(4).to_csv(TAB / "T1_team_price.csv", index=False, encoding="utf-8-sig")
    drop.round(4).to_csv(TAB / "T2_team_drop.csv", index=False, encoding="utf-8-sig")
    info = {"시험 기간": f"{T0} ~ {T1}", "급락일": "휴일·주말만, 11~15시 최저 SMP 30원 미만",
            "모델": {reg: v["model"] for reg, v in SRC.items()},
            "학습 기간 (현재 모델, 팀 규칙 2023~ 과 다름)": {"육지 급락": "2023~2025 (R1, 팀 규칙)", "제주 급락": "2024~2025",
                                                   "일주일 잔차 보정": "2023~ (매월 재학습)", "한 달 mx v2": "2023~ (표 모델 매월, LSTM·GRU 분기 재학습)"},
            "주의": "현재 모델의 2026 성적은 모델을 고른 뒤 처음 채점한 값. 경보 문턱은 평일 포함으로 정했음"}
    (TAB / "T_team.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = ["계통", "예측", "구분", "표본수", "MAE", "rMAE", "편향(공표−예측)", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수"]
    print(price[price["구분"].isin(["전체", "일평균", "한낮 11~15시", "저녁 18~22시", "한낮 최저", "D+7"])][show].round(3).to_string(index=False))
    print(drop.round(3).T.to_string())


if __name__ == "__main__":
    main()
