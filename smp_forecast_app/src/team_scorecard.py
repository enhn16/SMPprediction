"""최종 모델 성적표 — 확정된 네 예측(내일 급락·급등 경보, 일주일, 한 달)의 2026 공식 성적을 같은 코드·같은 기간으로 한곳에 모은다.
(2026-10-26 '팀 비교표'로 시작 → 2026-11-03 모델이 모두 확정돼 최종 성적표로 정리. 웹 참고 문서 '핵심 성능 지표'·발표 숫자의 출처)

기간: 2026-01-01 ~ 2026-10-18 (공식 성적 기간). 급락 = 모든 날(평일 포함) 공식 + 휴일·주말 참고, 급등 = 모든 날.
지표 (참고 문서 '성적 지표 읽는 법'과 같은 정의)
  경보: PR-AUC·무작위 PR-AUC(= 사건 비율)·PR-AUC 배수·ROC-AUC·Brier, 헛경보@재현율80%·90%, 실전(미리 정한 문턱) 적중·놓침·헛경보·재현율·정밀도.
        PR-AUC·헛경보@재현율은 날짜 부트스트랩 2,000회 95% 구간 병기 (사건이 적어 차이가 우연일 수 있음)
  가격: MAE·RMSE·MAPE(공표 20원 이상)·rMAE·편향·포함률·하한 아래·상한 위·평균 폭·구간점수
        단순 기준(rMAE): 일주일 = 지난주 같은 시간, 내일(D+1) = 전날 같은 시간, 한 달 = 기준일까지 같은 유형일 4일 평균
  일주일 범위는 **보정 전 모델 범위**로 채점한다 (2026-11-03 사용자: 경보·휴일 하한 보정은 웹에서 보기 좋게 하는 것이지 모델 성능이 아님).
입력 (읽기만): 육지 wz3/WZ3_K3_backtest_hourly.csv(일주일 K3)·mm/MM_predictions_daily.csv(한 달)·drop_육지_15h.csv(R1)·surge_육지_15h.csv(R4),
      제주 jeju/의 일주일·한 달·급락 파일
출력: outputs/tables/T1_team_price.csv(가격), T2_team_drop.csv(급락), T3_team_surge.csv(급등), T_team.json(설명)
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
from smp_forecast_app.src.jz_month import daily_stats

DATA = PROCESSED_DIR / "smp_forecast_app"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
T0, T1 = "2026-01-01", "2026-10-18"
STATS = {"mean": "일평균", "mid": "한낮 11~15시", "eve": "저녁 18~22시", "midmin": "한낮 최저"}
SRC = {"육지": {"week": DATA / "wz3" / "WZ3_K3_backtest_hourly.csv", "month": DATA / "mm" / "MM_predictions_daily.csv", "drop": ROOT / "drop_classifier" / "outputs" / "tables" / "drop_육지_15h.csv",
               "model": {"drop": "LSTM + LightGBM(확장 변수) 평균 (예상 한낮 최저)", "week": "K3: Chronos-2 (평일 미세조정판·휴무일 공변량판) + 0.5 × 잔차 보정",
                         "month": "MM: LightGBM 단독 + 가스공사 발전용 요금 (월말 판단 → 다음 달 일평균)"}},
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
    w = w.reset_index(drop=True)   # 범위 보정(웹 표시용)은 붙이지 않음 — 모델 범위 그대로
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
    mm = "H" not in m   # 육지 MM: 월말 판단 → 다음 달, 주차 = 대상 달 1~7·8~14·15~21·22일~
    if mm:
        m["H"] = (m["date"] - m["base_date"]).dt.days
    for s, nm in STATS.items():
        if f"pred_{s}" not in m:
            continue
        x = m[m[f"y_{s}"].notna() & m[f"pred_{s}"].notna()]
        naive = np.where(x["offday"].astype(bool), same4[True][s].reindex(x["base_date"]).values, same4[False][s].reindex(x["base_date"]).values)
        rows.append({"계통": region, "예측": "한 달", "구분": nm, **metrics(x[f"y_{s}"], x[f"pred_{s}"], x[f"lo_{s}"], x[f"hi_{s}"], naive)})
        wk = np.minimum((x["H"] - 1) // 7 + 1, 4).values if mm else ((x["H"] - 7) // 7 + 1).values   # 롤링: 1주차 = 7~13일 뒤 … 4주차 = 28~34일 뒤
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


SCOPES = ("전체 날 (공식)", "휴일·주말")   # 팀 규칙 3 (10/28): 급락은 모든 날, 휴일·주말은 참고 행


def drop_rows(region, scope=SCOPES[0], n_boot=2000):
    d = pd.read_csv(SRC[region]["drop"], parse_dates=["date"])
    d = d[in_test(d["date"])]
    off = d["date"].map(build_calendar(T0, T1).set_index("date")["is_offday"].astype(bool)).values
    if scope == SCOPES[0] and d.loc[~off, "label"].isna().all():
        return None                                                    # 평일 레이블이 없으면(전체 날 산출 전) 공식 행을 내지 않음
    d = (d if scope == SCOPES[0] else d[off]).dropna(subset=["label"])
    y = d["label"].astype(int).values
    s = (-d["score"] if region == "육지" else d["prob"]).values      # 위험 점수 (육지 = −예상 한낮 최저, 확률과 순서 같음)
    st = drop_stats(y, s)
    rng, bs = np.random.default_rng(SEED), []
    while len(bs) < n_boot:
        i = rng.integers(0, len(y), len(y))
        if 3 <= y[i].sum() < len(i):
            bs.append(drop_stats(y[i], s[i]))
    bs = pd.DataFrame(bs)
    row = {"계통": region, "범위": scope, "모델": SRC[region]["model"]["drop"], "일수": len(y), "급락일": int(y.sum()), **st,
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


def surge_row(n_boot=2000):
    """내일 급등 경보 (육지, R4 v3): 모든 날. 점수 = 확률 (휴무일은 모델 규칙상 0)."""
    p = ROOT / "surge_classifier" / "outputs" / "tables" / "surge_육지_15h.csv"
    d = pd.read_csv(p, parse_dates=["date"])
    d = d[in_test(d["date"])].dropna(subset=["label"])
    y, s = d["label"].astype(int).values, d["prob"].values
    st = drop_stats(y, s)
    rng, bs = np.random.default_rng(SEED), []
    while len(bs) < n_boot:
        i = rng.integers(0, len(y), len(y))
        if 2 <= y[i].sum() < len(i):
            bs.append(drop_stats(y[i], s[i]))
    bs = pd.DataFrame(bs)
    row = {"계통": "육지", "범위": "모든 날", "모델": "부호 고정 점수 + 스플라인 로지스틱 + LightGBM 그루터기 평균 (v3)", "일수": len(y), "급등일": int(y.sum()),
           **st, "무작위 PR-AUC": y.mean(), "PR-AUC 배수": st["PR-AUC"] / y.mean(), "Brier": brier_score_loss(y, s)}
    bs["PR-AUC 배수"] = bs["PR-AUC"] / y.mean()
    for k in ("PR-AUC", "PR-AUC 배수", "헛경보@재현율80%", "헛경보@재현율90%"):
        row[f"{k} 95%하한"], row[f"{k} 95%상한"] = bs[k].quantile(.025), bs[k].quantile(.975)
    for nm, a in (("주의 이상", (d["grade"] != "없음").values), ("경보", (d["grade"] == "경보").values)):
        hit, fa = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
        row.update({f"{nm} 적중": hit, f"{nm} 놓침": int(y.sum()) - hit, f"{nm} 헛경보": fa,
                    f"{nm} 재현율": hit / y.sum(), f"{nm} 정밀도": hit / max(hit + fa, 1)})
    return row


def main():
    price = pd.DataFrame([r for reg in SRC for r in price_rows(reg)])
    drop = pd.DataFrame([r for reg in SRC for sc in SCOPES if (r := drop_rows(reg, sc)) is not None])
    price.round(4).to_csv(TAB / "T1_team_price.csv", index=False, encoding="utf-8-sig")
    drop.round(4).to_csv(TAB / "T2_team_drop.csv", index=False, encoding="utf-8-sig")
    surge = pd.DataFrame([surge_row()])
    surge.round(4).to_csv(TAB / "T3_team_surge.csv", index=False, encoding="utf-8-sig")
    info = {"시험 기간": f"{T0} ~ {T1}", "급락일": "모든 날(평일 포함), 11~15시 최저 SMP 30원 미만 — 공식 = 전체 날, 휴일·주말은 참고 (제주는 전체 날 산출 전이라 휴일·주말만)",
            "급등일": "그날 최고 SMP ≥ 직전 28일 일 최고 중앙값 × 1.25, 모든 날 (휴무일은 경보 없음)",
            "모델": {reg: v["model"] for reg, v in SRC.items()},
            "범위": "일주일·한 달 범위는 모델 범위 그대로 (웹의 경보·휴일 하한 보정 전)",
            "주의": "모든 모델은 2025 이전 자료로 고른 뒤 2026을 한 번 채점. 한 달 연료 입력(가스요금)과 급등 v3 세부는 2026을 본 뒤 정해져 2026 성적은 참고"}
    (TAB / "T_team.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = ["계통", "예측", "구분", "표본수", "MAE", "rMAE", "편향(공표−예측)", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수"]
    print(price[price["구분"].isin(["전체", "일평균", "한낮 11~15시", "저녁 18~22시", "한낮 최저", "D+7"])][show].round(3).to_string(index=False))
    print(drop.round(3).T.to_string())
    print(surge.round(3).T.to_string())


if __name__ == "__main__":
    main()
