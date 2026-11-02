"""과제3 예측 근거 문장 — 내일 급등 경보 (육지, 팀 합의 v3: 부호 고정 점수 + 스플라인 로지스틱 + LightGBM 그루터기 확률 평균).
R5 담당, 모델 코드·결과는 읽기만 (저장된 v3_models.pkl을 그대로 씀).

점수 = 세 확률의 평균, 휴무일 = 0 → 평일만 판단. 비교 대상은 결과에 따라 고름 ('왜 이렇게 나왔나', 사용자 2026-11-02):
  · 주의·경보 → 평소 평일(2023~2025 학습 구간 평일)과 비교해 위험을 높인 근거(up)만
  · 없음     → 급등했던 평일(학습 구간 급등일)과 비교해 위험을 낮춘 근거(down)만 (휴무일은 고정 한 줄)
세 모델 모두 로짓이 피처별 항의 합이라 기여가 정확함:
  · 부호 고정: 기울기 × 부호 × z ÷ 피처 수
  · 스플라인: 피처마다 기저함수 계수 × 기저값 (피처끼리 섞이지 않음)
  · 그루터기: pred_contrib (깊이 1이라 피처마다 계단 함수의 합)
  각 피처 항에서 비교 대상 날들의 평균을 빼 기준을 옮긴 뒤, 모델마다 '확률 − 기준 확률'을 로짓 기여 비율대로 나누고
  (비율 = 두 로짓 사이 시그모이드 평균 기울기라 항상 유한), 세 모델 ÷ 3을 같은 묶음끼리 더함. 기여 > 0 = 위험을 높임(up).
문장: reason_drop과 같은 방식 (묶음 |기여| ≥ MIN_P, 최대 3줄, 값과 방향이 어긋나면 그 틀은 건너뜀). 휴무일은 고정 한 줄.
실행: python -m smp_forecast_app.src.reason_surge 2026-01-01 2026-10-18  → 근거 줄 수만 출력 (시험용, 파일은 reason_web이 만듦)
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd

from surge_classifier.src import team_model2 as T2
from surge_classifier.src import team_model2_fa as FA
from surge_classifier.src import team_model3 as V3
from surge_classifier.src.label_days import OUT_TAB as SURGE_TAB

REGION = "육지"
MIN_P = 0.05            # 이보다 작은 묶음 기여(점수 0~1)는 문장으로 내지 않음 ('확실한 근거만', 사용자 2026-11-01)
# 가격 피처는 뜻별로 나눔 — 한 묶음에 섞으면 가장 큰 단일 피처의 흐린 문장이 구체적 사실을 가림 (2026-01-22 확인)
GROUPS = {
    "오늘 가격": ["prev_max", "prev_max_rel", "prev_mean_rel", "prev_evening_rel", "prev_max_vs_q90", "prev_high_hours", "prev_excess"],
    "최근 최고 가격": ["max7_rel", "vol7", "peak60_rel", "near_peak_28"],
    "가격 수준 변화": ["ref_max", "ref_change28", "ref_vs_year"],
    "최근 급등": ["spikes_28d", "spikes_90d", "days_since_spike"],
    "전력 수요": ["prev_dem_max", "prev_dem_rel28", "same_type_dem_rel28", "dem_trend7", "kpx_peak"],
    "기온": ["wx_tmax", "wx_tmin", "wx_tmean", "wx_hdd", "wx_cdd", "wx_tchg"],
    "흐림·비": ["wx_cloud", "wx_precip", "wk_sky_pm", "wk_cloud_mid_jma"],
    "달력": ["dow", "is_off", "is_holiday", "is_major", "after_off", "off_run_len", "month_sin", "month_cos"],
}
OFFDAY_TEXT = "내일은 휴일이라 전력 수요가 적어 급등 경보를 내지 않습니다"


def _sig(v):
    return 1 / (1 + np.exp(-v))


def _logit_terms(models, x: pd.DataFrame, feats: list[str]) -> dict:
    """모델별 (기준 로짓 상수, 피처별 로짓 항 DataFrame). 상수 + 항의 합 = 모델 로짓."""
    out = {}
    s = models["sign"]
    a, b = s.cal.coef_[0, 0], s.cal.intercept_[0]
    z = (x[list(T2.SIGNS)].fillna(s.med) - s.mu) / s.sd
    t = z * pd.Series(T2.SIGNS) * a / len(T2.SIGNS)
    out["sign"] = (b, t.reindex(columns=feats, fill_value=0.0))
    sp = models["spline"]
    sc, st, lr = sp.m.named_steps["standardscaler"], sp.m.named_steps["splinetransformer"], sp.m.named_steps["logisticregression"]
    B = st.transform(sc.transform(x[feats].fillna(sp.med)))
    w = lr.coef_[0]
    k = B.shape[1] // len(feats)                         # 피처마다 기저 k개, 피처 순서대로 붙어 있음
    out["spline"] = (lr.intercept_[0], pd.DataFrame((B * w).reshape(len(x), len(feats), k).sum(axis=2), columns=feats, index=x.index))
    c = models["stump"].m.booster_.predict(x[feats], pred_contrib=True)
    out["stump"] = (float(c[0, -1]), pd.DataFrame(c[:, :-1], columns=feats, index=x.index))
    return out


def foil_rows(d: pd.DataFrame, ref: str) -> pd.DataFrame:
    """비교 대상: usual = 학습 구간 평일(평소 평일), risky = 학습 구간 급등 평일."""
    lab = d[d["y"].notna() & d["date_s"].between(T2.DEV_FROM, T2.DEV_TO) & (d["is_off"] == 0)]
    return (lab if ref == "usual" else lab[lab["y"] == 1]).reset_index(drop=True)


def contributions(d: pd.DataFrame, dates, models, ref: str = "usual") -> pd.DataFrame:
    feats = list(T2.FEATS)
    x = d.set_index("date").reindex(pd.to_datetime(list(dates))).reset_index().rename(columns={"index": "date"})
    lab = foil_rows(d, ref)
    p = T2.predict(models, x)
    tx, tb = _logit_terms(models, x, feats), _logit_terms(models, lab, feats)
    total = pd.DataFrame(0.0, index=x.index, columns=feats)
    base_p = 0.0
    for m in T2.KEYS:
        const, terms = tx[m]
        mu = tb[m][1].mean()                               # 비교 대상 날들의 피처 항 평균 → 기준으로 옮김
        L0, phi = const + mu.sum(), terms - mu
        L = L0 + phi.sum(axis=1)
        assert np.allclose(_sig(L), p[m]), f"{m} 로짓 분해가 모델 확률과 다름"
        p0 = _sig(L0)
        dL = L - L0
        slope = np.where(np.abs(dL) > 1e-9, (p[m] - p0) / np.where(np.abs(dL) > 1e-9, dL, 1), p0 * (1 - p0))
        total += phi.mul(slope, axis=0) / 3
        base_p += p0 / 3
    r = pd.DataFrame({"date": x["date"].dt.strftime("%Y-%m-%d"), "score": FA.combine(p, x, V3.CFG).values,
                      "avg": p["avg"].values, "base": base_p, "is_off": x["is_off"].values, "ref": ref})
    for g, fs in GROUPS.items():
        r[f"g:{g}"] = total[fs].sum(axis=1).values
    for f in feats:
        r[f"f:{f}"] = total[f].values
    gap = (r["base"] + r[[c for c in r if c.startswith("g:")]].sum(axis=1) - r["avg"]).abs().max()
    assert gap < 1e-6, f"기여 합이 점수와 다름 ({gap})"
    return r


def _texts(f: str, up: bool, x: pd.Series, med: pd.Series, usual: pd.Series):
    """피처 하나 → (문장, 값이 방향과 맞는지). up = 급등 위험을 높이는 쪽.
    med = 비교 대상 중앙값(방향 확인용), usual = 평소 평일 중앙값('낮다'고 말해도 되는지)."""
    hi = lambda c: x[c] > med[c]
    ref = x["ref_max"]
    if f in ("prev_max", "prev_max_rel", "prev_max_vs_q90", "prev_excess", "prev_mean_rel"):
        pct = (x["prev_max_rel"] - 1) * 100
        return (f"오늘 최고 가격이 {x['prev_max']:,.0f}원으로 최근 4주 평소 최고({ref:,.0f}원)보다 {pct:.0f}% 높았습니다" if up
                else f"오늘 최고 가격이 {x['prev_max']:,.0f}원으로 최근 4주 평소 최고({ref:,.0f}원) 수준에 머물렀습니다"), \
            (x["prev_max_rel"] >= 1.05) if up else (x["prev_max_rel"] <= 1.0)
    if f == "prev_evening_rel":
        return ("오늘 저녁 가격이 최근 4주 평소 최고보다 높게 올랐습니다" if up
                else "오늘 저녁 가격은 평소 수준이었습니다"), (x[f] > 1) if up else (x[f] <= 1)
    # 일주일·두 달 최고가 바로 오늘 값이면 '오늘 가격' 줄과 같은 사실이라 건너뜀
    not_today = lambda c: x[c] > x["prev_max_rel"] + 0.01
    if f == "max7_rel":
        pct = (x[f] - 1) * 100
        return (f"최근 일주일 사이 최고 가격이 평소 최고보다 {pct:.0f}% 높았던 날이 있습니다" if up
                else "최근 일주일 최고 가격이 평소 수준에 머물렀습니다"), (x[f] >= 1.1 and not_today(f)) if up else (x[f] <= 1.05)
    if f == "peak60_rel":
        pct = (x[f] - 1) * 100
        return (f"최근 두 달 사이 최고 가격이 평소 최고보다 {pct:.0f}% 높았던 날이 있습니다" if up
                else "최근 두 달 동안 가격이 평소 최고를 크게 넘은 적이 없습니다"), (x[f] >= 1.1 and not_today(f)) if up else (x[f] <= 1.05)
    if f == "near_peak_28":
        n = int(x[f]) if x[f] == x[f] else 0
        return (f"최근 4주 동안 두 달 내 최고 가격 가까이 오른 날이 {n}일 있었습니다" if up
                else "최근 4주 동안 두 달 내 최고 가격 가까이 오른 날이 없었습니다"), (n > 0) if up else (n == 0)
    if f == "vol7":
        return ("최근 일주일 가격이 크게 오르내렸습니다" if up else "최근 일주일 가격이 안정적이었습니다"), hi(f) == up
    if f == "prev_high_hours":
        n = int(x[f]) if x[f] == x[f] else 0
        return (f"오늘 평소 최고보다 15% 넘게 비싼 시간이 {n}시간 있었습니다" if up else ""), (n > 0) if up else False
    if f in ("ref_max", "ref_change28", "ref_vs_year"):
        ch = (x["ref_change28"] - 1) * 100
        return (f"최근 4주 가격 수준이 한 달 전보다 {abs(ch):.0f}% {'올랐' if ch > 0 else '내렸'}습니다"), \
            (ch >= 5) if up else (ch <= -5)
    if f in ("spikes_28d", "spikes_90d"):
        n28, n90 = x["spikes_28d"], x["spikes_90d"]
        if up:
            return (f"최근 4주 동안 급등이 {int(n28)}번 있었습니다" if n28 > 0 else f"최근 석 달 동안 급등이 {int(n90)}번 있었습니다"), n90 > 0
        return "최근 석 달 동안 급등이 없었습니다", n90 == 0
    if f == "days_since_spike":
        n = x[f]
        return (f"마지막 급등이 {n:.0f}일 전으로 얼마 되지 않았습니다" if up
                else f"마지막 급등이 {n:.0f}일 전으로 오래됐습니다"), (n <= 30) if up else (n > 90)
    if f == "kpx_peak":
        return (f"내일 최대 전력수요가 {x[f]:.0f}GW로 높게 전망됩니다 (주간 전망)" if up
                else f"내일 최대 전력수요가 {x[f]:.0f}GW로 {'낮게' if x[f] < usual[f] else '급등했던 날들보다 낮게'} 전망됩니다 (주간 전망)"), hi(f) == up
    if f in ("prev_dem_max", "prev_dem_rel28", "same_type_dem_rel28", "dem_trend7"):
        pct = (x["prev_dem_rel28"] - 1) * 100
        return (f"오늘 최대 전력수요가 최근 4주 평균보다 {abs(pct):.0f}% {'많' if up else '적'}습니다"), \
            (pct >= 3) if up else (pct <= -3)
    if f in ("wx_hdd", "wx_tmin", "wx_tmean", "wx_tmax", "wx_cdd", "wx_tchg"):
        tm, tmin, tmax, chg = x["wx_tmean"], x["wx_tmin"], x["wx_tmax"], x["wx_tchg"]
        if up:
            if tm < 5:
                drop_ = f", 오늘보다 {abs(chg):.0f}℃ 떨어집니다" if chg <= -3 else ""
                return f"내일 아침 기온이 {tmin:.0f}℃로 추워 난방 수요가 많겠습니다{drop_}", True
            if tmax >= 30:
                return f"내일 낮 최고 기온이 {tmax:.0f}℃로 더워 냉방 수요가 많겠습니다", True
            return "", False
        return f"내일 기온이 평균 {tm:.0f}℃로 냉난방 수요가 크지 않겠습니다", 8 <= tm <= 22
    if f in ("wx_cloud", "wk_sky_pm", "wk_cloud_mid_jma"):
        cloudy = x["wk_sky_pm"] >= 3 or x["wx_cloud"] >= 70
        clear = x["wk_sky_pm"] <= 2 and x["wx_cloud"] < 40
        return ("내일 낮 흐릴 것으로 예보돼 태양광 발전이 줄고, 그만큼 비싼 발전기가 더 돌아야 합니다" if up
                else "내일 낮 맑을 것으로 예보돼 태양광 발전이 많겠습니다"), cloudy if up else clear
    if f == "wx_precip":
        return ("내일 비 소식이 있어 태양광 발전이 적겠습니다" if up else ""), (x[f] >= 0.3) if up else False
    if f == "after_off":
        return ("내일은 휴일 다음 첫 평일이라 전력 수요가 갑자기 늘어납니다" if up else ""), bool(x[f]) if up else False
    return "", False


def reasons(r: pd.Series, x: pd.Series, med: pd.Series, grade: str, usual: pd.Series) -> list[dict]:
    """판정과 같은 방향의 근거만 (주의·경보 = up, 없음 = down) — 결과와 어긋나는 줄을 보이지 않음."""
    if r["is_off"]:
        return [{"t": OFFDAY_TEXT, "dir": "down", "group": "달력", "feature": "is_off", "p": np.nan}]
    want_up = grade in ("주의", "경보")
    items = []
    for g, fs in GROUPS.items():
        c = r[f"g:{g}"]
        if abs(c) < MIN_P or (c > 0) != want_up:
            continue
        up = c > 0
        for f in sorted(fs, key=lambda f: -(r[f"f:{f}"] * (1 if up else -1))):
            t, ok = _texts(f, up, x, med, usual)
            if t and ok:
                items.append({"t": t, "dir": "up" if up else "down", "group": g, "feature": f, "p": round(float(c), 3)})
                break
    items.sort(key=lambda d: -abs(d["p"]))
    return items[:3]


def explain(dates, d: pd.DataFrame | None = None):
    """대상일 목록 → (근거 표, 기여 표). p = 점수(세 확률 평균) 변화."""
    ch = json.loads((SURGE_TAB / f"surge_choice_{REGION}_15h.json").read_text(encoding="utf-8"))
    models = pickle.loads(V3.MODEL_PKL.read_bytes())
    FA.configure(V3.CFG)
    try:
        d = V3.data() if d is None else d
        rows = d.set_index("date").reindex(pd.to_datetime(list(dates)))
        if rows[["prev_max", "wx_tmean"]].isna().any().any():          # ops_surge.predict와 같은 확인 (빈칸이면 점수가 운영과 달라짐)
            raise ValueError("대상일 입력 없음 — 판단일 SMP 실적 또는 Open-Meteo(collect_extra openmeteo) 확인")
        cb = contributions(d, dates, models, "usual")
        cb["grade"] = V3.grade(cb["score"].values, ch["문턱"])
        low = cb.loc[(cb["grade"] == "없음") & (cb["is_off"] == 0), "date"]
        if len(low):                                       # '없음' 평일은 급등했던 평일과 비교해 다시 계산
            cr = contributions(d, pd.to_datetime(low), models, "risky").set_index("date")
            cb = cb.set_index("date")
            cb.loc[cr.index, cr.columns] = cr
            cb = cb.reset_index()
        med = {ref: foil_rows(d, ref)[T2.FEATS].median() for ref in ("usual", "risky")}
    finally:
        FA.configure({})
    xs = d.set_index("date_s")
    rows = []
    for _, r in cb.iterrows():
        grade = r["grade"]
        its = reasons(r, xs.loc[r["date"]], med[r["ref"]], grade, med["usual"])
        base = {"date": r["date"], "score": round(r["score"], 4), "base": round(r["base"], 4), "grade": grade, "ref": r["ref"]}
        if not its:
            rows.append({**base, "rank": 0, "t": "", "dir": "", "group": "", "feature": "", "p": np.nan})
        for i, it in enumerate(its, 1):
            rows.append({**base, "rank": i, **it})
    return pd.DataFrame(rows), cb


def main():
    a, b = (sys.argv[1:3] + ["2026-01-01", "2026-10-18"][len(sys.argv[1:3]):])
    rs, cb = explain(pd.date_range(a, b))
    wk = rs[rs["feature"] != "is_off"]
    print(f"{rs['date'].nunique()}일 (평일 {wk['date'].nunique()}), 평일 근거 줄 수 분포:",
          wk.groupby("date")["rank"].max().value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
