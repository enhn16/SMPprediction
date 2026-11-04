"""과제3 예측 근거 문장 — 내일 급락 경보 (육지, 팀 합의 LSTM + LightGBM 평균). R5 담당, 모델 코드·결과는 읽기만.

점수 = 두 모델이 예측한 내일 한낮(11~15시) 최저 SMP(원)의 평균 → 근거도 '예상 한낮 최저를 몇 원 올리고 내렸나'(원)로 계산.
비교 대상은 결과에 따라 고름 ('왜 이렇게 나왔나', 사용자 2026-11-02): 기여의 합 + 기준값 = 점수 (정확히 맞음).
  · 주의·경보 → 평소 날(학습 구간 전체)과 비교해 위험을 높인 근거(up)만
  · 없음     → 위험했던 날(학습 구간에서 한낮 최저가 주의 문턱 이하였던 날)과 비교해 위험을 낮춘 근거(down)만
  · LightGBM: TreeSHAP(pred_contrib). team_lstm.lgbm과 같은 설정으로 다시 학습(결정적이라 같은 모델 — 예측 일치 확인).
  · LSTM: 묶음 가리기 Shapley. 묶음 7개(최근 168시간 흐름 · 지난 같은 유형 날 · 최근 흐름 · 전력 수요 · 날씨 예보 · 달력 · 태양광 설비)를
    평소 값(흐름 = 학습 구간 시간 위치별 평균, 표 피처 = 학습 구간 중앙값)으로 바꾼 128가지 조합을 예측해 정확한 Shapley 값.
    가린 행을 예측 표에 같이 넣어 team_lstm.lstm을 그대로 부름 (학습은 예측 표를 쓰지 않고 시드 고정 → 같은 모델).
  · 두 모델 기여 ÷ 2를 같은 묶음끼리 더함. 기여 < 0(예상 최저를 낮춤) = 위험을 높임(up).
문장: 묶음마다 기여가 가장 큰 피처로 틀을 고르고 실제 값을 채움. 값과 방향이 어긋나면(비선형) 그 피처는 건너뜀.
      |기여| ≥ MIN_WON인 묶음만, 최대 3줄, 판정과 같은 방향이 먼저.
실행: python -m smp_forecast_app.src.reason_drop 2026-01-01 2026-10-18  → 근거 줄 수만 출력 (시험용, 파일은 reason_web이 만듦)
"""
import json
import sys
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common.config import ROOT
from drop_classifier.src.ens8 import COLS, LGBM
from drop_classifier.src.team_lstm import XSETS, lgbm, lstm, target_y
from smp_forecast_app.src.ops_drop import TEAM_START, team_features

REGION = "육지"
MIN_WON = 10.0          # 이보다 작은 묶음 기여는 문장으로 내지 않음 ('확실한 근거만', 사용자 2026-11-01)
_CH = json.loads((ROOT / "drop_classifier" / "outputs" / "tables" / f"drop_choice_{REGION}.json").read_text(encoding="utf-8"))
WARN, STRONG = _CH["문턱"]["주의"], _CH["문턱"]["경보"]
X = COLS + XSETS["연료비+수준"]
GROUPS = {
    "지난 같은 유형 날": ["prev_same_type_min_smp", "prev_day_min_smp"],
    "최근 흐름": ["drops_last14", "min_smp_last7", "smp_max7", "smp_mean7"],    # + LSTM 168시간 흐름(seq)
    "전력 수요": ["demand_fcst_prev", "demand_prev_rel28", "kpx_peak", "kpx_peak_rel"],
    "날씨 예보": ["rf_sky_pm", "rf_sky_am", "rf_clear_pm", "rf_pop_pm", "rf_rain_pm", "rf_ta_pm"],
    "달력": ["is_offday", "is_weekend_only", "is_holiday_single", "is_long_holiday", "off_block_len", "off_block_day",
           "is_myeongjeol", "is_labor_day", "is_bridge", "is_after_offday", "month_sin", "month_cos"],
    "태양광 설비": ["cap_solar_gw", "trend_years"],
    "연료비": ["fc_lng", "fc_coal", "fc_lng_coal_gap"],
}
FEAT_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}
assert sorted(FEAT_GROUP) == sorted(X)
PLAYERS = ["seq"] + [g for g in GROUPS if g != "연료비"]          # LSTM 입력 = 흐름 + 표 피처(COLS, 연료비·7일 수준 없음)
PLAYER_COLS = {g: [c for c in GROUPS[g] if c in COLS] for g in PLAYERS[1:]}


# ---------------------------------------------------------------- 기여 계산

def _ira(word: str) -> str:
    """서술격 조사 '(이)라': 끝 글자에 받침이 있으면 '이라', 없으면 '라' (예: 주말이라, 명절 연휴라, 연휴 3일째라)."""
    c = word[-1:] if word else ""
    if "가" <= c <= "힣":
        return "이라" if (ord(c) - 0xAC00) % 28 else "라"
    return "이라"

def _shapley(v: dict, n: int) -> np.ndarray:
    """v[frozenset(조합)] = 예측 배열(날짜 수) → 선수별 Shapley 값 (n × 날짜 수)."""
    out = []
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        phi = 0
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(rest, k):
                phi = phi + w * (v[frozenset(S) | {i}] - v[frozenset(S)])
        out.append(phi)
    return np.array(out)


def _lstm_shapley(tr: pd.DataFrame, te: pd.DataFrame, foil: pd.DataFrame):
    """반환: (비교 대상 값으로 바꾼 예측, 실제 예측, 선수별 기여 DataFrame[날짜 × PLAYERS]). foil = 비교 대상 날들."""
    base_seq = np.nanmean(np.stack(foil["seq168"]), axis=0).astype("float32")
    med = foil[COLS].median()
    n = len(PLAYERS)
    coals = [frozenset(S) for k in range(n + 1) for S in combinations(range(n), k)]
    parts = []
    for S in coals:                                      # S = 실제 값을 쓰는 선수, 나머지는 평소 값
        t = te.copy()
        if 0 not in S:
            t["seq168"] = [base_seq] * len(t)
        for j, g in enumerate(PLAYERS[1:], 1):
            if j not in S:
                for c in PLAYER_COLS[g]:
                    t[c] = med[c]
        parts.append(t)
    p = lstm(tr, pd.concat(parts, ignore_index=True), "T2", 168).reshape(len(coals), len(te))
    v = dict(zip(coals, p))
    phi = _shapley(v, n)
    return v[frozenset()], v[frozenset(range(n))], pd.DataFrame(phi.T, columns=PLAYERS, index=te.index)


def foil_rows(tr: pd.DataFrame, ref: str) -> pd.DataFrame:
    """비교 대상: usual = 학습 구간 전체(평소 날), risky = 학습 구간에서 한낮 최저가 주의 문턱 이하였던 날."""
    return tr if ref == "usual" else tr[tr["min_smp_11_15"] <= WARN]


def contributions(dfe: pd.DataFrame, dates, ref: str = "usual") -> pd.DataFrame:
    """대상일별 점수(예상 한낮 최저 원)·기준값·묶음 기여·피처 기여(원, 두 모델 평균 기준 = 각 모델 ÷ 2).
    학습 구간 = predict_team과 같음 (대상일 전년도 말까지). ref = 비교 대상 (foil_rows).
    LightGBM은 피처별 TreeSHAP에서 비교 대상 날들의 평균을 빼 기준을 옮김 (합은 그대로 정확)."""
    dates = pd.to_datetime(pd.Series(list(dates))).drop_duplicates()
    out = []
    for yr, ds in dates.groupby(dates.dt.year):
        tr = dfe[dfe["label"].notna() & dfe["date"].between(TEAM_START, pd.Timestamp(f"{yr - 1}-12-31"))]
        te = dfe[dfe["date"].isin(ds)].reset_index(drop=True)
        foil = foil_rows(tr, ref)
        m = LGBMRegressor(**{k: v for k, v in LGBM.items() if k != "class_weight"}).fit(tr[X], target_y(tr, "T2"))
        p_lg = m.predict(te[X])
        assert np.allclose(p_lg, lgbm(tr, te, "T2", "연료비+수준")), "LightGBM 재학습이 운영 모델과 다름"
        sh = m.predict(te[X], pred_contrib=True)
        mu = m.predict(foil[X], pred_contrib=True)[:, :-1].mean(axis=0)
        lg = pd.DataFrame(sh[:, :-1] - mu, columns=X) / 2
        b_ls, p_ls, ls = _lstm_shapley(tr, te, foil)
        r = pd.DataFrame({"date": te["date"].dt.strftime("%Y-%m-%d"), "score": (p_ls + p_lg) / 2,
                          "base": (b_ls + sh[:, -1] + mu.sum()) / 2, "lstm": p_ls, "lgbm": p_lg, "ref": ref})
        for g in GROUPS:
            r[f"g:{g}"] = lg[GROUPS[g]].sum(axis=1) + (ls[g] / 2 if g in ls else 0)
        r["g:최근 흐름"] += ls["seq"] / 2
        for c in X:
            r[f"f:{c}"] = lg[c]
        r["f:seq"] = ls["seq"] / 2                        # 최근 168시간 흐름 (LSTM)
        for g in PLAYER_COLS:
            r[f"l:{g}"] = ls[g] / 2                       # LSTM의 표 피처 묶음 몫 (문장 틀 고르기엔 안 씀)
        out.append(r)
    r = pd.concat(out, ignore_index=True)
    gap = (r["base"] + r[[c for c in r if c.startswith("g:")]].sum(axis=1) - r["score"]).abs().max()
    assert gap < 1e-3, f"기여 합이 점수와 다름 ({gap})"
    return r


# ---------------------------------------------------------------- 문장
def _won(v):
    return f"{v:,.0f}원"


def _texts(feat: str, up: bool, x: pd.Series, med: pd.Series, prev_label: str, usual: pd.Series):
    """피처 하나 → (문장, 값이 방향과 맞는지). up = 위험을 높이는 쪽(예상 최저를 낮춤).
    med = 비교 대상의 중앙값(방향 확인용), usual = 평소 날 중앙값('높다'고 말해도 되는지 — 위험했던 날보다 높아도 평소보다 낮으면 '높다'고 하지 않음)."""
    high = lambda c: x[c] >= usual[c]
    v = x.get(feat)
    lo = lambda c: x[c] < med[c]                         # 평소(학습 구간 중앙값)보다 낮은가
    if feat == "prev_same_type_min_smp":
        return (f"지난 {prev_label} 한낮 가격이 {_won(v)}까지 내려갔습니다" if up
                else f"지난 {prev_label} 한낮 가격이 {_won(v)}으로 {'높게 유지됐습니다' if high(feat) else '크게 내려가지 않았습니다'}"), lo(feat) == up
    if feat == "prev_day_min_smp":
        return (f"오늘 한낮 가격이 {_won(v)}까지 내려갔습니다" if up
                else f"오늘 한낮 가격이 {_won(v)} 이상으로 유지됐습니다"), lo(feat) == up
    if feat == "min_smp_last7":
        return (f"최근 일주일 사이 한낮 가격이 {_won(v)}까지 내려간 날이 있었습니다" if up
                else f"최근 일주일 동안 한낮 가격이 {_won(v)} 아래로 내려간 적이 없습니다"), lo(feat) == up
    if feat == "drops_last14":
        return (f"최근 2주 동안 한낮 가격이 30원 아래로 떨어진 날이 {int(v)}일 있었습니다" if up
                else "최근 2주 동안 한낮 가격이 30원 아래로 떨어진 날이 없었습니다"), (v > 0) == up
    if feat in ("smp_mean7", "smp_max7", "seq"):
        m7 = x["smp_mean7"]
        return (f"최근 일주일 평균 가격이 {_won(m7)}으로 낮은 편입니다" if up
                else f"최근 일주일 평균 가격이 {_won(m7)}으로 {'높은 편입니다' if high('smp_mean7') else '크게 낮지 않습니다'}"), lo("smp_mean7") == up
    if feat in ("kpx_peak", "kpx_peak_rel"):
        return (f"내일 최대 전력수요가 {x['kpx_peak']:.0f}GW로 낮게 전망됩니다 (주간 전망)" if up
                else f"내일 최대 전력수요가 {x['kpx_peak']:.0f}GW로 높게 전망됩니다 (주간 전망)"), lo(feat) == up
    if feat in ("demand_fcst_prev", "demand_prev_rel28"):
        pct = abs(x["demand_prev_rel28"] - 1) * 100
        return (f"오늘 낮 전력수요가 최근 4주보다 {pct:.0f}% 적습니다" if up
                else f"오늘 낮 전력수요가 최근 4주 수준 이상입니다"), lo("demand_prev_rel28") == up
    if feat in ("rf_sky_pm", "rf_sky_am", "rf_clear_pm"):
        clear = x["rf_clear_pm"] >= 0.5 and x["rf_sky_pm"] <= 2.5
        cloudy = x["rf_sky_pm"] >= 3
        return ("내일 오후 대체로 맑을 것으로 예보돼 태양광 발전이 많겠습니다" if up
                else "내일 오후 흐릴 것으로 예보돼 태양광 발전이 적겠습니다"), clear if up else cloudy
    if feat in ("rf_pop_pm", "rf_rain_pm"):
        rainy = x["rf_pop_pm"] >= 50 or x["rf_rain_pm"] >= 0.5
        dry = x["rf_pop_pm"] < 20 and x["rf_rain_pm"] == 0
        return ("내일 오후 비 소식이 없어 태양광 발전이 많겠습니다" if up
                else "내일 오후 비 소식이 있어 태양광 발전이 적겠습니다"), dry if up else rainy
    if feat == "rf_ta_pm":
        mild, extreme = 12 <= v <= 26, v < 5 or v > 30
        return (f"내일 오후 기온이 {v:.0f}℃로 냉난방이 거의 필요 없는 날씨입니다" if up
                else f"내일 오후 기온이 {v:.0f}℃로 {'난방' if v < 15 else '냉방'} 수요가 많겠습니다"), mild if up else extreme
    if feat in GROUPS["달력"][:10]:
        if up:
            if x["is_myeongjeol"]:
                kind = "명절 연휴"
            elif x["is_long_holiday"]:
                kind = f"연휴 {int(x['off_block_day'])}일째"
            elif x["is_holiday_single"]:
                kind = "공휴일"
            elif x["is_offday"]:
                kind = "주말"
            else:
                kind = "징검다리 평일" if x["is_bridge"] else ""
            return f"내일은 {kind}{_ira(kind)} 공장·사무실 전력 수요가 적습니다", bool(kind)
        return "내일은 평일이라 낮 전력 수요가 많습니다", not x["is_offday"] and not x["is_bridge"]
    if feat in ("month_sin", "month_cos"):
        mo = pd.Timestamp(x["date"]).month
        spring_fall = mo in (3, 4, 5, 9, 10)
        return ("봄·가을은 햇빛이 좋고 냉난방 수요가 적어 한낮 가격이 내려가기 쉬운 계절입니다" if up
                else f"{'여름' if 6 <= mo <= 8 else '겨울'}철은 냉난방 수요가 많아 한낮 가격이 잘 내려가지 않습니다"), \
            spring_fall if up else mo in (1, 2, 6, 7, 8, 12)
    if feat in ("cap_solar_gw", "trend_years"):
        return (f"태양광 설비가 {x['cap_solar_gw']:.1f}GW로 늘어 한낮 공급이 많습니다" if up else ""), \
            (not lo("cap_solar_gw")) if up else False
    if feat in GROUPS["연료비"]:
        return ("발전 연료비(LNG)가 낮은 편이라 전체 가격 수준이 낮습니다" if up
                else "발전 연료비(LNG)가 높은 편이라 전체 가격 수준이 높습니다"), lo("fc_lng") == up
    return None, False


def reasons(r: pd.Series, x: pd.Series, med: pd.Series, grade: str, prev_label: str, usual: pd.Series) -> list[dict]:
    """판정과 같은 방향의 근거만 (주의·경보 = up, 없음 = down) — 결과와 어긋나는 줄을 보이지 않음."""
    want_up = grade in ("주의", "경보")
    items = []
    for g in GROUPS:
        c = r[f"g:{g}"]
        if abs(c) < MIN_WON or (c < 0) != want_up:
            continue
        up = c < 0
        feats = (GROUPS[g] + (["seq"] if g == "최근 흐름" else []))
        # 같은 방향으로 큰 순. LSTM 몫(표 피처 묶음)은 피처별로 안 나뉘므로, 앞 틀이 값과 어긋나면 묶음 안 다음 틀을 시도
        feats = sorted(feats, key=lambda f: -(r[f"f:{f}"] * (-1 if up else 1)))
        for f in feats:
            t, ok = _texts(f, up, x, med, prev_label, usual)
            if t and ok:
                items.append({"t": t, "dir": "up" if up else "down", "group": g, "feature": f, "won": round(float(c), 1)})
                break
    items.sort(key=lambda d: -abs(d["won"]))
    return items[:3]


def grade_of(s):
    return "경보" if s <= STRONG else "주의" if s <= WARN else "없음"


def explain(dates, dfe: pd.DataFrame | None = None) -> pd.DataFrame:
    """대상일 목록 → 근거 표 (date, score, grade, ref, rank, t, dir, group, feature, won). won = 예상 한낮 최저 변화(원).
    먼저 평소 날 기준으로 계산(점수·등급 확정) → '없음'인 날만 위험했던 날 기준으로 다시 계산 (LSTM 학습 한 번 더)."""
    dfe = team_features() if dfe is None else dfe
    cb = contributions(dfe, dates, "usual")
    low = cb.loc[cb["score"].map(grade_of) == "없음", "date"]
    if len(low):
        cr = contributions(dfe, pd.to_datetime(low), "risky").set_index("date")
        cb = cb.set_index("date")
        cb.loc[cr.index, cr.columns] = cr
        cb = cb.reset_index()
    rows = []
    for _, r in cb.iterrows():
        d = pd.Timestamp(r["date"])
        x = dfe.set_index("date").loc[d].copy()
        x["date"] = d
        tr = dfe[dfe["label"].notna() & dfe["date"].between(TEAM_START, pd.Timestamp(f"{d.year - 1}-12-31"))]
        med, usual = foil_rows(tr, r["ref"])[X].median(), tr[X].median()
        same = dfe[(dfe["date"] < d - pd.Timedelta(days=0)) & (dfe["date"] <= d - pd.Timedelta(days=1))
                   & (dfe["is_offday"] == x["is_offday"])]
        pd_ = same["date"].max()
        prev_label = f"{'휴일' if x['is_offday'] else '평일'}({pd_.month}/{pd_.day})"
        s = r["score"]
        grade = grade_of(s)
        its = reasons(r, x, med, grade, prev_label, usual)
        base = {"date": r["date"], "score": round(s, 1), "base": round(r["base"], 1), "grade": grade, "ref": r["ref"]}
        if not its:
            rows.append({**base, "rank": 0, "t": "", "dir": "", "group": "", "feature": "", "won": np.nan})
        for i, it in enumerate(its, 1):
            rows.append({**base, "rank": i, **it})
    return pd.DataFrame(rows), cb


def main():
    a, b = (sys.argv[1:3] + ["2026-01-01", "2026-10-18"][len(sys.argv[1:3]):])
    dates = pd.date_range(a, b)
    rs, cb = explain(dates)
    print(f"{rs['date'].nunique()}일, 근거 줄 수 분포:", rs.groupby("date")["rank"].max().value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
