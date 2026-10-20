"""예측 범위(상·하한) 산출과 평가 — 과제2 공용.

입력 표 d: [date, hour, y(실적), pred(예측), offday(휴무일 여부)] 시간 단위, 날짜·시간 순.
범위는 "실제 − 예측" 오차의 분위수를 예측값에 더해 만든다 (과제2 Q&A Q2의 1단계 방식).
누수 방지: 대상일 D의 범위에는 D 이전 날짜의 오차만 쓴다 (D-1까지의 실적은 판단 시각에 공표되어 있음).
"""
import numpy as np
import pandas as pd

ALPHA = 0.2          # 80% 범위
Q_LO, Q_HI = ALPHA / 2, 1 - ALPHA / 2

# 시간대 묶음: 표본이 적은 매일 갱신형에서 사용
BAND = {**{h: "밤" for h in range(1, 9)}, 9: "오전", 10: "오전", **{h: "한낮" for h in range(11, 16)},
        16: "오후", 17: "오후", **{h: "저녁" for h in range(18, 25)}}


def add_keys(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["k_hour"] = d["hour"].astype(str)
    d["k_hour_off"] = d["hour"].astype(str) + "_" + d["offday"].astype(int).astype(str)
    d["k_band_off"] = d["hour"].map(BAND) + "_" + d["offday"].astype(int).astype(str)
    d["err"] = d["y"] - d["pred"]
    # 배수형: 오차를 예측값 대비 비율로 (예측값이 매우 낮을 때 발산 방지로 분모 하한 20원 — PRD MAPE 기준 승계)
    d["rerr"] = d["err"] / d["pred"].clip(lower=20)
    return d


def _apply(target: pd.DataFrame, q: pd.DataFrame, key: str, form: str) -> tuple[np.ndarray, np.ndarray]:
    t = target[[key, "pred"]].merge(q, left_on=key, right_index=True, how="left")
    base = t["pred"].clip(lower=20) if form == "mult" else 1.0
    lo = t["pred"] + t["qlo"] * base
    hi = t["pred"] + t["qhi"] * base
    return lo.clip(lower=0).values, hi.values  # 육지 SMP는 음수 없음(최저 0원)


def _quant(rows: pd.DataFrame, key: str, form: str, min_n: int = 30) -> pd.DataFrame:
    col = "rerr" if form == "mult" else "err"
    g = rows.groupby(key)[col]
    q = pd.DataFrame({"qlo": g.quantile(Q_LO), "qhi": g.quantile(Q_HI), "n": g.size()})
    # 표본이 적은 칸은 전체 분위수로 대체
    allq = rows[col].quantile([Q_LO, Q_HI]).values
    q.loc[q["n"] < min_n, ["qlo", "qhi"]] = allq
    return q[["qlo", "qhi"]]


def static_interval(d, eval_mask, key="k_hour", form="add", cal_days=None, gap=1):
    """고정형: 평가 구간 시작 전 오차로 분위수를 한 번 계산해 평가 구간 전체에 적용.
    gap = 지평 H: 대상일 D의 판단일은 D-H이므로 D-H일까지의 오차만 쓸 수 있다 (H=1이면 D-1까지)."""
    ev = d[eval_mask]
    start = ev["date"].min() - pd.Timedelta(days=gap - 1)
    cal = d[d["date"] < start]
    if cal_days:
        cal = cal[cal["date"] >= start - pd.Timedelta(days=cal_days)]
    return _apply(ev, _quant(cal, key, form), key, form)


def rolling_interval(d, eval_mask, key="k_hour_off", form="add", window=90, gap=1):
    """매일 갱신형: 대상일 D마다 [D-H-window+1, D-H] 오차로 분위수를 다시 계산 (H=gap, 기본 1)."""
    ev = d[eval_mask]
    lo = pd.Series(np.nan, index=ev.index)
    hi = pd.Series(np.nan, index=ev.index)
    hist = d[["date", key, "err", "rerr"]]
    for day, part in ev.groupby("date"):
        end = day - pd.Timedelta(days=gap - 1)
        rows = hist[(hist["date"] < end) & (hist["date"] >= end - pd.Timedelta(days=window))]
        l, h = _apply(part, _quant(rows, key, form, min_n=max(8, window // 10)), key, form)
        lo[part.index], hi[part.index] = l, h
    return lo.values, hi.values


def methods(gap: int = 1) -> dict:
    """비교할 범위 방법 (이름 → 함수(d, eval_mask)). 2026 채점 전에 고정. gap = 지평 H."""
    g = gap
    m = {
        "고정·시간대": lambda d, e: static_interval(d, e, "k_hour", gap=g),
        "고정·시간대×휴무일": lambda d, e: static_interval(d, e, "k_hour_off", gap=g),
        "고정·시간대×휴무일·최근1년": lambda d, e: static_interval(d, e, "k_hour_off", cal_days=365, gap=g),
        "고정·시간대×휴무일·배수형·최근1년": lambda d, e: static_interval(d, e, "k_hour_off", "mult", 365, gap=g),
    }
    for w in (90, 180, 365):
        m[f"갱신·시간대×휴무일·{w}일"] = (lambda w: lambda d, e: rolling_interval(d, e, "k_hour_off", "add", w, g))(w)
        m[f"갱신·시간대×휴무일·배수형·{w}일"] = (lambda w: lambda d, e: rolling_interval(d, e, "k_hour_off", "mult", w, g))(w)
    for w in (28, 56):
        m[f"갱신·시간대묶음×휴무일·{w}일"] = (lambda w: lambda d, e: rolling_interval(d, e, "k_band_off", "add", w, g))(w)
        m[f"갱신·시간대묶음×휴무일·배수형·{w}일"] = (lambda w: lambda d, e: rolling_interval(d, e, "k_band_off", "mult", w, g))(w)
    return m


def score(y, lo, hi, pred=None) -> dict:
    """포함률·이탈 방향·평균 폭·구간 점수(Winkler, 낮을수록 좋음)."""
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    below, above = y < lo, y > hi
    width = hi - lo
    wink = width + (2 / ALPHA) * ((lo - y) * below + (y - hi) * above)
    out = {"시간수": len(y), "포함률": 1 - below.mean() - above.mean(), "하한아래": below.mean(),
           "상한위": above.mean(), "평균폭": width.mean(), "폭중앙값": np.median(width), "구간점수": wink.mean()}
    if pred is not None:
        out["MAE"] = np.abs(y - np.asarray(pred, float)).mean()
    return out


def segments(d: pd.DataFrame) -> dict:
    """보고용 구간 마스크: 전체, 평일, 휴무일, 휴무일 한낮(11~15시), 급락 시간(30원 미만)."""
    mid = d["hour"].between(11, 15)
    return {"전체": np.ones(len(d), bool), "평일": ~d["offday"].values, "휴무일": d["offday"].values,
            "휴무일 한낮(11~15시)": (d["offday"] & mid).values, "급락 시간(30원 미만)": (d["y"] < 30).values}


def report(d: pd.DataFrame, name: str, period: str) -> list[dict]:
    rows = []
    for seg, m in segments(d).items():
        if m.sum() == 0:
            continue
        g = d[m]
        rows.append({"방법": name, "기간": period, "구간": seg, **score(g["y"], g["lo"], g["hi"], g["pred"])})
    return rows


def select(sel: pd.DataFrame) -> str:
    """선택 규칙 (2026 채점 전 고정): 선택 구간 전체 포함률 75~85% 중 구간점수 최소. 없으면 포함률이 80%에 가장 가까운 것."""
    s = sel[sel["구간"] == "전체"]
    ok = s[s["포함률"].between(0.75, 0.85)]
    if len(ok):
        return ok.sort_values(["구간점수", "평균폭"]).iloc[0]["방법"]
    return s.assign(gap=(s["포함률"] - 0.8).abs()).sort_values("gap").iloc[0]["방법"]
