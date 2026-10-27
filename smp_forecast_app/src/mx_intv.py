"""한 달 예측 v3 — 범위만 다시 선택 (점예측 = v2 그대로). 한 달 6차 시험.

v2 범위는 2025(가장 조용한 해, 월 변화 표준편차 5.8원)로만 골라 2026(13.9원)에 좁았다 → 두 가지를 바꾼다 (2026 채점 전 고정).
  1) 선택 구간: 대상일 2024-07-01 ~ 2025-12-31 (2024 상반기는 오차 이력이 없어 제외). '고정 1년' 방법은 해마다 그 전 오차로 다시 맞춤.
  2) 후보 추가 — 월 경계: 연료비 단가가 매월 1일 바뀌므로 칸을 '주차 × 휴무일 × 기준일~대상일 사이 바뀐 달 수(0·1·2)'로 나눔
     (표본 30개 미만 칸은 '주차 × 휴무일'로). 오차 분위수(고정 1년·갱신 180·365일, 가산·배수) + Chronos 일별 분위수 보정(갱신 365일).
  3) 후보 추가 — 2년 오차: 고정 2년·갱신 730일 (가산·배수, 칸 = 주차×휴무일 또는 월 경계) + Chronos 일별 분위수 보정 갱신 730일.
     조용한 2025 오차만으로 2026 범위가 좁아지는 것을 막으려는 것 (1차 실행에서 월 경계 후보가 안 뽑혀 채점 전 추가).
  기존 후보(v2와 같은 10종)도 같은 구간으로 다시 평가. 규칙: 포함률 75~85% 중 구간점수 최소,
  단 1위와 0.1 이내면 오차 이력이 긴 방법 (select_v3: 선택 구간에선 2년·1년 창이 같은 오차라 구별 불가 → 더 긴 이력 쪽).
입력: mx/MX_predictions_daily_v2.csv (2024·2025 = 2024로 맞춘 선택 예측, 2026 = 재적합 예측), mx/chr_dc2.csv.gz, mx 피처의 mon_cross
출력: outputs/tables/MX3_intervals_v3.csv, MX_choice_v3.json, mx/MX_predictions_daily_v3.csv, mx/mx_ens_coef_v3.json(= v2)
      --test: MX4_test2026_v3.csv (v3 vs v2 vs 지금 모델 범위, 같은 기준일·지평)
실행: python -m smp_forecast_app.src.mx_intv [--test]
"""
import json
import shutil
import sys

import numpy as np
import pandas as pd

from smp_forecast_app.src.final_metrics import metrics
from smp_forecast_app.src.intervals import ALPHA, Q_HI, Q_LO, score, select
from smp_forecast_app.src.model_monthly import WEEK
from smp_forecast_app.src.mx_ens import DATA, SN, TAB, TEST, interval_methods
from smp_forecast_app.src.mx_features import MX, STATS, load

EVAL = ("2024-07-01", "2025-12-31")
MIN_N = 30


def table(s):
    """지표 s의 [base_date, H, date, offday, y, pred, dc2, qlo_dc2, qhi_dc2, cross, W]."""
    X = pd.read_csv(MX / "MX_predictions_daily_v2.csv", parse_dates=["base_date", "date"])
    c = pd.read_csv(MX / "chr_dc2.csv.gz", parse_dates=["base_date"])[["base_date", "H", f"pred_{s}", f"qlo_{s}", f"qhi_{s}"]]
    f = load()[["base_date", "H", "mon_cross"]]
    d = X[["base_date", "H", "date", "offday", f"y_{s}", f"pred_{s}"]].merge(
        c.rename(columns={f"pred_{s}": "dc2", f"qlo_{s}": "qlo_dc2", f"qhi_{s}": "qhi_dc2"}), on=["base_date", "H"], how="left"
    ).merge(f, on=["base_date", "H"], how="left").rename(columns={f"y_{s}": "y", f"pred_{s}": "pred"})
    d = d[d["pred"].notna()].sort_values(["base_date", "H"]).reset_index(drop=True)
    d["W"] = WEEK(d["H"])
    d["cross"] = d["mon_cross"].clip(upper=2)
    d["k1"] = d["W"].astype(str) + "_" + d["offday"].astype(str)
    d["k2"] = d["k1"] + "_" + d["cross"].astype(str)
    return d


def _qmap(rows, col, ev, q, key="k2"):
    """ev의 칸별 분위수. key=k2(주차×휴무일×바뀐 달 수)는 표본 부족하면 k1(주차×휴무일), 그래도 없으면 전체."""
    g1 = rows.groupby("k1")[col]
    v1 = g1.quantile(q).where(g1.size() >= MIN_N)
    out = ev["k1"].map(v1)
    if key == "k2":
        g2 = rows.groupby("k2")[col]
        out = ev["k2"].map(g2.quantile(q).where(g2.size() >= MIN_N)).fillna(out)
    return out.fillna(rows[col].quantile(q)).values


def err_cross(d, ev_mask, window, form, key="k2", fixed_days=365):
    """오차 분위수 (칸 = key). window=None → 평가 구간 시작 전 fixed_days일로 고정, 정수 → 매일 직전 window일.
    기준일 I의 범위에는 대상일 ≤ I 오차만."""
    d = d.assign(err=(d["y"] - d["pred"]) if form == "add" else (d["y"] - d["pred"]) / d["pred"].clip(lower=20))
    ev = d[ev_mask]
    hist = d[d["err"].notna()]
    lo, hi = pd.Series(np.nan, index=ev.index), pd.Series(np.nan, index=ev.index)
    groups = [(ev["base_date"].min(), ev)] if window is None else list(ev.groupby("base_date"))
    for i, part in groups:
        w = fixed_days if window is None else window
        rows = hist[(hist["date"] <= i) & (hist["date"] > i - pd.Timedelta(days=w))]
        ql, qh = _qmap(rows, "err", part, Q_LO, key), _qmap(rows, "err", part, Q_HI, key)
        base = part["pred"].clip(lower=20).values if form == "mult" else 1.0
        lo[part.index] = np.clip(part["pred"].values + ql * base, 0, None)
        hi[part.index] = part["pred"].values + qh * base
    return lo.values, hi.values


def cqr_cross(d, ev_mask, window, key="k2"):
    """Chronos 일별 분위수 폭을 예측값에 옮기고, 점수 max(lo−y, y−hi)의 80% 분위수로 칸(주차×휴무일×바뀐 달 수)별 보정."""
    d = d.assign(lo0=d["pred"] + (d["qlo_dc2"] - d["dc2"]), hi0=d["pred"] + (d["qhi_dc2"] - d["dc2"]))
    d["cs"] = np.maximum(d["lo0"] - d["y"], d["y"] - d["hi0"])
    ev = d[ev_mask]
    hist = d[d["cs"].notna()]
    lo, hi = pd.Series(np.nan, index=ev.index), pd.Series(np.nan, index=ev.index)
    for i, part in ev.groupby("base_date"):
        rows = hist[(hist["date"] <= i) & (hist["date"] > i - pd.Timedelta(days=window))]
        c = _qmap(rows, "cs", part, 1 - ALPHA, key)
        lo[part.index] = np.minimum(np.clip(part["lo0"].values - c, 0, None), part["pred"].values)
        hi[part.index] = np.maximum(part["hi0"].values + c, part["pred"].values)
    return lo.values, hi.values


def methods():
    m = dict(interval_methods())          # v2와 같은 10종 (칸 = 주차×휴무일)
    for w, wn in ((None, "고정1년"), (180, "갱신180일"), (365, "갱신365일")):
        for form, fn in (("add", "가산"), ("mult", "배수")):
            m[f"월경계·{wn}·{fn}"] = (lambda w, form: lambda d, e: err_cross(d, e, w, form))(w, form)
    m["월경계·보정분위수(Chronos-2 일별)·갱신365일"] = lambda d, e: cqr_cross(d, e, 365)
    # 2년 오차 (2026 범위에 2024 오차까지 들어감 — 조용한 2025만으로 좁아지는 것 방지). 2026 채점 전 추가.
    for pre, key in (("", "k1"), ("월경계·", "k2")):
        for w, wn in ((None, "고정2년"), (730, "갱신730일")):
            for form, fn in (("add", "가산"), ("mult", "배수")):
                m[f"{pre}{wn}·{fn}"] = (lambda w, form, key: lambda d, e: err_cross(d, e, w, form, key, 730))(w, form, key)
        m[f"{pre}보정분위수(Chronos-2 일별)·갱신730일"] = (lambda key: lambda d, e: cqr_cross(d, e, 730, key))(key)
    return m


def window_days(name):
    """방법이 쓰는 오차 이력 길이 (일)."""
    for pat, n in (("2년", 730), ("730일", 730), ("1년", 365), ("365일", 365), ("180일", 180), ("90일", 90)):
        if pat in name:
            return n
    return 0


def select_v3(it, tol=0.1):
    """intervals.select와 같되 동률 규칙 추가 (2026 채점 전 고정): 포함률 75~85% 중 구간점수 1위와 tol 이내면 오차 이력이 긴 방법.
    이유: 백테스트 예측이 2024-01부터라 선택 구간에서는 2년 창과 1년 창이 같은 오차를 써서 구별되지 않는다."""
    s = it[(it["구간"] == "전체") & it["포함률"].between(0.75, 0.85)]
    if s.empty:
        return select(it)
    near = s[s["구간점수"] <= s["구간점수"].min() + tol]
    return near.assign(w=near["방법"].map(window_days)).sort_values(["w", "구간점수"], ascending=[False, True]).iloc[0]["방법"]


def run_years(fn, d, mask):
    """'고정' 방법은 대상 연도마다 그 전 1년 오차로 다시 맞춤 (운영과 같게). 갱신형은 한 번에."""
    lo, hi = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for yr in sorted(d.loc[mask, "date"].dt.year.unique()):
        m = mask & (d["date"].dt.year == yr).values
        a, b = fn(d, m)
        lo[m], hi[m] = a, b
    return lo, hi


def main():
    rows, choice, outs = [], {}, None
    for s in STATS:
        d = table(s)
        ev = ((d["date"] >= EVAL[0]) & (d["date"] <= EVAL[1]) & d["y"].notna()).values
        ms = methods()
        part = []
        for name, fn in ms.items():
            lo, hi = run_years(fn, d, ev) if name.startswith(("고정", "월경계·고정")) else fn(d, ev)
            lo, hi = (lo[ev], hi[ev]) if len(lo) == len(d) else (lo, hi)
            g = d[ev]
            r = {"지표": SN[s], "방법": name, "구간": "전체", **score(g["y"], lo, hi, g["pred"])}
            for yr in (2024, 2025):
                m = (g["date"].dt.year == yr).values
                r[f"{yr} 포함률"] = score(g["y"][m], lo[m], hi[m])["포함률"]
            part.append(r)
        it = pd.DataFrame(part)
        ich = select_v3(it)
        it["선택"] = np.where(it["방법"] == ich, "✓", "")
        rows.append(it)
        # 산출: 2025~ (연도별 고정 재적합 포함)
        out_m = (d["date"] >= "2025-01-01").values
        fn = ms[ich]
        lo, hi = run_years(fn, d, out_m) if ich.startswith(("고정", "월경계·고정")) else fn(d, out_m)
        lo, hi = (lo[out_m], hi[out_m]) if len(lo) == len(d) else (lo, hi)
        o = d[out_m][["base_date", "H"]].assign(**{f"lo_{s}": lo, f"hi_{s}": hi})
        outs = o if outs is None else outs.merge(o, on=["base_date", "H"], how="outer")
        r = it[it["방법"] == ich].iloc[0]
        print(f"[{SN[s]}] 범위 {ich} — 2024하~2025 포함률 {r['포함률']:.1%} (2024 {r['2024 포함률']:.1%}, 2025 {r['2025 포함률']:.1%}), "
              f"폭 {r['평균폭']:.1f}, 구간점수 {r['구간점수']:.1f}", flush=True)
        choice[s] = {"범위": ich}
    pd.concat(rows).round(4).to_csv(TAB / "MX3_intervals_v3.csv", index=False, encoding="utf-8-sig")
    X = pd.read_csv(MX / "MX_predictions_daily_v2.csv", parse_dates=["base_date", "date"])
    X = X.drop(columns=[c for c in X if c.startswith(("lo_", "hi_"))]).merge(outs, on=["base_date", "H"], how="left")
    X.round(2).to_csv(MX / "MX_predictions_daily_v3.csv", index=False, encoding="utf-8-sig")
    v2 = json.loads((TAB / "MX_choice_v2.json").read_text(encoding="utf-8"))
    for s in STATS:
        v2["지표별"][s]["범위"] = choice[s]["범위"]
    v2["범위 선택"] = f"대상일 {EVAL[0]} ~ {EVAL[1]} 포함률 75~85% 중 구간점수 최소 (v3: 2024 하반기 포함, 월 경계 후보 추가)"
    v2["시험"] = f"{TEST[0]} ~ {TEST[1]} (한 달 예측 6차 시험: v2 점예측 + v3 범위)"
    (TAB / "MX_choice_v3.json").write_text(json.dumps(v2, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy(MX / "mx_ens_coef_v2.json", MX / "mx_ens_coef_v3.json")


def test():
    V3 = pd.read_csv(MX / "MX_predictions_daily_v3.csv", parse_dates=["base_date", "date"])
    V2 = pd.read_csv(MX / "MX_predictions_daily_v2.csv", parse_dates=["base_date", "date"])
    Z = pd.read_csv(DATA / "MZ_predictions_daily.csv", parse_dates=["base_date", "date"])
    rows = []
    for s in STATS:
        g = V3[["base_date", "H", "date", f"y_{s}", f"pred_{s}", f"lo_{s}", f"hi_{s}"]].merge(
            V2[["base_date", "H", f"lo_{s}", f"hi_{s}"]].rename(columns={f"lo_{s}": "lo2", f"hi_{s}": "hi2"}), on=["base_date", "H"]).merge(
            Z[["base_date", "H", f"pred_{s}", f"lo_{s}", f"hi_{s}"]].rename(columns={f"pred_{s}": "p0", f"lo_{s}": "lo0", f"hi_{s}": "hi0"}),
            on=["base_date", "H"])
        g = g.rename(columns={f"y_{s}": "y", f"pred_{s}": "p", f"lo_{s}": "lo", f"hi_{s}": "hi"})
        g = g[(g["date"] >= TEST[0]) & (g["date"] <= TEST[1]) & g["y"].notna() & g["lo"].notna() & g["p0"].notna()]
        for w in ["전체", 1, 2, 3, 4]:
            x = g if w == "전체" else g[WEEK(g["H"]) == w]
            r = {"지표": SN[s], "주차": w}
            for tag, (p, lo, hi) in {"v3": ("p", "lo", "hi"), "v2": ("p", "lo2", "hi2"), "지금": ("p0", "lo0", "hi0")}.items():
                m = metrics(x["y"], x[p], x[lo], x[hi])
                r.update({f"{tag} {k}": m[k] for k in ("MAE", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수")})
            rows.append(r)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "MX4_test2026_v3.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t[["지표", "주차", "v3 포함률", "v2 포함률", "지금 포함률", "v3 평균 폭", "v2 평균 폭", "지금 평균 폭",
             "v3 구간점수", "v2 구간점수", "지금 구간점수", "v3 하한 아래", "v3 상한 위"]].round(3).to_string(index=False))


if __name__ == "__main__":
    test() if "--test" in sys.argv[1:] else main()
