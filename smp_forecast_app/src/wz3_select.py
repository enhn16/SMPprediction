"""일주일 v3 (R3) — 후보 선택(2024 맞춤 → 2025 비교) → 2024~2025 재맞춤 → 2026 한 번 채점.

선택 규칙은 docs/WEEK_V3.md ②절에 2025 결과를 보기 전에 고정했다 (아래 P_ORDER·A 후보·문턱이 그대로).
  점예측: 2025 전체 MAE 최소, 0.05원 이내면 단순한 쪽 / 범위: 2025 포함률 75~85% 중 구간점수 최소, 1% 이내면 단순한 쪽
단계 (인자)
  point  : 점예측 후보 P0~P7 → outputs/tables/WZ3_S1_point_2025.csv, WZ3_choice.json의 '점예측'
  range  : 고른 점예측 위 범위 후보 A0~A3 → WZ3_S2_range_2025.csv, WZ3_choice.json의 '범위'
  test   : 2024~2025로 λ·β 재맞춤 → 2026-01-01 ~ 10-18 한 번 채점, 지금 모델과 같은 날짜·시간끼리
           → WZ3_T1_test2026.csv, data/processed/smp_forecast_app/wz3/WZ3_predictions_hourly.csv
실행: python -m smp_forecast_app.src.wz3_select point|range|test
"""
import json
import sys

import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from common.loaders import load_smp_actual
from smp_forecast_app.src import range_adjust
from smp_forecast_app.src.final_metrics import hourly_segments, metrics
from smp_forecast_app.src.intervals import BAND
from smp_forecast_app.src.wz_common import OUT as WZOO
from smp_forecast_app.src.wz3_resid import RES3, run as run_resid
from smp_forecast_app.src.wz3_diag import BANDS

DATA = PROCESSED_DIR / "smp_forecast_app"
W3 = DATA / "wz3"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
CHOICE = TAB / "WZ3_choice.json"
FIT, SEL, TEST = ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-10-18")
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
HALFLIFE = 3
TIE_MAE, TIE_RANGE = 0.05, 0.01
# 점예측 후보: 이름 → (잔차 설정, λ 방식, 편향 보정). 순서 = 단순한 순
P_ORDER = {"P0 현재": None, "P1 피처 표 재생성": ("new", "one", False), "P2 λ 일 유형×시간대": ("new", "seg", False),
           "P3 최근 편향 보정": ("new", "one", True), "P4 휴무일 맥락·추세 피처": ("hol", "one", False),
           "P5 Open-Meteo 3~7일 전 실행": ("om", "one", False), "P6 Chronos 공변량 확장": ("ext", "one", False)}
COMBO_PARTS = {"P2 λ 일 유형×시간대": "seg", "P3 최근 편향 보정": "bias", "P4 휴무일 맥락·추세 피처": "hol",
               "P5 Open-Meteo 3~7일 전 실행": "om", "P6 Chronos 공변량 확장": "ext"}


# ---------------------------------------------------------------- 공통 표
def base_table(cfg) -> pd.DataFrame:
    """잔차 설정 cfg의 [date, hour, H, c, qlo, qhi, rhat] + y·휴무일·구간 열.
    cfg = 'mix' (P8 사후 후보): 휴무일 행은 ext(Chronos 공변량 확장), 평일 행은 new(피처 표 재생성)."""
    if cfg.startswith("mix"):      # mix = 평일 new·휴무일 ext, mix|평일설정|휴무일설정 으로 일반화
        wk, of = ("new", "ext") if cfg == "mix" else cfg.split("|")[1:]
        a, b = base_table(wk), base_table(of)
        return pd.concat([a[~a["offday"]], b[b["offday"]]]).sort_values(["H", "date", "hour"]).reset_index(drop=True)
    if not (RES3 / f"{cfg}.csv.gz").exists():
        run_resid(cfg)
    d = pd.read_csv(RES3 / f"{cfg}.csv.gz", parse_dates=["date"])
    act = load_smp_actual("육지")[["date", "hour", "smp"]].rename(columns={"smp": "y"})
    d = d.merge(act, on=["date", "hour"], how="left")
    cal = build_calendar(d["date"].min(), d["date"].max()).set_index("date")["is_offday"]
    d["offday"] = d["date"].map(cal).astype(bool)
    d["seg"] = np.where(d["offday"], "휴무일 ", "평일 ") + d["hour"].map(BANDS)
    d["base_date"] = d["date"] - pd.to_timedelta(d["H"], unit="D")
    return d.sort_values(["H", "date", "hour"]).reset_index(drop=True)


def between(d, per):
    return d["date"].between(*per).values & d["y"].notna().values


def fit_lambda(d, per, mode):
    """λ (전체 하나 또는 일 유형×시간대 칸별)를 per 구간 MAE로."""
    m = between(d, per)
    best = lambda x: min(GRID, key=lambda l: np.abs(x["y"] - x["c"] - l * x["rhat"]).mean())
    if mode == "one":
        return {"전체": best(d[m])}
    return {s: best(g) for s, g in d[m].groupby("seg")}


def apply_lambda(d, lam):
    l = d["seg"].map(lam).fillna(lam.get("전체", np.nan)).values if "전체" not in lam else lam["전체"]
    return d["c"].values + l * d["rhat"].values


def recent_bias(d, pred) -> np.ndarray:
    """기준일까지 확정된 같은 지평·같은 시간 최종 오차의 지수평균(반감기 3일). 대상일 D = 기준일 + H에 붙임."""
    x = d[["date", "hour", "H", "y"]].assign(e=d["y"].values - pred)
    out = pd.Series(np.nan, index=d.index)
    for (H, h), g in x.groupby(["H", "hour"]):
        s = g.set_index("date")["e"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max()))
        b = s.ewm(halflife=HALFLIFE, ignore_na=True).mean()      # 날짜 t까지의 오차 평균 (t의 실적은 t에 확정)
        b.index = b.index + pd.Timedelta(days=H)                  # 대상일 t+H(기준일 t)에 사용
        out[g.index] = g["date"].map(b).values
    return out.fillna(0).values


def fit_beta(d, pred, b, per):
    m = between(d, per)
    return min(GRID, key=lambda k: np.abs(d["y"].values[m] - pred[m] - k * b[m]).mean())


def build_point(spec, fit_per):
    """spec = (잔차 설정, λ 방식, 편향 보정) → (표, 예측, 설정값)."""
    cfg, mode, bias = spec
    d = base_table(cfg)
    lam = fit_lambda(d, fit_per, mode)
    pred = apply_lambda(d, lam)
    info = {"잔차 설정": cfg, "λ": lam}
    if bias:
        b = recent_bias(d, pred)
        beta = fit_beta(d, pred, b, fit_per)
        pred = pred + beta * b
        info["β"] = beta
    return d, pred, info


def p0_point():
    """현재 운영 모델 = 잔차 설정 old(운영 잔차 예측과 2025~ 완전히 같음, 2024분 포함) + λ 0.5 고정."""
    d = base_table("old")
    return d, d["c"].values + 0.5 * d["rhat"].values


def level_change(d) -> np.ndarray:
    """수준 변화(사후, 진단용): 대상일 일평균 ÷ 기준일까지 28일 평균."""
    act = load_smp_actual("육지")
    daily = act.groupby("date")["smp"].mean()
    ma28 = daily.rolling(28, min_periods=20).mean()
    r = d["date"].map(daily) / d["base_date"].map(ma28)
    return np.select([r < 0.9, r > 1.1], ["하락", "상승"], "보합")


def point_segments(d, pred, mask) -> dict:
    x = d[mask]
    e = np.abs(x["y"].values - pred[mask])
    mid, eve = x["hour"].between(11, 15).values, x["hour"].between(18, 22).values
    off = x["offday"].values
    lv = level_change(x)
    h1 = (x["date"] < "2025-07-01").values
    segs = {"전체": np.ones(len(x), bool), "상반기": h1, "하반기": ~h1, "평일": ~off, "휴무일": off,
            "휴무일 한낮": off & mid, "휴무일 저녁 피크": off & eve, "평일 저녁 피크": ~off & eve,
            "D+1": (x["H"] == 1).values, "D+2~7": (x["H"] >= 2).values, "수준 하락일(사후)": lv == "하락",
            "수준 상승일(사후)": lv == "상승", "급락 시간": (x["y"] < 30).values}
    return {k: e[m].mean() for k, m in segs.items()}


# ---------------------------------------------------------------- ① 점예측
def select_point():
    rows, preds, infos = [], {}, {}
    for name, spec in P_ORDER.items():
        if spec is None:
            d, pred = p0_point()
            info = {"설명": "Chronos-2 small + 0.5 × 잔차(외부만, 이전 피처 표)"}
        else:
            d, pred, info = build_point(spec, FIT)
        m = between(d, SEL) & ~np.isnan(pred)
        preds[name], infos[name] = (d, pred), info
        rows.append({"후보": name, **point_segments(d, pred, m), "설정": json.dumps(info, ensure_ascii=False)})
        print(f"{name}: 2025 MAE {rows[-1]['전체']:.3f}", flush=True)
    t = pd.DataFrame(rows)
    # P7 조합: P1보다 0.05원 넘게 좋은 것
    p1 = t.loc[t["후보"] == "P1 피처 표 재생성", "전체"].iloc[0]
    good = [COMBO_PARTS[n] for n in COMBO_PARTS if t.loc[t["후보"] == n, "전체"].iloc[0] < p1 - TIE_MAE]
    if len(good) >= 2:
        cfg = "+".join(p for p in ("ext", "hol", "om") if p in good) or "new"
        spec = (cfg, "seg" if "seg" in good else "one", "bias" in good)
        d, pred, info = build_point(spec, FIT)
        m = between(d, SEL)
        name = "P7 조합(" + ", ".join(good) + ")"
        preds[name], infos[name] = (d, pred), info
        rows.append({"후보": name, **point_segments(d, pred, m), "설정": json.dumps(info, ensure_ascii=False)})
        print(f"{name}: 2025 MAE {rows[-1]['전체']:.3f}", flush=True)
        t = pd.DataFrame(rows)
    best = t["전체"].min()
    pick = t[t["전체"] <= best + TIE_MAE].iloc[0]["후보"]          # 표 순서 = 단순한 순
    t["선택"] = np.where(t["후보"] == pick, "✓", "")
    t.round(3).to_csv(TAB / "WZ3_S1_point_2025.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print(t.drop(columns="설정").round(2).to_string(index=False))
    ch = {"점예측": {"후보": pick, "2024 맞춤 설정": json.loads(t.loc[t["후보"] == pick, "설정"].iloc[0]),
                    "spec": None if pick == "P0 현재" else _spec_of(pick, t)},
          "점예측 규칙": "2024로 λ·β → 2025 전체 MAE 최소, 0.05원 이내면 단순한 쪽"}
    CHOICE.write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
    print("선택:", pick)


def _spec_of(name, t):
    if name in P_ORDER:
        return list(P_ORDER[name])
    info = json.loads(t.loc[t["후보"] == name, "설정"].iloc[0])
    return [info["잔차 설정"], "seg" if len(info["λ"]) > 1 else "one", "β" in info]


def posthoc_point():
    """P8 사후 후보(2025를 본 뒤 추가, 사용자 결정 2026-10-28): 휴무일 = P6, 평일 = P1. λ는 2024로.
    근거: P6이 2024·2025 모두 휴무일 −0.35원·평일 +0.2원대 (λ 고정이라 2024도 표본 밖). 점예측을 P8로 바꿔 둠."""
    d, pred, info = build_point(("mix", "one", False), FIT)
    p0d, p0 = p0_point()
    rows = []
    for name, dd, pp in (("P0 현재", p0d, p0), ("P8 혼합(휴무일 P6·평일 P1)", d, pred)):
        for per in (FIT, SEL):
            m = between(dd, per) & ~np.isnan(pp)
            rows.append({"후보": name, "기간": per[0][:4], **point_segments(dd, pp, m)})
    t = pd.DataFrame(rows)
    t.round(3).to_csv(TAB / "WZ3_S1b_point_posthoc.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print(t.round(2).to_string(index=False))
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))
    ch["점예측 (규칙 선택)"] = ch["점예측"]
    ch["점예측"] = {"후보": "P8 혼합(휴무일 P6·평일 P1)", "2024 맞춤 설정": info, "spec": ["mix", "one", False],
                 "주의": "2025 결과를 본 뒤 추가한 사후 후보 — 2024에서 같은 방향 확인, 사용자 결정(2026-10-28)"}
    ch.pop("범위", None)
    CHOICE.write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")


def select_final():
    """④-2 최종 조합: C0, P8, ④-0 선택 Chronos 판, ④-1 FK, K(둘을 합침), K-mix(K의 휴무일 행 = ext판).
    2024·2025 모두 C0보다 좋은 후보 중 2025 MAE 최소, 0.05원 이내면 단순한 쪽(이 순서)."""
    s3 = pd.read_csv(TAB / "WZ3_S3_chronos.csv")
    s4 = pd.read_csv(TAB / "WZ3_S4_features.csv")
    ch_pick = s3[s3["선택"] == "✓"]["설정"].tolist()
    ch_cfg = ch_pick[0] if ch_pick else None
    ok4 = s4[s4["인정"].astype(bool)]
    fk = ok4[ok4["후보"].str.startswith("FK")]["설정"].tolist() or (ok4["설정"].tolist()[:1] if len(ok4) == 1 else [])
    if len(ok4) >= 2 and not fk:
        fk = [ok4.sort_values("2025 전체").iloc[0]["설정"]]
    fparts = fk[0].split("+")[1:] if fk else []
    cands = {"C0 기준(P1)": "new", "P8 v3(휴무일 ext·평일 new)": "mix"}
    if ch_cfg:
        cands[f"④-0 Chronos({ch_cfg})"] = ch_cfg
    if fk:
        cands[f"④-1 피처({fk[0]})"] = fk[0]
    k = "+".join([ch_cfg or "new"] + fparts)
    if ch_cfg and fparts:
        cands[f"K({k})"] = k
    if fparts or ch_cfg:
        cands[f"K-mix(평일 {k}·휴무일 {'+'.join(['ext'] + fparts)})"] = f"mix|{k}|{'+'.join(['ext'] + fparts)}"
    t = pd.DataFrame([eval_cfg(n, c) for n, c in cands.items()])
    c0 = t.iloc[0]
    t["인정"] = [i > 0 and accept(x, c0) for i, x in t.iterrows()]
    ok = t[t["인정"]]
    pick = ok[ok["2025 전체"] <= ok["2025 전체"].min() + TIE_MAE].iloc[0] if len(ok) else t.iloc[0]
    t["선택"] = np.where(t["후보"] == pick["후보"], "✓", "")
    t.round(3).to_csv(TAB / "WZ3_S5_final.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print(t.round(3).drop(columns=["설정"]).to_string(index=False))
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))
    if "점예측 (v3 P8)" not in ch:
        ch["점예측 (v3 P8)"], ch["범위 (v3 P8)"] = ch["점예측"], ch.pop("범위", None)
    ch.pop("범위", None)
    ch.pop("2026 채점", None)
    ch["점예측"] = {"후보": pick["후보"], "spec": [pick["설정"], "one", False], "단계": "④-2 최종 조합"}
    CHOICE.write_text(json.dumps(ch, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("선택:", pick["후보"])


C_ORDER = {"C0 기준(P1)": "new", "C3 수요예측 이력(과거 전용)": "dem", "C2 과거 1년(8,192시간)": "c8k",
           "C5 대형판": "big", "C1 미세조정": "ft", "C6 small·대형 평균": "avg_cal_big"}


def eval_cfg(name, cfg) -> dict:
    """잔차 설정 cfg(λ 2024 맞춤)의 2024·2025 MAE 요약 한 줄."""
    d, pred, info = build_point((cfg, "one", False), FIT)
    r = {"후보": name, "설정": cfg, "λ": info["λ"]["전체"]}
    for per in (FIT, SEL):
        m = between(d, per) & ~np.isnan(pred)
        seg = point_segments(d, pred, m)
        r |= {f"{per[0][:4]} {k}": v for k, v in seg.items()
              if k in ("전체", "상반기", "하반기", "평일", "휴무일", "휴무일 한낮", "D+1", "D+2~7", "수준 하락일(사후)")}
    c = between(d, SEL)
    r["2025 Chronos 단독"] = np.abs(d["y"].values[c] - d["c"].values[c]).mean()
    print(f"{name}: 2024 {r['2024 전체']:.3f} · 2025 {r['2025 전체']:.3f} (Chronos 단독 {r['2025 Chronos 단독']:.3f})", flush=True)
    return r


def accept(r, c0) -> bool:
    """인정 규칙: 2025 MAE가 기준보다 0.05원 넘게 좋고 2024에서도 좋을 것."""
    return bool((r["2025 전체"] < c0["2025 전체"] - TIE_MAE) and (r["2024 전체"] < c0["2024 전체"]))


def select_features():
    """④-1 잔차 피처·학습 후보 F1~F7 → 인정된 것을 합친 FK."""
    order = {"C0 기준(P1)": "new", "F1 수요예측 이력": "new+dmd", "F2 SMP 실적 수준": "new+smp", "F3 연료비 변화": "new+fuel",
             "F4 계절·연휴 달력": "new+cal", "F5 지평 통합 학습": "new+pool", "F6 최근 1년 학습": "new+w365",
             "F7 LightGBM 설정": "new+hp"}
    t = pd.DataFrame([eval_cfg(n, c) for n, c in order.items()])
    c0 = t.iloc[0]
    t["인정"] = [i > 0 and accept(x, c0) for i, x in t.iterrows()]
    ok = t[t["인정"]]
    if len(ok) >= 2:
        cfg = "new+" + "+".join(c.split("+")[1] for c in ok["설정"])
        r = eval_cfg(f"FK 조합({cfg})", cfg)
        r["인정"] = accept(r, c0)
        t = pd.concat([t, pd.DataFrame([r])], ignore_index=True)
    t.round(3).to_csv(TAB / "WZ3_S4_features.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print(t.round(3).drop(columns=["설정"]).to_string(index=False))


def select_chronos():
    """④-0 Chronos 고도화: 2025 MAE가 C0보다 0.05원 넘게 좋고 2024에서도 좋은 후보만 인정 → 둘 이상이면 C4(평균)."""
    t = pd.DataFrame([eval_cfg(n, c) for n, c in C_ORDER.items()])
    c0 = t.iloc[0]
    t["인정"] = [i > 0 and accept(x, c0) for i, x in t.iterrows()]
    ok = t[t["인정"]]
    if len(ok) >= 2:
        cfg = "avg_" + "_".join(ok["설정"])
        r = eval_cfg(f"C4 평균({', '.join(ok['설정'])})", cfg)
        r["인정"] = accept(r, c0)
        t = pd.concat([t, pd.DataFrame([r])], ignore_index=True)
    ok = t[t["인정"]]
    pick = None
    if len(ok):
        best = ok["2025 전체"].min()
        pick = ok[ok["2025 전체"] <= best + TIE_MAE].iloc[0]       # 표 순서 = 단순한 순
    t["선택"] = np.where(t["후보"] == (pick["후보"] if pick is not None else ""), "✓", "")
    t.round(3).to_csv(TAB / "WZ3_S3_chronos.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print(t.round(3).to_string(index=False))
    if pick is None:
        print("인정된 후보 없음 → 2026은 보지 않음")
        return
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))
    ch["점예측 (v3 P8)"] = ch["점예측"]
    ch["범위 (v3 P8)"] = ch.pop("범위", None)
    ch.pop("2026 채점", None)
    ch["점예측"] = {"후보": pick["후보"], "spec": [pick["설정"], "one", False], "단계": "④-0 Chronos 고도화"}
    CHOICE.write_text(json.dumps(ch, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("선택:", pick["후보"])


# ---------------------------------------------------------------- ② 범위
def c2_quantiles() -> pd.DataFrame:
    h = pd.read_csv(W3 / "chronos" / "c2_cal_2023-01-01.csv.gz", parse_dates=["date"])
    b = pd.read_csv(WZOO / "c2.csv.gz", parse_dates=["date"])
    return pd.concat([h, b], ignore_index=True)[["date", "hour", "H", "qlo", "qhi"]]


def conformal(d, qlo, qhi, pred, ev, window, mode, gamma=None):
    """칸(시간대묶음×휴무일)별, 매일 직전 window일(대상일 ≤ 기준일) 점수의 분위수로 보정. 지평 H 하나짜리 표 d.
    mode: sym(대칭, 80%) | asym(하한·상한 각 90%) | aci(asym + 적응형 수준, γ)."""
    H = int(d["H"].iloc[0])
    y = d["y"].values
    key = d["hour"].map(BAND).values + np.where(d["offday"].values, "_1", "_0")
    dates = d["date"].values.astype("datetime64[D]")
    lo, hi = np.full(len(d), np.nan), np.full(len(d), np.nan)
    s_sym = np.maximum(qlo - y, y - qhi)
    s_lo, s_hi = qlo - y, y - qhi
    days = np.unique(dates[ev])
    for k in np.unique(key):
        ik = np.where(key == k)[0]
        dk, yk = dates[ik], y[ik]
        okk = ~np.isnan(yk)
        a_lo = a_hi = 0.1
        miss_log = {}
        for day in days:
            tgt = ik[(dk == day)]
            tgt = tgt[ev[tgt]]
            if len(tgt) == 0:
                continue
            end = day - np.timedelta64(H - 1, "D")
            hist = ik[(dk < end) & (dk >= end - np.timedelta64(window, "D")) & okk]
            if len(hist) < 20:
                continue
            if mode == "sym":
                c = np.quantile(s_sym[hist], 0.8)
                lo[tgt], hi[tgt] = qlo[tgt] - c, qhi[tgt] + c
            else:
                if mode == "aci":
                    known = day - np.timedelta64(H, "D")      # 기준일에 확정된 마지막 대상일
                    if known in miss_log:
                        ml, mh = miss_log[known]
                        a_lo = float(np.clip(a_lo + gamma * (0.1 - ml), 0.005, 0.5))
                        a_hi = float(np.clip(a_hi + gamma * (0.1 - mh), 0.005, 0.5))
                lo[tgt] = qlo[tgt] - np.quantile(s_lo[hist], 1 - a_lo)
                hi[tgt] = qhi[tgt] + np.quantile(s_hi[hist], 1 - a_hi)
                if mode == "aci":
                    yy = y[tgt]
                    ok = ~np.isnan(yy)
                    if ok.any():
                        miss_log[day] = ((yy[ok] < lo[tgt][ok]).mean(), (yy[ok] > hi[tgt][ok]).mean())
    lo = np.clip(np.minimum(lo, pred), 0, None)
    hi = np.maximum(hi, pred)
    return lo, hi


def range_candidates():
    c = [("A0 현재", dict(kind="current"))]
    for fam, mode in (("A1 대칭 보정", "sym"), ("A2 하한·상한 따로 보정", "asym")):
        for src in ("Chronos-2", "Chronos-2 small"):
            for w in (90, 180, 365):
                c.append((f"{fam} · {src} · {w}일", dict(kind=mode, src=src, window=w)))
    for src in ("Chronos-2", "Chronos-2 small"):
        for g in (0.005, 0.02):
            c.append((f"A3 적응형 · {src} · 365일 · γ={g}", dict(kind="aci", src=src, window=365, gamma=g)))
    return c


def make_range(d, pred, cand, ev_per, q2=None):
    """d(전 지평) + 예측 → lo, hi (ev_per 구간 행만 값, 나머지 NaN)."""
    kind = cand["kind"]
    lo, hi = np.full(len(d), np.nan), np.full(len(d), np.nan)
    shift = pred - d["c"].values
    for H in range(1, 8):
        ih = np.where(d["H"].values == H)[0]
        g = d.iloc[ih].reset_index(drop=True)
        ev = g["date"].between(*ev_per).values
        if kind == "current":
            qq = g[["date", "hour"]].merge(q2[q2["H"] == H], on=["date", "hour"], how="left")
            ql, qh = qq["qlo"].values, qq["qhi"].values
            if H == 1:
                l, h = conformal(g, ql, qh, pred[ih], ev, 90, "sym")
            else:
                l, h = np.clip(ql, 0, None), qh
        else:
            if cand["src"] == "Chronos-2":
                qq = g[["date", "hour"]].merge(q2[q2["H"] == H], on=["date", "hour"], how="left")
                ql, qh = qq["qlo"].values, qq["qhi"].values
            else:
                ql, qh = g["qlo"].values, g["qhi"].values
            ql, qh = ql + shift[ih], qh + shift[ih]
            l, h = conformal(g, ql, qh, pred[ih], ev, cand["window"], kind, cand.get("gamma"))
        lo[ih[ev]], hi[ih[ev]] = np.asarray(l)[ev], np.asarray(h)[ev]
    return lo, hi


def rscore(y, lo, hi):
    b, a = y < lo, y > hi
    pen = 10 * ((lo - y) * b + (y - hi) * a)
    return {"포함률": 1 - b.mean() - a.mean(), "하한 아래": b.mean(), "상한 위": a.mean(), "평균 폭": (hi - lo).mean(),
            "구간점수": (hi - lo + pen).mean()}


def range_segments(d, lo, hi, m):
    x = d[m]
    y, l, h = x["y"].values, lo[m], hi[m]
    off, mid, eve = x["offday"].values, x["hour"].between(11, 15).values, x["hour"].between(18, 22).values
    segs = {"전체": np.ones(len(x), bool), "휴무일": off, "휴무일 한낮": off & mid, "휴무일 저녁 피크": off & eve,
            "D+1": (x["H"] == 1).values, "D+2~7": (x["H"] >= 2).values, "수준 하락일(사후)": level_change(x) == "하락"}
    out = {}
    for k, s in segs.items():
        r = rscore(y[s], l[s], h[s])
        out |= {f"{k} {n}": v for n, v in r.items()} if k != "전체" else r
    return out


def point_from_choice(ch, fit_per):
    spec = ch["점예측"]["spec"]
    if spec is None:
        return (*p0_point(), {"λ": {"전체": 0.5}, "잔차 설정": "old"})
    return build_point(tuple(spec), fit_per)


def select_range():
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))
    d, pred, _ = point_from_choice(ch, FIT)
    q2 = c2_quantiles()
    m = between(d, SEL)
    made = []
    for name, cand in range_candidates():
        lo, hi = make_range(d, pred, cand, SEL, q2)
        for adj in (False, True):
            l = lo
            if adj:
                w = d[["date", "hour", "H"]].assign(lo=lo)
                l = range_adjust.apply(w, "육지", offday=d["offday"].values)["lo"].values
            made.append((name, adj, l, hi))
    ok = m.copy()                      # 모든 후보가 값을 가진 같은 시간끼리 비교
    for _, _, l, h in made:
        ok &= ~np.isnan(l) & ~np.isnan(h)
    print(f"공통 채점 시간 {ok.sum():,} / 2025 전체 {m.sum():,}", flush=True)
    rows = []
    for name, adj, l, h in made:
        rows.append({"후보": name, "휴일·주말 한낮 하한 보정": "있음" if adj else "없음", "시간수": int(ok.sum()),
                     **range_segments(d, l, h, ok)})
        if adj:
            print(f"{name}: 포함률 {rows[-2]['포함률']:.3f} / {rows[-1]['포함률']:.3f}, 구간점수 {rows[-2]['구간점수']:.2f} / {rows[-1]['구간점수']:.2f}", flush=True)
    t = pd.DataFrame(rows)
    ok = t["포함률"].between(0.75, 0.85)
    best = t.loc[ok, "구간점수"].min()
    # 단순한 순서: 표 순서(후보) → 하한 보정 없음 먼저
    cand = t[ok & (t["구간점수"] <= best * (1 + TIE_RANGE))]
    pick = cand.iloc[0]
    t["선택"] = np.where((t["후보"] == pick["후보"]) & (t["휴일·주말 한낮 하한 보정"] == pick["휴일·주말 한낮 하한 보정"]), "✓", "")
    t.round(4).to_csv(TAB / "WZ3_S2_range_2025.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(t[["후보", "휴일·주말 한낮 하한 보정", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수", "휴무일 포함률",
             "휴무일 구간점수", "D+2~7 구간점수", "선택"]].round(3).to_string(index=False))
    ch["범위"] = {"후보": pick["후보"], "휴일·주말 한낮 하한 보정": pick["휴일·주말 한낮 하한 보정"],
                "설정": dict(range_candidates())[pick["후보"]]}
    ch["범위 규칙"] = "2025 포함률 75~85% 중 구간점수 최소, 1% 이내면 단순한 쪽, 모든 지평 같은 방식"
    CHOICE.write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
    print("선택:", pick["후보"], pick["휴일·주말 한낮 하한 보정"])


# ---------------------------------------------------------------- ③ 2026 채점
def final_predictions(ch, fit_per=("2024-01-01", "2025-12-31"), ev_per=("2025-01-01", "2100-01-01")):
    """고른 구성을 fit_per로 다시 맞춰 ev_per 대상일의 [base_date, H, date, hour, y, pred, lo, hi, offday]."""
    d, pred, info = point_from_choice(ch, fit_per)
    cand = ch["범위"]["설정"]
    lo, hi = make_range(d, pred, cand, ev_per, c2_quantiles())
    if ch["범위"]["휴일·주말 한낮 하한 보정"] == "있음":
        lo = range_adjust.apply(d[["date", "hour", "H"]].assign(lo=lo), "육지", offday=d["offday"].values)["lo"].values
    w = d.assign(pred=pred, lo=lo, hi=hi)
    w = w[w["date"].between(*ev_per) & w["lo"].notna()]
    return w[["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday"]], info


def score_test():
    ch = json.loads(CHOICE.read_text(encoding="utf-8"))
    new, info = final_predictions(ch)
    new.assign(stage="15h").round(2).to_csv(W3 / "WZ3_predictions_hourly.csv", index=False, encoding="utf-8-sig")
    cur = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["base_date", "date"])
    cur = range_adjust.apply(cur[cur["y"].notna()].reset_index(drop=True), "육지")
    act = load_smp_actual("육지")[["date", "hour", "smp"]]
    lag7 = act.assign(date=act["date"] + pd.Timedelta(days=7)).rename(columns={"smp": "lag7"})
    m = cur.merge(new, on=["date", "hour", "H"], suffixes=("_cur", "_new"))
    p8f = W3 / "WZ3_predictions_hourly_P8.csv"
    models = {"cur": "지금 모델", "new": "v3 최종"}
    if p8f.exists() and ch["점예측"]["spec"] != ["mix", "one", False]:
        p8 = pd.read_csv(p8f, parse_dates=["date"])[["date", "hour", "H", "pred", "lo", "hi"]]
        m = m.merge(p8.rename(columns={"pred": "pred_p8", "lo": "lo_p8", "hi": "hi_p8"}), on=["date", "hour", "H"])
        models = {"cur": "지금 모델", "p8": "v3 P8", "new": "v3 최종"}
    m = m[m["date"].between(*TEST) & m["y_cur"].notna()].merge(lag7, on=["date", "hour"], how="left")
    m["y"], m["offday"] = m["y_cur"], m["offday_cur"].astype(bool)
    rows = []
    segs = hourly_segments(m)
    off, eve = m["offday"].values, m["hour"].between(18, 22).values
    segs |= {"휴무일 저녁 피크(18~22시)": off & eve, "평일 저녁 피크(18~22시)": ~off & eve, "D+2~7": (m["H"] >= 2).values}
    segs |= {f"D+{H}": (m["H"] == H).values for H in range(1, 8)}
    for seg, s in segs.items():
        g = m[s]
        for who, label in models.items():
            rows.append({"구분": seg, "모델": label,
                         **metrics(g["y"], g[f"pred_{who}"], g[f"lo_{who}"], g[f"hi_{who}"], g["lag7"])})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "WZ3_T1_test2026.csv", index=False, encoding="utf-8-sig")
    ch["2026 채점"] = {"기간": f"{TEST[0]} ~ {TEST[1]}", "시간수": int(len(m)), "재맞춤 설정(2024~2025)": info}
    CHOICE.write_text(json.dumps(ch, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 300)
    print(t[["구분", "모델", "표본수", "MAE", "rMAE", "편향(공표−예측)", "포함률", "하한 아래", "상한 위", "평균 폭", "구간점수"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    {"point": select_point, "posthoc": posthoc_point, "chronos": select_chronos, "features": select_features,
     "final": select_final, "range": select_range,
     "test": score_test}[sys.argv[1]]()
