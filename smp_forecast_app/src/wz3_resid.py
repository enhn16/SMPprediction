"""일주일 v3 (R3) — Chronos 잔차 보정 LightGBM 후보 (wz_resid와 같은 모델·설정, 2024-01부터 매월 재학습).

  최종 예측 = Chronos + λ × 잔차 예측 (λ 등은 wz3_select가 2024로 정하고 2025로 비교)
  잔차 = 실적 − Chronos, LightGBM(L1) 같은 설정, 지평 H마다, 매월 1일 재학습 (실적은 재학습일 − H일까지, 학습 2023-01~)
후보 설정 (2026 채점 전 고정)
  old  : 현재 운영과 같은 피처 표(공용 features_hourly_육지*) + 외부만 — 재현 점검용
  new  : wz3 피처 표(2023 날씨·주간전망 보강 반영) + 외부만                                  [B6]
  hol  : new + 휴무일 맥락(연휴 길이·며칠째, 직전 같은 유형일 평균·한낮최저, 전날 평균·한낮최저)
         + 추세(7일 같은 시간 평균 ÷ 28일 평균, Chronos 최근 7일 편향: 같은 시간·하루 전체)         [B2]
  om   : new + Open-Meteo 3~7일 전 실행 일사·운량을 D+2~6에 채움 (wz3_openmeteo)               [B5]
  ext  : Chronos-2 small 공변량 확장판(wz3_chronos ext) + new 피처                              [B4]
  ④-1 잔차 피처·학습 후보 (new 위에 덧붙임): dmd 수요예측 이력, smp SMP 실적 수준, fuel 연료비 변화,
       cal 계절·연휴 달력, pool 지평 통합 학습, w365 최근 1년만 학습, hp LightGBM 설정(800그루·잎 31)
  dem·c8k·ft·big: Chronos 판만 바꾼 C3(수요 이력)·C2(과거 1년)·C1(미세조정)·C5(대형판), 피처는 new
  avg_cal_big: C6 small·대형 평균
  ⑤ 변동비 순위: cvc = Chronos 변동비 계단 공변량판(wz3_chronos vc) + new 피처, new+vsS1 등 (vs 계단·vk 누적출력 위치·va 둘 다·vd 변화량+위치(사후) × S1·S2), wz3_vc
  ⑥: nuc·sup·curt 공급 피처(wz3_supply), tirex·tfm 기초모델 판(wz3_fm), ftl·ftf 미세조정 재시도, offonly 휴무일 행만 학습
  ravg: C0·F1~F7 잔차 예측 단순 평균 (사후 등록, 같은 인정 규칙)
  조합(예: hol+om, ext+hol)은 이름을 '+'로 이어 씀
출력: data/processed/smp_forecast_app/wz3/resid/{설정}.csv.gz [date, hour, H, c, qlo, qhi, rhat]
실행: python -m smp_forecast_app.src.wz3_resid old new hol om ext
"""
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR
from smp_forecast_app.src.features import load_features
from smp_forecast_app.src.wz_chronos import HIST
from smp_forecast_app.src.wz_common import OUT as WZOO
from smp_forecast_app.src.wz_resid import CD, EXO, LGB

warnings.filterwarnings("ignore")

W3 = PROCESSED_DIR / "smp_forecast_app" / "wz3"
RES3 = W3 / "resid"
HS = range(1, 8)
TRAIN_START = "2023-01-01"
EVAL_START = "2024-01-01"
DMD = ["d_d1_h", "d_d7_h", "d_d1_mean", "d_same_h", "d_d1_shape", "kpx_vs_dem"]
SMPX = ["s_d1", "s_d7", "s_ma7_h", "s_ma28", "s_same_h", "s_d1_min", "s_d1_max", "s_d1_midmin", "s_same_midmin"]
FUEL = ["fc_lng_chg", "fc_coal_chg", "fc_ratio", "smp_vs_lng"]
CALX = ["month", "doy_sin", "doy_cos", "off_block_len", "off_block_day", "is_labor_day"]
LGB_HP = dict(LGB, n_estimators=800, learning_rate=0.02, num_leaves=31, min_child_samples=100)
HOL = ["off_block_len", "off_block_day", "s_same_mean", "s_same_midmin", "s_d1_mean", "s_d1_midmin",
       "trend_7_28", "cbias7_h", "cbias7"]


CHRONOS_FILES = {"ext": ["c2s_ext_2023-01-01"], "dem": ["c2s_dem_2023-01-01"], "c8k": ["c2s_cal_2023-01-01_ctx8192"],
                 "cvc": ["c2s_vc_2023-01-01"], "tirex": ["tirex_2023-01-01"], "tfm": ["tfm_2023-01-01"],
                 "fty": ["c2s_fty_2023-01-01"], "ftq": ["c2s_ftq_2023-01-01"],   # ⑦ 미세조정 1년·분기 (wz3_ftq, 2025까지)
                 "fts": ["c2s_fts_2023-01-01"]}   # K3 확정판: 1년 미세조정을 기준일 기준으로 고름 (wz3_ops.assemble_fts, 2026 포함)


def chronos(kind="cal") -> pd.DataFrame:
    """Chronos-2 small [date, hour, H, pred, qlo, qhi], 대상일 2023~.
    ft(C1): 2023 제로샷 + 2024 ft2023판 + 2025 ft2024판 (+ 2026 ft2025판이 있으면) — 모두 그 판의 학습 마감 뒤 예측."""
    rd = lambda n: pd.read_csv(W3 / "chronos" / f"{n}.csv.gz", parse_dates=["date"])
    hist = pd.read_csv(HIST / "c2s.csv.gz", parse_dates=["date"])
    if kind.startswith("avg_"):     # C4: 여러 판의 예측·분위수 평균 (예: avg_dem_ft), 모든 판에 있는 행만
        fs = [chronos(k).set_index(["date", "hour", "H"]) for k in kind.split("_")[1:]]
        idx = fs[0].index
        for f in fs[1:]:
            idx = idx.intersection(f.index)
        return (sum(f.loc[idx, ["pred", "qlo", "qhi"]] for f in fs) / len(fs)).reset_index()
    if kind == "big":               # C5: Chronos-2 대형판 제로샷 (2023~24 wz3 + 2025~ wzoo)
        return pd.concat([rd("c2_cal_2023-01-01"), pd.read_csv(WZOO / "c2.csv.gz", parse_dates=["date"])], ignore_index=True)
    if kind in CHRONOS_FILES:
        return pd.concat([rd(n) for n in CHRONOS_FILES[kind]], ignore_index=True)
    if kind in ("ft", "ftl", "ftf"):     # ⑥ ftl = LoRA 1e-5, ftf = 전체 1e-6
        v = kind[2:]
        parts = [hist[hist["date"] < "2024-01-01"], rd(f"c2s_cal_2024-01-01_ft2023{v}"), rd(f"c2s_cal_2025-01-01_ft2024{v}")]
        if (W3 / "chronos" / f"c2s_cal_2026-01-01_ft2025{v}.csv.gz").exists():
            parts.append(rd(f"c2s_cal_2026-01-01_ft2025{v}"))
        return pd.concat(parts, ignore_index=True)
    return pd.concat([hist, pd.read_csv(WZOO / "c2s.csv.gz", parse_dates=["date"])], ignore_index=True)


def chronos_kind(cfg) -> str:
    return next((p for p in cfg.split("+") if p in ("ext", "dem", "c8k", "ft", "ftl", "ftf", "big", "cvc", "tirex", "tfm", "fty", "ftq", "fts") or p.startswith("avg_")), "cal")


def feats(H, src="new") -> pd.DataFrame:
    if src == "old":
        return load_features(H)
    return pd.read_csv(W3 / "features" / f"features_h{H}.csv", parse_dates=["date"])


def chronos_bias(c, y) -> pd.DataFrame:
    """기준일 I까지 알려진 Chronos 하루 전(H=1) 오차의 최근 7일 평균 → [base_date, hour, cbias7_h, cbias7]."""
    e = c[c["H"] == 1][["date", "hour", "pred"]].merge(y, on=["date", "hour"])
    e["e"] = e["y"] - e["pred"]
    w = e.pivot(index="date", columns="hour", values="e").sort_index()
    w = w.reindex(pd.date_range(w.index.min(), w.index.max()))
    bh = w.rolling(7, min_periods=5).mean()
    b = bh.stack().rename("cbias7_h").reset_index().rename(columns={"level_0": "base_date"})
    b["cbias7"] = b["base_date"].map(w.mean(axis=1).rolling(7, min_periods=5).mean())
    return b


def table(H, cfg, c, bias=None, f=None) -> tuple[pd.DataFrame, list]:
    """f: 피처 표를 직접 넘길 때 (운영 wz3_ops: features.build(H) 결과). 없으면 wz3/features 표."""
    parts = cfg.split("+")
    if f is None:
        f = feats(H, "old" if "old" in parts else "new")
    if "om" in parts and 2 <= H <= 6:
        from smp_forecast_app.src.wz3_openmeteo import load
        om = load()
        om = om[om["H"] == H].drop(columns="H")
        f = f.drop(columns=["om_rad_mid", "om_cloud_mid"]).merge(om, on="date", how="left")
    d = f.merge(c[c["H"] == H][["date", "hour", "pred", "qlo", "qhi"]], on=["date", "hour"])
    d["c_spread"] = d["qhi"] - d["qlo"]
    d["c_vs_d1"] = d["pred"] - d["s_d1"]
    d["c_vs_same"] = d["pred"] - d["s_same_h"]
    d["c_vs_ma28"] = d["pred"] - d["s_ma28"]
    d["r"] = d["y"] - d["pred"]
    cols = EXO + CD
    if "hol" in parts:
        d["trend_7_28"] = d["s_ma7_h"] / d["s_ma28"].clip(lower=1)
        d["base_date"] = d["date"] - pd.Timedelta(days=H)
        d = d.merge(bias, on=["base_date", "hour"], how="left").drop(columns="base_date")
        cols = cols + HOL
    if "dmd" in parts:      # F1 하루전 수요예측 이력(D-H까지 공표분)과 주간전망 대비
        d["kpx_vs_dem"] = d["kpx_peak"] / d["d_d1_mean"].clip(lower=1)
        cols = cols + DMD
    if "smp" in parts:      # F2 SMP 실적 수준 (Chronos 상대값 말고 원값)
        cols = cols + SMPX
    if "fuel" in parts:     # F3 연료비 변화·상대 수준 (모두 기준일에 알려진 값)
        prev = f[["date", "fc_lng", "fc_coal"]].drop_duplicates("date").set_index("date").sort_index()
        prev.index = prev.index + pd.Timedelta(days=35)
        d = d.merge(prev.rename(columns={"fc_lng": "lng_p", "fc_coal": "coal_p"}), left_on="date", right_index=True, how="left")
        d["fc_lng_chg"] = d["fc_lng"] / d["lng_p"] - 1
        d["fc_coal_chg"] = d["fc_coal"] / d["coal_p"] - 1
        d["fc_ratio"] = d["fc_lng"] / d["fc_coal"]
        d["smp_vs_lng"] = d["s_ma28"] / d["fc_lng"]
        cols = cols + FUEL
    if "cal" in parts:      # F4 계절·연휴 달력
        cols = cols + CALX
    vc = next((p for p in parts if p[:2] in ("vs", "vk", "va", "vd") and p[2:] in ("S1", "S2")), None)
    if vc:                  # ⑤ 변동비 순위 (wz3_vc): 계단 vs·누적출력 위치 vk·둘 다 va × 공개 가정 S1·S2
        from smp_forecast_app.src.wz3_vc import VK, VS, features as vc_features
        d = vc_features(d, H, vc[2:])
        d["vc_c_vs_stack"] = d["pred"] - d["vc_stack"]
        cols = cols + {"vs": VS, "vk": VK, "va": VS + VK, "vd": ["vc_dl_chg", "vc_gl_chg"] + VK}[vc[:2]]
    if {"nuc", "sup", "curt"} & set(parts):     # ⑥ 공급 쪽 (13종 밖, wz3_supply)
        from smp_forecast_app.src.wz3_supply import features as sup_features
        d, extra = sup_features(d, H, parts)
        cols = cols + extra
    if "stk" in parts:      # ⑩ G1 여러 Chronos 예측을 잔차 입력으로 (기준 Chronos 대비 차이, 자기 자신 제외)
        base = chronos_kind(cfg)
        for k in ("cal", "ext", "ftf", "big"):
            if k == base:
                continue
            o = _chronos_cached(k)
            o = o[o["H"] == H][["date", "hour", "pred", "qlo", "qhi"]]
            d = d.merge(o.rename(columns={"pred": f"p_{k}", "qlo": f"lo_{k}", "qhi": f"hi_{k}"}), on=["date", "hour"], how="left")
            d[f"stk_{k}"] = d[f"p_{k}"] - d["pred"]
            cols = cols + [f"stk_{k}"]
        if "lo_big" in d:
            d["stk_big_spread"] = d["hi_big"] - d["lo_big"]
            cols = cols + ["stk_big_spread"]
    if "pvx" in parts:      # ⑩ G2 예상 태양광 (맑은 날 일사 × 설비용량 × 구름·강수 감쇠)
        d = pv_features(d)
        cols = cols + PVX
    return d, cols


_CH = {}


def _chronos_cached(k):
    if k not in _CH:
        _CH[k] = chronos(k).drop_duplicates(["date", "hour", "H"], keep="last")
    return _CH[k]


PVX = ["pv_clr", "pv_exp", "pv_ratio", "pv_rel"]


def pv_features(d) -> pd.DataFrame:
    """맑은 날 일사(Haurwitz, 위도 36.5°·경도 127.5°, 시간 h = (h−1, h] 가운데) × cap_solar × 구름·강수 감쇠."""
    doy = d["date"].dt.dayofyear.values
    t = d["hour"].values - 0.5 + (127.5 - 135.0) / 15.0          # KST → 지방 평균 태양시
    g = 2 * np.pi / 365 * (doy - 1 + (t - 12) / 24)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians(15 * (t + eot / 60 - 12))
    lat = np.radians(36.5)
    sin_alt = np.clip(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha), 0, None)
    ghi = np.where(sin_alt > 0.01, 1098 * sin_alt * np.exp(-0.057 / np.maximum(sin_alt, 0.01)), 0.0)
    sky = np.where(d["hour"].values <= 12, d["rf_sky_am"].values, d["rf_sky_pm"].values)
    cloud = 0.1 + 0.85 * np.clip((sky - 1) / 3, 0, 1)
    att = (1 - 0.75 * cloud ** 3.4) * (1 - 0.4 * np.nan_to_num(d["rf_pop_pm"].values, nan=0.0) / 100)
    d = d.copy()
    d["pv_clr"] = d["cap_solar"].values * ghi / 1000 * 0.8
    d["pv_exp"] = d["pv_clr"] * att
    d["pv_ratio"] = d["pv_exp"] / (d["kpx_peak"] * 1000)
    d["pv_rel"] = d["pv_exp"] / d["d_d1_h"].clip(lower=1)
    return d


def rolling(d, H, cols, parts=()) -> pd.DataFrame:
    """매월 1일 재학습. d에 H열이 있으면(지평 통합 pool) 행마다 자기 H로 실적 마감을 적용.
    parts: w365 = 학습을 재학습일 직전 365일로, hp = LightGBM 설정 LGB_HP."""
    ps = list(pd.date_range(EVAL_START, d["date"].max(), freq="MS")) + [pd.Timestamp("2100-01-01")]
    hh = d["H"] if "H" in d else H
    params = LGB_HP if "hp" in parts else LGB
    out = []
    for a, b in zip(ps[:-1], ps[1:]):
        cut = d["date"] < a - pd.to_timedelta(hh - 1, unit="D")
        start = a - pd.Timedelta(days=365) if "w365" in parts else pd.Timestamp(TRAIN_START)
        tr = d[(d["date"] >= start) & cut & d["r"].notna()]
        if "offonly" in parts:      # ⑥ S1 휴무일 전용: 휴무일 행만으로 학습
            tr = tr[tr["offday"] == 1]
        te = d[(d["date"] >= a) & (d["date"] < b)]
        if te.empty:
            continue
        m = lgb.LGBMRegressor(**params).fit(tr[cols], tr["r"])
        o = te[["date", "hour", "pred", "qlo", "qhi"]].assign(rhat=m.predict(te[cols]))
        out.append(o.assign(H=te["H"].values if "H" in te else H))
    return pd.concat(out).rename(columns={"pred": "c"})


RAVG = ["new", "new+dmd", "new+smp", "new+fuel", "new+cal", "new+pool", "new+w365", "new+hp"]


def average(name="ravg", cfgs=RAVG):
    """R-avg: 등록한 잔차 모델(C0·F1~F7)의 rhat 단순 평균 (고르지 않고 모두 같은 비중) — 모든 파일에 있는 행만."""
    fs = [pd.read_csv(RES3 / f"{c}.csv.gz", parse_dates=["date"]).set_index(["date", "hour", "H"]) for c in cfgs]
    idx = fs[0].index
    for f in fs[1:]:
        idx = idx.intersection(f.index)
    out = fs[0].loc[idx, ["c", "qlo", "qhi"]].assign(rhat=sum(f.loc[idx, "rhat"] for f in fs) / len(fs))
    out.reset_index()[["date", "hour", "c", "qlo", "qhi", "rhat", "H"]].round(3).to_csv(RES3 / f"{name}.csv.gz", index=False, encoding="utf-8")
    print(f"저장 {name}.csv.gz ({len(idx):,}행)", flush=True)


def run(cfg):
    if cfg == "ravg":
        return average()
    RES3.mkdir(parents=True, exist_ok=True)
    p = RES3 / f"{cfg}.csv.gz"
    if p.exists():
        print(f"{cfg}: 이미 있음 → 건너뜀", flush=True)
        return
    c = chronos(chronos_kind(cfg))
    bias = None
    if "hol" in cfg.split("+"):
        y = feats(1)[["date", "hour", "y"]].dropna()
        bias = chronos_bias(c, y)
    t0, outs = time.time(), []
    parts = cfg.split("+")
    if "pool" in parts:     # F5 지평 통합: 7개 지평 행을 한 모델로 (H를 피처로)
        tabs = [table(H, cfg, c, bias) for H in HS]
        d = pd.concat([x[0].assign(H=H) for H, x in zip(HS, tabs)], ignore_index=True)
        outs.append(rolling(d, 0, tabs[0][1] + ["H"], parts))
        print(f"{cfg} 통합 ({time.time() - t0:.0f}s)", flush=True)
    for H in ([] if "pool" in parts else HS):
        d, cols = table(H, cfg, c, bias)
        outs.append(rolling(d, H, cols, parts))
        print(f"{cfg} H={H} ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(outs).round(3).to_csv(p, index=False, encoding="utf-8")
    print(f"저장 {p.name}", flush=True)


if __name__ == "__main__":
    for cfg in sys.argv[1:]:
        run(cfg)
