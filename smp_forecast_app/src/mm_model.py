"""한 달 예측 mm — 월말 판단 → 다음 달 모든 날 일평균 SMP + 80% 범위 (2026-11-03 확정, MONTH.md 0-4·0-5절).

모델 = 팀원 최종 코드(reference/한달예측_최종모델_20261103/smp-web/ml/monthly_final.py)의 'LightGBM + 대상 달 연료'를
       우리 로더(load_smp_actual·build_calendar·load_gas_tariff)로 옮긴 것. 연료는 가스공사 발전용 요금(기본) 또는 변동비(비교).
  · 판단 = 매월 마지막 날 15시. 입력 = 그날까지의 SMP(그날 SMP는 전날 저녁 공표) · 다음 달 달력 · 대상 달 연료값.
  · 학습 = 2023-01-01부터 매일을 판단일로 1~31일 뒤 행(대상일 < 학습 마감), 목표 = 대상일 일평균 ÷ 최근 28일 평균. 설정은 팀원 그대로.
  · 대상 달 연료값 = 그 달 전달 말일까지 쓸 수 있던 값(usable_from 기준, 없으면 직전 달 값).
    가스: 원/GJ × 7.2 ÷ 1000을 LNG·LNG(직도입) 두 자리에, 석탄 = 1.0. 변동비: 연료 3개 월별 중앙값(팀원 가정 = 월말에 다음 달 값 공표).
  · 범위 B2 = 표본 밖 상대 오차 ÷ (28일 변동성/0.08 + |LNG 대상 달÷이번 달 로그|/0.05) → 지평 주차별 10·90% 분위수 × 배율.
두 가지 학습 방식
  · 'year' (선택 규칙 재현): 2024·2025·2026 각각 그해 1/1까지 학습한 모델 하나. 2025 범위는 2024 오차, 2026 범위는 2024~2025 오차.
  · 'month' (운영): 월말마다 그날까지 학습. 범위는 2024-01-01 ~ 대상 해 전년 말 대상일의 운영식 오차 (해마다 1월에 늘어남).
실행: python -m smp_forecast_app.src.mm_model            (백테스트·성적·설정, 약 1분)
출력: data/processed/smp_forecast_app/mm/MM_predictions_daily.csv (판단 2024-12-31 ~ 2026-09-30, 운영 방식)
      mm/MM_predictions_calib2024.csv (2024 대상, 범위 보정용) · outputs/tables/MM_score.csv · MM_choice.json
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, RAW_DIR, SEED
from common.loaders import load_gas_tariff, load_smp_actual

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
MM = PROCESSED_DIR / "smp_forecast_app" / "mm"
TRAIN_START = "2023-01-01"
CALIB_START = "2024-01-01"
HMAX = 31
ALPHA = 0.2
MJ_PER_KWH = 7.2
LGB_PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, lambda_l1=1.0, lambda_l2=5.0,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED, deterministic=True)
ROUNDS = 400
FEAT = ["지평", "요일", "휴일", "공휴일", "대상 월", "7일÷28일", "91일÷28일", "평일/휴일 배율", "판단일÷28일", "4주 같은 요일÷28일",
        "LNG 대상달÷최근28일", "직도입 대상달÷최근28일", "석탄 대상달÷최근28일", "LNG 대상달÷이번달", "28일 평균÷LNG"]
FUELS = ["LNG", "LNG(직도입)", "석탄"]
COLS = ["base_date", "date", "day_type", "offday", "pred_mean", "lo_mean", "hi_mean"]


def fuel_table(fuel):
    """월(YYYY-MM) × 연료 3개 값, 그 달 값을 쓸 수 있는 날(usable_from)."""
    if fuel == "gas":
        g = load_gas_tariff()
        ym = g["month"].dt.strftime("%Y-%m").to_numpy()
        kwh = g["won_kwh"].to_numpy()
        return pd.DataFrame({"LNG": kwh, "LNG(직도입)": kwh, "석탄": 1.0}, index=ym), pd.Series(g["usable_from"].to_numpy(), index=ym)
    # 변동비 — 팀원 _load_vc 그대로 (빠진 2023-03은 앞뒤 달 '변동비 ÷ 열량단가' 평균 × 그달 열량단가)
    v = pd.read_csv(RAW_DIR / "변동비_연료별_월별요약_202203-202611.csv", encoding="utf-8-sig").pivot_table(
        index="연월", columns="연료구분", values="변동비_중앙값")[FUELS]
    v = v.reindex(pd.period_range(v.index.min(), v.index.max(), freq="M").astype(str))
    heat = pd.read_csv(RAW_DIR / "연료원별_열량단가_월별_202101-202608.csv", encoding="utf-8-sig").set_index("연월")
    heat_of = {"LNG": "LNG", "LNG(직도입)": "LNG", "석탄": "유연탄"}
    for ym in v.index[v.isna().all(axis=1)]:
        i = v.index.get_loc(ym)
        if ym not in heat.index or i == 0 or i + 1 >= len(v):
            continue
        prv, nxt = v.index[i - 1], v.index[i + 1]
        if v.loc[[prv, nxt]].isna().any().any() or prv not in heat.index or nxt not in heat.index:
            continue
        for f in FUELS:
            v.at[ym, f] = np.mean([v.at[m, f] / heat.at[m, heat_of[f]] for m in (prv, nxt)]) * heat.at[ym, heat_of[f]]
    v = v.ffill()
    return v, pd.Series(pd.to_datetime(v.index) - pd.Timedelta(days=1), index=v.index)


class Data:
    """일별 배열(팀원 horizon_forecast와 같은 꼴) + 모든 (판단일, 1~31일 뒤) 피처 표."""

    def __init__(self, fuel="gas"):
        self.fuel = fuel
        s = load_smp_actual("육지")
        w = s.pivot_table(index="date", columns="hour", values="smp").reindex(columns=range(1, 25))
        cal = build_calendar(str(w.index.min().date()), str((w.index.max() + pd.Timedelta(days=HMAX + 10)).date()))
        P = w.reindex(cal["date"]).to_numpy(float)
        self.full = ~np.isnan(P).any(1)
        self.DM = np.where(self.full, P.mean(1), np.nan)
        self.cal = cal
        self.d = cal["date"].dt.strftime("%Y-%m-%d").to_numpy()
        self.ym = cal["date"].dt.strftime("%Y-%m").to_numpy()
        self.idx = {d: i for i, d in enumerate(self.d)}
        self.OFF, self.HOL = cal["is_offday"].astype(int).to_numpy(), cal["is_holiday"].astype(int).to_numpy()
        self.DOW, self.MONTH = cal["weekday"].to_numpy(), cal["date"].dt.month.to_numpy()
        self.LAST = int(np.where(self.full)[0].max())
        self.T, self.U = fuel_table(fuel)
        self.VCD = np.stack([self.T[f].reindex(self.ym).ffill().bfill().to_numpy(float) for f in FUELS], 1)
        self.table = self._table()

    def vc_target(self, ym):
        """대상 달 연료값 = 전달 말일까지 쓸 수 있던 값 중 그 달 이하 가장 최근 달."""
        cut = pd.Timestamp(ym + "-01") - pd.Timedelta(days=1)
        ok = [m for m in self.T.index if m <= ym and self.U[m] <= cut]
        return self.T.loc[ok[-1]].to_numpy(float)

    def _nm(self, xs):
        xs = [v for v in xs if not np.isnan(v)]
        return float(np.mean(xs)) if xs else np.nan

    def level(self, t, n):
        return self._nm(self.DM[max(0, t - n + 1): t + 1])

    def _table(self):
        DM, full, DOW = self.DM, self.full, self.DOW
        vt_of = {m: self.vc_target(m) for m in np.unique(self.ym[self.idx[TRAIN_START]:])}
        rows = []
        for t in range(self.idx[TRAIN_START], self.LAST + 1):
            if not full[t]:
                continue
            l7, l28, l91 = self.level(t, 7), self.level(t, 28), self.level(t, 91)
            win = [j for j in range(t - 27, t + 1) if full[j]]
            off_f = self._nm([DM[j] for j in win if self.OFF[j]]) / l28
            wk_f = self._nm([DM[j] for j in win if not self.OFF[j]]) / l28
            s4 = {}
            for dw in range(7):
                same = [j for j in range(t, max(-1, t - 35), -1) if DOW[j] == dw and full[j]][:4]
                s4[dw] = self._nm([DM[j] for j in same]) / l28
            vw, vnow = np.nanmean(self.VCD[t - 27: t + 1], 0), self.VCD[t]
            vol = np.nanstd(DM[t - 27: t + 1]) / l28
            for h in range(1, HMAX + 1):
                i = t + h
                vt = vt_of[self.ym[i]]
                typ = off_f if self.OFF[i] else wk_f
                rows.append([t, i, h, DM[i] if full[i] else np.nan, l28, vol, float(np.log(vt[0] / vnow[0])),
                             h, DOW[i], self.OFF[i], self.HOL[i], self.MONTH[i], l7 / l28, l91 / l28, typ, DM[t] / l28, s4[DOW[i]],
                             *np.log(vt / vw), np.log(vt[0] / vnow[0]), l28 / vt[0]])
        return pd.DataFrame(rows, columns=["t", "i", "h", "y", "l28", "vol", "lng_chg", *FEAT])

    def month_end(self, day):
        """day 이하 가장 가까운 월말 판단일 (day가 월말이면 그날)."""
        b = pd.Timestamp(day)
        return b if (b + pd.Timedelta(days=1)).day == 1 else b.replace(day=1) - pd.Timedelta(days=1)

    def predict(self, t, train_end):
        """판단일 t(인덱스) → 다음 달 모든 날 점예측 (대상일 < train_end 행으로 학습)."""
        tb = self.table
        tr = tb[(tb["i"] < self.idx[train_end]) & tb["y"].notna()]
        r = (tr["y"] / tr["l28"]).to_numpy()
        m = ~np.isnan(r)
        model = lgb.train(LGB_PARAMS, lgb.Dataset(tr[FEAT].to_numpy(float)[m], r[m]), ROUNDS)
        ev = tb[(tb["t"] == t) & (self.ym[tb["i"].to_numpy()] == self.ym[t + 1])].copy()
        ev["p"] = model.predict(ev[FEAT].to_numpy(float)) * ev["l28"]
        i = ev["i"].to_numpy()
        ev["base_date"], ev["date"] = pd.Timestamp(self.d[t]), pd.to_datetime(self.d[i])
        ev["day_type"], ev["offday"] = self.cal["day_type"].to_numpy()[i], self.OFF[i].astype(bool)
        return ev[["base_date", "date", "h", "day_type", "offday", "p", "y", "vol", "lng_chg"]].reset_index(drop=True)


def band_scale(df):
    return df["vol"].to_numpy() / 0.08 + np.abs(df["lng_chg"].to_numpy()) / 0.05


def band(calib, ev):
    """B2: calib(표본 밖 [h, y, p, vol, lng_chg])의 상대 오차 ÷ 배율 → 주차별 10·90% → ev에 적용."""
    c = calib[calib["y"].notna()]
    rel = (c["y"] - c["p"]) / c["p"] / band_scale(c)
    wc, we = np.minimum((c["h"] - 1) // 7, 3), np.minimum((ev["h"] - 1) // 7, 3)
    q = {w: (np.nanquantile(rel[wc == w], 0.1), np.nanquantile(rel[wc == w], 0.9)) for w in range(4)}
    s = band_scale(ev)
    lo = ev["p"] * (1 + np.array([q[w][0] for w in we]) * s)
    hi = ev["p"] * (1 + np.array([q[w][1] for w in we]) * s)
    return lo.to_numpy(), hi.to_numpy()


def calib_of(hist, target_year):
    """운영 범위 보정 표본: 2024-01-01 ~ 대상 해 전년 말 대상일."""
    return hist[(hist["date"] >= CALIB_START) & (hist["date"] < f"{target_year}-01-01")]


def origins(D, first_target, last_target):
    return [D.idx[(f - pd.Timedelta(days=1)).strftime("%Y-%m-%d")] for f in pd.date_range(first_target, last_target, freq="MS")]


def run_month(D, first_target="2024-01-01", last_target="2026-10-01"):
    """운영 방식 백테스트: 월말마다 그날까지 학습 → 다음 달 예측, 범위는 calib_of."""
    out = [D.predict(t, D.d[t + 1]) for t in origins(D, first_target, last_target)]
    h = pd.concat(out, ignore_index=True)
    h["lo"], h["hi"] = np.nan, np.nan
    for y, g in h.groupby(h["date"].dt.year):
        c = calib_of(h, y)
        if len(c):
            h.loc[g.index, "lo"], h.loc[g.index, "hi"] = band(c, g)
    return h


def run_year(D):
    """선택 규칙 재현(팀원 FOLDS): 그해 1/1까지 학습한 모델 하나로 그해 월말 판단 전부."""
    f = {}
    for y in (2024, 2025, 2026):
        last = "2026-10-01" if y == 2026 else f"{y}-12-01"
        f[y] = pd.concat([D.predict(t, f"{y}-01-01") for t in origins(D, f"{y}-01-01", last)], ignore_index=True)
    for y in (2025, 2026):
        f[y]["lo"], f[y]["hi"] = band(pd.concat([f[k] for k in f if k < y]), f[y])
    return pd.concat([f[2025], f[2026]], ignore_index=True)


def score(df):
    m = df["y"].notna()
    y, p = df.loc[m, "y"].to_numpy(), df.loc[m, "p"].to_numpy()
    out = {"대상일 수": int(m.sum()), "MAE": np.abs(y - p).mean(), "±10% 적중률": (np.abs(p - y) <= 0.1 * y).mean()}
    if "lo" in df and df.loc[m, "lo"].notna().all():
        lo, hi = df.loc[m, "lo"].to_numpy(), df.loc[m, "hi"].to_numpy()
        out.update({"포함률": ((y >= lo) & (y <= hi)).mean(), "평균 폭": (hi - lo).mean(),
                    "구간점수": ((hi - lo) + 2 / ALPHA * ((lo - y) * (y < lo) + (y - hi) * (y > hi))).mean()})
    return out


PERIODS = {"2025 (검증)": ("2025-01-01", "2025-12-31"), "2026 ~10/18 (참고, 연료 자료는 2026을 본 뒤 추가)": ("2026-01-01", "2026-10-18")}


def main():
    t0 = time.time()
    MM.mkdir(parents=True, exist_ok=True)
    rows, live = [], {}
    for fuel, fname in (("gas", "가스요금"), ("vc", "변동비")):
        D = Data(fuel)
        print(f"{fname} 피처 표 {len(D.table)}행 ({time.time()-t0:.0f}s)", flush=True)
        res = {"연 1회 학습 (선택 규칙)": run_year(D), "월말마다 재학습 (운영)": run_month(D)}
        print(f"{fname} 백테스트 끝 ({time.time()-t0:.0f}s)", flush=True)
        for how, df in res.items():
            for pn, (a, b) in PERIODS.items():
                rows.append({"연료": fname, "학습 방식": how, "기간": pn, **score(df[(df["date"] >= a) & (df["date"] <= b)])})
        if fuel == "gas":
            h = res["월말마다 재학습 (운영)"]
            h.to_csv(MM / "MM_backtest_full.csv", index=False, encoding="utf-8-sig")
            out = h.rename(columns={"p": "pred_mean", "lo": "lo_mean", "hi": "hi_mean", "y": "y_mean"})
            out = out[COLS + ["y_mean"]].round(2)
            out[out["base_date"] >= "2024-12-31"].to_csv(MM / "MM_predictions_daily.csv", index=False, encoding="utf-8-sig")
            out[out["base_date"] < "2024-12-31"].to_csv(MM / "MM_predictions_calib2024.csv", index=False, encoding="utf-8-sig")
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "MM_score.csv", index=False, encoding="utf-8-sig")
    choice = {
        "모델": "LightGBM 단독 (팀원 최종 코드 monthly_final.py의 lgbm_vc, 연료 = 가스공사 발전용 요금)",
        "판단": "매월 마지막 날 15시 → 다음 달 1일~말일 일평균 SMP",
        "학습": {"시작": TRAIN_START, "행": "매일을 판단일로 1~31일 뒤(대상일 < 판단일 다음 날)", "목표": "대상일 일평균 ÷ 판단일 최근 28일 평균",
                 "LightGBM": {**LGB_PARAMS, "rounds": ROUNDS}, "재학습": "월말 판단마다 (약 10초)"},
        "피처": FEAT,
        "연료": {"가스": f"가스공사 일반발전 요금(원/GJ) × {MJ_PER_KWH} ÷ 1000 → LNG·LNG(직도입) 두 자리, 석탄 = 1.0",
                 "대상 달 값": "전달 말일까지 usable_from이 지난 값 중 가장 최근 달 (common.loaders.load_gas_tariff)"},
        "범위": {"방식": "B2 = 변동성 + LNG 변화 연동 (팀 규칙: 2025 포함률 75~85% 중 구간점수 최소, 팀원 선택)",
                 "배율": "28일 변동성 ÷ 0.08 + |log(LNG 대상 달 ÷ 이번 달)| ÷ 0.05", "분위수": "지평 주차(1~7·8~14·15~21·22일~)별 10·90%",
                 "보정 표본": f"운영식 백테스트 오차, 대상일 {CALIB_START} ~ 대상 해 전년 말 (해마다 1월에 늘어남)"},
        "선택 근거": "2025 검증 MAE: LightGBM+변동비 7.65 · LightGBM+가스 7.67 · 평균(LightGBM+Chronos)+변동비 7.71 (팀원 코드 재현, MONTH.md 0-4절)",
    }
    (TAB / "MM_choice.json").write_text(json.dumps(choice, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print(f"완료 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
