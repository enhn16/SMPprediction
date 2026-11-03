"""제주 한 달 v3 — 육지 한 달 최종(R6: 팀원 LightGBM 단독 + 가스요금, 월말 판단 → 다음 달 모든 날 일평균)에 맞춤 (R2, 2026-11-03).

방법: 팀원 최종 코드(reference/한달예측_최종모델_20261103/smp-web)를 data/ 아래 작업 폴더에 복사해서 돌린다 (원본은 건드리지 않음, R6와 같은 방식).
  연료 입력 = 가스공사 일반발전 요금(원/kWh = 원/GJ × 7.2 ÷ 1000)을 LNG·LNG(직도입) 두 열에, 석탄 = 1.0.
  월말 판단에 그달 게시가 안 된 달(usable_from > 판단일)은 전달 값. 시험 끝 = 2026-10-18 (팀 공식 기간)
  육지판 = 원래 days.json (smp = 육지), 제주판 = days.json의 smp를 smpJ(제주 SMP)로 바꾼 것
후보 (단순한 순, 결과 보기 전 고정 — JEJU.md 11절)
  M1 육지 LightGBM+가스 그대로       M2 M1 + 판단일 직전 28일 (제주 − 육지) 일평균 차
  M3 제주 단독 LightGBM+가스(제주판)  M4 (M2 + M3) / 2
선택: 2025 검증(2024년까지 학습) 제주 일평균 MAE 최소, 0.05원 이내면 단순한 쪽
범위: 팀원 band() 그대로 (B0 고정 · B1 변동성 · B2 변동성+연료 변화, 변동성·연료 변화는 제주판 값), 2025 포함률 75~85% 중 구간점수 최소
최종: 2026-01-01 ~ 10-18 한 번 채점, 지금 제주 한 달(JMZ_predictions_daily.csv, 육지 v2 그대로)과 같은 (판단일, 대상일)끼리
출력: outputs/tables/JMZ3_select_2025.csv, JMZ3_test2026.csv, JMZ3_choice.json, data/processed/smp_forecast_app/jeju/JMZ3_predictions_daily.csv
실행: python -m smp_forecast_app.src.jz_month2 [--rerun]
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from common.config import ROOT
from common.loaders import load_gas_tariff
from smp_forecast_app.src.jz_common import JDIR

REF = ROOT / "reference" / "한달예측_최종모델_20261103" / "smp-web"
WORK = JDIR / "jmz3"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
VC_REL = "data-raw/data/common/fuel_20261030/변동비_연료별_월별요약_202203-202611.csv"
TIE = 0.05
TEST_END = "2026-10-18"


def gas_vc() -> pd.DataFrame:
    """월별 가스요금을 팀원 변동비 파일 형식(연월 · 연료구분 · 변동비_중앙값)으로. 월말에 아직 게시 안 된 달은 전달 값."""
    g = load_gas_tariff().sort_values("month").reset_index(drop=True)
    val = g["won_kwh"].copy()
    late = g["usable_from"] > g["month"] - pd.Timedelta(days=1)
    val[late] = val.shift(1)[late]
    rows = []
    for m, v in zip(g["month"].dt.strftime("%Y-%m"), val):
        rows += [{"연월": m, "연료구분": "LNG", "변동비_중앙값": v}, {"연월": m, "연료구분": "LNG(직도입)", "변동비_중앙값": v},
                 {"연월": m, "연료구분": "석탄", "변동비_중앙값": 1.0}]
    return pd.DataFrame(rows), g.loc[late, "month"].dt.strftime("%Y-%m").tolist()


def prepare(region):
    w = WORK / region
    if w.exists():
        shutil.rmtree(w)
    (w / "ml").mkdir(parents=True)
    for f in ("monthly_final.py", "horizon_forecast.py"):
        shutil.copy2(REF / "ml" / f, w / "ml" / f)
    src = (w / "ml" / "monthly_final.py").read_text(encoding="utf-8")
    src = src.replace('FINAL = "avg_lc"', "FINAL = None").replace('TEST_END = "2026-09-27"', f'TEST_END = "{TEST_END}"')
    (w / "ml" / "monthly_final.py").write_text(src, encoding="utf-8")
    (w / "public" / "data").mkdir(parents=True)
    days = json.loads((REF / "public" / "data" / "days.json").read_text(encoding="utf-8"))
    if region == "제주":
        for x in days["days"]:
            x["smp"] = x.get("smpJ")
    (w / "public" / "data" / "days.json").write_text(json.dumps(days, ensure_ascii=False), encoding="utf-8")
    vc, late = gas_vc()
    (w / VC_REL).parent.mkdir(parents=True)
    vc.to_csv(w / VC_REL, index=False, encoding="utf-8-sig")
    (w / "data-raw" / "horizon").mkdir(parents=True)
    (w / "reports").mkdir()
    return w, late


def run(region):
    w, late = prepare(region)
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    print(f"[{region}] 팀원 코드 실행 (전달 값 쓴 달 {late}) …", flush=True)
    r = subprocess.run([sys.executable, "ml/monthly_final.py"], cwd=w, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"{region} 실행 실패")
    print("\n".join(l for l in r.stdout.splitlines() if "MAE" in l or "LightGBM + 대상 달" in l or "끝" in l)[:2000], flush=True)


def load(region):
    return pd.read_csv(WORK / region / "data-raw" / "horizon" / "month_final.csv")


def wk(h):
    return np.minimum((np.asarray(h) - 1) // 7, 3)


def band_scale(df, kind):
    if kind == "B0":
        return np.ones(len(df))
    s = df["vol"].to_numpy() / 0.08
    return s if kind == "B1" else s + np.abs(df["lng_chg"].to_numpy()) / 0.05


def band(cal, ev, col, kind):
    """팀원 band()와 같음: 앞선 해 상대 오차 ÷ 배율의 주차별 10·90% 분위수."""
    rel = (cal["y"] - cal[col]) / cal[col] / band_scale(cal, kind)
    q = {w: (np.nanquantile(rel[wk(cal["h"]) == w], 0.1), np.nanquantile(rel[wk(cal["h"]) == w], 0.9)) for w in range(4)}
    s, w = band_scale(ev, kind), wk(ev["h"])
    return (ev[col] * (1 + np.array([q[x][0] for x in w]) * s)).to_numpy(), (ev[col] * (1 + np.array([q[x][1] for x in w]) * s)).to_numpy()


def stats(y, p, lo=None, hi=None):
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    out = {"일수": int(m.sum()), "MAE": float(np.abs(y - p).mean()), "편향(실제−예측)": float((y - p).mean()),
           "±10% 적중": float((np.abs(p - y) <= 0.1 * y).mean())}
    if lo is not None:
        l, h = lo[m], hi[m]
        b, a = y < l, y > h
        out |= {"포함률": float(1 - b.mean() - a.mean()), "하한 아래": float(b.mean()), "상한 위": float(a.mean()),
                "평균 폭": float((h - l).mean()), "구간점수": float(((h - l) + 10 * ((l - y) * b + (y - h) * a)).mean())}
    return out


def table():
    L, J = load("육지"), load("제주")
    key = ["part", "t", "d", "h"]
    d = J[key + ["y", "vol", "lng_chg", "lgbm_vc"]].rename(columns={"lgbm_vc": "M3"}).merge(
        L[key + ["lgbm_vc", "y"]].rename(columns={"lgbm_vc": "M1", "y": "y_land"}), on=key, how="left")
    # 판단일 직전 28일 (제주 − 육지) 일평균 차 — days.json 기준 (판단일 t까지)
    days = json.loads((REF / "public" / "data" / "days.json").read_text(encoding="utf-8"))["days"]
    dm = lambda k: np.array([np.mean(x[k]) if x.get(k) and None not in x[k] else np.nan for x in days])
    gap = pd.Series(dm("smpJ") - dm("smp")).rolling(28, min_periods=20).mean().to_numpy()
    d["gap28"] = gap[d["t"].to_numpy()]
    d["M2"] = d["M1"] + d["gap28"]
    d["M4"] = (d["M2"] + d["M3"]) / 2
    d["date"] = pd.to_datetime(d["d"])
    d["base_date"] = pd.to_datetime([days[t]["d"] for t in d["t"]])
    return d


NAMES = {"M1": "M1 육지 LightGBM+가스 그대로", "M2": "M2 육지 + 최근 28일 제주−육지 차", "M3": "M3 제주 단독 LightGBM+가스",
         "M4": "M4 평균(M2, M3)"}


def main():
    if "--rerun" in sys.argv or not (WORK / "제주" / "data-raw" / "horizon" / "month_final.csv").exists():
        for r in ("육지", "제주"):
            run(r)
    d = table()
    v, te, y24 = d[d["part"] == "val"], d[(d["part"] == "test") & (d["d"] <= TEST_END)], d[d["part"] == "2024"]
    # 육지판 재현 확인 (R6: 2025 7.67, 2026 ~10/18 10.64)
    print(f"육지판 육지 MAE: 2025 {np.nanmean(np.abs(v['y_land'] - v['M1'])):.2f}, 2026 {np.nanmean(np.abs(te['y_land'] - te['M1'])):.2f}")
    rows = []
    for i, c in enumerate(NAMES):
        rows.append({"후보": NAMES[c], "단순 순서": i, **{f"2025 {k}": x for k, x in stats(v["y"].to_numpy(), v[c].to_numpy()).items()},
                     "2024 MAE (참고, 2023만 학습)": stats(y24["y"].to_numpy(), y24[c].to_numpy())["MAE"]})
    t = pd.DataFrame(rows)
    best = t["2025 MAE"].min()
    pi = t[t["2025 MAE"] <= best + TIE].sort_values("단순 순서").iloc[0]
    col = list(NAMES)[int(pi["단순 순서"])]
    t["선택"] = np.where(t["후보"] == pi["후보"], "✓", "")
    br = []
    for kind in ("B0", "B1", "B2"):
        lv, hv = band(y24, v, col, kind)
        br.append({"범위": kind, **stats(v["y"].to_numpy(), v[col].to_numpy(), lv, hv)})
    br = pd.DataFrame(br)
    ok = br[br["포함률"].between(0.75, 0.85)]
    bk = ok.sort_values("구간점수").iloc[0]["범위"] if len(ok) else br.iloc[(br["포함률"] - 0.8).abs().argmin()]["범위"]
    br["선택"] = np.where(br["범위"] == bk, "✓", "")
    t.round(4).to_csv(TAB / "JMZ3_select_2025.csv", index=False, encoding="utf-8-sig")
    br.round(4).to_csv(TAB / "JMZ3_range_2025.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print(br.round(3).to_string(index=False))

    # 2026 한 번 채점
    lt, ht = band(pd.concat([y24, v]), te, col, bk)
    te = te.assign(pred=te[col].to_numpy(), lo=lt, hi=ht)
    cur = pd.read_csv(JDIR / "JMZ_predictions_daily.csv", parse_dates=["base_date", "date"])
    cur = cur[["base_date", "date", "pred_mean", "lo_mean", "hi_mean"]]
    x = te.merge(cur, on=["base_date", "date"], how="left")
    same = x["pred_mean"].notna().to_numpy()
    rows = [{"비교": "새 (전체 대상일)", **stats(x["y"].to_numpy(), x["pred"].to_numpy(), x["lo"].to_numpy(), x["hi"].to_numpy())},
            {"비교": "새 (지금 모델과 같은 대상일)", **stats(x["y"].to_numpy()[same], x["pred"].to_numpy()[same], x["lo"].to_numpy()[same], x["hi"].to_numpy()[same])},
            {"비교": "지금 (육지 v2 그대로, 같은 대상일)", **stats(x["y"].to_numpy()[same], x["pred_mean"].to_numpy()[same], x["lo_mean"].to_numpy()[same], x["hi_mean"].to_numpy()[same])}]
    for c in NAMES:
        if c != col:
            rows.append({"비교": f"{NAMES[c]} (참고, 선택에 안 씀)", **stats(x["y"].to_numpy(), x[c].to_numpy())})
    r = pd.DataFrame(rows)
    r.round(4).to_csv(TAB / "JMZ3_test2026.csv", index=False, encoding="utf-8-sig")
    print(r.round(3).to_string(index=False))

    allp = pd.concat([v.assign(pred=v[col].to_numpy(), lo=band(y24, v, col, bk)[0], hi=band(y24, v, col, bk)[1]), te])
    allp["H"] = (allp["date"] - allp["base_date"]).dt.days
    allp[["base_date", "H", "date", "y", "pred", "lo", "hi"]].round(2).to_csv(JDIR / "JMZ3_predictions_daily.csv", index=False, encoding="utf-8-sig")
    ch = {"모델": NAMES[col], "범위": {"B0": "고정", "B1": "판단 전 28일 변동성 연동", "B2": "변동성 + 연료 변화 연동"}[bk],
          "설정": "월말 판단 → 다음 달 모든 날 일평균 (육지 한 달 최종 R6와 같은 방식)",
          "연료": "가스공사 일반발전 요금 원/kWh를 LNG·직도입 열에, 석탄 1.0. 월말에 미게시 달은 전달 값",
          "선택 규칙": "2025 제주 일평균 MAE 최소, 0.05원 이내면 단순한 쪽 (M1 → M4 순)",
          "범위 규칙": "2025 포함률 75~85% 중 구간점수 최소 (2025용 = 2024 오차, 2026용 = 2024+2025 오차)",
          "학습": "2023년부터 매일을 판단일로 1~31일 뒤 (팀원 코드), 2025 = 2024년까지 학습, 2026 = 2025년까지 학습",
          "2026 시험": f"2026-01-01 ~ {TEST_END} 한 번 (JMZ3_test2026.csv)",
          "작업 폴더": str(WORK.relative_to(ROOT))}
    (TAB / "JMZ3_choice.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
    print("선택:", ch["모델"], "/ 범위", ch["범위"])


if __name__ == "__main__":
    main()
