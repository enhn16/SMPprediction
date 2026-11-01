"""추가 자료 실험 (팀 13종 밖, 기간 2023부터): team_lstm.py 선택 구성(T2 회귀)에 자료를 더해 2025로만 판단.

결정: PLAN.md 결정 기록 2026-10-28 '추가 자료 실험 사전 고정'.
후보: E1 Open-Meteo JMA 2일 전 실행 한낮 일사·운량 / E2 원전 가용 용량·가용률 / E3 = E1+E2. 두 모델 모두에 더한다.
채택: 2023~2024·2024 학습 → 2025 전체 날 헛경보@재현율80% 나쁜 쪽이 기준보다 3일 이상 적은 후보 중 가장 적은 것
      (같으면 PR-AUC 나쁜 쪽 높은 쪽 → 자료 적은 쪽). 없으면 미채택, 2026 열지 않음.
실행: python -m drop_classifier.src.team_lstm_extra select → 23_extra_select_2025_육지.csv, 23_extra_pick.json (2026 안 봄)
      python -m drop_classifier.src.team_lstm_extra final  → 채택 후보만 22_extra_team_lstm_{score,daily}_육지.csv
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from common.config import API_RAW_DIR
from common.loaders import load_nuclear_available
from drop_classifier.src.final import threshold_for
from drop_classifier.src.team_lstm import (OUT_TAB, REGION, TR_SEL, VALID, combine, data, final, lgbm, lstm, split)
from smp_forecast_app.src.team_scorecard import drop_stats

warnings.filterwarnings("ignore")
OM = ["om_rad_mid", "om_cloud_mid"]
NUC = ["nuc_avail_gw", "nuc_avail_ratio"]
VARIANTS = {"기준": [], "E1 +Open-Meteo JMA": OM, "E2 +원전 가용": NUC, "E3 +Open-Meteo JMA+원전": OM + NUC}
MIN_GAIN = 3


def om_jma() -> pd.DataFrame:
    """JMA 2일 전 실행(previous_day2) 예보 → [date, om_rad_mid, om_cloud_mid]: 7개 도시 평균, 11~14시 시작(11~15시) 일사 합·운량 평균.
    common.loaders.load_openmeteo_prev2와 같은 집계에 2023 백필 폴더를 더함 (시간대 Asia/Seoul 확인)."""
    rows = []
    for folder in ("openmeteo_2023_backfill", "openmeteo"):
        for p in sorted((API_RAW_DIR / folder).glob("prev2_jma_*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            for loc in (d if isinstance(d, list) else [d]):
                h = pd.DataFrame(loc["hourly"])
                h["time"] = pd.to_datetime(h["time"])
                h = h[h["time"].dt.hour.between(11, 14)]
                g = h.groupby(h["time"].dt.normalize()).agg(rad=("shortwave_radiation_previous_day2", "sum"),
                                                           cloud=("cloud_cover_previous_day2", "mean"),
                                                           n=("shortwave_radiation_previous_day2", "count"))
                g.loc[g["n"] < 4, ["rad", "cloud"]] = np.nan
                rows.append(g.assign(lat=loc["latitude"], lon=loc["longitude"], src=p.name))
    d = pd.concat(rows).rename_axis("date").reset_index()
    d = d.sort_values("src").groupby(["date", "lat", "lon"]).last().reset_index()   # 겹치는 파일은 뒤(최근 수집) 우선
    return d.groupby("date").agg(om_rad_mid=("rad", "mean"), om_cloud_mid=("cloud", "mean")).reset_index()


def add_extra(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(om_jma(), on="date", how="left")
    n = load_nuclear_available()
    n["nuc_avail_gw"], n["nuc_avail_ratio"] = n["nuc_avail_mw"] / 1000, n["nuc_avail_mw"] / n["nuc_total_mw"]
    return df.merge(n[["date"] + NUC], on="date", how="left")


def cfg():
    return json.loads((OUT_TAB / "22_team_lstm_pick.json").read_text(encoding="utf-8"))["선택"]


def select():
    t0 = time.time()
    c = cfg()
    df = add_extra(data())
    df = df[df["date"] <= VALID[1]].reset_index(drop=True)              # 2026 행 제거 (열지 않음)
    va = split(df, VALID)
    y, off = va["label"].values, va["offday"].values
    print("2023~2025 빈칸:", {k: int(df[k].isna().sum()) for k in OM + NUC}, flush=True)
    rows = []
    for name, ex in VARIANTS.items():
        r = {"후보": name, "추가 열": ", ".join(ex), "자료 수": len(ex)}
        for w, rng in TR_SEL.items():
            tr = split(df, rng)
            s = combine(lstm(tr, va, c["target"], c["seq"], extra=ex), lgbm(tr, va, c["target"], c["x"], extra=ex),
                        c["target"], c["comb"])
            a = drop_stats(y, s)
            r[f"헛경보@80% {w}학습"], r[f"헛경보@90% {w}학습"] = a["헛경보@재현율80%"], a["헛경보@재현율90%"]
            r[f"PR-AUC {w}학습"] = a["PR-AUC"]
            r[f"평일 헛경보@80% {w}학습"] = int(((s >= threshold_for(y, s, .8)) & ~off & (y == 0)).sum())
        r["헛경보@80% 나쁜쪽"] = max(r[f"헛경보@80% {w}학습"] for w in TR_SEL)
        r["PR-AUC 나쁜쪽"] = min(r[f"PR-AUC {w}학습"] for w in TR_SEL)
        rows.append(r)
        print(f"  {name}: 헛경보@80% 나쁜쪽 {r['헛경보@80% 나쁜쪽']}, PR-AUC 나쁜쪽 {r['PR-AUC 나쁜쪽']:.3f} "
              f"({time.time() - t0:.0f}초)", flush=True)
    res = pd.DataFrame(rows)
    base = res.loc[res["후보"] == "기준", "헛경보@80% 나쁜쪽"].iloc[0]
    ok = res[(res["후보"] != "기준") & (res["헛경보@80% 나쁜쪽"] <= base - MIN_GAIN)]
    pick = None if ok.empty else ok.sort_values(["헛경보@80% 나쁜쪽", "PR-AUC 나쁜쪽", "자료 수"],
                                                ascending=[True, False, True]).iloc[0]["후보"]
    res["채택"] = np.where(res["후보"] == pick, "채택", "")
    res.round(4).to_csv(OUT_TAB / f"23_extra_select_2025_{REGION}.csv", index=False, encoding="utf-8-sig")
    (OUT_TAB / "23_extra_pick.json").write_text(json.dumps(
        {"채택": pick, "추가 열": VARIANTS[pick] if pick else [], "기준 헛경보@80% 나쁜쪽": int(base),
         "규칙": f"기준보다 {MIN_GAIN}일 이상 적을 때만 채택 (PLAN.md 2026-10-28)"}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print(f"채택: {pick or '없음 (추가 자료 미채택, 2026 열지 않음)'}")


def final_extra():
    p = json.loads((OUT_TAB / "23_extra_pick.json").read_text(encoding="utf-8"))
    if not p["채택"]:
        print("채택된 추가 자료 없음 — 2026 채점하지 않음")
        return
    final(extra=p["추가 열"], tag="_extra", df=add_extra(data()))


if __name__ == "__main__":
    if "select" in sys.argv:
        select()
    elif "final" in sys.argv:
        final_extra()
