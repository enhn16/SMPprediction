"""일주일 v3 (R3) — Open-Meteo 3~7일 전 실행 예보(일사·운량) 수집·읽기, 육지 7개 도시.

규칙 (제주 jeju_ext와 같음): k일 전 실행분(previous_dayk, k=3~7) → 기준일 I에서 H = k−1일 뒤 대상일에 사용.
  대상일 D = I+H의 k=H+1일 전 실행 = I−1일 실행분 → 기준일 15시에 확실히 확보 (팀 규칙 '2일 이상 전 실행분'도 충족).
  H=1은 기존 prev2(features.py)가 이미 쓰고, H=7은 k=8이 API에 없어 비워 둔다.
기간: 2024-01-01 ~ (팀 데이터 목록의 Open-Meteo 기간. 2023은 피처 결측 → LightGBM이 결측으로 처리)
저장: data/processed/smp_forecast_app/wz3/openmeteo/prevx_{model}_{시작}_{끝}.json (원문) — common/collect·data/raw는 건드리지 않음
실행: python -m smp_forecast_app.src.wz3_openmeteo [시작] [끝]
"""
import json
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from common.collect.openmeteo_prevruns import CITIES, MODELS, URL
from common.config import PROCESSED_DIR

OUT_DIR = PROCESSED_DIR / "smp_forecast_app" / "wz3" / "openmeteo"
KS = range(3, 8)
VARS = ",".join(f"{v}_previous_day{k}" for k in KS for v in ("shortwave_radiation", "cloud_cover"))
CHUNKS = [("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31"), ("2026-01-01", None)]


def collect(start=None, end=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lat = ",".join(str(v[0]) for v in CITIES.values())
    lon = ",".join(str(v[1]) for v in CITIES.values())
    chunks = [(start, end or date.today().isoformat())] if start else [(a, b or date.today().isoformat()) for a, b in CHUNKS]
    for a, b in chunks:
        for model in MODELS:
            out = OUT_DIR / f"prevx_{model}_{a}_{b}.json"
            if out.exists():
                continue
            for t in range(4):
                try:
                    r = requests.get(URL, params={"latitude": lat, "longitude": lon, "hourly": VARS, "models": model,
                                                  "start_date": a, "end_date": b, "timezone": "Asia/Seoul"}, timeout=600)
                    r.raise_for_status()
                    break
                except Exception as ex:  # noqa: BLE001
                    print(f"  {model} {a} 재시도 {t + 1}: {ex}", flush=True)
                    time.sleep(30 * (t + 1))
            else:
                raise RuntimeError(f"{model} {a}~{b} 수집 실패")
            out.write_text(r.text, encoding="utf-8")
            first = json.loads(r.text)[0]
            vals = first["hourly"]["shortwave_radiation_previous_day3"]
            print(f"{model} {a}~{b}: {len(vals)}시간 중 값 있음 {sum(v is not None for v in vals)}", flush=True)
            time.sleep(5)


def load() -> pd.DataFrame:
    """[date, H, om_rad_mid, om_cloud_mid]: H=2~6, 7개 도시·모델 평균, 11~15시(시작 시각 11~14시) 일사 합·운량 평균.
    같은 날짜가 여러 파일에 있으면 나중 파일 값."""
    rows = []
    for p in sorted(OUT_DIR.glob("prevx_*.json")):
        model = p.name.split("_")[1]
        for loc in json.loads(p.read_text(encoding="utf-8")):
            h = pd.DataFrame(loc["hourly"])
            h["time"] = pd.to_datetime(h["time"])
            h = h[h["time"].dt.hour.between(11, 14)]
            day = h["time"].dt.normalize()
            for k in KS:
                rad, cl = f"shortwave_radiation_previous_day{k}", f"cloud_cover_previous_day{k}"
                g = h.groupby(day).agg(rad=(rad, "sum"), cloud=(cl, "mean"), n=(rad, "count"))
                g.loc[g["n"] < 4, ["rad", "cloud"]] = np.nan
                rows.append(g.assign(model=model, H=k - 1, src=p.name).rename_axis("date").reset_index())
    d = pd.concat(rows).drop_duplicates(["date", "model", "H", "src"])
    d = d.sort_values("src").groupby(["date", "model", "H"]).last().reset_index()   # 같은 날은 나중 파일
    return d.groupby(["date", "H"]).agg(om_rad_mid=("rad", "mean"), om_cloud_mid=("cloud", "mean")).reset_index()


if __name__ == "__main__":
    collect(*sys.argv[1:3])
