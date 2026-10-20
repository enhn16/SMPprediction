"""Open-Meteo Previous Runs API — 과거 시점에 실제로 나왔던 예보(일사·운량) 수집 (키 불필요, 비상업 무료).

`*_previous_day2` = 대상 시각보다 2일 전에 실행된 예보값 → D일 값은 D-2 실행분이라 D-1 15시에 확실히 확보됨.
(`previous_day1`은 D-1 00UTC=09시 실행분이 섞여 공개 시각 확인이 필요하므로 쓰지 않음)
지점: 태양광이 많은 7개 도시(기상청 구역예보와 동일).
모델: JMA(2023 이전부터), ECMWF IFS 0.25(약 2024-03부터), GFS(약 2024-02부터)
실행: python -m common.collect.openmeteo_prevruns [시작 YYYY-MM-DD] [끝 YYYY-MM-DD]
저장: data/raw/api/openmeteo/prev2_{model}_{시작}_{끝}.json (원문)
"""
import json
import sys
import time
from datetime import date

import requests

from common.config import API_RAW_DIR

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
CITIES = {"서산": (36.78, 126.45), "전주": (35.82, 127.15), "광주": (35.16, 126.85), "목포": (34.81, 126.39),
          "안동": (36.57, 128.73), "대구": (35.87, 128.60), "진주": (35.18, 128.11)}
MODELS = ["jma_seamless", "ecmwf_ifs025", "gfs_seamless"]
VARS = "shortwave_radiation_previous_day2,cloud_cover_previous_day2"
OUT_DIR = API_RAW_DIR / "openmeteo"


def main(start: str = "2024-01-01", end: str | None = None):
    end = end or date.today().isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lat = ",".join(str(v[0]) for v in CITIES.values())
    lon = ",".join(str(v[1]) for v in CITIES.values())
    for model in MODELS:
        out = OUT_DIR / f"prev2_{model}_{start}_{end}.json"
        if out.exists():
            continue
        r = requests.get(URL, params={"latitude": lat, "longitude": lon, "hourly": VARS, "models": model,
                                      "start_date": start, "end_date": end, "timezone": "Asia/Seoul"}, timeout=300)
        r.raise_for_status()
        out.write_text(r.text, encoding="utf-8")
        data = json.loads(r.text)
        first = data[0] if isinstance(data, list) else data
        vals = first["hourly"]["shortwave_radiation_previous_day2"]
        n_ok = sum(v is not None for v in vals)
        print(f"{model}: {len(vals)}시간 중 값 있음 {n_ok}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main(*sys.argv[1:3])
