"""Open-Meteo Previous Runs API — 과거 시점에 실제로 나왔던 예보(일사·운량) 수집 (키 불필요, 비상업 무료).

`*_previous_day2` = 대상 시각보다 2일 전에 실행된 예보값 → D일 값은 D-2 실행분이라 D-1 15시에 확실히 확보됨.
(`previous_day1`은 D-1 00UTC=09시 실행분이 섞여 공개 시각 확인이 필요하므로 쓰지 않음)
지점: 태양광이 많은 7개 도시(기상청 구역예보와 동일).
모델: JMA(2023 이전부터), ECMWF IFS 0.25(약 2024-03부터), GFS(약 2024-02부터)
실행: python -m common.collect.openmeteo_prevruns [시작 YYYY-MM-DD] [끝 YYYY-MM-DD] [jeju]
저장: data/raw/api/openmeteo/prev2_{model}_{시작}_{끝}.json (원문)
제주(세 번째 인자 jeju, 3~7일 전 실행분은 jeju_ext → prevx_*.json): 제주 4지점 + 풍속(10m·100m) → data/raw/api/openmeteo_jeju/ (별도 폴더 필수:
  load_openmeteo_prev2()가 openmeteo/ 폴더의 모든 지점을 평균하므로 섞이면 육지 피처가 오염된다)
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
# 제주: ASOS 4지점 위치 (제주시·고산(서부 풍력)·성산(동부 풍력)·서귀포)
JEJU_CITIES = {"제주": (33.51, 126.53), "고산": (33.29, 126.16), "성산": (33.39, 126.88), "서귀포": (33.25, 126.57)}
JEJU_VARS = VARS + ",wind_speed_10m_previous_day2,wind_speed_100m_previous_day2"
JEJU_OUT_DIR = API_RAW_DIR / "openmeteo_jeju"
# 제주 일주일 예측용: k일 전 실행분(k=3~7) → H=k−1일 뒤 예측에 사용 (파일 접두 prevx_)
JEJU_EXT_VARS = ",".join(f"{v}_previous_day{k}" for k in range(3, 8)
                         for v in ("shortwave_radiation", "cloud_cover", "wind_speed_10m", "wind_speed_100m"))


def main(start: str = "2024-01-01", end: str | None = None, region: str = "육지"):
    end = end or date.today().isoformat()
    jeju = region in ("jeju", "jeju_ext")
    cities, hourly, out_dir = (JEJU_CITIES, JEJU_VARS, JEJU_OUT_DIR) if jeju else (CITIES, VARS, OUT_DIR)
    prefix = "prev2"
    if region == "jeju_ext":
        hourly, prefix = JEJU_EXT_VARS, "prevx"
    out_dir.mkdir(parents=True, exist_ok=True)
    lat = ",".join(str(v[0]) for v in cities.values())
    lon = ",".join(str(v[1]) for v in cities.values())
    for model in MODELS:
        out = out_dir / f"{prefix}_{model}_{start}_{end}.json"
        if out.exists():
            continue
        r = requests.get(URL, params={"latitude": lat, "longitude": lon, "hourly": hourly, "models": model,
                                      "start_date": start, "end_date": end, "timezone": "Asia/Seoul"}, timeout=300)
        r.raise_for_status()
        out.write_text(r.text, encoding="utf-8")
        data = json.loads(r.text)
        first = data[0] if isinstance(data, list) else data
        vals = first["hourly"][hourly.split(",")[0]]
        n_ok = sum(v is not None for v in vals)
        print(f"{model}: {len(vals)}시간 중 값 있음 {n_ok}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main(*sys.argv[1:4])
