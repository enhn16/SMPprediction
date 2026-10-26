"""기상청 단기예보(격자) 발표분 매일 축적 — 11시·14시 발표, 육지·제주 예보 지점.

공통 규칙의 판단 시각(D-1 15시)에 맞는 날씨는 14시 발표분이다. 그런데 API는 최근 3일분만 주므로
매일 받아 쌓지 않으면 사라진다 (2026-10-26 확인: "최근 3일 간의 자료만 제공합니다").
과거 이력이 없어 지금 모델은 11시 구역예보·Open-Meteo를 쓰며, 이 자료는 새 시험 구간에서 비교할 때 쓴다.
(17시 발표분 과거 수집기는 short_forecast.py — 판단 시각 뒤라 피처로 못 씀)

실행: python -m common.collect.kma_vilage_fcst [일수=3]   (오늘 포함 최근 N일의 11·14시 발표분 중 없는 것만, 발표 전 시각은 건너뜀)
저장: data/raw/api/vilage_fcst/vf_{YYYYMMDD}_{HHMM}.csv  (열: stn_id, baseDate, baseTime, category, fcstDate, fcstTime, fcstValue, nx, ny)
"""
import sys
from datetime import datetime, timedelta

import pandas as pd

from common.api import QuotaExceeded, fetch_items
from common.config import API_RAW_DIR
from common.stations import FORECAST_STATIONS, STATIONS, latlon_to_grid

URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
OUT_DIR = API_RAW_DIR / "vilage_fcst"
BASE_TIMES = ["1100", "1400"]
READY_MIN = 15   # 발표 후 API 제공까지 여유 (분)


def fetch(base_date: str, base_time: str, stn: int) -> pd.DataFrame:
    _, _, lat, lon = STATIONS[stn]
    nx, ny = latlon_to_grid(lat, lon)
    items, _ = fetch_items(URL, {"dataType": "JSON", "pageNo": 1, "numOfRows": 1500, "base_date": base_date,
                                 "base_time": base_time, "nx": nx, "ny": ny}, timeout=120)
    return pd.DataFrame(items).assign(stn_id=stn)


def main(days: str = "3"):
    now = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stns = sum(FORECAST_STATIONS.values(), [])
    for k in range(int(days) - 1, -1, -1):
        d = (now - timedelta(days=k)).strftime("%Y%m%d")
        for bt in BASE_TIMES:
            out = OUT_DIR / f"vf_{d}_{bt}.csv"
            ready = datetime.strptime(d + bt, "%Y%m%d%H%M") + timedelta(minutes=READY_MIN)
            if out.exists() or now < ready:
                continue
            try:
                df = pd.concat([fetch(d, bt, s) for s in stns], ignore_index=True)
            except QuotaExceeded:
                print("일일 호출 한도 초과 → 중단", flush=True)
                return
            except Exception as e:  # noqa: BLE001  하나가 실패해도 다음 발표분은 시도
                print(f"{d} {bt} 실패: {e}", flush=True)
                continue
            cols = ["stn_id", "baseDate", "baseTime", "category", "fcstDate", "fcstTime", "fcstValue", "nx", "ny"]
            df.reindex(columns=cols).to_csv(out, index=False, encoding="utf-8")
            print(f"{d} {bt}: {len(stns)}지점 {len(df)}행", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
