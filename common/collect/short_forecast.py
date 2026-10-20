"""기상청 단기예보 과거 발표분 수집 (예측 피처용).

대상일 D마다 D-1 17시 발표분을 받아, 예보 시각이 D일인 행만 저장한다
(발표 1회분에는 4일치가 들어 있으나 D일 판단에는 D일 예보만 필요).
지점·월별 파일로 저장하고, 이미 받은 지난 달은 건너뛴다. 일일 호출 한도를 넘으면 멈추며,
다음 날 같은 명령을 다시 실행하면 이어서 받는다.

실행: python -m common.collect.short_forecast [계통: 육지|제주|전체] [시작일] [끝일]  (지점은 common.stations.FORECAST_STATIONS)
      끝일 생략 시 내일(D+1, 오늘 17시 발표분)까지
저장: data/raw/api/short_forecast/fcst_{지점번호}_{YYYYMM}.csv
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

from common.api import QuotaExceeded, fetch_items
from common.config import API_RAW_DIR
from common.stations import FORECAST_STATIONS, STATIONS, latlon_to_grid

URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
OUT_DIR = API_RAW_DIR / "short_forecast"
BASE_TIME = "1700"
WORKERS = 6


def fetch_day(nx: int, ny: int, target: date) -> list[dict]:
    base = target - timedelta(days=1)
    items, _ = fetch_items(URL, {
        "dataType": "JSON", "pageNo": 1, "numOfRows": 1500,
        "base_date": base.strftime("%Y%m%d"), "base_time": BASE_TIME, "nx": nx, "ny": ny,
    }, timeout=120)
    t = target.strftime("%Y%m%d")
    return [i for i in items if i["fcstDate"] == t]


def main(region: str = "육지", start: str = "20240101", end: str | None = None):
    d0 = pd.Timestamp(start).date()
    d1 = pd.Timestamp(end).date() if end else date.today() + timedelta(days=1)
    this_month = date.today().strftime("%Y%m")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = sum(FORECAST_STATIONS.values(), []) if region == "전체" else FORECAST_STATIONS[region]
    stations = {k: STATIONS[k] for k in ids}

    with ThreadPoolExecutor(WORKERS) as pool:
        for stn, (name, _, lat, lon) in stations.items():
            nx, ny = latlon_to_grid(lat, lon)
            for month in pd.period_range(d0, d1, freq="M"):
                ym = month.strftime("%Y%m")
                out = OUT_DIR / f"fcst_{stn}_{ym}.csv"
                if out.exists() and ym < this_month:
                    continue
                days = [d.date() for d in pd.date_range(max(month.start_time.date(), d0), min(month.end_time.date(), d1))]
                try:
                    results = list(pool.map(lambda d: fetch_day(nx, ny, d), days))
                except QuotaExceeded:
                    print(f"일일 호출 한도 초과 → 중단. {name}({stn}) {ym}부터 다음 실행 때 이어서 받음", flush=True)
                    return
                rows = [r for day_rows in results for r in day_rows]
                df = pd.DataFrame(rows)
                df.insert(0, "stn_id", stn)
                df.to_csv(out, index=False, encoding="utf-8-sig")
                empty = [d.isoformat() for d, r in zip(days, results) if not r]
                print(f"{name}({stn}) {ym}: {len(days)}일, {len(df)}행" + (f", 빈 날짜 {empty}" if empty else ""), flush=True)
    print("완료", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
