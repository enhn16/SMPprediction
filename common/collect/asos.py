"""기상청 ASOS 시간자료 수집 (실측: 상한선 성능·요인 분석 전용, 예측 피처로 쓰지 않음).

지점별·연도별 파일로 저장하며, 이미 받은 지난 연도는 건너뛴다.
API는 어제까지만 조회되므로 끝일 기본값은 어제.

실행: python -m common.collect.asos [시작일 YYYYMMDD] [끝일 YYYYMMDD]
저장: data/raw/api/asos/asos_{지점번호}_{연도}.csv
"""
import sys
import time
from datetime import date, timedelta

import pandas as pd

from common.api import fetch_all
from common.config import API_RAW_DIR
from common.stations import STATIONS as _STATIONS

URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
OUT_DIR = API_RAW_DIR / "asos"

STATIONS = {k: v[0] for k, v in _STATIONS.items()}


def main(start: str = "20230101", end: str | None = None):
    d0 = pd.Timestamp(start).date()
    d1 = pd.Timestamp(end).date() if end else date.today() - timedelta(days=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for stn, name in STATIONS.items():
        for year in range(d0.year, d1.year + 1):
            out = OUT_DIR / f"asos_{stn}_{year}.csv"
            if out.exists() and year < date.today().year:
                continue
            rows = []
            # 한 번에 한 달씩 (최대 744행)
            for month in pd.period_range(max(date(year, 1, 1), d0), min(date(year, 12, 31), d1), freq="M"):
                s = max(month.start_time.date(), d0)
                e = min(month.end_time.date(), d1)
                rows.extend(fetch_all(URL, {
                    "dataType": "JSON", "dataCd": "ASOS", "dateCd": "HR",
                    "startDt": s.strftime("%Y%m%d"), "startHh": "00",
                    "endDt": e.strftime("%Y%m%d"), "endHh": "23", "stnIds": stn,
                }, num_rows=999))
                time.sleep(0.05)
            pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
            print(f"{name}({stn}) {year}: {len(rows)}행", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
