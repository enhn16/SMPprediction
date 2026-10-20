"""전력거래소 연료원별 SMP 결정 횟수(일별) 수집.

하루 24시간 중 각 연료원(LNG, 유연탄, 원자력, 급전가능 재생에너지 등)이 SMP를 결정한 시간 수, 육지·제주별.
0원인 시간은 결정 연료원이 없어 합계가 24보다 작을 수 있다.
월별 파일로 저장하고, 이미 받은 지난 달은 건너뛴다. 일일 호출 한도를 넘으면 멈추고 다음 실행 때 이어 받는다.

실행: python -m common.collect.smp_fuel [시작일] [끝일]
저장: data/raw/api/smp_fuel/smp_fuel_YYYYMM.csv
"""
import sys
from datetime import date, timedelta

import pandas as pd

from common.api import ApiError, QuotaExceeded, fetch_items
from common.config import API_RAW_DIR

URL = "https://apis.data.go.kr/B552115/SmpDecByFuel2/getSmpDecByFuel2"
OUT_DIR = API_RAW_DIR / "smp_fuel"


def main(start: str = "20240101", end: str | None = None):
    d0 = pd.Timestamp(start).date()
    d1 = pd.Timestamp(end).date() if end else date.today() - timedelta(days=1)
    this_month = date.today().strftime("%Y%m")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for month in pd.period_range(d0, d1, freq="M"):
        ym = month.strftime("%Y%m")
        out = OUT_DIR / f"smp_fuel_{ym}.csv"
        if out.exists() and ym < this_month:
            continue
        rows, failed = [], []
        for d in pd.date_range(max(month.start_time.date(), d0), min(month.end_time.date(), d1)):
            try:
                items, _ = fetch_items(URL, {"dataType": "json", "pageNo": 1, "numOfRows": 50,
                                             "tradeDay": d.strftime("%Y%m%d")}, timeout=120)
                rows.extend(items)
            except QuotaExceeded:
                print(f"일일 호출 한도 초과 → 중단. {ym}부터 다음 실행 때 이어서 받음", flush=True)
                return
            except ApiError:
                failed.append(d.strftime("%Y-%m-%d"))
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{ym}: {len(rows)}행" + (f", 실패 {failed}" if failed else ""), flush=True)
    print("완료", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
