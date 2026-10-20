"""천문연구원 특일정보(공휴일) 수집.

실행: python -m common.collect.holidays [시작연도] [끝연도]
저장: data/raw/api/holidays/holidays_{시작}_{끝}.csv
"""
import sys

import pandas as pd

from common.api import fetch_all
from common.config import API_RAW_DIR

URL = "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"


def main(start: int = 2022, end: int = 2027):
    rows = []
    for year in range(start, end + 1):
        # 이 API는 인증키 파라미터 이름이 ServiceKey(대문자 S)
        items = fetch_all(URL, {"solYear": year, "_type": "json"}, num_rows=100,
                          key_param="ServiceKey")
        print(f"{year}년 공휴일 {len(items)}건")
        rows.extend(items)
    df = pd.DataFrame(rows)
    out = API_RAW_DIR / "holidays" / f"holidays_{start}_{end}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out} ({len(df)}행)")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:3]))
