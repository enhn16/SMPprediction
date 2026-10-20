"""한국동서발전 '하루 전 전국 태양광 발전량 예측값 조회' API 수집 (data.go.kr 15144541, oamsFile31).

기준일(D-1) 14시 단기예보 기반, 17개 시도·시간별(06~19시) 일사 예측·설비용량(계량·소비자발전·PPA)·발전량 예측(MWh).
→ D-1 14시 생성이라 15시 판단에 사용 가능 (게시 지연 여부는 운영 중 확인).
자료 시작 2024-12-22, 중간 결측일 있음(약 20%).
실행: python -m common.collect.ewp_solar_forecast [시작 YYYY-MM-DD] [끝 YYYY-MM-DD]
저장: data/raw/api/ewp_solar_forecast/ewp_{기준일}.xml (원문, 결측일은 빈 파일 대신 empty_dates.txt 기록)
"""
import sys
import time
from datetime import date

import pandas as pd
import requests

from common.config import API_RAW_DIR, get_api_key

URL = "https://apis.data.go.kr/B552070/forecastService/oamsFile31"
OUT_DIR = API_RAW_DIR / "ewp_solar_forecast"


def main(start: str = "2024-12-20", end: str | None = None):
    end = end or date.today().isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    empty_file = OUT_DIR / "empty_dates.txt"
    empty = set(empty_file.read_text().split()) if empty_file.exists() else set()
    key = get_api_key()
    n_new = 0
    for d in pd.date_range(start, end):
        bd = d.strftime("%Y%m%d")
        out = OUT_DIR / f"ewp_{bd}.xml"
        if out.exists() or bd in empty:
            continue
        pdt = (d + pd.Timedelta(days=1)).strftime("%Y%m%d")
        r = requests.get(URL, params={"serviceKey": key, "pageNo": 1, "numOfRows": 1000, "baseDate": bd,
                                      "predDate": pdt}, timeout=60)
        if "LIMITED_NUMBER_OF_SERVICE_REQUESTS" in r.text or r.status_code == 429:
            print("일일 호출 한도 초과 → 중단, 다음 실행 때 이어서", flush=True)
            break
        if "<totalCount>0</totalCount>" in r.text:
            empty.add(bd)
            continue
        if "<item>" not in r.text:
            print(bd, "응답 이상:", r.text[:150], flush=True)
            continue
        out.write_text(r.text, encoding="utf-8")
        n_new += 1
        time.sleep(0.2)
    empty_file.write_text("\n".join(sorted(empty)))
    print(f"새로 받음 {n_new}일, 결측 {len(empty)}일, 보유 {len(list(OUT_DIR.glob('ewp_*.xml')))}일", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
