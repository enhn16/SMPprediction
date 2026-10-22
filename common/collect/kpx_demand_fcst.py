"""전력거래소 전력수급예보(data.go.kr 15158707, 한국전력거래소_전력수급예보조회_GW) 수집 — 날짜별 최대 예측수요.

엔드포인트: https://apis.data.go.kr/B552115/forecast1dMaxBaseDate/getForecast1dMaxBaseDate
응답 항목: fcDate(연-월-일), fcStime·fcEtime(예보 시작·종료 시간), fcMaxload(최대 예측수요), fcReservePwr(예측 예비력),
           fcLevel(경보 단계), rn(순번)
용도: 게시판 이미지로만 올라오는 '주간 전력수요 전망'(kpx_peak 피처)의 자동 대체 후보.
  - 2026-10-21 확인: 엔드포인트는 있으나 현재 키는 '등록되지 않은 서비스키' → 공공데이터포털에서 활용신청 필요.
  - 승인 후 확인할 것: 요청 변수(기준일 baseDate 추정), 과거 기준일 조회 가능 여부, 값이 주간 전망 최대부하와 같은 척도인지.
실행: python -m common.collect.kpx_demand_fcst [기준일 YYYYMMDD]   (기본 오늘)
저장: data/raw/api/kpx_demand_fcst/fc_{기준일}.json (응답 원문)
"""
import sys
from datetime import date

import requests

from common.config import API_RAW_DIR, get_api_key

URL = "https://apis.data.go.kr/B552115/forecast1dMaxBaseDate/getForecast1dMaxBaseDate"
OUT_DIR = API_RAW_DIR / "kpx_demand_fcst"


def main(day: str | None = None):
    day = day or date.today().strftime("%Y%m%d")
    r = requests.get(URL, params={"serviceKey": get_api_key(), "pageNo": 1, "numOfRows": 100, "dataType": "json",
                                  "baseDate": day}, timeout=60)
    if "SERVICE_KEY_IS_NOT_REGISTERED" in r.text:
        print("활용신청 필요: 공공데이터포털 '한국전력거래소_전력수급예보조회_GW'(15158707) — 현재 키 미등록")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fc_{day}.json"
    out.write_text(r.text, encoding="utf-8")
    print(f"저장: {out.name} (HTTP {r.status_code}, {len(r.text)}자) {r.text[:200]}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
