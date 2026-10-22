"""전력거래소 전력수급예보(data.go.kr 15158707, 한국전력거래소_전력수급예보조회_GW) 수집 — 내일 최대 예측수요.

엔드포인트: https://apis.data.go.kr/B552115/forecast1dMaxBaseDate/getForecast1dMaxBaseDate
응답 항목: fcDate(대상일), fcStime·fcEtime(최대부하 예상 시간대), fcMaxload(최대 예측수요 MW), fcReservePwr(예측 예비력 MW),
           fcLevel(경보 단계), rn(순번)
용도: 게시판 이미지로만 올라오는 '주간 전력수요 전망'(kpx_peak 피처)의 자동 대체·보완 후보.
  - 2026-10-22 승인 확인: baseDate를 무엇으로 줘도 '현재 공표된 다음 날' 한 건만 돌려줌 (과거 조회 불가)
    → 매 실행 응답을 대상일·수집 시각과 함께 쌓아야 함. 공표 시각(D-1 15시 이전 확보 여부)은 쌓인 수집 시각으로 판정.
  - 판정 전에는 예측 피처로 쓰지 않는다 (루트 CLAUDE.md: 내일 수요예측 미사용).
실행: python -m common.collect.kpx_demand_fcst
저장: data/raw/api/kpx_demand_fcst/fc_{대상일}_{수집 YYYYMMDDHHMM}.json (응답 원문)
"""
import json
from datetime import datetime

import requests

from common.config import API_RAW_DIR, get_api_key

URL = "https://apis.data.go.kr/B552115/forecast1dMaxBaseDate/getForecast1dMaxBaseDate"
OUT_DIR = API_RAW_DIR / "kpx_demand_fcst"


def main():
    now = datetime.now()
    r = requests.get(URL, params={"serviceKey": get_api_key(), "pageNo": 1, "numOfRows": 100, "dataType": "json",
                                  "baseDate": now.strftime("%Y%m%d")}, timeout=60)
    if "SERVICE_KEY_IS_NOT_REGISTERED" in r.text:
        print("활용신청 필요: 공공데이터포털 '한국전력거래소_전력수급예보조회_GW'(15158707) — 현재 키 미등록")
        return
    try:
        items = json.loads(r.text)["response"]["body"]["items"]["item"]
        fc_date = items[0]["fcDate"]
    except (ValueError, KeyError, IndexError, TypeError):
        print(f"응답 해석 실패 (HTTP {r.status_code}): {r.text[:200]}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fc_{fc_date}_{now:%Y%m%d%H%M}.json"
    out.write_text(r.text, encoding="utf-8")
    it = items[0]
    print(f"저장: {out.name} — 대상일 {fc_date} 최대 {it['fcMaxload']}MW ({it['fcStime']}~{it['fcEtime']}시), "
          f"예비력 {it['fcReservePwr']}MW")


if __name__ == "__main__":
    main()
