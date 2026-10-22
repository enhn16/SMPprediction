"""전력통계정보시스템(EPSIS) 시장참여 설비용량 월별 수집 (전력거래 > 시장참여설비용량, 지역합계).

사용자가 내려받았던 파일(data/raw/HOME_전력거래_시장참여설비용량(전력시장·PPA)_연료원별.csv)과 같은 값.
응답 열 대응(2026/08로 대조 확인): c1 원자력, c2 유연탄, c3 무연탄, c23 석탄계, c4 유류, c5 LNG, c6 양수,
  c91 연료전지, c92 석탄가스화, c93 태양, c94 풍력, c95 수력, c96 해양, c97 바이오, c98 폐기물, c912 신재생계,
  c7 기타, c8 전력시장 계 (MW)
공표 시점 미확인 → M월 값은 M+2월 1일부터 사용 (loaders.load_market_capacity).
실행: python -m common.collect.epsis_capacity
저장: data/raw/api/epsis_capacity/gcp_{수집월}.txt (응답 원문, UTF-8)
"""
from datetime import date

import requests

from common.config import API_RAW_DIR

PAGE = "https://epsis.kpx.or.kr/epsisnew/selectEkmaGcpBftChart.do?menuId=040301"
URL = "https://epsis.kpx.or.kr/epsisnew/selectEkmaGcpBft.ajax"
OUT_DIR = API_RAW_DIR / "epsis_capacity"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    s.get(PAGE, timeout=60)  # 세션 쿠키
    r = s.post(URL, data={"selYear": "N", "selRegion": "1"}, timeout=120)
    r.raise_for_status()
    out = OUT_DIR / f"gcp_{date.today():%Y%m}.txt"
    out.write_text(r.text, encoding="utf-8")
    print(f"저장: {out.name}, {r.text.count('gridData.push')}개월")


if __name__ == "__main__":
    main()
