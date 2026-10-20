"""전력통계정보시스템(EPSIS) 월별 연료비용 수집 (전력거래 > 연료비용).

중앙급전발전기의 연료원별(원자력·유연탄·무연탄·유류·LNG) 월 기준
연료단가(c1~c5), 열량단가(c6~c10, 원/Gcal), 연료비단가(c11~c15, 원/kWh).
출처: 비용평가위원회 심의·의결 발전비용. 작성주기: 매월 초 전월 통계치 갱신.
→ 판단 시각 누수 방지를 위해 M월 값은 M+2월 1일부터 쓴다 (loaders.load_fuel_cost).

실행: python -m common.collect.epsis_fuel_cost [시작 YYYYMM] [끝 YYYYMM]
저장: data/raw/api/epsis_fuel_cost/fuc_{시작}_{끝}.txt (응답 원문, UTF-8)
"""
import sys
from datetime import date

import requests

from common.config import API_RAW_DIR

PAGE = "https://epsis.kpx.or.kr/epsisnew/selectEkmaFucUpfChart.do?menuId=040100"
URL = "https://epsis.kpx.or.kr/epsisnew/selectEkmaFucUpf.ajax"
OUT_DIR = API_RAW_DIR / "epsis_fuel_cost"


def main(start: str = "202301", end: str | None = None):
    end = end or date.today().strftime("%Y%m")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.get(PAGE, timeout=60)  # 세션 쿠키
    r = s.post(URL, data={"beginDate": start, "endDate": end, "selYear": "N"}, timeout=120)
    r.raise_for_status()
    out = OUT_DIR / f"fuc_{start}_{end}.txt"
    out.write_text(r.text, encoding="utf-8")
    print(f"저장: {out.name}, {r.text.count('\"Period\"')}개월")


if __name__ == "__main__":
    main(*sys.argv[1:3])
