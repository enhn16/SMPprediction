"""전력거래소 '발전원별 실시간 전력수급현황' 5분 실적 엑셀 수집 (키 불필요).

열: 태양광(BTM,추정)·태양광(PPA,추정)·태양광(전력시장)·양수·수력·가스·풍력·신재생·ESS·유류·국내탄·유연탄·석탄·원자력 (MW)
BTM·PPA 추정값은 2024-11-23부터 (그 전 0). 실시간 공개라 D-1 15시 이전 구간은 판단 시각에 확보됨.
방법: 페이지 GET으로 쿠키·_csrf 확보 → /xlsxdownload.es?act=powersourceV2 POST (한 번에 최대 31일)
실행: python -m common.collect.kpx_power_source [시작 YYYY-MM-DD] [끝 YYYY-MM-DD]
저장: data/raw/api/kpx_power_source/ps_{YYYYMM}.xlsx (월 단위, 원문)
"""
import re
import sys
import time
from datetime import date

import pandas as pd
import requests

from common.config import API_RAW_DIR

BASE = "https://new.kpx.or.kr"
PAGE = BASE + "/powerSource.es?mid=a10404030000&device=chart"
OUT_DIR = API_RAW_DIR / "kpx_power_source"


def session():
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    html = s.get(PAGE, timeout=60).text
    m = re.search(r'name="_csrf"\s+(?:content|value)="([^"]+)"', html) or re.search(r'_csrf[^"]*"\s*value="([^"]+)"', html)
    return s, m.group(1)


def fetch_month(s, csrf, start: date, end: date) -> bytes:
    r = s.post(BASE + "/xlsxdownload.es?act=powersourceV2",
               data={"view_sdate31": start.isoformat(), "view_edate31": end.isoformat(), "_csrf": csrf},
               headers={"Referer": PAGE}, timeout=120)
    r.raise_for_status()
    return r.content


def main(start: str = "2024-01-01", end: str | None = None):
    end = end or date.today().isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    this_month = date.today().strftime("%Y%m")
    s, csrf = session()
    for m in pd.period_range(start, end, freq="M"):
        ym = m.strftime("%Y%m")
        out = OUT_DIR / f"ps_{ym}.xlsx"
        if out.exists() and ym < this_month:
            continue
        a, b = m.start_time.date(), min(m.end_time.date(), date.fromisoformat(end))
        content = fetch_month(s, csrf, a, b)
        if not content.startswith(b"PK"):
            s, csrf = session()
            content = fetch_month(s, csrf, a, b)
        out.write_bytes(content)
        print(f"{ym}: {len(content):,} bytes", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
