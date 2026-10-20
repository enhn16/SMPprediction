"""기상청 API허브 단기예보 구역별 육상예보(fct_afs_dl) 과거 발표분 수집.

발표: 매일 05·11·17시. 발표마다 3~4일 뒤까지 오전(00시 발효)·오후(12시 발효) 구간별
하늘상태(SKY: DB01 맑음 ~ DB04 흐림), 강수확률(ST, %), 기온(TA), 강수유무(PREP), 예보 문구(WF).
판단 시각 15시에는 전날 11시 발표분까지 쓸 수 있다.

호출 한 번에 최대 약 한 달. 구역·월별 파일로 저장, 지난 달은 건너뜀.
실행: python -m common.collect.kma_regional_fcst [시작월 YYYYMM] [끝월 YYYYMM]
저장: data/raw/api/kma_afs/afs_{구역코드}_{YYYYMM}.txt (원문 그대로, UTF-8)
"""
import os
import sys
import time
from datetime import date

import pandas as pd
import requests

from common.config import API_RAW_DIR

URL = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_dl.php"
OUT_DIR = API_RAW_DIR / "kma_afs"
# 태양광이 많은 7개 지점 (단기예보 도시 구역 코드)
REGIONS = {"11C20101": "서산", "11F10201": "전주", "11F20501": "광주", "21F20801": "목포",
           "11H10501": "안동", "11H10701": "대구", "11H20701": "진주"}


def main(start: str = "202312", end: str | None = None):
    key = os.getenv("KMA_APIHUB_KEY")  # common.config import 시 .env 로드됨
    end = end or date.today().strftime("%Y%m")
    this_month = date.today().strftime("%Y%m")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for reg, name in REGIONS.items():
        for m in pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M"):
            ym = m.strftime("%Y%m")
            out = OUT_DIR / f"afs_{reg}_{ym}.txt"
            if out.exists() and ym < this_month:
                continue
            t1 = m.start_time.strftime("%Y%m%d00")
            t2 = m.end_time.strftime("%Y%m%d23")
            for attempt in range(3):
                try:
                    r = requests.get(URL, params={"reg": reg, "tmfc1": t1, "tmfc2": t2, "disp": 0, "help": 0,
                                                  "authKey": key}, timeout=120)
                    text = r.content.decode("euc-kr", "replace")
                    if "#START7777" in text or reg in text:
                        break
                except requests.RequestException:
                    pass
                time.sleep(3)
            rows = [l for l in text.splitlines() if l.startswith(reg)]
            out.write_text("\n".join(rows), encoding="utf-8")
            print(f"{name}({reg}) {ym}: {len(rows)}행", flush=True)
            time.sleep(0.3)


if __name__ == "__main__":
    main(*sys.argv[1:3])
