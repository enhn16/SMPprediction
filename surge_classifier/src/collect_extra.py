"""급등 피처 후보용 추가 자료 수집 (13종 밖 포함, 2026-10-28 사용자 요청 "모을 수 있는 건 다 모아봐").

공용 수집기(common/collect, 총괄 소유)는 고치지 않고, 원문은 data/raw/surge_extra/ 아래에 따로 둔다.
- supply : EPSIS 일별 최대전력기준 전력수급(설비용량·공급능력·최대전력·공급예비력·예비율·최대전력 시각), 일 단위 실적
           (data.go.kr 15150737 = epsis selectEkgeEpsMep). 15시 판단에는 D-2일분까지 확실 (D-1 최대는 저녁일 수 있음)
- source : 전력거래소 발전원별 5분 실적 2023분 (common.collect.kpx_power_source의 함수 재사용, 2024~는 공용 폴더에 이미 있음)
- warn   : 기상청 API허브 특보 발표 이력 (wrn_met_data)

- openmeteo : Open-Meteo 2일 전 실행 best_match 기온·운량·강수·일사 (급등 v3 운영 입력, **매일 15시 전 실행 필요**)

실행: python -m surge_classifier.src.collect_extra [supply|source|warn|openmeteo ...]
"""
import os
import sys
import time
from datetime import date

import pandas as pd
import requests

from common.config import RAW_DIR

OUT = RAW_DIR / "surge_extra"
EPSIS_PAGE = "https://epsis.kpx.or.kr/epsisnew/selectEkgeEpsMepChart.do?menuId=030100"
EPSIS_URL = "https://epsis.kpx.or.kr/epsisnew/selectEkgeEpsMep.ajax"


def supply(start: str = "2022-11-01", end: str | None = None):
    """연 단위로 나눠 받아 원문 저장 → epsis_supply_{연}.txt"""
    end = end or date.today().isoformat()
    d = OUT / "epsis_supply"
    d.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    s.get(EPSIS_PAGE, timeout=60)
    for y in range(int(start[:4]), int(end[:4]) + 1):
        a = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01"))
        b = min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31"))
        r = s.post(EPSIS_URL, data={"beginDate": f"{a:%Y%m%d}", "endDate": f"{b:%Y%m%d}", "selYear": "", "selMonth": ""},
                   timeout=180)
        r.raise_for_status()
        (d / f"epsis_supply_{y}.txt").write_text(r.text, encoding="utf-8")
        print(f"공급능력 {y}: {r.text.count('gridData.push')}일", flush=True)
        time.sleep(1)


def source(start: str = "2023-01-01", end: str = "2023-12-31"):
    from common.collect import kpx_power_source as ps
    d = OUT / "kpx_power_source"
    d.mkdir(parents=True, exist_ok=True)
    s, csrf = ps.session()
    for m in pd.period_range(start, end, freq="M"):
        out = d / f"ps_{m.strftime('%Y%m')}.xlsx"
        if out.exists():
            continue
        a, b = m.start_time.date(), m.end_time.date()
        content = ps.fetch_month(s, csrf, a, b)
        if not content.startswith(b"PK"):
            s, csrf = ps.session()
            content = ps.fetch_month(s, csrf, a, b)
        out.write_bytes(content)
        print(f"발전원별 {m.strftime('%Y%m')}: {len(content):,} bytes", flush=True)
        time.sleep(1)


def warn(start: str = "2022-12-01", end: str | None = None):
    """특보 발표 이력: 월 단위로 받아 원문 저장 → wrn_{YYYYMM}.txt"""
    end = end or date.today().isoformat()
    d = OUT / "kma_warning"
    d.mkdir(parents=True, exist_ok=True)
    key = os.environ["KMA_APIHUB_KEY"]   # common.config가 .env를 읽음
    for m in pd.period_range(start, end, freq="M"):
        out = d / f"wrn_{m.strftime('%Y%m')}.txt"
        if out.exists() and m < pd.Period(date.today(), "M"):
            continue
        r = requests.get("https://apihub.kma.go.kr/api/typ01/url/wrn_met_data.php",
                         params={"wrn": "A", "reg": "0", "tmfc1": f"{m.start_time.strftime('%Y%m%d')}0000",
                                 "tmfc2": f"{m.end_time.strftime('%Y%m%d')}2359", "disp": "0", "help": "1", "authKey": key},
                         timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"특보 {m.strftime('%Y%m')}: {len(r.content):,} bytes", flush=True)
        time.sleep(0.5)


def openmeteo(start: str | None = None, end: str | None = None):
    """팀원 급등 모델(spike_simple3) 입력과 같은 Open-Meteo 2일 전 실행 예보: 모델 지정 없음(best_match),
    육지 7개 도시, 시간별 기온·운량·강수·일사 → openmeteo_best/prev2_best_{start}_{end}.json
    인자 없이 부르면 매일 운영용: 오늘 −10일 ~ 내일+1일 (내일의 2일 전 실행분은 어제 실행이라 15시 판단에 사용 가능)."""
    from common.collect.openmeteo_prevruns import CITIES, URL
    today = pd.Timestamp(date.today())
    start = start or (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = end or (today + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    d = OUT / "openmeteo_best"
    d.mkdir(parents=True, exist_ok=True)
    hourly = ",".join(f"{v}_previous_day2" for v in ("temperature_2m", "cloud_cover", "precipitation", "shortwave_radiation"))
    r = requests.get(URL, params={"latitude": ",".join(str(v[0]) for v in CITIES.values()),
                                  "longitude": ",".join(str(v[1]) for v in CITIES.values()), "hourly": hourly,
                                  "start_date": start, "end_date": end, "timezone": "Asia/Seoul"}, timeout=600)
    r.raise_for_status()
    (d / f"prev2_best_{start}_{end}.json").write_text(r.text, encoding="utf-8")
    first = r.json()[0]
    v = first["hourly"]["temperature_2m_previous_day2"]
    print(f"Open-Meteo best_match: {len(v)}시간 중 기온 값 있음 {sum(x is not None for x in v)}", flush=True)


if __name__ == "__main__":
    for job in (sys.argv[1:] or ["supply", "source", "warn"]):
        globals()[job]()
