"""기상청 API허브 KIM 예보 공개 지연 측정 — 호출 시각에 확보된 최신 실행분(tmfc)을 기록한다.

D-1 00UTC(09시) 실행분이 15시 판단 전에 확보되는지 확인용. 일회성 예약 작업(SMP_kim_latency)이 30분마다 호출.
기록: data/raw/api/kim/latency_log.csv (확인시각, 모델, 최신 실행분(UTC), 해당 실행분 hf=30 확보 여부)
실행: python -m common.collect.kim_latency
"""
import os
from datetime import datetime, timedelta, timezone

import requests

from common.config import API_RAW_DIR

URL = "https://apihub.kma.go.kr/api/typ06/url/nwp_nc_vars_down.php"
BOX = dict(lon1=126.8, lat1=35.1, lon2=126.9, lat2=35.2)
VARS = {"KIMR": "SWDDIR2", "KIML": "SWDDIR2", "KIMG": "dswrsfc"}


def available(key, group, tmfc, hf):
    r = requests.get(URL, params=dict(group=group, data="U", vars=VARS[group], tmfc=tmfc, hf=hf, authKey=key, **BOX),
                     timeout=120)
    return "ERROR" not in r.text[:300]


def main():
    key = os.getenv("KMA_APIHUB_KEY")
    out = API_RAW_DIR / "kim" / "latency_log.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.write_text("checked_kst,group,latest_tmfc_utc,hf30_ok\n", encoding="utf-8")
    now = datetime.now()
    utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    runs = [utc - timedelta(hours=h) for h in range(0, 25) if (utc - timedelta(hours=h)).hour % 6 == 0]
    for group in VARS:
        latest, hf_ok = "", ""
        for r in runs:
            tm = r.strftime("%Y%m%d%H")
            if available(key, group, tm, 0):
                latest, hf_ok = tm, available(key, group, tm, 30)
                break
        with out.open("a", encoding="utf-8") as f:
            f.write(f"{now:%Y-%m-%d %H:%M},{group},{latest},{hf_ok}\n")
        print(now.strftime("%H:%M"), group, latest, hf_ok, flush=True)


if __name__ == "__main__":
    import common.config  # noqa: F401  (.env 로드)
    main()
