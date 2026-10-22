"""기상청 중기 육상예보(data.go.kr MidFcstInfoService/getMidLandFcst) 매일 수집 — 4~10일 뒤 오전·오후 날씨·강수확률.

API는 과거 발표분을 주지 않으므로(오늘 기준 1일 전까지) 매일 그날 06시 발표분을 받아 쌓는다.
저장 형식은 기상자료개방포털에서 받은 과거 파일(data/raw/FCT_MEDM_*.csv)과 같게 맞춘다 → common.loaders.load_mid_forecast가 함께 읽음.
  열: 발표시각("YYYY-MM-DD 06시"), 지역, 예보시각("YYYY-MM-DD 오전|오후", 8일 뒤부터는 하루 1개라 오전·오후에 같은 값), 예보, 강수확률(%)
실행: python -m common.collect.mid_fcst [발표일 YYYYMMDD]   (기본 오늘, 06시 발표분)
저장: data/raw/api/mid_fcst/mid_{발표일}06.csv
"""
import sys
from datetime import date

import pandas as pd

from common.api import fetch_items
from common.config import API_RAW_DIR

URL = "https://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst"
OUT_DIR = API_RAW_DIR / "mid_fcst"
# 과거 파일의 지역 이름에 맞춤
REG = {"11B00000": "서울.인천.경기", "11D10000": "강원영서", "11D20000": "강원영동", "11C20000": "충청남도",
       "11C10000": "충청북도", "11F20000": "전라남도", "11F10000": "전라북도", "11H10000": "경상북도",
       "11H20000": "경상남도", "11G00000": "제주도"}


def main(day: str | None = None):
    day = day or date.today().strftime("%Y%m%d")
    issue = pd.Timestamp(day)
    rows = []
    for reg, name in REG.items():
        items, _ = fetch_items(URL, {"dataType": "JSON", "pageNo": 1, "numOfRows": 10, "regId": reg, "tmFc": f"{day}0600"})
        if not items:
            print(f"{name}: 발표분 없음", flush=True)
            continue
        it = items[0]
        for k in range(4, 11):
            tgt = (issue + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
            for ap, suf in (("오전", "Am"), ("오후", "Pm")):
                wf = it.get(f"wf{k}{suf}", it.get(f"wf{k}"))
                pop = it.get(f"rnSt{k}{suf}", it.get(f"rnSt{k}"))
                if wf is not None:
                    rows.append({"발표시각": f"{issue:%Y-%m-%d} 06시", "지역": name, "예보시각": f"{tgt} {ap}",
                                 "예보": wf, "강수확률(%)": pop})
    if not rows:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"mid_{day}06.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out} ({len(rows)}행)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
