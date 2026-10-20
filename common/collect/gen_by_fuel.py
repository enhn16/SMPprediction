"""전력거래소 발전원별 발전량(시간별, 정산 데이터) 수집 — 공급 측 요인 분석용 (실적, 예측 피처로 쓰지 않음).

발전원: LNG, 무연탄, 수력, 신재생·기타, 양수, 원자력, 유연탄, 유전 (태양광 별도 구분 없음).
정산 데이터라 약 1.5개월 늦게 공개되고, 최근분은 신재생이 0으로 나올 수 있음.
일일 호출 한도가 작아(약 100회) 필요한 날짜만 골라 받는 방식을 기본으로 한다.

실행: python -m common.collect.gen_by_fuel 20250101 20250226 ...   (날짜 나열, 이미 받은 날짜는 건너뜀)
저장: data/raw/api/gen_by_fuel/gen_by_fuel_day_YYYYMMDD.csv (월 단위로 받은 gen_by_fuel_YYYYMM.csv도 함께 사용)
"""
import sys

import pandas as pd

from common.api import QuotaExceeded, fetch_items
from common.config import API_RAW_DIR

URL = "https://apis.data.go.kr/B552115/PvAmountByPwrGen/getPvAmountByPwrGen"
OUT_DIR = API_RAW_DIR / "gen_by_fuel"


EMPTY_FILE = OUT_DIR / "empty_dates.txt"  # 조회했지만 데이터가 없던 날짜 (정산 전) → 다시 받지 않음


def have_dates() -> set[str]:
    done = set()
    for p in OUT_DIR.glob("gen_by_fuel_*.csv"):
        done |= set(pd.read_csv(p, dtype=str, usecols=["tradeYmd"])["tradeYmd"])
    if EMPTY_FILE.exists():
        done |= set(EMPTY_FILE.read_text().split())
    return done


def main(dates: list[str]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = have_dates()
    todo = sorted({pd.Timestamp(d).strftime("%Y%m%d") for d in dates} - done)
    print(f"요청 {len(dates)}일 중 새로 받을 날짜 {len(todo)}일", flush=True)
    got, empty = 0, []
    for d in todo:
        try:
            items, _ = fetch_items(URL, {"dataType": "json", "pageNo": 1, "numOfRows": 500, "tradeYmd": d}, timeout=120)
        except QuotaExceeded:
            print(f"일일 호출 한도 초과 → 중단 ({got}일 받음, 남은 {len(todo) - got - len(empty)}일은 다음 실행 때)", flush=True)
            return
        if items:
            pd.DataFrame(items).to_csv(OUT_DIR / f"gen_by_fuel_day_{d}.csv", index=False, encoding="utf-8-sig")
            got += 1
        else:
            empty.append(d)
            with open(EMPTY_FILE, "a") as f:
                f.write(d + " ")
    print(f"완료: {got}일 저장" + (f", 데이터 없는 날짜 {len(empty)}일(정산 전): {empty[:5]}..." if empty else ""), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
