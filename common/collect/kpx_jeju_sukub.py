"""전력거래소 제주계통운영정보(송전단 포함) 5분 실적 수집 — data.go.kr 15158505.

열: suppAbility 공급능력, currPwrTot 발전단 수요, currNtPwrTot 송전단 수요(2026-06 이전 0), renewPwrTot 신재생 합계,
    renewPwrSolar 태양광, renewPwrWind 풍력 (MW). baseDatetime = 5분 시각 (0000~2355).
실시간 공개: D-1 15시 판단에는 D-1 14:55까지 사용 가능 (예측값 없음, 실적만). 2019-12-01부터.
호출 1회 = 하루(288행). 월별 파일로 저장하고 지난 달은 건너뜀. 일일 호출 한도를 넘으면 멈추고 다음 실행에 이어 받음.
실행: python -m common.collect.kpx_jeju_sukub [시작일 YYYYMMDD] [끝일 YYYYMMDD]   (끝일 생략 시 오늘)
저장: data/raw/api/kpx_jeju_sukub/jeju5m_{YYYYMM}.csv
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

from common.api import QuotaExceeded, fetch_items
from common.config import API_RAW_DIR

URL = "https://apis.data.go.kr/B552115/JejuSukub5mToday/getJejuSukub5mToday"
OUT_DIR = API_RAW_DIR / "kpx_jeju_sukub"
COLS = ["baseDatetime", "suppAbility", "currPwrTot", "currNtPwrTot", "renewPwrTot", "renewPwrSolar", "renewPwrWind"]
WORKERS = 4


def fetch_day(d: str) -> list[dict]:
    items, _ = fetch_items(URL, {"baseDate": d, "dataType": "json", "pageNo": 1, "numOfRows": 300})
    return items


def main(start: str = "20230101", end: str | None = None):
    end = end or date.today().strftime("%Y%m%d")
    this_month = date.today().strftime("%Y%m")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for m in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
        ym = m.strftime("%Y%m")
        out = OUT_DIR / f"jeju5m_{ym}.csv"
        days = [d.strftime("%Y%m%d") for d in pd.date_range(max(m.start_time, pd.Timestamp(start)), min(m.end_time, pd.Timestamp(end)))]
        if out.exists() and ym < this_month:
            # 지난 달: 288행이 다 찬 날은 건너뛰고 모자란 날만 다시 받음 (2026-11-06: 달이 바뀌기 전 며칠이 빈 채로 굳던 문제)
            have = pd.read_csv(out, usecols=["baseDatetime"], dtype=str)["baseDatetime"].str[:8].value_counts()
            days = [d for d in days if have.get(d, 0) < 288]
            if not days:
                continue
        try:
            with ThreadPoolExecutor(WORKERS) as ex:
                res = list(ex.map(fetch_day, days))
        except QuotaExceeded:
            print("일일 호출 한도 초과 → 중단 (다음에 같은 명령으로 이어 받기)", flush=True)
            return
        df = pd.DataFrame([r for rs in res for r in rs])
        if df.empty:
            print(f"{ym}: 0행", flush=True)
            continue
        df = df.reindex(columns=COLS).drop_duplicates("baseDatetime").sort_values("baseDatetime")
        if out.exists():   # 이번 달: 기존 파일과 합침
            old = pd.read_csv(out, dtype={"baseDatetime": str})
            df = pd.concat([old, df.astype({"baseDatetime": str})]).drop_duplicates("baseDatetime", keep="last").sort_values("baseDatetime")
        df.to_csv(out, index=False, encoding="utf-8")
        n_day = df["baseDatetime"].astype(str).str[:8].value_counts()
        print(f"{ym}: {len(df)}행, 288행 미만 날 {int((n_day < 288).sum())}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
