"""전력거래소 계통한계가격 및 수요예측(하루전 발전계획용) 수집.

하루 48행(24시간 × 육지/제주). mlfd=육지 수요예측, jlfd=제주 수요예측, slfd=합계 (MW).
날짜 없이 호출하면 2020-01-01 ~ 다음 날 전체가 최신순으로 나오므로,
한 페이지 5,000건씩 전체를 받은 뒤(약 8분) 48행이 안 되는 날짜만 날짜 지정으로 다시 받는다.

실행: python -m common.collect.smp_forecast [시작일 YYYYMMDD]
      python -m common.collect.smp_forecast --recent 5   (최근 5일 ~ 내일만 날짜 지정으로 다시 받아 월별 파일에 합침)
저장: data/raw/api/smp_forecast/smp_forecast_YYYYMM.csv (월별)
"""
import sys

import pandas as pd

from common.api import fetch_all, fetch_items
from common.config import API_RAW_DIR

URL = "https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand"
OUT_DIR = API_RAW_DIR / "smp_forecast"
ROWS_PER_DAY = 48


def main(start: str = "20220101"):
    # 1) 전체 일괄 수집
    rows, page = [], 1
    while True:
        items, total = fetch_items(URL, {"dataType": "json", "pageNo": page, "numOfRows": 5000})
        rows.extend(items)
        print(f"페이지 {page}: 누적 {len(rows)}/{total}", flush=True)
        if len(rows) >= total or not items:
            break
        page += 1
    df = pd.DataFrame(rows)
    df["hour"] = df["hour"].astype(int)  # "1"과 "01"이 섞여 있음
    df = df.drop_duplicates(["date", "hour", "areaName"])
    df = df[df["date"].astype(str) >= start]

    # 2) 48행이 안 되는 날짜는 날짜 지정으로 재수집
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(str(df["date"].max()))).strftime("%Y%m%d")
    counts = df.groupby(df["date"].astype(str)).size()
    short = [d for d in days if counts.get(d, 0) < ROWS_PER_DAY]
    if short:
        print(f"재수집 대상 {len(short)}일: {short[:10]}{' ...' if len(short) > 10 else ''}", flush=True)
        for d in short:
            df = df[df["date"].astype(str) != d]
            extra = pd.DataFrame(fetch_all(URL, {"dataType": "json", "date": d}, num_rows=100))
            if not extra.empty:
                extra["hour"] = extra["hour"].astype(int)
                df = pd.concat([df, extra.drop_duplicates(["date", "hour", "areaName"])])
    counts = df.groupby(df["date"].astype(str)).size()
    still = [d for d in days if counts.get(d, 0) < ROWS_PER_DAY]
    print(f"48행 미만 날짜(재수집 후): {still}", flush=True)

    # 3) 월별 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["date", "areaName", "hour"])
    ym = df["date"].astype(str).str[:6]
    for m, g in df.groupby(ym):
        g.to_csv(OUT_DIR / f"smp_forecast_{m}.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_DIR} ({ym.nunique()}개월, {len(df)}행, {df['date'].min()}~{df['date'].max()})")


def update_recent(n_days: int = 5):
    """오늘 기준 최근 n일 ~ 내일을 날짜 지정으로 받아 기존 월별 파일에 덮어 합친다 (48행 미만 날은 공표 전이라 저장 안 함, 단 육지 24행은 저장)."""
    today = pd.Timestamp.today().normalize()
    days = pd.date_range(today - pd.Timedelta(days=n_days), today + pd.Timedelta(days=1)).strftime("%Y%m%d")
    new = []
    for d in days:
        items = pd.DataFrame(fetch_all(URL, {"dataType": "json", "date": d}, num_rows=100))
        print(f"{d}: {len(items)}행", flush=True)
        if len(items) >= ROWS_PER_DAY // 2:
            items["hour"] = items["hour"].astype(int)
            new.append(items.drop_duplicates(["date", "hour", "areaName"]))
    if not new:
        return
    new = pd.concat(new)
    for m, g in new.groupby(new["date"].astype(str).str[:6]):
        path = OUT_DIR / f"smp_forecast_{m}.csv"
        old = pd.read_csv(path, dtype={"date": str}) if path.exists() else pd.DataFrame()
        if len(old):
            old["hour"] = old["hour"].astype(int)
            old = old[~old["date"].astype(str).isin(g["date"].astype(str).unique())]
        out = pd.concat([old, g.assign(date=g["date"].astype(str))]).sort_values(["date", "areaName", "hour"])
        out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"저장: {path.name} ({out['date'].min()}~{out['date'].max()})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--recent"]:
        update_recent(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    else:
        main(*sys.argv[1:2])
