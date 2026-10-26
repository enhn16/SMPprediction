"""전력거래소 지역별 시간별 태양광 발전량 — data.go.kr 한국전력거래소_지역별 시간별 태양광 발전량 정보 (2026-10-26 승인).

열: tradeYmd(YYYYMMDD), tradeNo(1~24시), regionNm(시·도), amgo(발전량, MWh). 2021-01-01부터.
API는 날짜 인자를 무시하고 **전체를 최신 날짜부터** 페이지로 준다. 2026-10-26 확인 시 최신 2026-09-05 (약 7주 늦게 갱신).
→ 받은 마지막 날짜보다 새 날짜만 앞 페이지부터 받고 멈춘다. 2025-12까지는 원본 파일(15065269)이 있으므로 기본 시작은 2026-01-01.
실적이라 피처로 쓰려면 판단 시각에 공개돼 있었는지(공개 지연) 따져야 한다. 지금은 해석·후속 검토용.
실행: python -m common.collect.kpx_pv_region [시작 YYYYMMDD=20260101]
저장: data/raw/api/kpx_pv_region/pv_region_{YYYYMM}.csv
"""
import sys

import pandas as pd

from common.api import QuotaExceeded, fetch_items
from common.config import API_RAW_DIR

URL = "https://apis.data.go.kr/B552115/PvAmountByLocHr/getPvAmountByLocHr"
OUT_DIR = API_RAW_DIR / "kpx_pv_region"
ROWS = 1000


def saved_max() -> str | None:
    fs = sorted(OUT_DIR.glob("pv_region_*.csv"))
    return str(pd.read_csv(fs[-1], dtype={"tradeYmd": str})["tradeYmd"].max()) if fs else None


def saved() -> pd.DataFrame:
    fs = sorted(OUT_DIR.glob("pv_region_*.csv"))
    return pd.concat([pd.read_csv(f, dtype={"tradeYmd": str, "tradeNo": str}) for f in fs]) if fs else pd.DataFrame(columns=["tradeYmd"])


def backfill(start: str):
    """한도 때문에 중간에 멈춘 경우: 받은 가장 옛날 날짜보다 앞(옛날) 구간을 이어 받음. 최신부터 정렬이라 받은 행 수로 시작 쪽을 정함."""
    s = saved()
    if s.empty or s["tradeYmd"].min() <= start:
        return
    page = len(s) // ROWS + 1
    rows = []
    while True:
        try:
            items, _ = fetch_items(URL, {"dataType": "json", "pageNo": page, "numOfRows": ROWS}, timeout=120)
        except QuotaExceeded:
            print("일일 호출 한도 초과 → 받은 만큼 저장 (다음 실행에 이어 받음)", flush=True)
            break
        if not items:
            break
        rows += items
        if min(i["tradeYmd"] for i in items) < start:
            break
        page += 1
    if rows:
        _save(pd.DataFrame(rows).query("tradeYmd >= @start"), "이어 받기")


def _save(df: pd.DataFrame, what: str):
    df = df[["tradeYmd", "tradeNo", "regionNm", "amgo"]].astype({"tradeNo": str})
    for ym, g in df.groupby(df["tradeYmd"].str[:6]):
        out = OUT_DIR / f"pv_region_{ym}.csv"
        if out.exists():
            g = pd.concat([pd.read_csv(out, dtype={"tradeYmd": str, "tradeNo": str}), g])
        g.drop_duplicates(["tradeYmd", "tradeNo", "regionNm"], keep="last").sort_values(["tradeYmd", "tradeNo", "regionNm"]) \
            .to_csv(out, index=False, encoding="utf-8")
    print(f"{what}: {df['tradeYmd'].min()} ~ {df['tradeYmd'].max()} {len(df)}행", flush=True)


def main(start: str = "20260101"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backfill(start)   # 지난번에 한도로 멈췄으면 옛날 쪽부터 마저 받음
    stop = max(start, (saved_max() or "0")) if saved_max() else start   # 이 날짜 이후(>)만 새로 받음, 처음이면 시작일 포함
    first = saved_max() is None
    rows, page = [], 1
    while True:
        try:
            items, total = fetch_items(URL, {"dataType": "json", "pageNo": page, "numOfRows": ROWS}, timeout=120)
        except QuotaExceeded:
            print("일일 호출 한도 초과 → 받은 만큼 저장", flush=True)
            break
        if not items:
            break
        rows += items
        oldest = min(i["tradeYmd"] for i in items)
        if (oldest < stop) if first else (oldest <= stop):
            break
        page += 1
    df = pd.DataFrame(rows)
    if df.empty:
        print("새 자료 없음", flush=True)
        return
    df = df[(df["tradeYmd"] >= stop) if first else (df["tradeYmd"] > stop)]
    if df.empty:
        print(f"새 자료 없음 (마지막 {stop})", flush=True)
        return
    _save(df, f"새 자료 ({page}쪽)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
