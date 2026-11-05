"""제주 5분 실적 보강 수집 (2019-12 ~ 2022-12 + 운영 로더 빈 날) — 확장 재채점(제주 급락)용.

운영 수집기(common/collect/kpx_jeju_sukub.py)는 고치지 않고 fetch_day만 import한다.
저장: data/raw/backfill_2020/kpx_jeju_sukub/jeju5m_{YYYYMM}.csv (API 원래 열 그대로)
이미 저장된 달(빈 날 없음)은 건너뛰고, 한도 초과 시 멈춘 뒤 같은 명령으로 이어 받는다.
실행(루트에서): python -m extended_backtest.src.jeju_collect_sukub
"""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common.api import QuotaExceeded
from common.collect.kpx_jeju_sukub import COLS, fetch_day
from extended_backtest.src.data import BF

OUT_DIR = BF / "kpx_jeju_sukub"
START, END = "2019-12-01", "2022-12-31"
GAP_DAYS = ["2023-07-02", "2026-10-29", "2026-10-30", "2026-10-31"]   # 운영 로더에 없는 날
WORKERS = 4


def _save(days: list[pd.Timestamp], out) -> bool:
    try:
        with ThreadPoolExecutor(WORKERS) as ex:
            res = list(ex.map(fetch_day, [d.strftime("%Y%m%d") for d in days]))
    except QuotaExceeded:
        print("일일 호출 한도 초과 → 중단 (다음에 같은 명령으로 이어 받기)", flush=True)
        return False
    df = pd.DataFrame([r for rs in res for r in rs]).reindex(columns=COLS)
    df["baseDatetime"] = df["baseDatetime"].astype(str)
    if out.exists():
        old = pd.read_csv(out, dtype={"baseDatetime": str})
        df = pd.concat([old, df])
    df = df.drop_duplicates("baseDatetime", keep="last").sort_values("baseDatetime")
    df.to_csv(out, index=False, encoding="utf-8")
    n = df["baseDatetime"].str[:8].value_counts()
    print(f"{out.name}: {len(df)}행, 날 {len(n)}, 288행 미만 날 {int((n < 288).sum())}", flush=True)
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    days = list(pd.date_range(START, END)) + [pd.Timestamp(d) for d in GAP_DAYS]
    for ym, grp in pd.Series(days).groupby(lambda i: days[i].strftime("%Y%m")):
        out = OUT_DIR / f"jeju5m_{ym}.csv"
        need = list(grp)
        if out.exists():
            have = set(pd.read_csv(out, dtype={"baseDatetime": str})["baseDatetime"].str[:8])
            need = [d for d in need if d.strftime("%Y%m%d") not in have]
        if need and not _save(need, out):
            return


if __name__ == "__main__":
    main()
