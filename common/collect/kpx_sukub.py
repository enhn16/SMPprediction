"""전력거래소 오늘전력수급현황(육지 5분) 매일 축적 — data.go.kr 한국전력거래소_오늘전력수급현황조회_GW (2026-10-26 승인).

열: baseDatetime(YYYYMMDDHHMMSS), currPwrTot 현재수요, suppAbility 공급능력, suppReservePwr·suppReserveRate 공급예비력(률),
    operReservePwr·operReserveRate 운영예비력(률), forecastLoad 당일 최대 예측수요 (MW, %).
API는 **오늘분만** 준다(baseDate 인자 무시) → 하루가 끝나기 전에 받아야 전체가 남는다. 매일 15:30·21:30·23:50에 호출.
실시간 공개라 D-1 15시 판단에는 D-1 14:55까지 사용 가능 (제주판 kpx_jeju_sukub과 같은 성격). 아직 피처로 쓰지 않음.
실행: python -m common.collect.kpx_sukub
저장: data/raw/api/kpx_sukub/sukub5m_{YYYYMM}.csv (같은 시각은 마지막 값으로 덮어씀)
"""
import pandas as pd

from common.api import fetch_items
from common.config import API_RAW_DIR

URL = "https://apis.data.go.kr/B552115/Sukub5mToday/getSukub5mToday"
OUT_DIR = API_RAW_DIR / "kpx_sukub"
COLS = ["baseDatetime", "currPwrTot", "suppAbility", "suppReservePwr", "suppReserveRate", "operReservePwr", "operReserveRate", "forecastLoad"]


def main():
    items, _ = fetch_items(URL, {"dataType": "json", "pageNo": 1, "numOfRows": 300})
    df = pd.DataFrame(items).reindex(columns=COLS)
    if df.empty:
        print("0행", flush=True)
        return
    df["baseDatetime"] = df["baseDatetime"].astype(str)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"sukub5m_{df['baseDatetime'].iloc[0][:6]}.csv"
    if out.exists():
        df = pd.concat([pd.read_csv(out, dtype={"baseDatetime": str}), df])
    df = df.drop_duplicates("baseDatetime", keep="last").sort_values("baseDatetime")
    df.to_csv(out, index=False, encoding="utf-8")
    day = df["baseDatetime"].str[:8]
    print(f"{out.name}: {len(df)}행, 오늘 {int((day == day.max()).sum())}행 (마지막 {df['baseDatetime'].iloc[-1]})", flush=True)


if __name__ == "__main__":
    main()
