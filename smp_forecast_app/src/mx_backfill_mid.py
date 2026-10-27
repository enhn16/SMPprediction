"""중기 육상예보 과거 발표분 보강 (기상청 API허브 fct_afs_wl) — data.go.kr API는 과거분을 주지 않아 2023년이 비어 있었음.

기상자료개방포털 과거 파일(data/raw/FCT_MEDM_*.csv)·매일 수집분(common.collect.mid_fcst)과 같은 열로 저장 → common.loaders.load_mid_forecast가 함께 읽음.
  열: 발표시각("YYYY-MM-DD HH시"), 지역, 예보시각("YYYY-MM-DD 오전|오후"; 24시간 구간(A01)은 포털 파일처럼 오전 한 행), 예보, 강수확률(%)
지역 = common.collect.mid_fcst.REG (육지 9개 권역 + 제주도), 발표 06·18시 모두.
실행: python -m smp_forecast_app.src.mx_backfill_mid [시작월 YYYYMM=202301] [끝월 YYYYMM=202312]
저장: data/raw/api/mid_fcst/mid_hist_{시작}_{끝}.csv
"""
import os
import re
import sys
import time

import pandas as pd
import requests

from common.collect.mid_fcst import OUT_DIR, REG

URL = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wl.php"
LINE = re.compile(r'^(\S+)\s+(\d{12})\s+(\d{12})\s+(A0[12])\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+"([^"]*)"\s+(-?\d+)')


def fetch(reg, t1, t2, key):
    for _ in range(3):
        try:
            r = requests.get(URL, params={"reg": reg, "tmfc1": t1, "tmfc2": t2, "disp": 0, "help": 0, "authKey": key}, timeout=120)
            text = r.content.decode("euc-kr", "replace")
            if "#START7777" in text:
                return text
        except requests.RequestException:
            pass
        time.sleep(3)
    raise RuntimeError(f"{reg} {t1}: 응답 없음")


def main(start="202301", end="202312"):
    key = os.getenv("KMA_APIHUB_KEY")
    rows = []
    for reg, name in REG.items():
        for m in pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M"):
            text = fetch(reg, m.start_time.strftime("%Y%m%d0000"), m.end_time.strftime("%Y%m%d2359"), key)
            n = 0
            for ln in text.splitlines():
                g = LINE.match(ln)
                if not g:
                    continue
                _, fc, ef, mod, wf, pop = g.groups()
                fc, ef = pd.Timestamp(fc), pd.Timestamp(ef)
                # 24시간 구간(A01, 8~10일 뒤)은 포털 과거 파일처럼 '오전' 한 행 (2024-01 대조: 겹치는 1,612행 예보·강수확률 100% 일치)
                aps = ("오전",) if mod == "A01" or ef.hour == 0 else ("오후",)
                for ap in aps:
                    rows.append({"발표시각": f"{fc:%Y-%m-%d} {fc:%H}시", "지역": name, "예보시각": f"{ef:%Y-%m-%d} {ap}",
                                 "예보": wf, "강수확률(%)": int(pop)})
                    n += 1
            print(f"{name} {m}: {n}행", flush=True)
            time.sleep(0.3)
    out = OUT_DIR / f"mid_hist_{start}_{end}.csv"
    pd.DataFrame(rows).drop_duplicates().to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out} ({len(rows)}행)")


if __name__ == "__main__":
    from common import config  # noqa: F401  (.env 로드)
    main(*sys.argv[1:3])
