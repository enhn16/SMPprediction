"""한국가스공사 발전용 천연가스 요금 수집 (홈페이지 > 요금안내 > 발전용, 연도별 엑셀).

요금표는 연도별 엑셀 하나에 월별로 쌓이고, 다음 달 요금이 전달 26~27일쯤 같은 파일에 추가된다
(파일 수정 시각: 2022-11-29, 2024-11-27, 2025-11-26, 2026-10-27 17:57).
매일 돌려도 되며, 내용이 바뀐 때만 저장하고 새로 보인 달을 first_seen.csv에 남긴다 (게시 시점 기록 = 누수 점검용).

실행: python -m common.collect.kogas_gas_tariff [시작연도]     (기본: 올해와 작년, 12월이면 내년 파일도 시도)
저장: data/raw/api/kogas_gas_tariff/{연도}.xlsx (원본), first_seen.csv [month, won_gj, first_seen, file_modified]
읽기: common.loaders.load_gas_tariff
"""
import io
import re
import sys
import zipfile
from datetime import date, datetime
from urllib.parse import quote

import pandas as pd
import requests

from common.config import API_RAW_DIR

PAGE = "https://www.kogas.or.kr/site/koGas/1040402000000"
BASE = "https://www.kogas.or.kr"
OUT_DIR = API_RAW_DIR / "kogas_gas_tariff"
SEEN = OUT_DIR / "first_seen.csv"


def parse(content: bytes, year: int) -> dict:
    """연도 엑셀 → {YYYY-MM: 일반발전 합계 요금(원/GJ)}."""
    x = pd.read_excel(io.BytesIO(content), header=None)
    mon, kind = x.iloc[2].ffill(), x.iloc[3]
    gj = x.index[x[0].astype(str).str.contains("원/GJ")][0]
    out = {}
    for j in range(2, x.shape[1]):
        m = str(mon[j]).replace("월", "").strip()
        v = pd.to_numeric(x.iloc[gj + 2, j], errors="coerce")
        if m.isdigit() and kind[j] == "일반발전" and v > 0:
            out[f"{year}-{int(m):02d}"] = float(v)
    return out


def modified(content: bytes) -> str:
    """엑셀 docProps/core.xml의 마지막 수정 시각 (UTC 그대로)."""
    try:
        core = zipfile.ZipFile(io.BytesIO(content)).read("docProps/core.xml").decode("utf-8")
        return re.search(r"<dcterms:modified[^>]*>([^<]+)<", core).group(1)
    except Exception:
        return ""


def main(start: int | None = None):
    today = date.today()
    years = list(range(start or today.year - 1, today.year + 1)) + ([today.year + 1] if today.month == 12 else [])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    page = s.get(PAGE, timeout=60)  # 세션 쿠키 (없으면 빈 HTML이 내려옴)
    page.raise_for_status()
    links = dict(re.findall(r"fileDownload\.do\?REAL_NAME=(\d{4})\.gas\.xlsx&(?:amp;)?fileName=([^\"'&<]+\.xlsx)", page.text))
    seen = pd.read_csv(SEEN, dtype=str) if SEEN.exists() else pd.DataFrame(columns=["month", "won_gj", "first_seen", "file_modified"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    new = []
    for y in years:
        if str(y) not in links:
            print(f"{y}: 게시된 요금표 없음")
            continue
        url = f"{BASE}/fileDownload.do?REAL_NAME={y}.gas.xlsx&fileName={quote(links[str(y)])}"
        r = s.get(url, headers={"Referer": PAGE}, timeout=120)
        r.raise_for_status()
        if not r.content.startswith(b"PK"):
            raise RuntimeError(f"{y}: 엑셀이 아닌 응답 ({r.headers.get('Content-Type')}) — 페이지 구조 변경 확인 필요")
        vals = parse(r.content, y)
        p = OUT_DIR / f"{y}.xlsx"
        if not p.exists() or p.read_bytes() != r.content:
            p.write_bytes(r.content)
            print(f"{y}: 저장 ({len(vals)}개월, 마지막 {max(vals) if vals else '-'})")
        else:
            print(f"{y}: 변경 없음 ({len(vals)}개월)")
        mod = modified(r.content)
        for m, v in sorted(vals.items()):
            if m not in set(seen["month"]):
                new.append({"month": m, "won_gj": f"{v:.2f}", "first_seen": now, "file_modified": mod})
    if new:
        seen = pd.concat([seen, pd.DataFrame(new)], ignore_index=True).sort_values("month")
        seen.to_csv(SEEN, index=False, encoding="utf-8-sig")
        print(f"새로 보인 달: {', '.join(n['month'] for n in new)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
