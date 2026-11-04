"""2020~2022 과거 자료 보강 (2026-11-04, 급락·급등 채점 기간 연장 시험용).

운영 모델이 읽는 폴더와 섞이지 않도록 data/raw/backfill_2020/ 아래에 따로 저장한다 (로더 미사용).
기존 수집기를 그대로 쓰고 저장 폴더만 바꾼다.

실행: python -m common.collect.backfill_2020 [작업 ...]
작업: smp holidays afs fuel openmeteo  (인자 없으면 전부)
"""
import re
import sys
import time

import pandas as pd
import requests

from common.config import RAW_DIR

BK = RAW_DIR / "backfill_2020"


def smp():
    """① 시간별 SMP + ② 하루전 수요예측 (data.go.kr, 2020-01-01 ~ 2021-12-31). 2022 이후는 기존 폴더에 있으므로 지움."""
    from common.collect import smp_forecast as m
    m.OUT_DIR = BK / "smp_forecast"
    m.main("20200101")
    for f in m.OUT_DIR.glob("smp_forecast_*.csv"):
        if f.stem[-6:] >= "202201":
            f.unlink()


def holidays():
    """④ 특일정보 2019~2021 (2020-01 초 연휴 판단에 2019 말 필요)."""
    from common.collect import holidays as m
    m.API_RAW_DIR = BK
    m.main(2019, 2021)


def afs():
    """⑧ 단기 구역별 육상예보 (기상청 API허브 fct_afs_dl), 육지 7개 지점, 2019-12 ~ 2022-12."""
    from common.collect import kma_regional_fcst as m
    m.OUT_DIR = BK / "kma_afs"
    m.main("201912", "202212", "육지")


def fuel():
    """⑥ 연료비단가 (EPSIS), 2019-10 ~ 2022-12 (M월 값은 M+2월부터라 2020-01용으로 2019-11 필요)."""
    from common.collect import epsis_fuel_cost as m
    m.OUT_DIR = BK / "epsis_fuel_cost"
    m.main("201910", "202212")


def openmeteo():
    """⑩ Open-Meteo 2일 전 실행 예보, 육지 7개 도시, 연도별 (best_match 기온·운량·강수·일사 = 급등 입력, JMA 일사·운량).
    best_match는 2022-06부터 기온만, 2022-12부터 전 변수. JMA는 2020-12 무렵부터 기온·운량·강수(일사 없음)."""
    from common.collect.openmeteo_prevruns import CITIES, URL
    lat = ",".join(str(v[0]) for v in CITIES.values())
    lon = ",".join(str(v[1]) for v in CITIES.values())
    jobs = {"best": (None, ("temperature_2m", "cloud_cover", "precipitation", "shortwave_radiation")),
            "jma": ("jma_seamless", ("temperature_2m", "cloud_cover", "precipitation", "shortwave_radiation"))}
    for name, (model, vars_) in jobs.items():
        d = BK / f"openmeteo_{name}"
        d.mkdir(parents=True, exist_ok=True)
        for y in (2020, 2021, 2022, 2023):
            out = d / f"prev2_{name}_{y}.json"
            if out.exists():
                continue
            p = {"latitude": lat, "longitude": lon, "hourly": ",".join(f"{v}_previous_day2" for v in vars_),
                 "start_date": f"{y}-01-01", "end_date": f"{y}-12-31", "timezone": "Asia/Seoul"}
            if model:
                p["models"] = model
            for attempt in range(3):
                r = requests.get(URL, params=p, timeout=600)
                if r.ok:
                    break
                print(f"Open-Meteo {name} {y}: {r.status_code} {r.text[:120]}", flush=True)
                time.sleep(60)
            r.raise_for_status()
            out.write_text(r.text, encoding="utf-8")
            h = r.json()[0]["hourly"]
            got = {v: sum(x is not None for x in h[f"{v}_previous_day2"]) for v in vars_}
            print(f"Open-Meteo {name} {y}: {len(h['time'])}시간, 값 있음 {got}", flush=True)
            time.sleep(5)


def _peaks(text: str, posted: pd.Timestamp) -> list[dict]:
    """'2. 최대부하 전망' 표: 날짜 9개(m/d(요일)) + '예측 최대부하' 값 9개 → [target_date, peak_gw].
    단위가 해마다 다름: 만㎾(6,300) / MW(63,000) / GW(63.0) → 크기로 판별해 GW로."""
    i = text.find("최대부하 전망")
    if i < 0:
        return []
    sec = text[i:]
    ds = re.findall(r"(\d{1,2})/(\d{1,2})\s*\(", sec)[:9]
    j = sec.find("최대부하", 4)
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d{1,3}(?:,\d{3})+|\d{2,3}\.\d", sec[j:])][:9] if j >= 0 else []
    if len(ds) < 7 or len(nums) < len(ds):
        return []
    out = []
    for (m, d), v in zip(ds, nums):
        y = posted.year + (1 if posted.month == 12 and int(m) == 1 else 0)
        gw = v / 100 if 1000 <= v < 20000 else v / 1000 if v >= 20000 else v
        out.append({"target_date": pd.Timestamp(y, int(m), int(d)).date(), "peak_gw": round(gw, 2)})
    return out


def weekly(start="2019-12-01", end="2022-12-24", pages=range(18, 44)):
    """⑤ 주간 전력수요전망: 전력거래소 게시판 과거 글 (2019-12 ~ 2022-12, 기존 표는 2022-12-25 게시분부터).
    2021-10 이전은 HWP(옛 한글) → 미리보기 텍스트(PrvText), 이후는 PDF 텍스트. 게시 다음 날부터 사용."""
    import olefile
    from drop_classifier.src.parse_weekly_outlook import tokens_pdf
    from common.collect.kpx_weekly_outlook import BOARD, VIEW
    d = BK / "kpx_weekly_outlook"
    d.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    posts = {}
    for pg in pages:
        t = s.get(f"{BOARD}&nPage={pg}", timeout=30).content.decode("utf-8", "replace")
        # 목록 한 줄 = 다음 'list_no=' 앞까지 (줄마다 길이가 달라 고정 길이로 자르면 줄을 놓침)
        cuts = [m.start() for m in re.finditer(r"list_no=\d+", t)] + [len(t)]
        for a, b in zip(cuts, cuts[1:]):
            m = re.match(r"list_no=(\d+)[^>]*>(.*?)</a>", t[a:b], re.S)
            if not m:
                continue
            n, ttl = m.group(1), re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(2))).strip()
            dt = re.search(r"(\d{4})/(\d{2})/(\d{2})", re.sub(r"<[^>]+>", " ", t[a + m.end():b]))
            if dt and "전망" in ttl:
                posts[int(n)] = (ttl, pd.Timestamp("-".join(dt.groups())))
    todo = {n: v for n, v in posts.items() if pd.Timestamp(start) <= v[1] <= pd.Timestamp(end)}
    print(f"게시물 {len(todo)}건", flush=True)
    rows, fails = [], []
    for n, (ttl, posted) in sorted(todo.items()):
        v = s.get(VIEW.format(n), timeout=30).content.decode("utf-8", "replace")
        links = [u.replace("&amp;", "&") for u in re.findall(r'href="([^"]*boardDownload\.es\?[^"]*)"', v)]
        got = []
        for u in links:
            f = next(d.glob(f"{n}_{links.index(u)}.*"), None)
            if f is None:
                c = s.get(u if u.startswith("http") else "https://www.kpx.or.kr" + u, timeout=60).content
                ext = ("pdf" if c[:4] == b"%PDF" else "hwp" if c[:4] == b"\xd0\xcf\x11\xe0" else "png" if c[:4] == b"\x89PNG" else "bin")
                f = d / f"{n}_{links.index(u)}.{ext}"
                f.write_bytes(c)
            try:
                if f.suffix == ".hwp":
                    got = _peaks(olefile.OleFileIO(str(f)).openstream("PrvText").read().decode("utf-16-le", "replace"), posted)
                elif f.suffix == ".pdf":
                    got = _peaks(tokens_pdf(f), posted)
            except Exception as e:  # noqa: BLE001
                print(f"{n} {f.name}: {e!r}"[:150], flush=True)
            if got:
                rows += [{"list_no": n, "title": ttl, "posted": posted.date(), "source": f.suffix[1:], **r} for r in got]
                break
        if not got:
            fails.append((n, ttl, posted.date()))
        time.sleep(0.3)
    out = pd.DataFrame(rows)
    out.to_csv(BK / "weekly_outlook_2020_2022.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fails, columns=["list_no", "title", "posted"]).to_csv(BK / "weekly_outlook_failed.csv", index=False, encoding="utf-8-sig")
    print(f"주간전망: 읽음 {out['list_no'].nunique() if len(out) else 0}건 {len(out)}행, 못 읽음 {len(fails)}건", flush=True)


if __name__ == "__main__":
    for job in (sys.argv[1:] or ["holidays", "fuel", "afs", "openmeteo", "smp", "weekly"]):
        print(f"== {job}", flush=True)
        try:
            globals()[job]()
        except Exception as e:  # noqa: BLE001 — 한 작업이 실패해도 나머지는 진행
            print(f"{job} 실패: {e!r}"[:300], flush=True)
