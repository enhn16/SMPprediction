"""전력거래소 주간 전력수요 전망 자동 수집 — 게시판 이미지(OCR) + 전력정보앱 텍스트(평일).

출처
- 게시판: https://www.kpx.or.kr/board.es?mid=a10109030100&bid=0014 (매주 마지막 근무일 게시, 2025-09 이후 표가 PNG 이미지)
  이미지 표: 대상 9일(토~다음 주 일)의 예측 최대부하(GW)·기온. → Windows 내장 OCR(한국어, 3배 확대)로 읽음
- 전력정보앱 주간 수급전망: https://www.kpx.or.kr/nationalPowerForecast.es?mid=a30501000000&device=mbl
  다음 주 평일 5일의 최대전력(MW)을 텍스트로 제공 → 평일 값은 이것을 우선(정확), OCR 검증에도 사용
공개 시점: 게시판 등록일(posted). 피처는 '등록 다음 날부터' 사용 (drop_classifier/src/outlook_features.py 규칙).
저장: data/raw/kpx_weekly_outlook/{list_no}.png, post_list.csv(목록 추가)
      data/processed/smp_forecast_app/kpx_outlook_auto.csv [list_no, format, written, target_date, peak_gw, posted, source]
실행: python -m common.collect.kpx_weekly_outlook            (새 게시물 수집)
      python -m common.collect.kpx_weekly_outlook --validate (과거 이미지 OCR 결과를 사람이 옮겨 적은 값과 비교)
      python -m common.collect.kpx_weekly_outlook --backfill [시작 등록일] [끝 등록일]  (과거 게시물 보강, 기본 2023년분)
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import requests

from common.config import PROCESSED_DIR, RAW_DIR, ROOT

BOARD = "https://www.kpx.or.kr/board.es?mid=a10109030100&bid=0014"
VIEW = "https://www.kpx.or.kr/board.es?mid=a10109030100&bid=0014&act=view&list_no={}"
APP = "https://www.kpx.or.kr/nationalPowerForecast.es?mid=a30501000000&device=mbl"
SRC = RAW_DIR / "kpx_weekly_outlook"
OUT = PROCESSED_DIR / "smp_forecast_app" / "kpx_outlook_auto.csv"
OCR_PS1 = ROOT / "smp_forecast_app" / "ops" / "ocr.ps1"


def ocr_lines(img: Path) -> list[str]:
    from PIL import Image, ImageOps
    im = Image.open(img).convert("L")
    im = ImageOps.autocontrast(im.resize((im.width * 3, im.height * 3), Image.LANCZOS))
    with tempfile.TemporaryDirectory(dir=PROCESSED_DIR) as td:   # 임시 파일도 프로젝트 안에
        p = Path(td) / "ocr.png"
        im.save(p)
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(OCR_PS1), str(p)],
                           capture_output=True, timeout=120)
    return r.stdout.decode("utf-8", "replace").splitlines()


def parse_peaks(lines: list[str], year: int) -> tuple[dict, str | None]:
    """'MM/DD(요일)' 줄 다음에 처음 나오는 50~110 사이 소수 한 자리 값을 그 날의 최대부하(GW)로 본다."""
    peaks, cur, written = {}, None, None
    for ln in lines:
        w = re.search(r"'?(\d{2})\.(\d{1,2})\.(\d{1,2})", ln)
        if w and written is None and "~" in ln or (w and "수요" in ln):
            written = f"20{w.group(1)}-{int(w.group(2)):02d}-{int(w.group(3)):02d}"
        d = re.match(r"\s*(\d{1,2})\s*/\s*(\d{1,2})", ln)
        if d:
            cur = (int(d.group(1)), int(d.group(2)))
            continue
        v = re.fullmatch(r"\s*(\d{2}\.\d)\s*", ln)
        if v and cur and cur not in peaks and 40 <= float(v.group(1)) <= 110:
            peaks[cur] = float(v.group(1))
    out = {}
    for (m, dd), val in peaks.items():
        y = year + 1 if (m == 1 and any(k[0] == 12 for k in peaks)) else year
        out[pd.Timestamp(y, m, dd)] = val
    return out, written


def app_weekdays(s) -> dict:
    t = s.get(APP, timeout=30).content.decode("utf-8", "replace")
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t))
    out = {}
    for mm, dd, _sup, mx in re.findall(r"(\d{1,2})\.(\d{1,2})\([월화수목금토일]\)\s*([\d,]+)\s+([\d,]+)", txt):
        out[(int(mm), int(dd))] = float(mx.replace(",", "")) / 1000
    return out


def collect():
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    pl_path = SRC / "post_list.csv"
    pl = pd.read_csv(pl_path)
    t = s.get(BOARD, timeout=30).content.decode("utf-8", "replace")
    # 링크 안 글자에서 태그를 걷어 내고 제목을 찾음 — 새 글에는 제목 앞에 '새글' 표시가 붙어 예전 정규식이 놓쳤음 (2026-10-27)
    posts = {}
    for n, inner in re.findall(r"list_no=(\d+)[^>]*>(.*?)</a>", t, re.S):
        m = re.search(r"(\d{4}년\s*\d{1,2}월\s*\d주\s*주간 전력수요 전망)", re.sub(r"<[^>]+>", " ", inner))
        if m:
            posts.setdefault(int(n), re.sub(r"\s+", " ", m.group(1)))
    new = {n: ttl for n, ttl in posts.items() if n not in set(pl["list_no"])}
    auto = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    app = app_weekdays(s)
    rows = []
    for n, ttl in sorted(new.items()):
        v = s.get(VIEW.format(n), timeout=30).content.decode("utf-8", "replace")
        posted = re.search(r"(\d{4})[./-](\d{2})[./-](\d{2})", v[v.find(ttl):] if ttl in v else v)
        posted = f"{posted.group(1)}/{posted.group(2)}/{posted.group(3)}" if posted else pd.Timestamp.today().strftime("%Y/%m/%d")
        link = re.search(r'href="([^"]*boardDownload\.es\?[^"]*list_no=' + str(n) + r'[^"]*)"', v)
        img = re.search(r'src="([^"]*(?:boardDownload|upload)[^"]*\.(?:png|jpg)[^"]*)"', v, re.I)
        url = (link or img).group(1).replace("&amp;", "&") if (link or img) else None
        if not url:
            print(f"{n} {ttl}: 첨부 못 찾음")
            continue
        url = url if url.startswith("http") else "https://www.kpx.or.kr" + url
        content = s.get(url, timeout=60).content
        ext = "png" if content[:4] == b"\x89PNG" else ("jpg" if content[:2] == b"\xff\xd8" else "bin")
        f = SRC / f"{n}.{ext}"
        f.write_bytes(content)
        year = int(re.search(r"(\d{4})년", ttl).group(1))
        peaks, written = parse_peaks(ocr_lines(f), year) if ext in ("png", "jpg") else ({}, None)
        if peaks:   # OCR이 놓친 날 중 앱 텍스트(평일)에 있는 날은 채움 — 같은 게시물 대상 기간 안만
            lo_d, hi_d = min(peaks), max(peaks)
            for (mm, dd), val in app.items():
                d = pd.Timestamp(year + (1 if mm < lo_d.month - 6 else 0), mm, dd)
                if lo_d <= d <= hi_d and d not in peaks:
                    peaks[d] = val
        for d, val in sorted(peaks.items()):
            src = "ocr"
            if (d.month, d.day) in app:
                if abs(app[(d.month, d.day)] - val) > 0.05:
                    print(f"  {d.date()} OCR {val} ≠ 앱 {app[(d.month, d.day)]} → 앱 값 사용")
                val, src = app[(d.month, d.day)], "앱 텍스트"
            rows.append({"list_no": n, "format": "auto", "written": written, "target_date": d.date(), "peak_gw": val,
                         "posted": pd.Timestamp(posted).date(), "source": src})
        pl = pd.concat([pl, pd.DataFrame([{"list_no": n, "title": ttl, "posted": posted, "attach": ext}])])
        print(f"{n} {ttl}: 등록 {posted}, {len(peaks)}일 판독")
    if rows:
        auto = pd.concat([auto, pd.DataFrame(rows)]).drop_duplicates(["list_no", "target_date"], keep="last")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        auto.to_csv(OUT, index=False, encoding="utf-8-sig")
        pl.sort_values("list_no", ascending=False).to_csv(pl_path, index=False, encoding="utf-8-sig")
    print(f"새 게시물 {len(new)}건, 추가 {len(rows)}행")


def mw_to_gw(t: str) -> str:
    """2023년 양식은 최대부하를 MW·천 단위 쉼표로 적음(예: 67,200) → '67.2'로 바꿔 기존 파서(GW 가정)가 읽게 함.
    쉼표가 없으면 그대로 (GW 양식)."""
    # 뒤에 숫자가 붙은 경우(예: '85,50072,000' — PDF 텍스트에서 칸 사이 공백이 빠짐)도 앞에서부터 끊어 읽음.
    # 한 번 바꾸면 다음 숫자 앞이 공백이 되므로 바뀌지 않을 때까지 되풀이
    while True:
        u = re.sub(r"(?<![\d.,])(\d{2,3}),(\d{3})", lambda m: f" {int(m.group(1) + m.group(2)) / 1000:.1f} ", t)
        if u == t:
            return t
        t = u


def fix_dates(rows: list[dict]) -> list[dict]:
    """표는 항상 연속 9일(토~다음 주 일). 문서 오타로 날짜가 거꾸로 가면(예: 4/22 다음 '4/13') 첫날부터 연속으로 고침."""
    if len(rows) >= 2:
        d0 = pd.Timestamp(rows[0]["target_date"])
        if any(pd.Timestamp(r["target_date"]) != d0 + pd.Timedelta(days=i) for i, r in enumerate(rows)):
            rows = [{**r, "target_date": (d0 + pd.Timedelta(days=i)).date()} for i, r in enumerate(rows)]
    return rows


def backfill(start="2022-12-25", end="2023-12-14", pages=range(12, 26)):
    """과거 게시물 보강 (2026-10-27, 팀 규칙 '2023년부터 학습'용): 게시판 여러 쪽에서 등록일이 start~end인 글을 받아
    PDF·HWPX는 기존 파서(drop_classifier.src.parse_weekly_outlook)로, PNG·JPG는 OCR로 읽어 자동분 파일에 덧붙인다.
    이미 표에 있는 글(list_no)은 건너뜀. 앱 텍스트는 과거분이 없어 OCR 값만 씀 → source로 구분."""
    from drop_classifier.src.parse_weekly_outlook import parse_text, tokens_hwpx, tokens_pdf
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    have = set(pd.read_csv(PROCESSED_DIR / "drop_classifier" / "kpx_weekly_outlook.csv")["list_no"])
    auto = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["list_no"])
    have |= set(auto["list_no"])
    posts = {}
    for pg in pages:
        t = s.get(f"{BOARD}&nPage={pg}", timeout=30).content.decode("utf-8", "replace")
        for n, inner, tail in re.findall(r"list_no=(\d+)[^>]*>(.*?)</a>(.{0,400})", t, re.S):
            m = re.search(r"(\d{4}년\s*\d{1,2}월\s*\d주\s*주간 전력수요 전망)", re.sub(r"<[^>]+>", " ", inner))
            d = re.search(r"(\d{4})/(\d{2})/(\d{2})", re.sub(r"<[^>]+>", " ", tail))
            if m and d:
                posts.setdefault(int(n), (re.sub(r"\s+", " ", m.group(1)), pd.Timestamp(f"{d.group(1)}-{d.group(2)}-{d.group(3)}")))
    todo = {n: v for n, v in posts.items() if pd.Timestamp(start) <= v[1] <= pd.Timestamp(end) and n not in have}
    print(f"대상 게시물 {len(todo)}건 (등록 {start} ~ {end}, 이미 있는 글 제외)", flush=True)
    rows, plist = [], []
    for n, (ttl, posted) in sorted(todo.items()):
        v = s.get(VIEW.format(n), timeout=30).content.decode("utf-8", "replace")
        links = [u.replace("&amp;", "&") for u in re.findall(r'href="([^"]*boardDownload\.es\?[^"]*)"', v)]
        got = None
        for u in links:   # 첨부가 여럿이면 텍스트로 읽히는 PDF·HWPX를 먼저
            c = s.get(u if u.startswith("http") else "https://www.kpx.or.kr" + u, timeout=60).content
            ext = ("pdf" if c[:4] == b"%PDF" else "hwpx" if c[:2] == b"PK" else "png" if c[:4] == b"\x89PNG"
                   else "jpg" if c[:2] == b"\xff\xd8" else "bin")
            if got is None or (ext in ("pdf", "hwpx") and got[0] not in ("pdf", "hwpx")):
                got = (ext, c)
        if not got or got[0] == "bin":
            print(f"{n} {ttl}: 읽을 첨부 없음", flush=True)
            continue
        ext, c = got
        f = SRC / f"{n}.{ext}"
        f.write_bytes(c)
        year = int(re.search(r"(\d{4})년", ttl).group(1))
        try:
            if ext in ("pdf", "hwpx"):
                rs = parse_text(mw_to_gw(tokens_pdf(f) if ext == "pdf" else tokens_hwpx(f)))
                got_rows = fix_dates([{"written": r["written"].date(), "target_date": r["target_date"].date(), "peak_gw": r["peak_gw"],
                                       "source": f"{ext} 텍스트(2023 보강)"} for r in rs])
            else:
                peaks, written = parse_peaks(ocr_lines(f), year)
                got_rows = [{"written": written, "target_date": d.date(), "peak_gw": val, "source": "ocr(2023 보강)"}
                            for d, val in sorted(peaks.items())]
        except Exception as e:  # noqa: BLE001
            print(f"{n} {ttl}: 판독 실패 {e!r}"[:160], flush=True)
            continue
        rows += [{"list_no": n, "format": ext, **r, "posted": posted.date()} for r in got_rows]
        plist.append({"list_no": n, "title": ttl, "posted": posted.strftime("%Y/%m/%d"), "attach": ext})
        print(f"{n} {ttl}: 등록 {posted.date()}, {ext}, {len(got_rows)}일", flush=True)
    if rows:
        auto = pd.concat([auto, pd.DataFrame(rows)]).drop_duplicates(["list_no", "target_date"], keep="last")
        auto.to_csv(OUT, index=False, encoding="utf-8-sig")
        pl_path = SRC / "post_list.csv"
        pl = pd.concat([pd.read_csv(pl_path), pd.DataFrame(plist)]).drop_duplicates("list_no", keep="last")
        pl.sort_values("list_no", ascending=False).to_csv(pl_path, index=False, encoding="utf-8-sig")
    print(f"보강 {len(plist)}건, {len(rows)}행", flush=True)


def validate():
    """과거 이미지(사람이 옮겨 적은 image_transcribed.csv)와 OCR 판독 비교."""
    ref = pd.read_csv(PROCESSED_DIR / "drop_classifier" / "image_transcribed.csv", parse_dates=["target_date"])
    pl = pd.read_csv(SRC / "post_list.csv")
    rows = []
    for n, g in ref.groupby("list_no"):
        f = next((p for p in (SRC / f"{n}.png", SRC / f"{n}_1.png", SRC / f"{n}.jpg") if p.exists()), None)
        if f is None:
            continue
        ttl = pl.loc[pl["list_no"] == n, "title"]
        year = int(re.search(r"(\d{4})년", ttl.iloc[0]).group(1)) if len(ttl) else g["target_date"].dt.year.min()
        peaks, _ = parse_peaks(ocr_lines(f), year)
        for r in g.itertuples():
            o = peaks.get(r.target_date)
            rows.append({"list_no": n, "target_date": r.target_date.date(), "사람": r.peak_gw, "OCR": o,
                         "일치": o is not None and abs(o - r.peak_gw) < 0.05})
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "smp_forecast_app" / "outputs" / "tables" / "O1_outlook_ocr_validation.csv", index=False, encoding="utf-8-sig")
    print(f"이미지 {d.list_no.nunique()}건, {len(d)}일: 일치 {d['일치'].mean():.1%}, 못 읽음 {d['OCR'].isna().mean():.1%}, "
          f"잘못 읽음 {((~d['일치']) & d['OCR'].notna()).mean():.1%}")
    print(d[(~d["일치"]) & d["OCR"].notna()].head(15).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--validate"]:
        validate()
    elif sys.argv[1:2] == ["--backfill"]:
        backfill(*sys.argv[2:4])
    else:
        collect()
