"""전력거래소 주간 전력수요 전망 게시물(data/raw/kpx_weekly_outlook) → 표.

게시물마다 '작성일'과 대상 9일의 '예측 최대부하(GW)', 전국 가중 최고·최저기온(℃)을 뽑는다.
- HWPX(한글) · PDF: 텍스트 추출로 자동 처리
- PNG · JPG: 이미지라 따로 옮겨 적은 표(`image_transcribed.csv`)를 합친다
공개 시점은 게시판 '등록일'(post_list.csv)로 본다 (작성일보다 늦게 올라오는 경우가 있음).

실행: python -m drop_classifier.src.parse_weekly_outlook
저장: data/processed/drop_classifier/kpx_weekly_outlook.csv
"""
import re
import zipfile

import pandas as pd
from pypdf import PdfReader

from common.config import PROCESSED_DIR, RAW_DIR

SRC = RAW_DIR / "kpx_weekly_outlook"
OUT = PROCESSED_DIR / "drop_classifier" / "kpx_weekly_outlook.csv"
NUM = r"-?\d+(?:\.\d+)?"


def tokens_hwpx(p) -> str:
    x = zipfile.ZipFile(p).read("Contents/section0.xml").decode("utf-8")
    return " ".join(re.findall(r"<hp:t>([^<]*)</hp:t>", x))


def tokens_pdf(p) -> str:
    return " ".join(page.extract_text() for page in PdfReader(p).pages)


def parse_text(t: str) -> list[dict]:
    # 작성일: '24.03.22 / ‘25.1.3. 등 (월·일 한 자리 허용)
    w = re.search(r"(\d{2})\.\s*(\d{1,2})\.\s*(\d{1,2})", t)
    written = pd.Timestamp(2000 + int(w.group(1)), int(w.group(2)), int(w.group(3)))
    # 작성일 뒤부터 '날짜(MM/DD)'와 '숫자'를 차례로 읽어, 처음 연달아 나오는 날짜 n개 → 바로 뒤 숫자 n개
    toks = re.findall(r"\d{1,2}/\d{1,2}|-?\d+(?:\.\d+)?", t[w.end():])
    dates, peaks = [], []
    for tok in toks:
        if "/" in tok:
            if peaks:
                break
            dates.append(tok)
        elif dates:
            peaks.append(float(tok))
            if len(peaks) == len(dates):
                break
    rows = []
    for i, md in enumerate(dates):
        mo, d = map(int, md.split("/"))
        y = written.year + (1 if mo < written.month - 6 else 0)
        v = peaks[i] if i < len(peaks) else None
        rows.append({"written": written, "target_date": pd.Timestamp(y, mo, d),
                     "peak_gw": v if v is not None and 30 <= v <= 110 else None})
    return rows


def main():
    posts = pd.read_csv(SRC / "post_list.csv")
    posts["posted"] = pd.to_datetime(posts["posted"])
    rows, fails = [], []
    for p in sorted(SRC.glob("*")):
        if p.suffix not in (".hwpx", ".pdf"):
            continue
        ln = int(p.stem.split("_")[0])
        try:
            t = tokens_hwpx(p) if p.suffix == ".hwpx" else tokens_pdf(p)
            for r in parse_text(t):
                rows.append({"list_no": ln, "format": p.suffix[1:], **r})
        except Exception as e:  # 양식이 다른 문서는 따로 확인
            fails.append((p.name, repr(e)[:80]))
    df = pd.DataFrame(rows)
    img = SRC.parent.parent / "processed" / "drop_classifier" / "image_transcribed.csv"
    if img.exists():
        im = pd.read_csv(img, parse_dates=["written", "target_date"])
        df = pd.concat([df, im.assign(format="image")], ignore_index=True)
    df = df.merge(posts[["list_no", "posted"]], on="list_no", how="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["written", "target_date"]).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} ({len(df)}행, 게시물 {df['list_no'].nunique()}건)")
    print("형식별 게시물 수:", df.groupby("format")["list_no"].nunique().to_dict())
    print("최대부하 결측:", int(df["peak_gw"].isna().sum()))
    if fails:
        print("실패:", fails)


if __name__ == "__main__":
    main()
