"""프로젝트 공통 설정: 경로, 시드, API 키."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
API_RAW_DIR = RAW_DIR / "api"  # API로 수집한 원본 (수정 금지)
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42

load_dotenv(ROOT / ".env")


def get_api_key() -> str:
    """공공데이터포털 인증키를 .env에서 읽는다."""
    key = os.getenv("DATA_GO_KR_API_KEY")
    if not key:
        raise RuntimeError(".env에 DATA_GO_KR_API_KEY가 없습니다.")
    return key
