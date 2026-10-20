"""매일 수집 작업(작업 스케줄러 'SMP_daily_collect')의 완료 여부 확인.

완료 조건
- 연료원별 SMP 결정 횟수: 2024-04 ~ 2024-12, 2026-01 ~ 2026-09 월 파일이 모두 있음
  (2025년은 사용자 제공 파일로 대신, 2026-10은 진행 중인 달이라 제외)
- 발전원별 발전량: target_dates.txt의 날짜가 모두 저장됐거나 '데이터 없음'으로 기록됨
완료면 종료 코드 0, 아니면 1. collect_daily.bat이 0일 때 작업을 삭제한다.

실행: python -m common.collect.check_done
"""
import sys

import pandas as pd

from common.collect.gen_by_fuel import OUT_DIR as GEN_DIR, have_dates
from common.config import API_RAW_DIR


def main() -> int:
    months = [p.strftime("%Y%m") for p in pd.period_range("2024-04", "2024-12", freq="M")] + \
             [p.strftime("%Y%m") for p in pd.period_range("2026-01", "2026-09", freq="M")]
    fuel_missing = [m for m in months if not (API_RAW_DIR / "smp_fuel" / f"smp_fuel_{m}.csv").exists()]
    targets = set((GEN_DIR / "target_dates.txt").read_text().split())
    gen_missing = sorted(targets - have_dates())
    print(f"[완료 확인] 연료원 남은 달 {len(fuel_missing)}개 {fuel_missing[:4]} / 발전원별 남은 날짜 {len(gen_missing)}개",
          flush=True)
    return 0 if not fuel_missing and not gen_missing else 1


if __name__ == "__main__":
    sys.exit(main())
