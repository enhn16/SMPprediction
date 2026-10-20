"""날짜별 요일유형·연휴 정보 (모든 과제 공용).

정의
- 휴무일: 토·일 또는 공휴일(대체·임시공휴일, 선거일 포함)
- 연속 휴무일 묶음(block) 길이가 3일 이상이면 '연휴'
- 연휴가 아닌 공휴일(예: 수요일 하루 공휴일)은 '공휴일'
- 연휴가 아닌 토·일은 '주말', 나머지는 '평일'
- 명절: 연휴 묶음 안에 설날·추석이 들어 있으면 그 묶음 전체
- 근로자의날(5/1)은 2025년까지 공휴일이 아니지만 쉬는 사업장이 많아 별도 표시 (2026년부터 '노동절' 공휴일)
"""
import pandas as pd

from common.loaders import load_holidays

WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]


def build_calendar(start: str, end: str) -> pd.DataFrame:
    # 앞뒤로 여유를 두고 계산해야 기간 경계에 걸친 연휴 길이가 맞음
    pad = pd.Timedelta(days=10)
    dates = pd.date_range(pd.Timestamp(start) - pad, pd.Timestamp(end) + pad)
    cal = pd.DataFrame({"date": dates})
    hol = load_holidays()
    cal = cal.merge(hol.rename(columns={"name": "holiday_name"}), on="date", how="left")

    cal["weekday"] = cal["date"].dt.weekday
    cal["weekday_ko"] = cal["weekday"].map(dict(enumerate(WEEKDAY_KO)))
    cal["is_weekend"] = cal["weekday"] >= 5
    cal["is_holiday"] = cal["holiday_name"].notna()
    cal["is_offday"] = cal["is_weekend"] | cal["is_holiday"]

    # 연속 휴무일 묶음 번호, 길이, 며칠째
    block_id = (cal["is_offday"] != cal["is_offday"].shift()).cumsum()
    cal["off_block_len"] = cal.groupby(block_id)["is_offday"].transform("size").where(cal["is_offday"], 0)
    cal["off_block_day"] = (cal.groupby(block_id).cumcount() + 1).where(cal["is_offday"], 0)

    is_long = cal["off_block_len"] >= 3
    cal["day_type"] = "평일"
    cal.loc[cal["is_weekend"], "day_type"] = "주말"
    cal.loc[cal["is_holiday"], "day_type"] = "공휴일"
    cal.loc[is_long, "day_type"] = "연휴"

    has_myeongjeol = cal["holiday_name"].fillna("").str.contains("설날|추석")
    cal["is_myeongjeol"] = has_myeongjeol.groupby(block_id).transform("any") & cal["is_offday"]
    cal["is_labor_day"] = (cal["date"].dt.month == 5) & (cal["date"].dt.day == 1) & ~cal["is_holiday"]

    cal = cal[(cal["date"] >= pd.Timestamp(start)) & (cal["date"] <= pd.Timestamp(end))]
    return cal.reset_index(drop=True)
