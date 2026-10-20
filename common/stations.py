"""관측·예보 지점 (ASOS 지점번호, 이름, 계통, 위도, 경도).

태양광 설비가 많은 지역 위주 + 수도권 + 제주. 좌표는 ASOS 지점 위치(근삿값, 단기예보 5km 격자 변환용).
"""
import math

STATIONS = {
    108: ("서울", "육지", 37.5714, 126.9658),
    129: ("서산", "육지", 36.7766, 126.4939),
    133: ("대전", "육지", 36.3720, 127.3721),
    146: ("전주", "육지", 35.8215, 127.1550),
    156: ("광주", "육지", 35.1729, 126.8916),
    165: ("목포", "육지", 34.8169, 126.3812),
    168: ("여수", "육지", 34.7393, 127.7406),
    143: ("대구", "육지", 35.8780, 128.6530),
    136: ("안동", "육지", 36.5729, 128.7073),
    138: ("포항", "육지", 36.0320, 129.3800),
    192: ("진주", "육지", 35.1638, 128.0400),
    184: ("제주", "제주", 33.5141, 126.5297),
    185: ("고산", "제주", 33.2938, 126.1628),
    188: ("성산", "제주", 33.3868, 126.8802),
    189: ("서귀포", "제주", 33.2461, 126.5653),
}


# 단기예보 수집 지점 (최소 구성). 육지 7개는 11개 지점 평균 운량과 상관 0.994 (2024~, ASOS 실측 기준)
FORECAST_STATIONS = {
    "육지": [129, 146, 156, 165, 136, 143, 192],  # 서산, 전주, 광주, 목포, 안동, 대구, 진주
    "제주": [184, 185, 188, 189],
}


def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """위경도 → 기상청 단기예보 격자(nx, ny). 기상청 Lambert 정각원추도법 공식."""
    RE, GRID = 6371.00877, 5.0
    SLAT1, SLAT2, OLON, OLAT, XO, YO = 30.0, 60.0, 126.0, 38.0, 43, 136
    d = math.pi / 180.0
    re = RE / GRID
    slat1, slat2, olon, olat = SLAT1 * d, SLAT2 * d, OLON * d, OLAT * d
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(
        math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5))
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn * math.cos(slat1) / sn
    ro = re * sf / math.tan(math.pi * 0.25 + olat * 0.5) ** sn
    ra = re * sf / math.tan(math.pi * 0.25 + lat * d * 0.5) ** sn
    theta = lon * d - olon
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    theta *= sn
    nx = int(ra * math.sin(theta) + XO + 0.5)
    ny = int(ro - ra * math.cos(theta) + YO + 0.5)
    return nx, ny
