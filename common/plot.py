"""matplotlib 한글 폰트 공통 설정. 모든 그래프 코드에서 먼저 setup_korean_font()를 호출한다."""
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 우선순위: Windows → macOS → Linux(나눔)
_CANDIDATES = ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic"]


def setup_korean_font() -> str:
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in _CANDIDATES:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
            return name
    raise RuntimeError(f"한글 폰트를 찾지 못했습니다. 후보: {_CANDIDATES}")
