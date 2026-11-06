"""발표용 그림: 분기별 성적 (구간 단위 성공·부분·실패 비율), 급락 / 급등(B 새 레이블).

입력: outputs/tables/P1_by_quarter.csv (python -m extended_backtest.src.period_table)
출력: outputs/figures/F1_quarterly.png
색: 성공(경보) 파랑 550, 부분(주의) 파랑 250 (순서형 한 색, 검사기 통과), 실패 회색. 축 하나, 100% 누적 막대.
"""
import re

import matplotlib.pyplot as plt
import pandas as pd

from common.config import ROOT
from common.plot import setup_korean_font

TAB = ROOT / "extended_backtest" / "outputs" / "tables"
FIG = ROOT / "extended_backtest" / "outputs" / "figures"
C_OK, C_PART, C_FAIL = "#1c5cab", "#86b6ef", "#d3d1ca"
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
PANELS = [("육지 급락", "LSTM + LightGBM 평균", "급락"), ("육지 급등", "B 새 레이블 (대표)", "급등")]


def n_of(s):
    m = re.match(r"(\d+)", str(s))
    return int(m.group(1)) if m else 0


def main():
    setup_korean_font()
    p = pd.read_csv(TAB / "P1_by_quarter.csv")
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), facecolor=SURF)
    for ax, (event, model, short) in zip(axes, PANELS):
        x = p[(p["사건"] == event) & (p["모델"] == model)].reset_index(drop=True)
        n = x["구간 수"].values
        ok, part, fail = (x[c].map(n_of).values for c in ("성공(경보)", "부분(주의)", "실패"))
        safe = n.clip(min=1)
        ok_r, part_r, fail_r = ok / safe * 100, part / safe * 100, fail / safe * 100
        idx = range(len(x))
        kw = dict(width=0.62, edgecolor=SURF, linewidth=2)
        ax.bar(idx, ok_r, color=C_OK, label="성공 (경보)", **kw)
        ax.bar(idx, part_r, bottom=ok_r, color=C_PART, label="부분 (주의만)", **kw)
        ax.bar(idx, fail_r, bottom=ok_r + part_r, color=C_FAIL, label="실패 (없음)", **kw)
        for i in idx:
            if n[i] == 0:
                ax.text(i, 50, "사건 없음", ha="center", va="center", color=MUTED, fontsize=10)
                continue
            ax.text(i, 103, f"{n[i]}구간", ha="center", va="bottom", color=INK2, fontsize=9.5)
            caught = ok_r[i] + part_r[i]
            if caught == 0:
                ax.text(i, 50, "0%", ha="center", va="center", color=INK, fontsize=10, fontweight="bold")
                continue
            on_dark = part_r[i] < 8          # 글자가 짙은 파랑(성공) 위에 놓이면 흰색, 옅은 파랑 위면 검정
            ax.text(i, caught - 3, f"{caught:.0f}%", ha="center", va="top",
                    color="white" if on_dark else INK, fontsize=10, fontweight="bold")
        one = lambda t: t.split("~")[0] if len(set(t.split("~"))) == 1 else t   # 42~42 → 42
        labels = [f"{q.replace(' ', chr(10), 1)}\n학습 {one(t)}일" for q, t in zip(x["분기"], x["학습한 사건일"])]
        labels = [l.replace("2026\n4분기", "2026\n4분기(10월)") for l in labels]
        ax.set_xticks(list(idx), labels, fontsize=9, color=INK2)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=INK2)
        ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(length=0)
        ax.set_facecolor(SURF)
        ax.set_title(f"{event} — {model}: 분기별 구간 판정 (막대 안 숫자 = 주의 이상으로 잡은 비율)",
                     loc="left", fontsize=11.5, color=INK, pad=14)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.965), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("매달 다시 학습한 모델의 시기별 성적 (시험 2025-01 ~ 2026-10, 전날 15시 판단, 2026 = 2차 시험)",
                 x=0.01, ha="left", fontsize=13, color=INK, y=0.995)
    fig.text(0.01, 0.005, "학습 ○일 = 그 분기 모델이 학습 때 본 사건일 수(2020-01부터). 문턱은 매달 앞 12개월 예측으로 다시 정함. "
             "급등 경보 = 정밀도 75% 문턱.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93), h_pad=3)
    out = FIG / "F1_quarterly.png"
    fig.savefig(out, dpi=180, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    main()
