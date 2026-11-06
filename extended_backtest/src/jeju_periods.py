"""제주 급락: 학습 기간(매달 늘어남)에 따라 무엇이 달라지나 — 고른 모델(S5P·LGB) 기준, 채점 뒤 해석용.

1) 달별 표: 그 달 모델의 학습 기간·학습 사건일 수, 1단계 잉여일 수, 문턱(주의·경보)과 문턱을 정한 근거,
   그 달 결과(사건·성공·부분·실패·헛경보, 주의 이상 비율)
2) 학습 사건일 구간별 성적 (같은 모델을 학습 자료 양으로 나눠 봄)
3) 주의 문턱이 낮아진 이유: 문턱을 정한 '지난 사건일'의 점수와, 그 예측을 낸 모델이 본 사건일 수
4) 회사 운영 채점과 같은 기간(2026-08-11 ~ 10-31) 결과
저장: outputs/tables/J6_by_month_jeju.csv, J6_by_train_size_jeju.csv, J6_caution_basis_jeju.csv, J6_company_period_jeju.csv,
      outputs/figures/J6_by_month_jeju.png
실행(루트에서): python -m extended_backtest.src.jeju_periods
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from extended_backtest.src.jeju_models import FEAT, OUT as PRED, TRAIN0
from extended_backtest.src.jeju_select import SEL, recall_threshold

TAB = Path(__file__).resolve().parents[1] / "outputs" / "tables"
FIG = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def main():
    pick = json.loads((TAB / "J2_pick_jeju.json").read_text(encoding="utf-8"))["선택"]
    p = pd.read_csv(PRED, parse_dates=["date"])
    p["score"] = p[pick]
    d = pd.read_csv(FEAT, parse_dates=["date"])
    g = pd.read_csv(TAB / "J3_daily_jeju.csv", parse_dates=["date"])
    ev = d.set_index("date")["event"]
    sur = d.set_index("date")["surplus"]

    # 예측한 달마다: 그 모델이 본 학습 사건일 수
    p["month"] = p["date"].dt.to_period("M")
    seen = {m: int(ev[(ev.index >= TRAIN0) & (ev.index < m.start_time)].sum()) for m in p["month"].unique()}
    p["train_events"] = p["month"].map(seen)

    rows = []
    for m, x in p.groupby("month"):
        r = {"예측 달": str(m), "학습 기간(0원 모델)": f"2024-01 ~ {(m - 1)}",
             "학습 날수": int(((ev.index >= TRAIN0) & (ev.index < m.start_time)).sum()),
             "학습 사건일": seen[m],
             "1단계 잉여일": int(sur[(sur.index < min(m.start_time, pd.Timestamp("2024-06-01")))].sum()),
             "사건일": int(x["event"].sum()), "구간": int(x["n_intervals"].sum()),
             "사건일 점수 중앙": x.loc[x["event"] == 1, "score"].median(), "평소 날 점수 중앙": x.loc[x["event"] == 0, "score"].median()}
        gm = g[g["date"].dt.to_period("M") == m]
        if len(gm):
            e1, e0 = gm[gm["event"] == 1], gm[gm["event"] == 0]
            r.update({"주의 문턱": gm["caution"].iloc[0], "경보 문턱": gm["alarm"].iloc[0],
                      "문턱 근거 예측 수(지난 사건일)": int(p[(p["date"] >= SEL[0]) & (p["date"] < m.start_time)]["event"].sum()),
                      "성공": int(e1.loc[e1["grade"] == "경보", "n_intervals"].sum()),
                      "부분": int(e1.loc[e1["grade"] == "주의", "n_intervals"].sum()),
                      "실패": int(e1.loc[e1["grade"] == "없음", "n_intervals"].sum()),
                      "헛경보 주의만": int((e0["grade"] == "주의").sum()), "헛경보 경보": int((e0["grade"] == "경보").sum()),
                      "주의 이상 비율": round((gm["grade"] != "없음").mean(), 2)})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "J6_by_month_jeju.csv", index=False, encoding="utf-8-sig")
    print(t.round(3).to_string(index=False))

    # 2) 학습 사건일 구간별 (예측 전체 2024-03 ~ 2026-10, 점수 자체의 분리력)
    bins = [-1, 5, 20, 60, 100, 200]
    lab = ["0~5", "6~20", "21~60", "61~100", "101~"]
    p["학습 사건일 구간"] = pd.cut(p["train_events"], bins, labels=lab)
    rows = []
    for b, x in p.groupby("학습 사건일 구간", observed=True):
        rows.append({"학습 사건일": b, "예측 달": f"{x['month'].min()} ~ {x['month'].max()}", "날": len(x),
                     "사건일": int(x["event"].sum()),
                     "ROC-AUC": roc_auc_score(x["event"], x["score"]) if x["event"].nunique() > 1 else np.nan,
                     "PR-AUC": average_precision_score(x["event"], x["score"]) if x["event"].sum() else np.nan,
                     "사건일 점수 중앙": x.loc[x["event"] == 1, "score"].median(),
                     "사건일 점수 최저": x.loc[x["event"] == 1, "score"].min(),
                     "평소 날 점수 상위10%": x.loc[x["event"] == 0, "score"].quantile(0.9)})
    s = pd.DataFrame(rows)
    s.round(4).to_csv(TAB / "J6_by_train_size_jeju.csv", index=False, encoding="utf-8-sig")
    print(s.round(3).to_string(index=False))

    # 3) 2025-01 주의 문턱의 근거: 2024-03 ~ 12 사건일 점수와 그 예측 모델이 본 사건일
    past = p[(p["date"] >= SEL[0]) & (p["date"] < "2025-01-01") & (p["event"] == 1)]
    b = past[["date", "score", "train_events"]].sort_values("score").reset_index(drop=True)
    b["90% 포착 문턱"] = recall_threshold(past["score"].values, 0.9)
    b.round(4).to_csv(TAB / "J6_caution_basis_jeju.csv", index=False, encoding="utf-8-sig")
    print(b.round(4).to_string(index=False))

    # 4) 회사 운영 채점과 같은 기간
    c = g[(g["date"] >= "2026-08-11") & (g["date"] <= "2026-10-31")]
    rows = []
    for labn, x in [("휴일", c[c["offday"] == 1]), ("평일", c[c["offday"] == 0]), ("계", c)]:
        e1, e0 = x[x["event"] == 1], x[x["event"] == 0]
        rows.append({"기간": "2026-08-11 ~ 10-31 (회사 운영 기록 기간, 2차 시험)", "구분": labn, "구간 수": int(e1["n_intervals"].sum()),
                     "성공(경보)": int(e1.loc[e1["grade"] == "경보", "n_intervals"].sum()),
                     "부분(주의만)": int(e1.loc[e1["grade"] == "주의", "n_intervals"].sum()),
                     "실패": int(e1.loc[e1["grade"] == "없음", "n_intervals"].sum()),
                     "헛경보 주의만": int((e0["grade"] == "주의").sum()), "헛경보 경보": int((e0["grade"] == "경보").sum())})
    cp = pd.DataFrame(rows)
    cp.to_csv(TAB / "J6_company_period_jeju.csv", index=False, encoding="utf-8-sig")
    print(cp.to_string(index=False))

    # 그림
    import matplotlib.pyplot as plt
    from common.plot import setup_korean_font
    setup_korean_font()
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    x = np.arange(len(t))
    ax[0].bar(x, t["학습 사건일"], color="#888")
    ax[0].set_ylabel("학습 사건일(누적)")
    ax[0].set_title("매달 다시 학습: 그 달 모델이 본 0원 사건일 수")
    ax[1].plot(x, t["주의 문턱"], marker="o", label="주의 문턱")
    ax[1].plot(x, t["경보 문턱"], marker="s", label="경보 문턱")
    ax[1].plot(x, t["사건일 점수 중앙"], ls="--", label="그 달 사건일 점수 중앙")
    ax[1].plot(x, t["평소 날 점수 중앙"], ls=":", label="그 달 평소 날 점수 중앙")
    ax[1].set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax[1].set_ylabel("점수(로그)")
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].set_title("문턱(시험 달만)과 점수 — 주의 문턱이 첫 해에 0.001 수준")
    ax[2].bar(x - 0.2, t["사건일"], 0.4, label="사건일")
    ax[2].bar(x + 0.2, t["주의 이상 비율"].fillna(0) * 30, 0.4, label="주의 이상 비율 × 30")
    ax[2].legend(fontsize=8)
    ax[2].set_title("그 달 사건일과 주의 이상을 낸 날의 비율")
    ax[2].set_xticks(x, t["예측 달"], rotation=90, fontsize=8)
    test0 = int(np.where(t["예측 달"] == "2025-01")[0][0])
    for a in ax:
        a.axvline(test0 - 0.5, color="red", lw=1)
    ax[0].text(test0 - 0.4, ax[0].get_ylim()[1] * 0.9, "← 고르기 | 시험 →", color="red", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "J6_by_month_jeju.png", dpi=150)


if __name__ == "__main__":
    main()
