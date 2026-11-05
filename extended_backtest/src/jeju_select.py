"""제주 급락(SMP ≤ 0원) 후보 고르기 → 매달 문턱 → 시험 채점.

고르기 (시험 자료 안 씀): 2024-03 ~ 2024-12 '앞으로만' 예측(사건일 17)
  순위 = PR-AUC(평균 정밀도) ↓. 최고값에서 0.03 이내면 단순한 쪽(규칙 < 단일 < 앙상블, 앙상블은 구성 수 적은 쪽).
문턱 (규칙 R2, 2026-11-06 채점 전 고정): 시험 달 m마다 2024-03 ~ m 직전의 앞으로만 예측으로
  주의 = 지난 사건일 90%를 잡는 가장 높은 점수, 경보 = 주의 이상에서 정밀도 50%·적중 3일 이상인 가장 낮은 점수
  (없으면 지난 사건일 50%를 잡는 점수, 주의보다 낮으면 주의와 같음). 참고: R1 = 확률 고정(주의 0.2, 경보 0.5).
채점 (--score, 한 번): 시험 2025-01-01 ~ 2026-10-31. 하루 경보 → 그날 모든 구간. 성공 = 경보, 부분 = 주의만, 실패 = 없음.
  헛경보 = 주의·경보를 냈는데 그날 0원 이하 시간 없는 날 수 (주의만/경보 나눔).
실행(루트에서): python -m extended_backtest.src.jeju_select [--score]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from extended_backtest.src.jeju_models import ENSEMBLES, OUT as PRED

TAB = Path(__file__).resolve().parents[1] / "outputs" / "tables"
FIG = Path(__file__).resolve().parents[1] / "outputs" / "figures"
SEL = ("2024-03-01", "2024-12-31")
TEST = ("2025-01-01", "2026-10-31")
TOL = 0.03
PERIODS = {"시험 전체 2025-01~2026-10": TEST, "2025": ("2025-01-01", "2025-12-31"),
           "2026 (2차 시험)": ("2026-01-01", "2026-10-31"), "참고 2025-07~2026-10": ("2025-07-01", "2026-10-31")}


def candidates(p):
    return [c for c in p.columns if c not in ("date", "event", "n_intervals", "offday", "s1")]


def complexity(c):
    if c.startswith(("규칙", "기준선")):
        return 0
    if c in ENSEMBLES:
        return 1 + len(ENSEMBLES[c])
    return 1


def fa_at_recall(y, s, r=0.9):
    t = recall_threshold(s[y == 1], r)
    return int(((s >= t) & (y == 0)).sum())


def recall_threshold(pos_scores, r):
    v = np.sort(np.asarray(pos_scores))[::-1]
    return v[int(np.ceil(r * len(v))) - 1]


def select(p):
    s = p[(p["date"] >= SEL[0]) & (p["date"] <= SEL[1])]
    h2 = s[s["date"] >= "2024-07-01"]
    rows = []
    for c in candidates(p):
        y, sc = s["event"].values, s[c].values
        rows.append({"후보": c, "단순함": complexity(c), "PR-AUC": average_precision_score(y, sc),
                     "ROC-AUC": roc_auc_score(y, sc), "헛경보@90%": fa_at_recall(y, sc),
                     "PR-AUC 하반기": average_precision_score(h2["event"], h2[c]),
                     "Brier": brier_score_loss(y, sc) if sc.min() >= 0 and sc.max() <= 1 and not c.startswith("규칙") else np.nan})
    t = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    best = t["PR-AUC"].iloc[0]
    near = t[t["PR-AUC"] >= best - TOL].sort_values(["단순함", "PR-AUC"], ascending=[True, False])
    return t, near.iloc[0]["후보"], int(s["event"].sum())


def thresholds_r2(past_y, past_s):
    caution = recall_threshold(past_s[past_y == 1], 0.9)
    alarm = None
    for t in np.sort(np.unique(past_s[past_s >= caution])):
        sel = past_s >= t
        hits = int((sel & (past_y == 1)).sum())
        if hits >= 3 and hits / sel.sum() >= 0.5:
            alarm = t
            break
    if alarm is None:
        alarm = max(recall_threshold(past_s[past_y == 1], 0.5), caution)
    return caution, alarm


def grade_test(p, c, rule="R2"):
    """시험 기간 날짜별 등급 [date, event, n_intervals, offday, score, caution, alarm, grade]."""
    te = p[(p["date"] >= TEST[0]) & (p["date"] <= TEST[1])].copy()
    te["score"] = te[c]
    cs, al = [], []
    for m, g in te.groupby(te["date"].dt.to_period("M")):
        if rule == "R1":
            a, b = 0.2, 0.5
        else:
            past = p[(p["date"] >= SEL[0]) & (p["date"] < m.start_time)]
            a, b = thresholds_r2(past["event"].values, past[c].values)
        cs += [a] * len(g)
        al += [b] * len(g)
    te["caution"], te["alarm"] = cs, al
    te["grade"] = np.select([te["score"] >= te["alarm"], te["score"] >= te["caution"]], ["경보", "주의"], "없음")
    return te


def tables(g):
    t1, t2, cm = [], [], []
    for pname, (a, b) in PERIODS.items():
        x = g[(g["date"] >= a) & (g["date"] <= b)]
        for lab, part in [("휴일", x[x["offday"] == 1]), ("평일", x[x["offday"] == 0]), ("계", x)]:
            n = int(part["n_intervals"].sum())
            ev = part[part["event"] == 1]
            k = {gr: int(ev.loc[ev["grade"] == gr, "n_intervals"].sum()) for gr in ["경보", "주의", "없음"]}
            pct = lambda v: f"{v} ({v / n:.0%})" if n else "0"
            t1.append({"기간": pname, "구분": lab, "구간 수": n, "성공(경보)": pct(k["경보"]), "부분(주의만)": pct(k["주의"]),
                       "실패": pct(k["없음"]), "사건일": len(ev)})
            no = part[part["event"] == 0]
            t2.append({"기간": pname, "구분": lab, "헛경보 주의만(일)": int((no["grade"] == "주의").sum()),
                       "헛경보 경보(일)": int((no["grade"] == "경보").sum()),
                       "헛경보 계(일)": int((no["grade"] != "없음").sum()), "사건 없는 날": len(no)})
        for lvl, flag in [("주의 이상", x["grade"] != "없음"), ("경보", x["grade"] == "경보")]:
            yv = x["event"] == 1
            cm.append({"기간": pname, "예측 기준": lvl, "실제 있음·예측 있음": int((yv & flag).sum()),
                       "실제 있음·예측 없음": int((yv & ~flag).sum()), "실제 없음·예측 있음": int((~yv & flag).sum()),
                       "실제 없음·예측 없음": int((~yv & ~flag).sum())})
    return pd.DataFrame(t1), pd.DataFrame(t2), pd.DataFrame(cm)


def summary_row(g):
    ev = g[g["event"] == 1]
    n = ev["n_intervals"].sum()
    no = g[g["event"] == 0]
    return {"구간": int(n), "성공": int(ev.loc[ev["grade"] == "경보", "n_intervals"].sum()),
            "부분": int(ev.loc[ev["grade"] == "주의", "n_intervals"].sum()),
            "실패": int(ev.loc[ev["grade"] == "없음", "n_intervals"].sum()),
            "헛경보 주의만": int((no["grade"] == "주의").sum()), "헛경보 경보": int((no["grade"] == "경보").sum()),
            "PR-AUC(일)": average_precision_score(g["event"], g["score"]), "ROC-AUC(일)": roc_auc_score(g["event"], g["score"])}


def plot_cm(cm, path):
    import matplotlib.pyplot as plt
    from common.plot import setup_korean_font
    setup_korean_font()
    x = cm[cm["기간"] == "시험 전체 2025-01~2026-10"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (_, r) in zip(axes, x.iterrows()):
        m = np.array([[r["실제 있음·예측 있음"], r["실제 있음·예측 없음"]], [r["실제 없음·예측 있음"], r["실제 없음·예측 없음"]]])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]}일", ha="center", va="center", fontsize=13,
                        color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_xticks([0, 1], [f"예측 있음\n({r['예측 기준']})", "예측 없음"])
        ax.set_yticks([0, 1], ["실제 0원 있음", "실제 0원 없음"])
        ax.set_title(f"제주 급락 혼동행렬 — {r['예측 기준']}")
    fig.suptitle("시험 2025-01-01 ~ 2026-10-31 (날짜 단위, 전날 15시 예측)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main(score=False):
    p = pd.read_csv(PRED, parse_dates=["date"])
    t, pick, nsel = select(p)
    t.round(4).to_csv(TAB / "J2_select_2024_jeju.csv", index=False, encoding="utf-8-sig")
    print(f"고르기 2024-03~12 사건일 {nsel}")
    print(t.head(20).round(3).to_string())
    print("선택:", pick)
    info = {"선택": pick, "구성": ENSEMBLES.get(pick, [pick]), "고르기 기간": SEL, "고르기 사건일": nsel,
            "규칙": f"PR-AUC 최고에서 {TOL} 이내면 단순한 쪽", "2024 PR-AUC": float(t.set_index("후보").loc[pick, "PR-AUC"]),
            "문턱": "R2 매달: 주의 = 지난 사건일 90% 포착, 경보 = 정밀도 50%·적중 3일 이상 가장 낮은 점수"}
    (TAB / "J2_pick_jeju.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    if not score:
        return
    g = grade_test(p, pick)
    g[["date", "offday", "event", "n_intervals", "score", "caution", "alarm", "grade"]].round(4).to_csv(
        TAB / "J3_daily_jeju.csv", index=False, encoding="utf-8-sig")
    t1, t2, cm = tables(g)
    t1.to_csv(TAB / "J3_table1_jeju.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(TAB / "J3_table2_jeju.csv", index=False, encoding="utf-8-sig")
    cm.to_csv(TAB / "J3_confusion_jeju.csv", index=False, encoding="utf-8-sig")
    plot_cm(cm, FIG / "J3_confusion_jeju.png")
    print(t1.to_string()); print(t2.to_string()); print(cm.to_string())
    # 참고: 모든 후보의 시험 결과 (고르기에 쓰지 않음), R2·R1
    rows = []
    for c in candidates(p):
        for rule in ["R2", "R1"]:
            if rule == "R1" and c.startswith("규칙"):
                continue
            rows.append({"후보": c, "문턱": rule, **summary_row(grade_test(p, c, rule))})
    a = pd.DataFrame(rows).merge(t[["후보", "PR-AUC"]].rename(columns={"PR-AUC": "2024 PR-AUC"}), on="후보")
    a.round(4).sort_values(["문턱", "PR-AUC(일)"], ascending=[True, False]).to_csv(
        TAB / "J3_all_candidates_test_jeju.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main("--score" in sys.argv)
