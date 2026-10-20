"""하루 전·일주일 최종 모델 학습과 검증 (육지).

search.py가 고른 (리드별) 모델을 사용한다.
- 경보 기준: 롤링 2회 예측(2025, 2026 상반기)에서 적중률 80% / 90%를 달성하는 가장 높은 점수 → 검증 기간에 그대로 적용
- 최종 모델: 학습 2024-01 ~ 2026-06 → 검증 2026-07 ~ 최신일
성공 기준표: 구간별 급락일 적중 수와 헛경보 수

실행: python -m drop_classifier.src.final
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import PROCESSED_DIR, ROOT
from common.plot import setup_korean_font
from drop_classifier.src.features_weekly import W_CALENDAR, W_MID, W_STATE
from drop_classifier.src.models import FOLDS, split
from drop_classifier.src.search import CANDIDATES, DAY_SETS, day_table

REGION = "육지"
OUT_DATA = PROCESSED_DIR / "drop_classifier"
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
OUT_FIG = ROOT / "drop_classifier" / "outputs" / "figures"
TARGETS = (0.8, 0.9)
CASES = ["2026-09-25", "2026-10-05"]
C_DROP, C_NORMAL, C_MUTED, C_INK, C_INK2 = "#eb6834", "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e"


def threshold_for(y, p, target):
    for t in np.sort(np.unique(p))[::-1]:
        if ((p >= t) & (y == 1)).sum() >= np.ceil(target * y.sum()):
            return float(t)
    return float(p.min())


def run_one(df: pd.DataFrame, cols: list[str], model: str) -> pd.DataFrame:
    """롤링1·2·검증 예측 점수와 경보 기준별 판정."""
    fn = CANDIDATES[model]
    parts = []
    for fold in FOLDS:
        tr, te = split(df, FOLDS[fold][0]), split(df, FOLDS[fold][1])
        parts.append(te.assign(score=fn(tr, te, cols), 구간=fold))
    out = pd.concat(parts, ignore_index=True)
    roll = out[out["구간"] != "검증"]
    for t in TARGETS:
        thr = threshold_for(roll["label"].values, roll["score"].values, t)
        out[f"경보@{int(t * 100)}%"] = (out["score"] >= thr).astype(int)
        out[f"기준@{int(t * 100)}%"] = thr
    return out


def success(out: pd.DataFrame, horizon: str) -> pd.DataFrame:
    rows = []
    for fold in list(FOLDS) + ["합계"]:
        d = out if fold == "합계" else out[out["구간"] == fold]
        r = {"모델": horizon, "구간": fold, "급락일": int(d["label"].sum())}
        for t in TARGETS:
            a = d[f"경보@{int(t * 100)}%"]
            r[f"적중@{int(t * 100)}%"] = int(((a == 1) & (d["label"] == 1)).sum())
            r[f"헛경보@{int(t * 100)}%"] = int(((a == 1) & (d["label"] == 0)).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def plot_week_grid(week: pd.DataFrame, key: str):
    """검증 기간: 대상일 × 며칠 전 경보 격자."""
    v = week[week["구간"] == "검증"].copy()
    grid = v.pivot_table(index="lead", columns="date", values=key, aggfunc="max")
    lab = v.groupby("date")["label"].max().reindex(grid.columns)
    setup_korean_font()
    fig, ax = plt.subplots(figsize=(14, 3.6))
    for (i, h) in enumerate(grid.index):
        for j, d in enumerate(grid.columns):
            a = grid.loc[h, d]
            if pd.isna(a):
                continue
            hit = lab[d] == 1
            color = C_DROP if (a == 1 and hit) else (C_MUTED if a == 1 else ("#f4f3ef" if not hit else "#fde3d6"))
            ax.add_patch(plt.Rectangle((j, i), 0.92, 0.85, color=color))
    for j, d in enumerate(grid.columns):
        if lab[d] == 1:
            ax.text(j + 0.46, len(grid.index) + 0.15, "▼", ha="center", fontsize=8, color=C_DROP)
    ax.set_xlim(0, len(grid.columns))
    ax.set_ylim(0, len(grid.index) + 0.6)
    ax.set_yticks(np.arange(len(grid.index)) + 0.42, [f"{h}일 전" for h in grid.index], fontsize=9)
    ticks = [j for j, d in enumerate(grid.columns) if d.day in (1, 15)]
    ax.set_xticks([t + 0.46 for t in ticks], [grid.columns[t].strftime("%m/%d") for t in ticks], fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{REGION} 일주일 모델 검증 기간 경보 ({key}): 주황 = 급락을 미리 경보, 연주황 = 급락인데 경보 없음, 회색 = 헛경보, ▼ = 실제 급락일",
                 loc="left", fontsize=10.5, color=C_INK)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"05_week_grid_{REGION}.png", dpi=150)
    plt.close(fig)


def main():
    sel = pd.read_csv(OUT_TAB / f"05_selected_{REGION}.csv")
    pd.set_option("display.width", 250)

    # 하루 전
    s1 = sel[sel["며칠 뒤"] == 1].iloc[0]
    day = run_one(day_table(), DAY_SETS[s1["피처"]], s1["모델"])
    day["lead"] = 1
    succ = [success(day, f"하루 전 ({s1['모델']}, {s1['피처']})")]

    # 일주일 (1일 뒤는 하루 전 모델 사용)
    w = pd.read_csv(OUT_DATA / f"features_weekly_{REGION}.csv", parse_dates=["date"])
    w["label"] = w["label"].astype(int)
    week = [day[["date", "label", "min_smp_11_15", "day_type", "구간", "score", "lead"]
                + [c for c in day.columns if c.startswith(("경보@", "기준@"))]]]
    for _, s in sel[sel["며칠 뒤"] >= 2].iterrows():
        h = int(s["며칠 뒤"])
        d = w[w["lead"] == h].reset_index(drop=True)
        cols = W_CALENDAR + W_STATE + (W_MID if h >= 5 else [])
        o = run_one(d, cols, s["모델"])
        week.append(o)
        succ.append(success(o, f"{h}일 전 ({s['모델']})"))
    week = pd.concat(week, ignore_index=True)

    # 일주일 모델 종합: 대상일 기준 '미리 경보한 날 수'
    v = week[week["구간"] == "검증"]
    lead_sum = []
    for t in TARGETS:
        k = f"경보@{int(t * 100)}%"
        first = v[(v[k] == 1)].groupby("date")["lead"].max()  # 가장 이른 경보 시점
        drops = v[v["label"] == 1]["date"].unique()
        for dd in drops:
            lead_sum.append({"기준": k, "급락일": pd.Timestamp(dd).date(), "가장 이른 경보": f"{int(first[dd])}일 전" if dd in first.index else "경보 없음",
                             "경보한 시점 수(1~7일 전)": int(v[(v["date"] == dd) & (v[k] == 1)].shape[0])})
    lead_sum = pd.DataFrame(lead_sum)

    succ = pd.concat(succ, ignore_index=True)
    succ.to_csv(OUT_TAB / f"05_success_{REGION}.csv", index=False, encoding="utf-8-sig")
    day.to_csv(OUT_TAB / f"05_day_predictions_{REGION}.csv", index=False, encoding="utf-8-sig")
    week.to_csv(OUT_TAB / f"05_week_predictions_{REGION}.csv", index=False, encoding="utf-8-sig")
    lead_sum.to_csv(OUT_TAB / f"05_week_earliest_alarm_{REGION}.csv", index=False, encoding="utf-8-sig")
    plot_week_grid(week, "경보@90%")

    # 과제2 연계: 하루 전 모델 점수·판정
    day[["date"]].assign(계통=REGION, 위험점수=day["score"].round(3), 판정=day["경보@90%"]).to_csv(
        OUT_TAB / f"drop_prob_{REGION}.csv", index=False, encoding="utf-8-sig")

    print(f"[성공 기준: 구간별 적중·헛경보]\n{succ.to_string(index=False)}")
    cases = day[day["date"].isin(pd.to_datetime(CASES))][["date", "label", "min_smp_11_15", "score", "경보@80%", "경보@90%"]]
    print(f"\n[실패 사례 — 하루 전]\n{cases.round(3).to_string(index=False)}")
    print(f"\n[검증 기간 급락일별 가장 이른 경보]\n{lead_sum.to_string(index=False)}")


if __name__ == "__main__":
    main()
