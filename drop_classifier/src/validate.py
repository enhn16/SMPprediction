"""④ 검증 결과 (육지). ③ models.py 산출물을 정리한다.

성공 기준: 과거 급락일 적중률(재현율)과 헛경보 수. 판정은 운영형(재현율 우선) 임계값.

실행: python -m drop_classifier.src.validate
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import ROOT
from common.plot import setup_korean_font

REGION = "육지"
OUT_TAB = ROOT / "drop_classifier" / "outputs" / "tables"
OUT_FIG = ROOT / "drop_classifier" / "outputs" / "figures"
MAIN_SET, CHOSEN = "예보", "로지스틱"
RULES = ["규칙: 휴무일", "규칙: 휴무일+맑음 예보"]
PERIOD = {"롤링1": "2025년", "롤링2": "2026년 1~6월", "검증": "2026년 7월~10/18"}
C_DROP, C_NORMAL, C_MUTED, C_INK, C_INK2 = "#eb6834", "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e"


def success_table(res: pd.DataFrame) -> pd.DataFrame:
    """구간별 적중률·헛경보: 선택 모델(운영형) vs 규칙 기준선."""
    rows = []
    for name, label in [(CHOSEN, "선택 모델(로지스틱, 운영형)")] + [(r, r) for r in RULES]:
        sub = res[(res["피처"] == MAIN_SET) & (res["모델"] == name)].set_index("구간")
        for fold in PERIOD:
            r = sub.loc[fold]
            rows.append({"모델": label, "구간": f"{fold} ({PERIOD[fold]})", "급락일": int(r["급락일"]),
                         "적중": int(r["적중(R)"]), "적중률": r["재현율(R)"],
                         "경보": int(r["예측 급락(R)"]), "헛경보": int(r["예측 급락(R)"] - r["적중(R)"])})
        tot = pd.DataFrame(rows[-3:])
        rows.append({"모델": label, "구간": "합계", "급락일": tot["급락일"].sum(), "적중": tot["적중"].sum(),
                     "적중률": tot["적중"].sum() / tot["급락일"].sum(), "경보": tot["경보"].sum(),
                     "헛경보": tot["헛경보"].sum()})
    return pd.DataFrame(rows)


def confusion(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for name, d in [("전체", df), ("평일", df[df["day_type"] == "평일"]), ("휴무일", df[df["day_type"] != "평일"])]:
        y, p = d["label"], d[pred_col]
        tp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
        fp, tn = int(((y == 0) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
        rows.append({"구분": name, "일수": len(d), "적중(TP)": tp, "놓침(FN)": fn, "헛경보(FP)": fp, "정상(TN)": tn,
                     "재현율": tp / (tp + fn) if tp + fn else np.nan,
                     "정밀도": tp / (tp + fp) if tp + fp else np.nan})
    return pd.DataFrame(rows)


def plot_timeline(v: pd.DataFrame, thr: float):
    setup_korean_font()
    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = np.arange(len(v))
    colors = [C_DROP if l == 1 else C_NORMAL for l in v["label"]]
    ax.bar(x, v["급락확률"], color=colors, width=0.8, alpha=0.85)
    ax.axhline(thr, color=C_INK2, linestyle="--", linewidth=1)
    ax.text(len(v) - 0.5, thr + 0.015, f"경보 기준 {thr:.2f}", ha="right", fontsize=9, color=C_INK2)
    off = v["day_type"] != "평일"
    ax.scatter(x[off], np.full(off.sum(), -0.04), marker="s", s=10, color=C_MUTED, clip_on=False)
    for i, r in v.reset_index(drop=True).iterrows():
        if r["date"].strftime("%Y-%m-%d") in ("2026-09-25", "2026-10-05"):
            ax.annotate(f"{r['date']:%m/%d}\n{r['급락확률']:.2f}", (i, r["급락확률"]), xytext=(0, 14),
                        textcoords="offset points", ha="center", fontsize=8.5, color=C_INK, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=C_INK2, lw=0.8))
    ticks = [i for i, d in enumerate(v["date"]) if d.day in (1, 15)]
    ax.set_xticks(ticks, [v["date"].iloc[i].strftime("%m/%d") for i in ticks])
    ax.set_ylim(-0.06, max(0.75, v["급락확률"].max() * 1.25))
    ax.set_ylabel("급락확률 (보정)", color=C_INK2)
    ax.set_title(f"{REGION} 검증 기간 날짜별 급락확률 (주황 = 실제 급락, 파랑 = 정상, 아래 회색 점 = 휴무일)",
                 loc="left", fontsize=11, color=C_INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"04_valid_timeline_{REGION}.png", dpi=150)
    plt.close(fig)


def main():
    res = pd.read_csv(OUT_TAB / f"03_model_comparison_{REGION}.csv")
    v = pd.read_csv(OUT_TAB / f"03_valid_pred_{REGION}.csv", parse_dates=["date"])
    allp = pd.read_csv(OUT_TAB / f"03_all_predictions_{REGION}.csv", parse_dates=["date"])
    thr = res[(res["피처"] == MAIN_SET) & (res["모델"] == CHOSEN) & (res["구간"] == "검증")]["임계값(재현율우선)"].iloc[0]

    succ = success_table(res)
    succ.round(3).to_csv(OUT_TAB / f"04_success_{REGION}.csv", index=False, encoding="utf-8-sig")

    # 검증 기간 날짜별 표
    daily = v.rename(columns={"weekday_ko": "요일", "day_type": "요일유형", "holiday_name": "공휴일",
                              "min_smp_11_15": "실제 한낮 최저 SMP", "smp_12h": "정오(12시) SMP", "label": "실제 급락"})
    daily["예측(운영형)"] = daily.pop("예측(재현율우선)")
    daily["판정"] = np.select([(daily["실제 급락"] == 1) & (daily["예측(운영형)"] == 1),
                             (daily["실제 급락"] == 1) & (daily["예측(운영형)"] == 0),
                             (daily["실제 급락"] == 0) & (daily["예측(운영형)"] == 1)], ["적중", "놓침", "헛경보"], "")
    cols = ["date", "요일", "요일유형", "공휴일", "급락확률", "예측(운영형)", "실제 급락", "판정", "실제 한낮 최저 SMP",
            "정오(12시) SMP"] + RULES
    daily[cols].to_csv(OUT_TAB / f"04_validation_daily_{REGION}.csv", index=False, encoding="utf-8-sig")

    v_conf = confusion(v, "예측(재현율우선)")
    a = allp.copy()
    a["pred_op"] = a["결과(재현율우선)"].isin(["적중", "헛경보"]).astype(int)
    all_conf = confusion(a, "pred_op")
    v_conf.round(3).to_csv(OUT_TAB / f"04_confusion_valid_{REGION}.csv", index=False, encoding="utf-8-sig")
    all_conf.round(3).to_csv(OUT_TAB / f"04_confusion_all_{REGION}.csv", index=False, encoding="utf-8-sig")

    # 예보 vs 상한선(실측 날씨)
    cmp_rows = []
    for s in ("기본(날씨 없음)", "예보", "상한선(실측 날씨)"):
        sub = res[(res["피처"] == s) & (res["모델"] == CHOSEN)].set_index("구간")
        cmp_rows.append({"피처": s, "PR-AUC(롤링 평균)": sub.loc[["롤링1", "롤링2"], "PR-AUC"].mean(),
                         "적중(3구간)": int(sub["적중(R)"].sum()), "급락일": int(sub["급락일"].sum()),
                         "헛경보(3구간)": int((sub["예측 급락(R)"] - sub["적중(R)"]).sum())})
    cmp = pd.DataFrame(cmp_rows)
    cmp.round(3).to_csv(OUT_TAB / f"04_forecast_vs_upper_{REGION}.csv", index=False, encoding="utf-8-sig")

    # 과제2 연계용
    a2 = allp[["date"]].assign(계통=REGION, 급락확률=allp["prob"].round(4), 판정=a["pred_op"])
    a2.to_csv(OUT_TAB / f"drop_prob_{REGION}.csv", index=False, encoding="utf-8-sig")

    plot_timeline(v, thr)

    pd.set_option("display.width", 250)
    print(f"운영형 임계값: {thr:.3f}\n\n[성공 기준: 적중률·헛경보]\n{succ.round(3).to_string(index=False)}")
    print(f"\n[혼동행렬 — 검증 기간]\n{v_conf.round(3).to_string(index=False)}")
    print(f"\n[혼동행렬 — 롤링+검증 전체 (2025-01 ~ 2026-10-18)]\n{all_conf.round(3).to_string(index=False)}")
    print(f"\n[예보 vs 상한선]\n{cmp.round(3).to_string(index=False)}")
    show = daily[(daily["실제 급락"] == 1) | (daily["예측(운영형)"] == 1) | daily["date"].isin(pd.to_datetime(["2026-10-05"]))]
    print(f"\n[검증 기간 경보·급락일]\n{show[cols].to_string(index=False)}")


if __name__ == "__main__":
    main()
