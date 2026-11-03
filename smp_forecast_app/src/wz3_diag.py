"""일주일 예측 고도화(R3, 육지) ① 진단 — 현재 모델 성능이 '어느 범위를 보느냐'에 따라 얼마나 다른가.

현재 모델: Chronos-2 small + 0.5 × 잔차 보정(외부만) / 80% 범위 D+1 = 보정 분위수 회귀(Chronos-2 분위수·갱신90일·
          시간대묶음×휴무일), D+2~7 = Chronos-2 분위수 그대로 / 휴일·주말 한낮 하한 보정(range_adjust, 육지 D+2~7 × 0.85)
진단 기간: 2025-01-01 ~ 2025-12-31 (검증 구간만). 2026은 이미 채점된 표(T1_team_price.csv)만 참고하고 여기서 다시 나누지 않는다.
  - 2025 하반기 예측은 WZ_predictions_hourly.csv와 같은지 확인한다(재현 점검).
  - 2025 상반기 점예측: λ=0.5를 고른 구간이라 약간 낙관적일 수 있음(잔차 모델 자체는 매월 재학습, 표본 밖).
  - D+1 범위는 직전 90일 보정 이력이 필요 → Chronos-2 분위수 이력이 2025-01부터라 2025-04-01 이전 D+1 범위는 빼고 집계.
구간: 일 유형(평일·휴무일) × 시간대(새벽 1~8 · 한낮 11~15 · 저녁 피크 18~22 · 기타 9·10·16·17·23·24),
      급락 시간(공표 30원 미만), 급등일(surge_classifier 정의: 일 최고 > 직전 28일 일 최고 중앙값 × 1.25),
      수준 변화(대상일 일평균 ÷ 기준일까지 28일 평균: 10% 넘게 하락·보합·상승) × 지평(D+1~7) × 분기
출력: outputs/tables/WZ3_D1_segments_2025.csv (축별 구간), WZ3_D2_cross_2025.csv (일 유형×시간대×지평),
      WZ3_D3_quarter_2025.csv (분기×일 유형×시간대), outputs/figures/WZ3_01~03_*.png,
      data/processed/smp_forecast_app/wz3/current_2025.csv (재현한 현재 모델 2025 시간별)
실행: python -m smp_forecast_app.src.wz3_diag   (약 1분)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.calendar import build_calendar
from common.config import PROCESSED_DIR, ROOT
from common.plot import setup_korean_font
from smp_forecast_app.src.intervals import ALPHA, add_keys
from smp_forecast_app.src.model import cqr
from smp_forecast_app.src import range_adjust
from smp_forecast_app.src.wz_common import OUT, actual_table
from smp_forecast_app.src.wz_resid import RES

DATA = PROCESSED_DIR / "smp_forecast_app"
W3 = DATA / "wz3"
TAB = ROOT / "smp_forecast_app" / "outputs" / "tables"
FIG = ROOT / "smp_forecast_app" / "outputs" / "figures"
PERIOD = ("2025-01-01", "2025-12-31")
D1_RANGE_FROM = "2025-04-01"
LAM = 0.5
BANDS = {**{h: "새벽(1~8)" for h in range(1, 9)}, **{h: "한낮(11~15)" for h in range(11, 16)},
         **{h: "저녁 피크(18~22)" for h in range(18, 23)}, **{h: "기타" for h in (9, 10, 16, 17, 23, 24)}}
BAND_ORDER = ["새벽(1~8)", "한낮(11~15)", "저녁 피크(18~22)", "기타"]


def current_2025() -> pd.DataFrame:
    """현재 일주일 모델을 2025 전체에 재현 [base_date, H, date, hour, y, pred, lo, hi, offday]."""
    c = pd.read_csv(OUT / "c2s.csv.gz", parse_dates=["date"])
    r = pd.read_csv(RES / "외부만.csv.gz", parse_dates=["date"])
    q = pd.read_csv(OUT / "c2.csv.gz", parse_dates=["date"])[["date", "hour", "H", "qlo", "qhi"]]
    d = c[["date", "hour", "H", "pred"]].merge(r, on=["date", "hour", "H"]).merge(q, on=["date", "hour", "H"])
    d["pred"] = d["pred"] + LAM * d["rhat"]
    d = d.merge(actual_table()[["date", "hour", "y"]], on=["date", "hour"], how="left")
    cal = build_calendar(d["date"].min(), d["date"].max()).set_index("date")["is_offday"]
    d["offday"] = d["date"].map(cal).astype(bool)
    d = d[d["date"] <= PERIOD[1]]
    outs = []
    for H, g in d.groupby("H"):
        k = add_keys(g[g["y"].notna()].sort_values(["date", "hour"]).reset_index(drop=True))
        if H == 1:
            ev = (k["date"] >= D1_RANGE_FROM).values
            lo, hi = np.full(len(k), np.nan), np.full(len(k), np.nan)
            lo[ev], hi[ev] = cqr(k, ev, window=90, key="k_band_off", gap=1)
        else:
            lo, hi = k["qlo"].clip(lower=0).values, k["qhi"].values
        outs.append(k.assign(lo=np.clip(lo, 0, None), hi=hi))
    w = pd.concat(outs, ignore_index=True)
    w["base_date"] = w["date"] - pd.to_timedelta(w["H"], unit="D")
    w = range_adjust.apply(w, "육지", offday=w["offday"].values)
    return w[["base_date", "H", "date", "hour", "y", "pred", "lo", "hi", "offday"]]


def check_against_wz(w):
    """2025 하반기가 운영 백테스트 파일(WZ_predictions_hourly.csv, 범위 보정 전)과 같은지."""
    z = pd.read_csv(DATA / "WZ_predictions_hourly.csv", parse_dates=["date"])
    z = range_adjust.apply(z, "육지")
    m = w.merge(z, on=["date", "hour", "H"], suffixes=("", "_wz"))
    m = m[m["date"] <= PERIOD[1]]
    print(f"재현 점검 (2025 하반기 {len(m):,}행): 예측 최대 차 {np.abs(m.pred - m.pred_wz).max():.3f}원, "
          f"하한 {np.nanmax(np.abs(m.lo - m.lo_wz)):.3f}, 상한 {np.nanmax(np.abs(m.hi - m.hi_wz)):.3f}")


def tag(w) -> pd.DataFrame:
    w = w.copy()
    w["일 유형"] = np.where(w["offday"], "휴무일", "평일")
    w["시간대"] = w["hour"].map(BANDS)
    w["분기"] = "Q" + w["date"].dt.quarter.astype(str)
    w["급락 시간"] = w["y"] < 30
    sd = pd.read_csv(ROOT / "surge_classifier" / "outputs" / "tables" / "01_surge_days_육지.csv", parse_dates=["date"])
    w["급등일"] = w["date"].isin(sd["date"])
    # 수준 변화: 대상일 일평균 ÷ 기준일까지 28일 평균 (사후 구분 — 진단용)
    act = actual_table()
    act = pd.concat([pd.read_csv(DATA / "features_hourly_육지.csv", usecols=["date", "hour", "y"], parse_dates=["date"]), act])
    daily = act.drop_duplicates(["date", "hour"]).groupby("date")["y"].mean()
    ma28 = daily.rolling(28, min_periods=20).mean()
    ratio = w["date"].map(daily) / w["base_date"].map(ma28)
    w["수준 변화"] = np.select([ratio < 0.9, ratio > 1.1], ["10% 넘게 하락", "10% 넘게 상승"], "보합(±10%)")
    return w


def metrics(g, tot_ae, tot_pen) -> dict:
    y, p = g["y"].values, g["pred"].values
    e = y - p
    out = {"시간수": len(g), "MAE": np.abs(e).mean(), "편향(실제−예측)": e.mean(), "오차 몫": np.abs(e).sum() / tot_ae}
    r = g[g["lo"].notna()]
    y, lo, hi = r["y"].values, r["lo"].values, r["hi"].values
    b, a = y < lo, y > hi
    pen = (2 / ALPHA) * ((lo - y) * b + (y - hi) * a)
    out |= {"범위 시간수": len(r), "포함률": 1 - b.mean() - a.mean() if len(r) else np.nan,
            "하한 아래": b.mean() if len(r) else np.nan, "상한 위": a.mean() if len(r) else np.nan,
            "평균 폭": (hi - lo).mean() if len(r) else np.nan, "구간점수": ((hi - lo) + pen).mean() if len(r) else np.nan,
            "이탈 벌점 몫": pen.sum() / tot_pen}
    return out


def penalty_total(w):
    r = w[w["lo"].notna()]
    return ((2 / ALPHA) * ((r.lo - r.y).clip(lower=0) + (r.y - r.hi).clip(lower=0))).sum()


def segments_table(w):
    tot_ae, tot_pen = np.abs(w.y - w.pred).sum(), penalty_total(w)
    axes = {"전체": pd.Series("전체", index=w.index), "일 유형": w["일 유형"], "시간대": w["시간대"],
            "일 유형×시간대": w["일 유형"] + " " + w["시간대"], "지평": "D+" + w["H"].astype(str), "분기": w["분기"],
            "급락 시간(30원 미만)": w["급락 시간"].map({True: "급락 시간", False: "그 외"}),
            "급등일": w["급등일"].map({True: "급등일", False: "그 외"}), "수준 변화(사후)": w["수준 변화"]}
    rows = []
    for ax, key in axes.items():
        for k, g in w.groupby(key, sort=True):
            rows.append({"축": ax, "구간": k, **metrics(g, tot_ae, tot_pen)})
    t = pd.DataFrame(rows)
    t["시간 몫"] = t["시간수"] / len(w)
    return t


def cross_table(w, by):
    tot_ae, tot_pen = np.abs(w.y - w.pred).sum(), penalty_total(w)
    rows = []
    for k, g in w.groupby(by):
        rows.append({**dict(zip(by, k)), **metrics(g, tot_ae, tot_pen)})
    return pd.DataFrame(rows)


def figures(w, cross):
    setup_korean_font()
    rows = [f"{t} {b}" for t in ("평일", "휴무일") for b in BAND_ORDER]
    c = cross.assign(행=cross["일 유형"] + " " + cross["시간대"])
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    specs = [("MAE", "Blues", None, "MAE (원)"), ("포함률", "RdBu", 0.8, "80% 범위 포함률 (가운데 흰색 = 80%)"),
             ("구간점수", "Blues", None, "구간점수 (낮을수록 좋음)")]
    for ax, (col, cmap, mid, title) in zip(axs, specs):
        m = c.pivot(index="행", columns="H", values=col).reindex(rows)
        if mid is None:
            im = ax.imshow(m.values, cmap=cmap, aspect="auto")
        else:
            span = max(abs(np.nanmin(m.values) - mid), abs(np.nanmax(m.values) - mid))
            im = ax.imshow(m.values, cmap=cmap, aspect="auto", vmin=mid - span, vmax=mid + span)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.0%}" if col == "포함률" else f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="black")
        ax.set_xticks(range(7), [f"D+{h}" for h in range(1, 8)])
        ax.set_yticks(range(len(rows)), rows if ax is axs[0] else [""] * len(rows))
        ax.set_title(title)
        ax.set_xlabel("지평 (기준일 15시 → 며칠 뒤)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("일주일 예측(현재 모델) 2025 검증 — 일 유형×시간대×지평 (D+1 범위는 2025-04~)")
    fig.tight_layout()
    fig.savefig(FIG / "WZ3_01_heatmap_2025.png", dpi=130)
    plt.close(fig)

    # 시간별: 평균 절대오차 / 이탈 방향 (평일·휴무일)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), sharex=True)
    col = {"평일": "#2a6fb0", "휴무일": "#d0702a"}
    r = w[w["lo"].notna()]
    for t, g in w.groupby("일 유형"):
        axs[0].plot(g.groupby("hour").apply(lambda x: np.abs(x.y - x.pred).mean()), color=col[t], lw=2, label=t)
    for t, g in r.groupby("일 유형"):
        axs[1].plot(g.groupby("hour").apply(lambda x: (x.y < x.lo).mean()), color=col[t], lw=2, label=t)
        axs[2].plot(g.groupby("hour").apply(lambda x: (x.y > x.hi).mean()), color=col[t], lw=2, label=t)
    for ax, ttl in zip(axs, ["시간별 MAE (원)", "하한 아래 이탈 비율 (목표 약 10%)", "상한 위 이탈 비율 (목표 약 10%)"]):
        ax.set_title(ttl)
        ax.set_xlabel("시 (1~24)")
        ax.grid(alpha=0.3)
        for a, b in ((11, 15), (18, 22)):
            ax.axvspan(a - 0.5, b + 0.5, color="0.92", zorder=0)
    for ax in axs[1:]:
        ax.axhline(0.1, color="0.4", ls="--", lw=1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axs[0].legend()
    fig.suptitle("일주일 예측(현재 모델) 2025 검증 — 시간별 오차와 이탈 방향 (회색 = 한낮·저녁 피크, 모든 지평)")
    fig.tight_layout()
    fig.savefig(FIG / "WZ3_02_by_hour_2025.png", dpi=130)
    plt.close(fig)

    # 분기 × 지평 MAE, 분기별 이탈 방향
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
    qcol = {"Q1": "#9ab8d8", "Q2": "#5b8fc4", "Q3": "#2a6fb0", "Q4": "#123f6e"}
    for qq, g in w.groupby("분기"):
        axs[0].plot(range(1, 8), g.groupby("H").apply(lambda x: np.abs(x.y - x.pred).mean()), marker="o", color=qcol[qq], lw=2, label=qq)
    axs[0].set_title("분기별 · 지평별 MAE (원)")
    axs[0].set_xlabel("지평 D+H")
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    s = r.groupby("분기").apply(lambda x: pd.Series({"하한 아래": (x.y < x.lo).mean(), "상한 위": (x.y > x.hi).mean()}))
    x = np.arange(len(s))
    axs[1].bar(x - 0.2, s["하한 아래"], 0.38, color="#2a6fb0", label="하한 아래")
    axs[1].bar(x + 0.2, s["상한 위"], 0.38, color="#d0702a", label="상한 위")
    axs[1].axhline(0.1, color="0.4", ls="--", lw=1)
    axs[1].set_xticks(x, s.index)
    axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axs[1].set_title("분기별 이탈 방향 (목표 각 약 10%)")
    axs[1].legend()
    fig.suptitle("일주일 예측(현재 모델) 2025 검증 — 분기 (Q1의 D+1 범위는 집계 제외)")
    fig.tight_layout()
    fig.savefig(FIG / "WZ3_03_quarter_2025.png", dpi=130)
    plt.close(fig)


def main():
    W3.mkdir(parents=True, exist_ok=True)
    w = current_2025()
    check_against_wz(w)
    w = w[w["date"].between(*PERIOD)]
    w.round(2).to_csv(W3 / "current_2025.csv", index=False, encoding="utf-8-sig")
    w = tag(w)
    seg = segments_table(w)
    seg.round(4).to_csv(TAB / "WZ3_D1_segments_2025.csv", index=False, encoding="utf-8-sig")
    cross = cross_table(w, ["일 유형", "시간대", "H"])
    cross.round(4).to_csv(TAB / "WZ3_D2_cross_2025.csv", index=False, encoding="utf-8-sig")
    quarter = cross_table(w, ["분기", "일 유형", "시간대"])
    quarter.round(4).to_csv(TAB / "WZ3_D3_quarter_2025.csv", index=False, encoding="utf-8-sig")
    figures(w, cross)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(seg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
