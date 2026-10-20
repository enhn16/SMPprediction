"""결과 페이지용 '휴무일 중 급락한 날 vs 급락하지 않은 날' 비교 섹션 생성 (육지, 2024-01 ~ 2026-10-18).

비교 대상: 주말·공휴일(휴무일)끼리. 평일은 급락이 거의 없어(682일 중 3일) 제외.
요인 선택 기준: 휴무일 급락/비급락 구분력(AUC, 0.5 = 못 가름 ~ 1 = 완벽)을 계산해 높은 순으로 고름.
  '15시' = 판단 시각(D-1 15시)에 알 수 있는 정보, '사후' = 그날이 지나야 아는 실측(상한선 참고용).
일사 예보는 계절 영향이 커서 '최근 31일(그날 포함 이전) 맑은 날(상위 10%) 대비 비율'로 바꿔 비교.
기술 통계(설명용)이며 모델 선택에는 쓰지 않는다.
출력: drop_classifier/outputs/compare_section.html (인라인 SVG, 색은 CSS 변수 --c-drop / --c-norm)
실행: python -m drop_classifier.src.compare_section
"""
import html
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.config import ROOT
from drop_classifier.src.explore2 import data2

warnings.filterwarnings("ignore")
OUT = ROOT / "drop_classifier" / "outputs" / "compare_section.html"

# 열: (이름, 단위, 배수, (하한, 상한, 눈금), 방향, 시점)
FACTORS = {
    "prev_day_min_smp": ("전날 한낮 최저 SMP", "원", 1, (0, 160, [0, 40, 80, 120, 160]), "낮을수록 위험", "15시"),
    "prev_same_type_min_smp": ("최근 같은 유형 날의 한낮 최저 SMP", "원", 1, (0, 160, [0, 40, 80, 120, 160]), "낮을수록 위험", "15시"),
    "rf_pop_pm": ("내일 오후 강수확률 예보", "%", 1, (0, 100, [0, 25, 50, 75, 100]), "낮을수록 위험", "15시"),
    "act_cloud": ("한낮 실측 운량 (0 맑음 ~ 10 흐림)", "", 1, (0, 10, [0, 2, 4, 6, 8, 10]), "적을수록 위험", "사후"),
    "rf_sky_pm": ("내일 오후 하늘상태 예보 (1 맑음 ~ 4 흐림)", "", 1, (1, 4, [1, 2, 3, 4]), "맑을수록 위험", "15시"),
    "act_ss": ("한낮 실측 일조 (시간당 비율)", "", 1, (0, 1, [0, .25, .5, .75, 1]), "길수록 위험", "사후"),
    "om_cloud": ("내일 한낮 운량 예보 (Open-Meteo)", "%", 1, (0, 100, [0, 25, 50, 75, 100]), "적을수록 위험", "15시"),
    "om_rad_rel": ("내일 일사 예보 ÷ 최근 맑은 날", "%", 100, (0, 130, [0, 25, 50, 75, 100, 125]), "높을수록 위험", "15시"),
    "demand_prev_rel28": ("전날 수요예측 ÷ 최근 28일 평균", "%", 100, (70, 110, [70, 80, 90, 100, 110]), "낮을수록 위험", "15시"),
    "kpx_peak_rel": ("주간전망 최대부하 ÷ 그 주 최고", "%", 100, (65, 100, [70, 80, 90, 100]), "낮을수록 위험", "15시"),
    "k_pump_d1": ("전날 한낮 양수 (음수 = 물을 퍼올리는 중)", "GW", 1, (-5, 1, [-5, -4, -3, -2, -1, 0, 1]), "많이 퍼올릴수록 위험", "15시"),
    "k_thermal_d1": ("전날 한낮 화력 발전 (가스·석탄·유류)", "GW", 1, (10, 50, [10, 20, 30, 40, 50]), "적을수록 위험", "15시"),
    "k_nuc_d1": ("전날 한낮 원전 발전", "GW", 1, (15, 25, [15, 20, 25]), "차이 거의 없음", "15시"),
    "act_icsr": ("한낮 실측 일사량 (원값)", "MJ/m²", 1, (0, 4, [0, 1, 2, 3, 4]), "원값으로는 차이 없음", "사후"),
}
PANELS = ["prev_same_type_min_smp", "rf_pop_pm", "rf_sky_pm", "om_rad_rel", "demand_prev_rel28", "act_cloud"]
W, ROWH, PADL, PADR = 360, 34, 118, 16
ROWS = ((1, "급락한 휴무일", "d"), (0, "급락 안 한 휴무일", "n"))


def add_rel_radiation(df: pd.DataFrame) -> pd.DataFrame:
    """일사 예보 ÷ 최근 31일(그날 포함 이전) 상위 10% — 그날 이후 값은 쓰지 않음."""
    df = df.sort_values("date").copy()
    df["om_rad_rel"] = df["om_rad"] / df["om_rad"].rolling(31, min_periods=10).quantile(0.9)
    return df


def auc(o, col):
    x = o[[col, "label"]].dropna()
    a = roc_auc_score(x["label"], x[col])
    return max(a, 1 - a)


def strip_panel(o, col):
    title, unit, mult, (lo, hi, ticks), hint, when = FACTORS[col]
    x = lambda v: PADL + (min(max(v, lo), hi) - lo) / (hi - lo) * (W - PADL - PADR)
    h = 2 * ROWH + 44
    rs = np.random.RandomState(0)
    parts = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{html.escape(title)}: 휴무일 중 급락한 날과 아닌 날의 분포">']
    for t in ticks:
        parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="8" y2="{2*ROWH+12}" class="cg"/>'
                     f'<text x="{x(t):.1f}" y="{2*ROWH+28}" class="ct" text-anchor="middle">{t:g}</text>')
    for i, (lab, name, cls) in enumerate(ROWS):
        v = o.loc[o["label"] == lab, col].dropna() * mult
        dates = o.loc[v.index, "date"].dt.strftime("%Y-%m-%d")
        cy = 10 + i * ROWH + ROWH / 2
        q1, med, q3 = v.quantile([.25, .5, .75])
        parts.append(f'<text x="{PADL-10}" y="{cy+4:.1f}" class="cl" text-anchor="end">{name} {len(v)}</text>')
        for val, d in zip(v, dates):
            parts.append(f'<circle cx="{x(val):.1f}" cy="{cy + rs.uniform(-9, 9):.1f}" r="2.6" class="dot {cls}">'
                         f'<title>{d} · {val:.1f}{unit}</title></circle>')
        parts.append(f'<rect x="{x(q1):.1f}" y="{cy-12:.1f}" width="{max(x(q3)-x(q1),2):.1f}" height="24" rx="4" class="iqr {cls}">'
                     f'<title>{name}: 중앙값 {med:.1f}{unit}, 가운데 50% {q1:.1f}~{q3:.1f}{unit}</title></rect>')
        parts.append(f'<line x1="{x(med):.1f}" x2="{x(med):.1f}" y1="{cy-14:.1f}" y2="{cy+14:.1f}" class="med"/>')
    parts.append("</svg>")
    md = o.groupby("label")[col].median() * mult
    tag = '<em class="tag post">사후 관측</em>' if when == "사후" else '<em class="tag">전날 15시</em>'
    return (f'<figure class="cmp"><figcaption><b>{html.escape(title)} {tag}</b><span>{hint} · 구분력 {auc(o, col):.2f} · '
            f'중앙값 급락 {md.get(1, np.nan):.1f}{unit} / 급락 안 함 {md.get(0, np.nan):.1f}{unit}</span></figcaption>'
            + "".join(parts) + "</figure>")


def auc_chart(o):
    items = sorted(((auc(o, c), c) for c in FACTORS), reverse=True)
    w, rowh, left, right = 640, 24, 250, 50
    h = len(items) * rowh + 34
    x = lambda a: left + (a - 0.5) / 0.5 * (w - left - right)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="요인별 구분력 순위">']
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="4" y2="{h-24}" class="cg"/>'
                     f'<text x="{x(t):.1f}" y="{h-8}" class="ct" text-anchor="middle">{t:.1f}</text>')
    for i, (a, c) in enumerate(items):
        title, *_, when = FACTORS[c]
        y0 = 6 + i * rowh
        cls = "barp" if when == "사후" else "barf"
        label = f"{title}{' (사후)' if when == '사후' else ''}"
        parts.append(f'<text x="{left-8}" y="{y0+13}" class="cl" text-anchor="end">{html.escape(label)}</text>'
                     f'<rect x="{x(0.5):.1f}" y="{y0+2}" width="{max(x(a)-x(0.5),2):.1f}" height="{rowh-8}" rx="4" class="{cls}">'
                     f'<title>{html.escape(label)}: 구분력 {a:.3f}</title></rect>'
                     f'<text x="{x(a)+6:.1f}" y="{y0+13}" class="cv">{a:.2f}</text>')
    parts.append("</svg>")
    return ('<figure class="cmp wide"><figcaption><b>요인별 구분력: 휴무일 중 급락한 날과 아닌 날을 얼마나 잘 가르나</b>'
            '<span>0.5 = 전혀 못 가름, 1.0 = 완벽히 가름 (ROC-AUC). 아래 그림은 이 순위에서 골랐습니다.</span></figcaption>'
            '<div class="legend cmp-legend" style="margin:0 0 6px"><span><span class="sw cf"></span> 전날 15시에 알 수 있는 정보</span>'
            '<span><span class="sw cp"></span> 그날이 지나야 아는 실측 (참고)</span></div>' + "".join(parts) + "</figure>")


def rate_bars(groups, title, note):
    """groups: [(이름, 일수, 급락수)] → 휴무일 급락률 막대."""
    n = len(groups)
    w, h, top, left = 640, 190, 22, 40
    step = (w - left - 8) / n
    bw = min(56, step * 0.72)
    ymax = max(0.4, max(d / max(c, 1) for _, c, d in groups) * 1.15)
    y = lambda r: top + (1 - r / ymax) * (h - top - 44)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">']
    for t in (0, .1, .2, .3, .4):
        if t <= ymax:
            parts.append(f'<line x1="{left}" x2="{w-4}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="cg"/>'
                         f'<text x="{left-6}" y="{y(t)+3:.1f}" class="ct" text-anchor="end">{int(t*100)}%</text>')
    for i, (name, c, d) in enumerate(groups):
        r = d / c if c else 0
        x0 = left + 4 + i * step + (step - bw) / 2
        parts.append(f'<rect x="{x0:.1f}" y="{y(r):.1f}" width="{bw:.1f}" height="{max(y(0)-y(r),0):.1f}" rx="4" class="bar">'
                     f'<title>{name}: 휴무일 {c}일 중 급락 {d}일 ({r:.0%})</title></rect>')
        parts.append(f'<text x="{x0+bw/2:.1f}" y="{y(r)-5:.1f}" class="cv" text-anchor="middle">{r:.0%}</text>')
        parts.append(f'<text x="{x0+bw/2:.1f}" y="{h-22}" class="cl" text-anchor="middle">{name}</text>'
                     f'<text x="{x0+bw/2:.1f}" y="{h-8}" class="ct" text-anchor="middle">{d}/{c}</text>')
    parts.append("</svg>")
    return (f'<figure class="cmp wide"><figcaption><b>{html.escape(title)}</b><span>{note}</span></figcaption>'
            + "".join(parts) + "</figure>")


def main():
    df = add_rel_radiation(data2())
    df = df[df["date"].between("2024-01-01", "2026-10-18")]
    o = df[df["is_offday"] == 1].copy()
    n_all, n_drop = len(o), int(o["label"].sum())
    wd = df[df["is_offday"] == 0]

    month = [(f"{m}월", int((o.date.dt.month == m).sum()), int(o.loc[o.date.dt.month == m, "label"].sum()))
             for m in range(1, 13)]

    def kind(r):
        if r.is_myeongjeol:
            return "명절"
        if r.off_block_len >= 3:
            return "연휴 3일째+" if r.off_block_day >= 3 else f"연휴 {int(r.off_block_day)}일째"
        return "단독 공휴일" if r.is_holiday_single else "일반 주말"
    o["kind"] = o.apply(kind, axis=1)
    order = ["일반 주말", "단독 공휴일", "연휴 1일째", "연휴 2일째", "연휴 3일째+", "명절"]
    kinds = [(k, int((o.kind == k).sum()), int(o.loc[o.kind == k, "label"].sum())) for k in order]

    med = o.groupby("label")[list(FACTORS)].median()
    ranked = sorted(FACTORS, key=lambda c: -auc(o, c))
    rows = "".join(
        f'<tr><td>{FACTORS[c][0]}</td><td>{"그날 지나야 앎" if FACTORS[c][5] == "사후" else "전날 15시"}</td>'
        f'<td class="n">{auc(o, c):.2f}</td><td class="n">{med.loc[1, c]*FACTORS[c][2]:.1f}{FACTORS[c][1]}</td>'
        f'<td class="n">{med.loc[0, c]*FACTORS[c][2]:.1f}{FACTORS[c][1]}</td><td>{FACTORS[c][4]}</td></tr>'
        for c in ranked)

    sec = f"""
<section id="compare">
  <h2>휴무일 중 급락한 날은 무엇이 달랐나</h2>
  <p><b>주말·공휴일(휴무일)끼리</b> 비교했습니다. 2024-01 ~ 2026-10-18의 휴무일 {n_all}일 중 <b>급락한 {n_drop}일</b>과 <b>급락하지 않은 {n_all-n_drop}일</b>입니다. 평일은 {len(wd)}일 중 급락이 {int(wd.label.sum())}일뿐이라 비교에서 뺐습니다. 이 그림은 설명용이며, 모델을 고르는 데는 2025년까지 자료만 썼습니다.</p>
  <p>요인은 두 무리를 얼마나 잘 가르는지(구분력)를 계산해 높은 순서로 골랐습니다. 판단 시각에 알 수 있는 정보에는 "전날 15시", 그날이 지나야 아는 실측에는 "사후 관측"을 붙였습니다.</p>
  {auc_chart(o)}
  <p class="callout"><b>가장 잘 가르는 것은 최근 SMP 흐름(0.84)</b>이고, 그다음이 <b>구름·강수 예보(0.70~0.75)</b>입니다. 전날 예보만으로도 그날 실측 운량(0.73)과 비슷하게 가릅니다. 반면 <b>일사량 원값은 거의 못 가릅니다(0.51)</b>. 급락이 해가 약한 11~2월에 몰려, 맑은 겨울날 일사량이 흐린 봄날보다 낮기 때문입니다. 같은 시기 맑은 날과 비교한 비율로 바꿔야 차이가 보입니다.</p>
  <div class="legend cmp-legend"><span><span class="sw cd"></span> 휴무일 중 급락한 날 ({n_drop}일)</span><span><span class="sw cn"></span> 휴무일 중 급락하지 않은 날 ({n_all-n_drop}일)</span><span class="muted">점 하나 = 하루 · 막대 = 가운데 50% · 세로선 = 중앙값</span></div>
  <div class="cmp-grid">{''.join(strip_panel(o, c) for c in PANELS)}</div>
  <p class="callout">두 무리가 많이 겹쳐 있어서 한 가지 조건만으로는 가를 수 없습니다. 최근 SMP가 낮아진 국면에, 맑고, 수요가 낮은 조건이 겹칠 때 급락합니다. 그래서 여러 조건을 함께 보는 모델이 필요합니다.</p>
  <div class="cmp-grid one">
    {rate_bars(month, "월별 휴무일 급락률", "태양광이 가장 강한 5월이 아니라 11~2월에 몰립니다")}
    {rate_bars(kinds, "휴무일 유형별 급락률", "설·추석 명절 연휴가 39%로 가장 높고, 명절이 아닌 연휴는 0%입니다")}
  </div>
  <div class="panel scroll"><table><thead><tr><th>요인 (휴무일끼리 비교)</th><th>알 수 있는 때</th><th class="n">구분력</th><th class="n">급락한 휴무일 중앙값</th><th class="n">급락 안 한 휴무일 중앙값</th><th>방향</th></tr></thead><tbody>{rows}</tbody></table></div>
</section>
"""
    OUT.write_text(sec, encoding="utf-8")
    print(f"저장: {OUT} (휴무일 {n_all}일, 급락 {n_drop}일)")
    for c in ranked:
        print(f"  {auc(o, c):.3f}  {FACTORS[c][0]} ({FACTORS[c][5]})")


if __name__ == "__main__":
    main()
