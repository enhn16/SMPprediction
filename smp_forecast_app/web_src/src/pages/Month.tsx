import { useMemo } from 'react'
import { D, REGION } from '../data'
import { avg, fmt, lab, md, sgn } from '../fmt'
import { bandOption, EChart } from '../components/Chart'
import { Card, Legend, PageHeader, Pill } from '../components/ui'
import { P, mix } from '../theme'

// 하루 모양을 함께 보는 보조 선: 저녁(수요 피크, 따뜻한 색) · 한낮(태양광, 청록)
const EVE = P.orange, MID = mix(P.cyan, '#000000', 0.35)

export function Month() {
  const M = D.month
  const n = M.d.length
  const next = M.mode === 'next_month'   // 육지 MM: 월말 판단 → 다음 달 전체 (일평균만), 제주: 매일 7~34일 뒤 4지표
  const tm = +M.d[0].slice(5, 7)
  // 주차: 앞 3주는 7일씩, 4주차는 나머지 전부 (다음 달 29~31일 포함)
  const wkIdx = (w: number) => [...Array(n).keys()].filter((j) => (w < 3 ? Math.floor(j / 7) === w : j >= 21))
  // 주인공 = 일평균(80% 범위 포함), 한낮·저녁·한낮 최저는 함께 보여 줌 (지표 탭 없음)
  const s = { p: M.p_mean, lo: M.lo_mean, hi: M.hi_mean, y: M.y_mean }
  const opt = useMemo(() => bandOption({
    x: M.d, ...s, tip: (i) => `${lab(M.d[i])} ${M.dt[i] && M.dt[i] !== '평일' ? M.dt[i] : ''} · 일평균`,
    extra: M.p_eve ? [{ name: '저녁 18~22시', data: M.p_eve, color: EVE }, { name: '한낮 11~15시', data: M.p_mid, color: MID }] : [],
    xLabel: (i) => (i % (n > 28 ? 3 : 2) === 0 ? md(M.d[i]) : ''), symbols: true, daily: true,
    shade: M.d.map((_, i) => i).filter((i) => M.dt[i] && M.dt[i] !== '평일').map((i) => [i, i] as [number, number]), dividers: [7, 14, 21],
  }), [])
  // 주차 평균 (예측·하한·상한)과 범위 막대 공통 눈금
  const weeks = [0, 1, 2, 3].map((w) => {
    const r = wkIdx(w)
    return { p: avg(r.map((j) => s.p[j])), lo: avg(r.map((j) => s.lo[j])), hi: avg(r.map((j) => s.hi[j])) }
  })
  const ext = weeks.flatMap((x) => [x.lo, x.hi]).filter((v): v is number => v != null)
  const a0 = Math.min(...ext), a1 = Math.max(...ext)
  const pos = (v: number | null) => `${v == null || a1 === a0 ? 50 : ((v - a0) / (a1 - a0)) * 90 + 5}%`
  const byMonth: Record<string, number[]> = {}
  M.d.forEach((d, i) => { (byMonth[d.slice(0, 7)] ||= []).push(i) })

  return (
    <div className="space-y-5">
      <PageHeader eyebrow={`예측 · ${REGION}`} title={next ? `${tm}월 한 달 예측` : '한 달 예측'}
        sub={<>{next ? '매월 말 판단 → 다음 달 일평균 SMP와 80% 범위' : '7~34일 뒤 일별 SMP와 80% 범위'} <Pill tone="gray">참고 전망</Pill></>}
        meta={[{ k: '기준일', v: <>{M.base} 15시 판단{next ? ' (월말)' : ''}</>, hl: true }, { k: '대상', v: <>{lab(M.d[0])} ~ {lab(M.d[n - 1])}</> }]} />
      {/* 주차 요약: 한 면에 여백으로 4칸, 범위 막대는 네 주 공통 눈금 */}
      <section className="card px-6 py-6 lg:px-7">
        <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
          <h2 className="h-sec">주차별 일평균</h2>
          <div className="flex flex-wrap gap-x-5 text-[12.5px]">
            {Object.entries(byMonth).map(([ym, idx]) => (
              <span key={ym} className="text-sub tabular-nums">
                <b className="font-semibold text-ink">{+ym.slice(5)}월</b> {idx.length}일분 평균 <b className="font-semibold text-brand-dark">{fmt(avg(idx.map((i) => s.p[i])))}원</b>
                <span className="text-mute"> · 범위 {fmt(avg(idx.map((i) => s.lo[i])))}~{fmt(avg(idx.map((i) => s.hi[i])))}</span>
              </span>
            ))}
          </div>
        </div>
        <div className="mt-5 grid grid-cols-2 gap-x-10 gap-y-7 lg:grid-cols-4">
          {[0, 1, 2, 3].map((w) => {
            const r = wkIdx(w)
            if (!r.length) return null
            const ys = r.map((j) => s.y[j]).filter((v: any) => v != null)
            const wk = weeks[w]
            return (
              <div key={w} className="min-w-0">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="t-label">{w + 1}주차</span>
                  <span className="text-[11.5px] text-mute tabular-nums">{md(M.d[r[0]])} ~ {md(M.d[r[r.length - 1]])}</span>
                </div>
                <div className="v-kpi mt-1.5 text-ink">{fmt(wk.p)}<span className="unit">원</span></div>
                <div className="relative mt-3 h-1.5 rounded-full bg-track">
                  <span className="absolute inset-y-0 rounded-full bg-band" style={{ left: pos(wk.lo), width: `calc(${pos(wk.hi)} - ${pos(wk.lo)})` }} />
                  <span className="absolute -inset-y-[3px] w-[3px] -translate-x-1/2 rounded-full bg-brand" style={{ left: pos(wk.p) }} />
                </div>
                <div className="mt-2 flex justify-between gap-2 text-[11.5px] tabular-nums text-mute">
                  <span>범위 {fmt(wk.lo)}~{fmt(wk.hi)}</span>
                  <span>{ys.length ? <>공표 <b className="font-semibold text-ink">{fmt(avg(ys))}</b> ({ys.length}일)</> : '공표 전'}</span>
                </div>
              </div>
            )
          })}
        </div>
      </section>
      <Card title="일별 예측" sub={next ? '회색 = 휴무일 · 세로선 = 주차 경계 · 지난 날은 공표 일평균과 비교' : '회색 = 휴무일 · 세로선 = 주차 경계'}
        right={<Legend items={[{ t: '일평균 · 80% 범위', c: 'band' }, { t: '일평균 예측', c: 'pred' }, ...(M.p_eve ? [{ t: '저녁 18~22시', c: EVE }, { t: '한낮 11~15시', c: MID }] : []), { t: '공표 일평균', c: 'act' }]} />}>
        <EChart option={opt} height={360} />
      </Card>
      <Card title="일별 표" sub={M.p_mid ? '일평균(80% 범위) · 한낮 · 저녁 · 한낮 최저 · 공표 일평균' : '일평균(80% 범위) · 공표 일평균'} flush>
        <div className="tbl-wrap max-h-[480px]">
          <table className="tbl">
            <thead className="sticky top-0 z-[1]"><tr><th className="l">날짜</th>{!next && <th>며칠 뒤</th>}<th>일평균</th><th>80% 범위</th>{M.p_mid && <><th>한낮 11~15시</th><th>저녁 18~22시</th><th>한낮 최저</th></>}<th>공표 일평균</th><th>오차 (공표−예측)</th></tr></thead>
            <tbody>
              {M.d.map((d: string, i: number) => (
                <tr key={d}>
                  <td className={`l ${M.dt[i] && M.dt[i] !== '평일' ? 'text-bad' : ''}`}>{lab(d)} {M.dt[i] && M.dt[i] !== '평일' ? M.dt[i] : ''}</td>
                  {!next && <td>{i + 7}</td>}<td className="font-semibold text-brand-dark">{fmt(s.p[i])}</td><td className="text-sub">{fmt(s.lo[i])} ~ {fmt(s.hi[i])}</td>
                  {M.p_mid && <><td>{fmt(M.p_mid[i])}</td><td>{fmt(M.p_eve[i])}</td><td>{fmt(M.p_midmin[i])}</td></>}<td className="font-semibold">{fmt(s.y[i])}</td>
                  <td>{s.y[i] != null && s.p[i] != null ? sgn(s.y[i] - s.p[i]) : '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <p className="muted">{next
        ? <>매월 말일 15시에 다음 달 모든 날의 일평균을 한 번 예측하고, 그달 내내 같은 예측을 보여 줍니다. 대상 달의 가스공사 발전용 요금을 반영합니다. 1~4주 앞은 수급 국면 변화로 빗나갈 수 있으니 참고 전망으로 보세요 (참고 문서 &gt; 한 달 예측).</>
        : <>1~4주 앞은 연료비·수급 국면 변화가 가격을 좌우해 정확도가 낮습니다. <b>한 달 범위는 평소 달 기준이며, 연료비·수급 변화가 있는 달은 벗어날 수 있습니다.</b> 참고 전망으로 보세요 (참고 문서 &gt; 한 달 예측).</>}</p>
    </div>
  )
}
