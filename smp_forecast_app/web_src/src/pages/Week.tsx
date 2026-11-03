import { useMemo, useState } from 'react'
import { D, REGION } from '../data'
import { avg, downloadCsv, fmt, inside, lab, pct } from '../fmt'
import { bandOption, EChart } from '../components/Chart'
import { Alerts } from '../components/Alerts'
import { Card, CsvButton, Legend, PageHeader } from '../components/ui'
import { LV } from '../theme'

/** 최근 7일(공표가 있는 날) 하루 전 예측 성적 한 줄 — 지금 예측을 얼마나 믿을지 가늠 */
function recentText() {
  const src = D.past.hourly['1'] ?? {}
  const ds = Object.keys(src).filter((d) => (D.past.actual[d] ?? []).some((v) => v != null)).sort().slice(-7)
  if (!ds.length) return null
  const e: number[] = [], inn: boolean[] = []
  ds.forEach((d) => {
    const [p, l, h] = src[d], y = D.past.actual[d]
    y.forEach((v, i) => {
      if (v == null || p[i] == null || l[i] == null || h[i] == null) return
      e.push(Math.abs(v - (p[i] as number))); inn.push(inside(v, l[i], h[i]))
    })
  })
  return `오차 ${fmt(avg(e))}원 · 범위 포함 ${pct(inn.filter(Boolean).length / inn.length, 0)}`
}

export function Week() {
  const W = D.week, days = W.days
  const [view, setView] = useState<'all' | number>('all')
  const [showTable, setShowTable] = useState(false)
  const opt = useMemo(() => {
    if (view === 'all') {
      const cat = (k: 'p' | 'lo' | 'hi' | 'y') => days.flatMap((d) => d[k])
      return bandOption({
        x: days.flatMap((d) => d.p.map((_, i) => `${d.date} ${i + 1}`)), p: cat('p'), lo: cat('lo'), hi: cat('hi'), y: cat('y'),
        tip: (i) => `${lab(days[Math.floor(i / 24)].date)} ${(i % 24) + 1}시`, xLabel: (i) => (i % 24 === 11 ? lab(days[Math.floor(i / 24)].date) : ''),
        shade: days.map((_, k) => [k * 24 + 10, k * 24 + 14] as [number, number]), dividers: days.map((_, k) => k * 24).slice(1),
      })
    }
    const d = days[view]
    return bandOption({ x: d.p.map((_, i) => `${i + 1}시`), p: d.p, lo: d.lo, hi: d.hi, y: d.y, tip: (i) => `${lab(d.date)} ${i + 1}시`,
      xLabel: (i) => (i % 2 === 0 ? `${i + 1}시` : ''), shade: [[10, 14]], symbols: true })
  }, [view, days])

  // 날짜 고르기: [7일] + 날짜 7개 (누르면 아래 차트가 그날로). 선택 = 진한 파랑 채움
  const pick = (on: boolean) => `rounded-ui px-3 py-2.5 text-left transition ${on ? 'bg-brand-dark text-white' : 'hover:bg-track'}`
  return (
    <div className="space-y-5">
      <PageHeader eyebrow={`예측 · ${REGION}`} title="일주일 예측" sub="내일부터 7일 동안의 시간별 SMP와 80% 범위"
        meta={[{ k: '기준일', v: <>{W.base} 15시 판단</>, hl: true }, { k: '대상', v: <>{lab(days[0].date)} ~ {lab(days[days.length - 1].date)}</> }, { k: '범위', v: '80%' }, ...(recentText() ? [{ k: '최근 7일 정확도', v: recentText() }] : [])]} />
      <Alerts iso={days[0].date} />
      <section className="card">
        <div className="panel-head">
          <div className="flex flex-wrap items-baseline gap-x-3">
            <h2 className="h-sec">{view === 'all' ? '시간별 예측 · 7일' : `${lab(days[view].date)} 시간별 예측`}</h2>
            <span className="muted">회색 = 한낮 11~15시</span>
          </div>
          <Legend items={[{ t: '80% 범위', c: 'band' }, { t: '예측', c: 'pred' }, { t: '공표 SMP', c: 'act' }, { t: '범위 밖', c: 'out' }]} />
        </div>
        <div className="grid grid-cols-4 gap-1 px-4 pt-4 xl:grid-cols-8 lg:px-5">
          <button onClick={() => setView('all')} aria-pressed={view === 'all'} className={pick(view === 'all')}>
            <div className={`text-[12px] font-medium ${view === 'all' ? 'text-white/75' : 'text-sub'}`}>전체 평균</div>
            <div className="mt-1 text-[19px] font-semibold leading-tight tracking-tight">7일</div>
            <div className={`mt-0.5 text-[11.5px] tabular-nums ${view === 'all' ? 'text-white/70' : 'text-mute'}`}>{fmt(avg(days.flatMap((d) => d.p)))}원</div>
          </button>
          {days.map((d, i) => {
            const mn = Math.min(...[10, 11, 12, 13, 14].map((j) => d.p[j] ?? Infinity))
            const on = view === i
            const hol = d.dt && d.dt !== '평일'
            return (
              <button key={d.date} onClick={() => setView(i)} aria-pressed={on} className={pick(on)}>
                <div className={`flex items-baseline justify-between gap-1 text-[12px] font-medium ${on ? 'text-white/75' : d.off ? 'text-bad' : 'text-sub'}`}
                  style={on && d.off ? { color: LV.strong.soft } : undefined}>
                  <span className="truncate">{lab(d.date)}{hol && <span className="ml-1 text-[11px]">{d.dt}</span>}</span>
                  {i === 0 && <span className={`text-[11px] font-semibold ${on ? 'text-white' : 'text-brand'}`}>내일</span>}
                </div>
                <div className={`mt-1 text-[19px] font-semibold leading-tight tracking-tight tabular-nums ${on ? 'text-white' : 'text-ink'}`}>
                  {fmt(avg(d.p))}<span className={`ml-0.5 text-[11.5px] font-medium ${on ? 'text-white/70' : 'text-mute'}`}>원</span>
                </div>
                <div className={`mt-0.5 text-[11.5px] leading-4 tabular-nums ${on ? 'text-white/70' : 'text-mute'}`}>{fmt(avg(d.lo))}~{fmt(avg(d.hi))}<br />한낮 최저 {fmt(mn)}</div>
              </button>
            )
          })}
        </div>
        <div className="px-3 pb-4 pt-2 sm:px-5"><EChart option={opt} height={360} /></div>
      </section>
      <Card title="시간별 표" sub="하한 · 예측 · 상한 · 공표 SMP" flush right={<div className="flex gap-2">
        <CsvButton onClick={() => {
          const ds = view === 'all' ? days : [days[view]]
          downloadCsv(`${REGION}_일주일예측_기준${W.base}${view === 'all' ? '' : '_' + ds[0].date}`, ['기준일', '날짜', '요일유형', '시간', '하한', '예측', '상한', '공표 SMP'],
            ds.flatMap((d) => d.p.map((p, i) => [W.base, d.date, d.dt ?? '', i + 1, d.lo[i], p, d.hi[i], d.y[i]])))
        }} />
        <button onClick={() => setShowTable(!showTable)} className="btn">{showTable ? '접기' : '펼치기'}</button>
      </div>}>
        {showTable ? (
          <div className="tbl-wrap max-h-[480px]">
            <table className="tbl">
              <thead className="sticky top-0 z-[1]"><tr><th className="l">날짜</th><th className="l">시간</th><th>하한</th><th>예측</th><th>상한</th><th>공표 SMP</th></tr></thead>
              <tbody>
                {(view === 'all' ? days : [days[view]]).flatMap((d) => d.p.map((p, i) => (
                  <tr key={d.date + i}>
                    <td className={`l ${d.off ? 'text-bad' : ''}`}>{lab(d.date)}</td><td className="l">{i + 1}시</td>
                    <td>{fmt(d.lo[i])}</td><td className="font-semibold text-brand-dark">{fmt(p)}</td><td>{fmt(d.hi[i])}</td><td className="font-semibold">{fmt(d.y[i])}</td>
                  </tr>
                )))}
              </tbody>
            </table>
          </div>
        ) : null}
      </Card>
      {REGION === '제주'
        ? <p className="muted">제주 가격은 대부분의 시간에 육지와 같아(2026년 91%) 육지 일주일 예측에 제주 잔차 보정(0.25배)을 더해 냅니다. 범위 하한은 음수 가격이 있어 0원에서 자르지 않습니다. 위의 급락·급등 경보는 별도 모델이 계산합니다.</p>
        : <p className="muted">내일(첫날) 시간별 가격은 일주일 모델(K3)의 D+1입니다. 내일 급락 경보가 주의 이상이면 내일 11~15시 범위의 아래쪽을 0원까지 넓힙니다. 위의 급락·급등 경보는 별도 모델이 계산합니다.</p>}
    </div>
  )
}
