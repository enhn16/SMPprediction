import { useMemo, useState } from 'react'
import { D } from '../data'
import { avg, fmt, lab } from '../fmt'
import { bandOption, EChart } from '../components/Chart'
import { DropMeter } from '../components/Drop'
import { Card, Legend, PageHeader, Seg } from '../components/ui'

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

  return (
    <div className="space-y-5">
      <PageHeader title="일주일 예측" sub={<>{lab(days[0].date)} ~ {lab(days[days.length - 1].date)} 시간별 SMP · 기준일 {W.base} 15시 판단 · 80% 범위</>}
        right={<Seg value={view === 'all' ? 'all' : 'day'} onChange={(v) => setView(v === 'all' ? 'all' : 0)} options={[{ v: 'all', t: '7일 전체' }, { v: 'day', t: '날짜별' }]} />} />
      <DropMeter d={W.drop} date={lab(days[0].date)} />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 xl:grid-cols-7">
        {days.map((d, i) => {
          const mn = Math.min(...[10, 11, 12, 13, 14].map((j) => d.p[j] ?? Infinity))
          const on = view === i
          return (
            <button key={d.date} onClick={() => setView(i)}
              className={`card px-3.5 py-3 text-left transition hover:border-brand/60 ${on ? 'border-brand ring-2 ring-brand/20' : ''}`}>
              <div className={`text-[12px] font-semibold ${d.off ? 'text-bad' : 'text-sub'}`}>{lab(d.date)} {d.dt && d.dt !== '평일' ? d.dt : ''}</div>
              <div className="mt-1 text-[21px] font-bold text-brand-dark tabular-nums">{fmt(avg(d.p))}<span className="ml-0.5 text-[12px] font-medium text-sub">원</span></div>
              <div className="text-[11.5px] text-mute">범위 {fmt(avg(d.lo))} ~ {fmt(avg(d.hi))}</div>
              <div className="text-[11.5px] text-mute">한낮 최저 {fmt(mn)}</div>
            </button>
          )
        })}
      </div>
      <Card title={view === 'all' ? '시간별 예측 (7일)' : `${lab(days[view].date)} 시간별 예측`} sub="회색 = 한낮 11~15시"
        right={<Legend items={[{ t: '80% 범위', c: 'band' }, { t: '예측', c: 'pred' }, { t: '공표 SMP', c: 'act' }, { t: '범위 밖', c: 'out' }]} />}>
        <EChart option={opt} height={360} />
      </Card>
      <Card title="시간별 표" right={<button onClick={() => setShowTable(!showTable)} className="text-[13px] font-semibold text-brand-dark">{showTable ? '접기' : '펼치기'}</button>}>
        {showTable && (
          <div className="max-h-[480px] overflow-auto rounded-lg border border-line">
            <table className="tbl">
              <thead className="sticky top-0"><tr><th className="l">날짜</th><th className="l">시간</th><th>하한</th><th>예측</th><th>상한</th><th>공표 SMP</th></tr></thead>
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
        )}
      </Card>
      <p className="muted">내일(첫날) 시간별 가격은 일주일 모델(Chronos-2 small + 잔차 보정)의 D+1입니다. 위의 급락 판별은 과제1 모델이 따로 계산합니다.</p>
    </div>
  )
}
