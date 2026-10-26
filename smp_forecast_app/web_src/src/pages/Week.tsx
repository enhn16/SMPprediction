import { useMemo, useState } from 'react'
import { D, REGION } from '../data'
import { avg, fmt, inside, lab, md, pct } from '../fmt'
import { bandOption, EChart } from '../components/Chart'
import { DropMeter } from '../components/Drop'
import { Card, Legend, PageHeader, Seg } from '../components/ui'

/** 최근 7일(공표가 있는 날) 하루 전 예측 성적 — 지금 예측을 얼마나 믿을지 가늠 */
function RecentScore() {
  const src = D.past.hourly['1'] ?? {}
  const ds = Object.keys(src).filter((d) => (D.past.actual[d] ?? []).some((v) => v != null)).sort().slice(-7)
  if (!ds.length) return null
  const e: number[] = [], inn: boolean[] = [], lo: boolean[] = []
  ds.forEach((d) => {
    const [p, l, h] = src[d], y = D.past.actual[d]
    y.forEach((v, i) => {
      if (v == null || p[i] == null || l[i] == null || h[i] == null) return
      e.push(Math.abs(v - (p[i] as number))); inn.push(inside(v, l[i], h[i])); lo.push(v < (l[i] as number))
    })
  })
  const cov = inn.filter(Boolean).length / inn.length, below = lo.filter(Boolean).length / lo.length
  return (
    <div className="card flex flex-wrap items-center gap-x-6 gap-y-1 px-5 py-3 text-[13px] text-sub">
      <b className="text-ink">최근 성적</b>
      <span>{md(ds[0])} ~ {md(ds[ds.length - 1])} ({ds.length}일) 하루 전 예측</span>
      <span>평균 오차 <b className="text-brand-dark">{fmt(avg(e))}원</b></span>
      <span>80% 범위 포함률 <b className={cov >= 0.75 && cov <= 0.85 ? 'text-good' : 'text-bad'}>{pct(cov)}</b> <span className="text-mute">(목표 75~85%)</span></span>
      <span>하한 아래 {pct(below)} · 상한 위 {pct(1 - cov - below)}</span>
    </div>
  )
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

  return (
    <div className="space-y-5">
      <PageHeader title="일주일 예측" sub={<>{lab(days[0].date)} ~ {lab(days[days.length - 1].date)} 시간별 SMP · 기준일 {W.base} 15시 판단 · 80% 범위</>}
        right={<Seg value={view === 'all' ? 'all' : 'day'} onChange={(v) => setView(v === 'all' ? 'all' : 0)} options={[{ v: 'all', t: '7일 전체' }, { v: 'day', t: '날짜별' }]} />} />
      <DropMeter d={W.drop} date={lab(days[0].date)} />
      <RecentScore />
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
      {REGION === '제주'
        ? <p className="muted">제주 가격은 대부분의 시간에 육지와 같아(2026년 91%) 육지 일주일 예측에 제주 잔차 보정(0.25배)을 더해 냅니다. 범위 하한은 음수 가격이 있어 0원에서 자르지 않습니다. 위의 급락 판별은 과제1 제주 모델(KNN 분류)이 따로 계산합니다.</p>
        : <p className="muted">내일(첫날) 시간별 가격은 일주일 모델(Chronos-2 small + 잔차 보정)의 D+1입니다. 위의 급락 판별은 과제1 모델이 따로 계산합니다.</p>}
    </div>
  )
}
