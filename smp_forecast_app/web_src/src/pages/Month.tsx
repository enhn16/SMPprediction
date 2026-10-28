import { useMemo, useState } from 'react'
import { D, REGION, STAT_NAME, type Stat } from '../data'
import { avg, fmt, lab, md } from '../fmt'
import { bandOption, EChart } from '../components/Chart'
import { Card, Legend, PageHeader, Pill, Seg } from '../components/ui'

export function Month() {
  const M = D.month
  const [k, setK] = useState<Stat>('mean')
  const n = M.d.length
  const s = { p: M['p_' + k], lo: M['lo_' + k], hi: M['hi_' + k], y: M['y_' + k] }
  const opt = useMemo(() => bandOption({
    x: M.d, ...s, tip: (i) => `${lab(M.d[i])} ${M.dt[i] && M.dt[i] !== '평일' ? M.dt[i] : ''} · ${STAT_NAME[k]}`,
    xLabel: (i) => (i % 2 === 0 ? md(M.d[i]) : ''), symbols: true, daily: true,
    shade: M.d.map((_, i) => i).filter((i) => M.dt[i] && M.dt[i] !== '평일').map((i) => [i, i] as [number, number]), dividers: [7, 14, 21],
  }), [k])
  // 주차 평균 (예측·하한·상한)과 범위 막대 공통 눈금
  const weeks = [0, 1, 2, 3].map((w) => {
    const r = [...Array(7).keys()].map((j) => w * 7 + j).filter((j) => j < n)
    return { p: avg(r.map((j) => s.p[j])), lo: avg(r.map((j) => s.lo[j])), hi: avg(r.map((j) => s.hi[j])) }
  })
  const ext = weeks.flatMap((x) => [x.lo, x.hi]).filter((v): v is number => v != null)
  const a0 = Math.min(...ext), a1 = Math.max(...ext)
  const pos = (v: number | null) => `${v == null || a1 === a0 ? 50 : ((v - a0) / (a1 - a0)) * 90 + 5}%`
  const byMonth: Record<string, number[]> = {}
  M.d.forEach((d, i) => { (byMonth[d.slice(0, 7)] ||= []).push(i) })

  return (
    <div className="space-y-5">
      <PageHeader eyebrow={`예측 · ${REGION}`} title="한 달 예측" sub={<>7~34일 뒤 일별 SMP와 80% 범위 <Pill tone="gray">참고 전망</Pill></>}
        meta={[{ k: '기준일', v: <>{M.base} 15시 판단</>, hl: true }, { k: '대상', v: <>{lab(M.d[0])} ~ {lab(M.d[n - 1])}</> }, { k: '지표', v: STAT_NAME[k] }]}
        right={<Seg value={k} onChange={setK} options={(Object.keys(STAT_NAME) as Stat[]).map((v) => ({ v, t: STAT_NAME[v] }))} />} />
      {/* 주차 요약: 한 면에 여백으로 4칸, 범위 막대는 네 주 공통 눈금 */}
      <section className="card px-6 py-6 lg:px-7">
        <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
          <h2 className="h-sec">주차별 평균 · {STAT_NAME[k]}</h2>
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
            const r = [...Array(7).keys()].map((j) => w * 7 + j).filter((j) => j < n)
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
      <Card title={`일별 예측 · ${STAT_NAME[k]}`} sub="회색 = 휴무일 · 세로선 = 주차 경계"
        right={<Legend items={[{ t: '80% 범위', c: 'band' }, { t: '예측', c: 'pred' }, { t: '공표 SMP', c: 'act' }, { t: '범위 밖', c: 'out' }]} />}>
        <EChart option={opt} height={360} />
      </Card>
      <Card title="일별 표" sub="하한 · 예측 · 상한 · 공표" flush>
        <div className="tbl-wrap max-h-[480px]">
          <table className="tbl">
            <thead className="sticky top-0 z-[1]"><tr><th className="l">날짜</th><th>며칠 뒤</th><th>하한</th><th>예측</th><th>상한</th><th>공표</th></tr></thead>
            <tbody>
              {M.d.map((d: string, i: number) => (
                <tr key={d}>
                  <td className={`l ${M.dt[i] && M.dt[i] !== '평일' ? 'text-bad' : ''}`}>{lab(d)} {M.dt[i] && M.dt[i] !== '평일' ? M.dt[i] : ''}</td>
                  <td>{i + 7}</td><td>{fmt(s.lo[i])}</td><td className="font-semibold text-brand-dark">{fmt(s.p[i])}</td><td>{fmt(s.hi[i])}</td><td className="font-semibold">{fmt(s.y[i])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <p className="muted">1~4주 앞은 연료비·수급 국면 변화가 가격을 좌우해 정확도가 낮습니다. <b>한 달 범위는 평소 달 기준이며, 연료비·수급 변화가 있는 달은 벗어날 수 있습니다</b> (2026 포함률 57~64%). 참고 전망으로 보세요 (참고 문서 &gt; 한 달 예측).</p>
    </div>
  )
}
