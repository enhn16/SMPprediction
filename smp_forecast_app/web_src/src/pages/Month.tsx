import { useMemo, useState } from 'react'
import { D, STAT_NAME, type Stat } from '../data'
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
  const byMonth: Record<string, number[]> = {}
  M.d.forEach((d, i) => { (byMonth[d.slice(0, 7)] ||= []).push(i) })

  return (
    <div className="space-y-5">
      <PageHeader title="한 달 예측" sub={<>{lab(M.d[0])} ~ {lab(M.d[n - 1])} 일별 · 기준일 {M.base} 15시 판단 · 7~34일 뒤 <Pill tone="gray">참고 전망</Pill></>}
        right={<Seg value={k} onChange={setK} options={(Object.keys(STAT_NAME) as Stat[]).map((v) => ({ v, t: STAT_NAME[v] }))} />} />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {[0, 1, 2, 3].map((w) => {
          const r = [...Array(7).keys()].map((j) => w * 7 + j).filter((j) => j < n)
          if (!r.length) return null
          const ys = r.map((j) => s.y[j]).filter((v: any) => v != null)
          return (
            <div key={w} className="card px-4 py-3.5">
              <div className="text-[12px] font-semibold text-sub">{w + 1}주차 · {md(M.d[r[0]])} ~ {md(M.d[r[r.length - 1]])}</div>
              <div className="mt-1 text-[22px] font-bold text-brand-dark tabular-nums">{fmt(avg(r.map((j) => s.p[j])))}<span className="ml-0.5 text-[12px] font-medium text-sub">원</span></div>
              <div className="text-[12px] text-mute">범위 {fmt(avg(r.map((j) => s.lo[j])))} ~ {fmt(avg(r.map((j) => s.hi[j])))}</div>
              {ys.length > 0 && <div className="text-[12px] font-semibold text-ink">공표 {fmt(avg(ys))}원 ({ys.length}일)</div>}
            </div>
          )
        })}
      </div>
      <div className="flex flex-wrap gap-2">
        {Object.entries(byMonth).map(([ym, idx]) => (
          <span key={ym} className="rounded-lg border border-line bg-white px-3 py-1.5 text-[13px]">
            <b className="text-brand-dark">{+ym.slice(5)}월</b> ({idx.length}일분) 평균 예상 <b>{fmt(avg(idx.map((i) => s.p[i])))}원</b>
            <span className="text-mute"> · 범위 {fmt(avg(idx.map((i) => s.lo[i])))}~{fmt(avg(idx.map((i) => s.hi[i])))}</span>
          </span>
        ))}
      </div>
      <Card title={`일별 예측 · ${STAT_NAME[k]}`} sub="회색 = 휴무일 · 세로선 = 주차 경계"
        right={<Legend items={[{ t: '80% 범위', c: 'band' }, { t: '예측', c: 'pred' }, { t: '공표 SMP', c: 'act' }, { t: '범위 밖', c: 'out' }]} />}>
        <EChart option={opt} height={360} />
      </Card>
      <Card title="일별 표">
        <div className="max-h-[480px] overflow-auto rounded-lg border border-line">
          <table className="tbl">
            <thead className="sticky top-0"><tr><th className="l">날짜</th><th>며칠 뒤</th><th>하한</th><th>예측</th><th>상한</th><th>공표</th></tr></thead>
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
      <p className="muted">1~4주 앞은 연료비·수급 국면 변화가 가격을 좌우해 정확도가 낮습니다. 참고 전망으로 보세요 (참고 문서 &gt; 한 달 예측).</p>
    </div>
  )
}
