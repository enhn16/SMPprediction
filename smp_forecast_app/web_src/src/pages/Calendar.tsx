import React, { useMemo, useState } from 'react'
import { D, REGION, info } from '../data'
import { fmt, inside, lab, pct } from '../fmt'
import { bandOption, EChart } from '../components/Chart'
import { DropChip } from '../components/Drop'
import { SurgeChip } from '../components/Surge'
import { Card, Legend, PageHeader, Select } from '../components/ui'
import { LV, P } from '../theme'

// 하루 24시간 중 80% 범위 밖 시간 수 → 색 (고정 경계: 80% 범위는 하루 약 5시간 벗어나는 것이 정상)
const LEVELS = [
  { max: 5, name: '정상', range: '0~5시간', fill: LV.low.fill, soft: LV.low.soft, text: LV.low.text },
  { max: 10, name: '많이 빗나감', range: '6~10시간', fill: P.orange, soft: `color-mix(in srgb, ${P.orange} 22%, white)`, text: `color-mix(in srgb, ${P.orange} 55%, black)` },
  { max: 24, name: '크게 빗나감', range: '11시간 이상', fill: LV.strong.fill, soft: LV.strong.soft, text: LV.strong.text },
]
const level = (n: number) => LEVELS.find((l) => n <= l.max) ?? LEVELS[2]
const WD = ['일', '월', '화', '수', '목', '금', '토']
const iso = (y: number, m: number, d: number) => `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`

/** 예측 달력: 날짜마다 그날 시간별 예측이 80% 범위를 벗어난 시간 수를 색 뱃지로 */
export function Calendar() {
  const [H, setH] = useState('1')
  // 날짜별 범위 밖 시간 수 (공표 SMP가 있는 시간만 셈)
  const out = useMemo(() => {
    const src = D.past.hourly[H] ?? {}, r: Record<string, { n: number; k: number; lo: number; hi: number }> = {}
    Object.entries(src).forEach(([d, [p, lo, hi]]) => {
      const y = D.past.actual[d]
      if (!y) return
      let n = 0, k = 0, a = 0, b = 0
      y.forEach((v, i) => {
        if (v == null || p[i] == null || lo[i] == null || hi[i] == null) return
        n++
        if (!inside(v, lo[i], hi[i])) { k++; if (v < (lo[i] as number)) a++; else b++ }
      })
      if (n) r[d] = { n, k, lo: a, hi: b }
    })
    return r
  }, [H])
  const days = Object.keys(out).sort()
  const last = days[days.length - 1] ?? new Date().toISOString().slice(0, 10)
  const months = [...new Set(days.map((d) => d.slice(0, 7)))]
  const [ym, setYm] = useState(last.slice(0, 7))
  const cur = months.includes(ym) ? ym : months[months.length - 1] ?? ym
  const [y, m] = cur.split('-').map(Number)
  const first = new Date(y, m - 1, 1).getDay(), nd = new Date(y, m, 0).getDate()
  const cells: (number | null)[] = [...Array(first).fill(null), ...Array.from({ length: nd }, (_, i) => i + 1)]
  while (cells.length % 7) cells.push(null)
  const mi = months.indexOf(cur)
  const [pick, setPick] = useState<string | null>(null)
  const sel = pick && pick.startsWith(cur) && out[pick] ? pick : days.filter((d) => d.startsWith(cur)).pop() ?? null

  // 이 달 요약
  const md = days.filter((d) => d.startsWith(cur)).map((d) => out[d])
  const cnt = LEVELS.map((l, i) => md.filter((x) => level(x.k) === LEVELS[i]).length)
  const cov = md.length ? 1 - md.reduce((a, x) => a + x.k, 0) / md.reduce((a, x) => a + x.n, 0) : null

  const nav = (to: number) => months[to] && setYm(months[to])
  const btn = 'grid h-8 w-8 place-items-center rounded-ui text-sub transition hover:bg-track hover:text-brand-dark disabled:opacity-30 disabled:hover:bg-transparent'
  return (
    <div className="space-y-5">
      <PageHeader eyebrow={`검증 · ${REGION}`} title="예측 달력" sub="날짜마다 그날 시간별 예측이 80% 범위를 벗어난 시간 수"
        right={<Select label="예측 시점" value={H} onChange={setH} options={Object.keys(D.past.hourly).sort().map((h) => ({ v: h, t: h === '1' ? '하루 전 (D+1)' : `${h}일 전 (D+${h})` }))} />} />
      <section className="card px-5 py-5 sm:px-6">
        {/* 월 이동 · 이 달 요약 · 범례 */}
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
          <div className="flex items-center gap-1">
            <button className={btn} onClick={() => nav(mi - 1)} disabled={mi <= 0} aria-label="이전 달">‹</button>
            <h2 className="min-w-[120px] text-center text-[18px] font-bold tabular-nums text-ink">{y}년 {m}월</h2>
            <button className={btn} onClick={() => nav(mi + 1)} disabled={mi >= months.length - 1} aria-label="다음 달">›</button>
          </div>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-[12.5px] text-sub">
            {LEVELS.map((l, i) => (
              <span key={l.name} className="inline-flex items-center gap-1.5">
                <i className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: l.fill }} />{l.name} <span className="text-mute">{l.range}</span>
                <b className="ml-0.5 font-semibold text-ink tabular-nums">{cnt[i]}일</b>
              </span>
            ))}
            {cov != null && <span className="text-mute">이 달 포함률 <b className="font-semibold text-ink">{pct(cov)}</b></span>}
          </div>
        </div>
        {/* 달력 */}
        <div className="mt-5 grid grid-cols-7 gap-1.5">
          {WD.map((w, i) => <div key={w} className={`pb-1 text-center text-[12px] font-medium ${i === 0 ? 'text-bad' : 'text-mute'}`}>{w}</div>)}
          {cells.map((d, i) => {
            if (d == null) return <div key={i} />
            const k = iso(y, m, d), o = out[k], l = o ? level(o.k) : null, off = info(k).off || i % 7 === 0
            return (
              <button key={i} type="button" disabled={!o} onClick={() => setPick(k)} title={o ? `${k} · 범위 밖 ${o.k}시간 / ${o.n}시간 — 누르면 그날 그래프` : `${k} · 공표 전 또는 예측 없음`}
                className={`flex min-h-[76px] flex-col justify-between rounded-ui px-2.5 py-2 text-left transition sm:min-h-[92px] ${o ? 'hover:bg-hover' : 'cursor-default'}`}
                style={{ background: l ? undefined : 'var(--color-track)', boxShadow: `inset 0 0 0 ${sel === k ? '2px var(--color-brand)' : '1px var(--color-line)'}` }}>
                <span className={`text-[13px] font-semibold tabular-nums ${off ? 'text-bad' : o ? 'text-ink' : 'text-mute'}`}>{d}</span>
                {l && o && (
                  <span className="self-start rounded-[4px] px-1.5 py-0.5 text-[12px] font-bold tabular-nums sm:px-2 sm:text-[13px]" style={{ background: l.soft, color: l.text }}>
                    {o.k}<span className="hidden font-medium sm:inline">시간</span>
                  </span>
                )}
              </button>
            )
          })}
        </div>
        <p className="mt-4 text-[12px] text-mute">80% 범위는 하루 24시간 중 약 5시간(20%)은 벗어나는 것이 정상입니다. 날짜를 누르면 아래에 그날 그래프가 나옵니다.</p>
      </section>
      {sel && <DayDetail d={sel} H={H} o={out[sel]} />}
    </div>
  )
}

/** 고른 날: 시간별 예측·80% 범위 vs 공표 SMP. 범위 아래로 벗어났으면 급락 쪽, 위로면 급등 쪽 */
function DayDetail({ d, H, o }: { d: string; H: string; o: { n: number; k: number; lo: number; hi: number } }) {
  const [p, lo, hi] = D.past.hourly[H]?.[d] ?? [[], [], []]
  const y = D.past.actual[d] ?? []
  const opt = useMemo(() => bandOption({
    x: p.map((_, i) => `${i + 1}시`), p, lo, hi, y, tip: (i) => `${lab(d)} ${i + 1}시`,
    xLabel: (i) => (i % 2 === 0 ? `${i + 1}시` : ''), shade: [[10, 14]], symbols: true,
  }), [d, H])
  const ys = y.filter((v): v is number => v != null)
  const mid = y.slice(10, 15).filter((v): v is number => v != null)
  // 벗어난 방향은 '시간 수'가 아니라 '벗어난 크기(원)'로: 조금씩 아래로 13시간보다 크게 위로 6시간이 그날의 사건
  let down = 0, up = 0
  y.forEach((v, i) => {
    if (v == null || lo[i] == null || hi[i] == null) return
    if (v < (lo[i] as number)) down += (lo[i] as number) - v
    else if (v > (hi[i] as number)) up += v - (hi[i] as number)
  })
  const way = o.k === 0 ? '모두 범위 안' : up > down ? '위로 크게 (급등 쪽)' : '아래로 크게 (급락 쪽)'
  // 실제 판정 (서비스 정의): 급락 = 한낮 11~15시 최저 30원 미만, 급등 = 하루 최고가 직전 28일 일 최고 중앙값의 1.25배 이상
  const prevMax = Object.keys(D.past.actual).filter((x) => x < d).sort().slice(-28)
    .map((x) => Math.max(...(D.past.actual[x].filter((v) => v != null) as number[]))).filter((v) => isFinite(v)).sort((a, b) => a - b)
  const ref = prevMax.length >= 20 ? prevMax[Math.floor(prevMax.length / 2)] : null
  const dayMax = ys.length ? Math.max(...ys) : null, dayMin = mid.length ? Math.min(...mid) : null
  const event = dayMin != null && dayMin < 30 ? '급락일' : dayMax != null && ref != null && dayMax >= ref * 1.25 ? `급등일 · 평소의 ${Math.round((dayMax / ref) * 100)}%` : '평소'
  const stat = (k: string, v: React.ReactNode) => (
    <div className="min-w-0"><div className="t-label">{k}</div><div className="v-sm mt-1 text-ink">{v}</div></div>
  )
  return (
    <Card title={`${lab(d)} 시간별 예측 vs 공표 SMP`} sub={`${H === '1' ? '하루 전' : H + '일 전'} 예측 · 회색 = 한낮 11~15시`}
      right={<Legend items={[{ t: '80% 범위', c: 'band' }, { t: '예측', c: 'pred' }, { t: '공표 SMP', c: 'act' }, { t: '범위 밖', c: 'out' }]} />}>
      <div className="mb-4 grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4 lg:grid-cols-7">
        {stat('범위 밖', <>{o.k}<span className="unit">/ {o.n}시간</span></>)}
        {stat('아래로 · 위로', <>{o.lo} · {o.hi}<span className="unit">시간</span></>)}
        {stat('벗어난 쪽', <span className="text-[13.5px]">{way}</span>)}
        {stat('그날 실제', <span className="text-[13.5px]">{event}</span>)}
        {stat('공표 한낮 최저 · 최고', <>{fmt(dayMin)} · {fmt(dayMax)}<span className="unit">원</span></>)}
        {stat('그날 급락 경보', <DropChip d={D.past.drop[d]} date={d} />)}
        {stat('그날 급등 경보', <SurgeChip iso={d} />)}
      </div>
      <EChart option={opt} height={320} />
    </Card>
  )
}
