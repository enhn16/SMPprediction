import { D, LEVEL, T_STRONG, T_WARN, type Drop } from '../data'
import { fmt } from '../fmt'

const COLOR = { low: '#4fb386', warn: '#f2c200', strong: '#d83f4b' }
const SOFT = { low: '#d6f0e3', warn: '#fdf1b0', strong: '#f8cdd1' }

// 2026년 등급별 실제 급락 비율 (확률 보정 점검표)
const real: Record<string, any> = {}
;(D.perf.dropCal || []).filter((r) => r['연도'] === 2026 && String(r['구분']).startsWith('등급 ')).forEach((r) => { real[String(r['구분']).slice(3)] = r })

export function DropChip({ d }: { d?: Drop }) {
  if (!d) return <span className="text-mute">-</span>
  const lv = LEVEL[d.grade] ?? LEVEL['없음']
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-0.5 text-[12px] font-semibold"
      style={{ background: SOFT[lv.key], color: lv.key === 'strong' ? '#9b1c27' : lv.key === 'warn' ? '#6b5300' : '#1d6b49' }}>
      <i className="inline-block h-2 w-2 rounded-full" style={{ background: COLOR[lv.key] }} />
      {d.prob == null ? '-' : Math.round(d.prob * 100) + '%'} · {lv.name}
    </span>
  )
}

/** 다음 날 급락 판별: 확률 + 낮음·주의·강한 경보 눈금 막대 */
export function DropMeter({ d, date }: { d?: Drop; date?: string }) {
  if (!d) return null
  const lv = LEVEL[d.grade] ?? LEVEL['없음']
  const p = Math.min(100, Math.max(0, (d.prob ?? 0) * 100))
  const w1 = T_WARN * 100, w2 = (T_STRONG - T_WARN) * 100, w3 = 100 - T_STRONG * 100
  const rr = real[d.grade]
  return (
    <div className="card flex flex-col gap-5 p-5 md:flex-row md:items-center" style={{ borderLeft: `5px solid ${COLOR[lv.key]}` }}>
      <div className="min-w-[200px]">
        <div className="text-[12.5px] font-semibold text-sub">다음 날 급락 판별{date ? ` · ${date}` : ''}</div>
        <div className="mt-1 flex items-baseline gap-2">
          <span className="text-[38px] font-extrabold leading-none tabular-nums" style={{ color: lv.key === 'low' ? '#005082' : lv.key === 'warn' ? '#8a6a00' : COLOR.strong }}>
            {d.prob == null ? '-' : Math.round(d.prob * 100)}<span className="text-[20px]">%</span>
          </span>
          <span className="rounded-full px-3 py-1 text-[13px] font-bold" style={{ background: SOFT[lv.key], color: lv.key === 'strong' ? '#9b1c27' : lv.key === 'warn' ? '#6b5300' : '#1d6b49' }}>{lv.name}</span>
        </div>
        <div className="mt-1 text-[12px] text-mute">한낮(11~15시) 최저 SMP가 30원 미만일 확률</div>
      </div>
      <div className="min-w-0 flex-1">
        <div className="relative mt-3 flex h-3.5 overflow-visible rounded-full">
          <span className="h-full rounded-l-full" style={{ width: `${w1}%`, background: lv.key === 'low' ? COLOR.low : SOFT.low }} />
          <span className="h-full" style={{ width: `${w2}%`, background: lv.key === 'warn' ? COLOR.warn : SOFT.warn }} />
          <span className="h-full rounded-r-full" style={{ width: `${w3}%`, background: lv.key === 'strong' ? COLOR.strong : SOFT.strong }} />
          <span className="absolute -top-2 h-7.5 w-[3px] -translate-x-1/2 rounded bg-ink" style={{ left: `${p}%` }} />
          <span className="absolute -top-7 -translate-x-1/2 rounded bg-ink px-1.5 py-0.5 text-[11px] font-bold text-white" style={{ left: `${p}%` }}>{Math.round(p)}%</span>
        </div>
        <div className="mt-1.5 flex text-[11.5px] font-medium text-sub">
          <span style={{ width: `${w1}%` }}>낮음</span>
          <span style={{ width: `${w2}%` }}>주의 {Math.round(T_WARN * 100)}%~</span>
          <span style={{ width: `${w3}%` }}>강한 경보 {Math.round(T_STRONG * 100)}%~</span>
        </div>
        <div className="mt-2 text-[12px] text-sub">
          과제1 모델(Ridge+EBM+랜덤포레스트)의 예상 한낮 최저 <b className="text-ink">{fmt(d.min)}원</b>
          {rr && <> · 2026년 이 등급 날 실제 급락 {Math.round(rr['실제 급락 비율'] * 100)}% ({rr['급락일']}/{rr['일수']}일)</>}
        </div>
      </div>
    </div>
  )
}
