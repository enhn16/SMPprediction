import { D, DROP_REAL, LEVEL, T_STRONG, T_WARN, dropTarget, type Drop } from '../data'
import { fmt } from '../fmt'

const COLOR = { low: '#4fb386', warn: '#f2c200', strong: '#d83f4b' }
const SOFT = { low: '#d6f0e3', warn: '#fdf1b0', strong: '#f8cdd1' }

export function DropChip({ d, date }: { d?: Drop; date?: string }) {
  if (date && !dropTarget(date)) return <span className="rounded-full bg-[#eef1f4] px-2.5 py-0.5 text-[12px] font-semibold text-mute">평일 · 대상 아님</span>
  if (!d) return <span className="text-mute">-</span>
  const lv = LEVEL[d.grade] ?? LEVEL['없음']
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-0.5 text-[12px] font-semibold"
      style={{ background: SOFT[lv.key], color: lv.key === 'strong' ? '#9b1c27' : lv.key === 'warn' ? '#6b5300' : '#1d6b49' }}>
      <i className="inline-block h-2 w-2 rounded-full" style={{ background: COLOR[lv.key] }} />
      {lv.name} · {d.prob == null ? '-' : Math.round(d.prob * 100) + '%'}
    </span>
  )
}

/** 내일 급락 경보: 확률 + 낮음·주의·경보 눈금 막대 */
export function DropMeter({ d, date, iso }: { d?: Drop; date?: string; iso?: string }) {
  if (iso && !dropTarget(iso)) {
    return (
      <div className="card flex flex-wrap items-baseline gap-x-3 gap-y-1 p-5" style={{ borderLeft: '5px solid #c9d4de' }}>
        <span className="text-[12.5px] font-semibold text-sub">내일 급락 경보{date ? ` · ${date}` : ''}</span>
        <span className="text-[20px] font-extrabold text-mute">평일 — 판정 대상 아님</span>
        <span className="text-[12.5px] text-mute">급락 경보는 휴일·주말만 냅니다 (팀 공통 규칙)</span>
      </div>
    )
  }
  if (!d) return null
  const lv = LEVEL[d.grade] ?? LEVEL['없음']
  const p = Math.min(100, Math.max(0, (d.prob ?? 0) * 100))
  const w1 = T_WARN * 100, w2 = (T_STRONG - T_WARN) * 100, w3 = 100 - T_STRONG * 100
  const rr = DROP_REAL[d.grade]
  return (
    <div className="card flex flex-col gap-5 p-5 md:flex-row md:items-center" style={{ borderLeft: `5px solid ${COLOR[lv.key]}` }}>
      <div className="min-w-[200px]">
        <div className="text-[12.5px] font-semibold text-sub">내일 급락 경보{date ? ` · ${date}` : ''}</div>
        {/* 등급이 먼저: 표시 확률은 높은 구간에서 과신(육지 경보 평균 70% → 2026 실제 37%)이라 보조로 */}
        <div className="mt-1 flex items-baseline gap-2">
          <span className="text-[34px] font-extrabold leading-none" style={{ color: lv.key === 'low' ? '#1d6b49' : lv.key === 'warn' ? '#8a6a00' : COLOR.strong }}>{lv.name}</span>
          <span className="text-[13px] font-semibold text-sub">모델 추정 {d.prob == null ? '-' : Math.round(d.prob * 100) + '%'}</span>
        </div>
        <div className="mt-1 text-[12px] text-mute">내일 한낮(11~15시) 최저 SMP 30원 미만 여부</div>
        {rr && <div className="mt-2 text-[13px] font-semibold text-ink">2026년 '{lv.name}' 날 실제 급락 {Math.round(rr.rate * 100)}% <span className="font-normal text-mute">({rr.k}/{rr.n}일)</span></div>}
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
          <span style={{ width: `${w3}%` }}>경보 {Math.round(T_STRONG * 100)}%~</span>
        </div>
        <div className="mt-2 text-[12px] text-sub">
          {D.dropMeta.model}{d.min != null && <>의 예상 한낮 최저 <b className="text-ink">{fmt(d.min)}원</b></>}
          {' · 막대는 모델 추정 확률과 등급 문턱'}
        </div>
      </div>
    </div>
  )
}
