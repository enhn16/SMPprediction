import { D, DROP_REAL, LEVEL, T_STRONG, T_WARN, dropTarget, type Drop } from '../data'
import { fmt } from '../fmt'
import { LV } from '../theme'

const COLOR = { low: LV.low.fill, warn: LV.warn.fill, strong: LV.strong.fill }
const SOFT = { low: LV.low.soft, warn: LV.warn.soft, strong: LV.strong.soft }
const INK = { low: LV.low.text, warn: LV.warn.text, strong: LV.strong.text }

export function DropChip({ d, date }: { d?: Drop; date?: string }) {
  if (date && !dropTarget(date)) return <span className="text-[12px] text-mute">평일 · 대상 아님</span>
  if (!d) return <span className="text-mute">-</span>
  const lv = LEVEL[d.grade] ?? LEVEL['없음']
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-[4px] px-2 py-0.5 text-[12px] font-semibold"
      style={{ background: SOFT[lv.key], color: INK[lv.key] }}>
      <i className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: COLOR[lv.key] }} />
      {lv.name} · {d.prob == null ? '-' : Math.round(d.prob * 100) + '%'}
    </span>
  )
}

/** 내일 급락 경보 (틀 없이 내용만 — 놓이는 면은 쓰는 쪽이 정함): 등급(크게) + 확률 눈금 막대 */
export function DropMeter({ d, date, iso }: { d?: Drop; date?: string; iso?: string }) {
  const head = (
    <div className="flex flex-wrap items-baseline gap-x-2">
      <h2 className="h-sec">내일 급락 경보</h2>
      {date && <span className="muted tabular-nums">{date} · 한낮 11~15시</span>}
    </div>
  )
  if (iso && !dropTarget(iso)) {
    return (
      <div>
        {head}
        <div className="mt-4 flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="v-kpi text-mute">평일</span>
          <span className="text-[13px] text-sub">판정 대상 아님 — 급락 경보는 휴일·주말만 냅니다 (팀 공통 규칙)</span>
        </div>
      </div>
    )
  }
  if (!d) return null
  const lv = LEVEL[d.grade] ?? LEVEL['없음']
  const p = Math.min(100, Math.max(0, (d.prob ?? 0) * 100))
  const w1 = T_WARN * 100, w2 = (T_STRONG - T_WARN) * 100, w3 = 100 - T_STRONG * 100
  const rr = DROP_REAL[d.grade]
  const lines = D.dropMeta.lines
  return (
    <div>
      {head}
      <div className="mt-4 flex flex-col gap-6 md:flex-row md:items-end">
        <div className="min-w-[190px]">
          {/* 등급이 먼저: 표시 확률은 높은 구간에서 과신(육지 경보 평균 70% → 2026 실제 37%)이라 보조로 */}
          <div className="flex items-baseline gap-2.5">
            <span className="v-hero" style={{ color: INK[lv.key] }}>{lv.name}</span>
            <span className="v-sm text-ink">{d.prob == null ? '-' : Math.round(d.prob * 100)}<span className="unit">% 모델 추정</span></span>
          </div>
          {rr && <div className="mt-2 text-[12.5px] text-sub">2026년 '{lv.name}' 날 실제 급락 <b className="font-semibold text-ink tabular-nums">{Math.round(rr.rate * 100)}%</b> <span className="text-mute">({rr.k}/{rr.n}일)</span></div>}
        </div>
        <div className="min-w-0 flex-1">
          <div className="relative flex h-2 gap-[3px]">
            <span className="h-full rounded-l-full" style={{ width: `${w1}%`, background: lv.key === 'low' ? COLOR.low : SOFT.low }} />
            <span className="h-full" style={{ width: `${w2}%`, background: lv.key === 'warn' ? COLOR.warn : SOFT.warn }} />
            <span className="h-full rounded-r-full" style={{ width: `${w3}%`, background: lv.key === 'strong' ? COLOR.strong : SOFT.strong }} />
            <span className="absolute -top-1 h-4 w-4 -translate-x-1/2 rounded-full border-[3px] border-surface bg-ink" style={{ left: `${p}%` }} />
          </div>
          <div className="mt-2 flex text-[11.5px] text-mute">
            <span style={{ width: `${w1}%` }}>낮음</span>
            <span style={{ width: `${w2}%` }}>주의 {Math.round(T_WARN * 100)}%~</span>
            <span style={{ width: `${w3}%` }}>경보 {Math.round(T_STRONG * 100)}%~</span>
          </div>
        </div>
      </div>
      {d.min != null && (
        /* 육지: 등급을 정하는 실제 기준 = 예상 한낮 최저 vs 원 단위 문턱 (제주 분류 모델은 예상 최저가 없음) */
        <div className="mt-4 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[12.5px]">
          <span className="text-sub">예상 한낮 최저 <b className="v-sm text-ink">{fmt(d.min)}</b><span className="unit">원</span></span>
          {lines && <span className="text-mute tabular-nums">주의 ≤ {fmt(lines['주의'])}원 · 경보 ≤ {fmt(lines['경보'])}원 · 급락 &lt; 30원</span>}
        </div>
      )}
      <div className="mt-2 text-[12px] text-mute">{D.dropMeta.model}</div>
    </div>
  )
}
