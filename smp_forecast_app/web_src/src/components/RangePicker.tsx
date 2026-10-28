import { useEffect, useRef, useState } from 'react'
import { DayPicker } from 'react-day-picker'
import { ko } from 'react-day-picker/locale/ko'
import 'react-day-picker/style.css'
import { addDays, dayDiff, fromIso, lab, toIso } from '../fmt'
import { info } from '../data'

/** 항공권식 기간 선택: 시작일 → 종료일 순서, 시작일 포함 최대 maxDays일 */
export function RangePicker({ from, to, min, max, maxDays = 14, onChange }: {
  from: string; to: string; min: string; max: string; maxDays?: number; onChange: (a: string, b: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [start, setStart] = useState<string | null>(null) // 시작일만 고른 상태
  const [hover, setHover] = useState<string | null>(null)
  const [month, setMonth] = useState(fromIso(to))
  const box = useRef<HTMLDivElement>(null)
  const wide = typeof window !== 'undefined' && window.innerWidth >= 768

  useEffect(() => {
    if (!open) return
    const h = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) close() }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  })
  const openAt = (side: 'from' | 'to') => {
    const ref = fromIso(side === 'from' ? from : to)
    setMonth(wide ? new Date(ref.getFullYear(), ref.getMonth() - 1, 1) : ref)
    setStart(side === 'to' ? from : null)
    setOpen(true)
  }
  const close = () => { setOpen(false); setStart(null); setHover(null) }
  const click = (d: Date) => {
    const s = toIso(d)
    if (!start || s < start) { setStart(s); return }
    onChange(start, s); close()
  }
  const last = start ? addDays(start, maxDays - 1) : null
  const disabled = [{ before: fromIso(min) }, { after: fromIso(max) }, ...(last ? [{ after: fromIso(last < max ? last : max) }] : [])]
  const shownTo = start ? (hover && hover >= start && (!last || hover <= last) ? hover : undefined) : to
  const selected = { from: fromIso(start ?? from), to: shownTo ? fromIso(shownTo) : undefined }
  const quick = (k: number) => { onChange(addDays(max, -k + 1), max); close() }

  const boxCls = (on: boolean) => `flex min-w-[132px] flex-1 flex-col rounded-ui px-3.5 py-1.5 text-left transition ${on ? 'bg-brand-soft ring-1 ring-brand' : 'bg-track hover:bg-brand-soft'}`
  return (
    <div ref={box} className="relative">
      <div className="flex items-center gap-2">
        <button className={boxCls(open && !start)} onClick={() => openAt('from')}>
          <span className="text-[11px] font-medium text-sub">시작일</span>
          <span className="text-[15px] font-bold text-ink tabular-nums">{lab(start ?? from)}</span>
        </button>
        <span className="text-mute">→</span>
        <button className={boxCls(open && !!start)} onClick={() => openAt('to')}>
          <span className="text-[11px] font-medium text-sub">종료일</span>
          <span className="text-[15px] font-bold text-ink tabular-nums">{start ? (shownTo ? lab(shownTo) : '선택') : lab(to)}</span>
        </button>
        <span className="hidden whitespace-nowrap rounded-[4px] bg-brand-soft px-2 py-0.5 text-[12px] font-semibold text-brand-dark tabular-nums sm:inline">
          {dayDiff(start ?? from, shownTo ?? (start ? start : to)) + 1}일
        </span>
      </div>
      {open && (
        <div className="absolute left-0 top-[calc(100%+8px)] z-30 w-max max-w-[calc(100vw-32px)] rounded-panel border border-line bg-surface p-4 shadow-[0_10px_28px_rgba(0,56,91,0.12)]">
          <div className="mb-1 flex items-center justify-between gap-3">
            <span className="text-[13px] text-sub">{start ? <><b className="text-brand-dark">종료일</b>을 고르세요 · 최대 {maxDays}일</> : <><b className="text-brand-dark">시작일</b>을 고르세요</>}</span>
          </div>
          <DayPicker
            locale={ko}
            mode="range"
            selected={selected}
            onSelect={() => {}}
            onDayClick={(d, m) => { if (!m.disabled) click(d) }}
            onDayMouseEnter={(d) => setHover(toIso(d))}
            month={month}
            onMonthChange={setMonth}
            numberOfMonths={wide ? 2 : 1}
            startMonth={fromIso(min)}
            endMonth={fromIso(max)}
            disabled={disabled}
            modifiers={{ off: (d: Date) => info(toIso(d)).off || d.getDay() === 0 }}
            modifiersClassNames={{ off: 'day-off' }}
          />
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 pt-3">
            <div className="flex gap-1.5">
              {[7, 14].map((k) => (
                <button key={k} onClick={() => quick(k)} className="btn">
                  최근 {k === 7 ? '1주' : '2주'}
                </button>
              ))}
            </div>
            <span className="text-[12px] text-mute">선택 가능 {min} ~ {max}</span>
          </div>
        </div>
      )}
    </div>
  )
}
