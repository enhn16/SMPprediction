import { LEVEL, dropTarget, type Drop } from '../data'
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
