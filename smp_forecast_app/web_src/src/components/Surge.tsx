import { surgeData } from '../data'
import { LV } from '../theme'

/** 표·목록용 작은 표시 */
export function SurgeChip({ iso }: { iso: string }) {
  const S = surgeData()
  const s = S?.past[iso]
  if (!s) return <span className="text-mute">-</span>
  if (s.grade === '경보') return <span className="rounded-[4px] px-2 py-0.5 text-[12px] font-semibold text-white" style={{ background: LV.strong.fill }}>경보</span>
  if (s.grade === '주의') return <span className="rounded-[4px] px-2 py-0.5 text-[12px] font-semibold" style={{ background: LV.warn.soft, color: LV.warn.text }}>주의</span>
  return <span className="text-[12px] text-mute">낮음</span>
}
