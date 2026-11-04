import { REGION, surgeData } from '../data'
import { LV } from '../theme'

/** 표·목록용 작은 표시 — 급락 칩(DropChip)과 같은 꼴: 색 점 + 등급 · 확률 */
export function SurgeChip({ iso }: { iso: string }) {
  if (REGION === '제주') return <span className="text-[12px] text-mute">제공 안 함</span>
  const s = surgeData()?.past[iso]
  if (!s) return <span className="text-mute">-</span>
  const k = s.grade === '경보' ? 'strong' : s.grade === '주의' ? 'warn' : 'low'
  const name = s.grade === '경보' ? '경보' : s.grade === '주의' ? '주의' : '낮음'
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-[4px] px-2 py-0.5 text-[12px] font-semibold"
      style={{ background: LV[k].soft, color: LV[k].text }}>
      <i className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: LV[k].fill }} />
      {name} · {s.prob == null ? '-' : Math.round(s.prob * 100) + '%'}
    </span>
  )
}
