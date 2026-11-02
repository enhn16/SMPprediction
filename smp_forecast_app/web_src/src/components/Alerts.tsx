import { D, LEVEL, REGION, T_STRONG, T_WARN, dropTarget, surgeData, type Why } from '../data'
import { lab } from '../fmt'
import { LV } from '../theme'

type Lv = 'low' | 'warn' | 'strong' | 'na'
const NAME: Record<Lv, string> = { low: '낮음', warn: '주의', strong: '경보', na: '준비 중' }
const pc = (v: number) => `${Math.round(v * 100)}%`

/** 등급 막대: 확률 그대로의 비율(0~100%), 등급 경계에 % 표시 */
function Meter({ lv, p, warn, strong }: { lv: Lv; p: number; warn: number; strong: number }) {
  const seg = (k: 'low' | 'warn' | 'strong', w: number, r: string) => (
    <span className={`h-full ${r}`} style={{ width: `${w * 100}%`, background: lv === k ? LV[k].fill : LV[k].soft }} />
  )
  const mark = (v: number, t: string) => (
    <span className="absolute top-0 -translate-x-1/2 whitespace-nowrap" style={{ left: `${v * 100}%` }}>{t} {pc(v)}</span>
  )
  return (
    <div className="mt-3">
      <div className="relative flex h-1.5 gap-[2px]">
        {seg('low', warn, 'rounded-l-full')}{seg('warn', strong - warn, '')}{seg('strong', 1 - strong, 'rounded-r-full')}
        <span className="absolute -top-[5px] h-4 w-4 -translate-x-1/2 rounded-full border-[3px] border-surface bg-ink" style={{ left: `${Math.min(100, Math.max(0, p * 100))}%` }} />
      </div>
      <div className="relative mt-1.5 h-4 text-[11px] text-mute tabular-nums">{mark(warn, '주의')}{mark(strong, '경보')}</div>
    </div>
  )
}

function Card({ kind, lv, prob, date, sub, why, note, warn, strong }: { kind: 'drop' | 'surge'; lv: Lv; prob?: number | null; date: string; sub: string; why?: Why; note?: string; warn: number; strong: number }) {
  const c = lv === 'na' ? null : LV[lv]
  return (
    <section className="card px-5 py-4">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h2 className="h-sec">내일 {kind === 'drop' ? '급락' : '급등'} 경보</h2>
        <span className="muted tabular-nums">{date} · {sub}</span>
      </div>
      <div className="mt-2 flex flex-wrap items-baseline gap-x-2.5">
        <span className="text-[20px] font-bold leading-tight" style={{ color: c ? c.text : 'var(--color-mute)' }}>{NAME[lv]}</span>
        {lv !== 'na' && prob != null && <span className="text-[13px] font-semibold text-sub tabular-nums">{pc(prob)}</span>}
        {lv === 'na' && <span className="text-[13px] text-sub">{note}</span>}
      </div>
      {lv !== 'na' && prob != null && <Meter lv={lv} p={prob} warn={warn} strong={strong} />}
      {/* 근거: 모델이 점수에 쓴 이유 중 확실한 것만 1~3줄. 근거가 없으면 회색 문장 한 줄, 자료가 없으면(제주 등) 표시 안 함 */}
      {lv !== 'na' && why && why.length > 0 && (
        <ul className="mt-2 space-y-1">
          {/* 낮음인데 근거가 모두 위험을 높이는 쪽이면: 평소보다 조금 높을 뿐 주의 문턱에는 못 미친다는 걸 먼저 밝힘 */}
          {lv === 'low' && why.every((r) => r.dir === 'up') && (
            <li className="text-[13px] leading-5 text-mute">평소보다 위험을 높인 점이 있지만 주의 수준에는 못 미칩니다</li>
          )}
          {why.slice(0, 3).map((r) => r.dir === 'none' ? (
            <li key={r.t} className="text-[13px] leading-5 text-mute">{r.t}</li>
          ) : (
            <li key={r.t} className="flex gap-2 text-[13px] leading-5 text-ink">
              <span className="mt-[1px] shrink-0 text-[10px]" style={{ color: r.dir === 'up' ? LV.strong.text : LV.low.text }}>{r.dir === 'up' ? '▲' : '▼'}</span>
              {r.t}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

/** 내일 급락·급등 경보 두 장 (서비스 화면: 상태·확률 + 막대 + 근거 3줄, 세부는 참고 문서) */
export function Alerts({ iso }: { iso: string }) {
  const d = D.week.drop, s = surgeData()?.past[iso]
  const dropLv: Lv = !dropTarget(iso) || !d ? 'na' : LEVEL[d.grade]?.key ?? 'low'
  const surgeLv: Lv = !s ? 'na' : s.grade === '경보' ? 'strong' : s.grade === '주의' ? 'warn' : 'low'
  const why = D.why ?? {}
  return (
    <div className="grid gap-5 md:grid-cols-2">
      <Card kind="drop" lv={dropLv} prob={d?.prob} date={lab(iso)} sub="한낮 11~15시" why={why.drop?.[iso]} warn={T_WARN} strong={T_STRONG}
        note={REGION === '제주' ? '제주 평일 급락 예측을 준비하고 있습니다.' : '예측이 아직 없습니다.'} />
      <Card kind="surge" lv={surgeLv} prob={s?.prob} date={lab(iso)} sub="하루 최고 가격" why={why.surge?.[iso]}
        warn={surgeData()?.meta.warnP ?? 0.04} strong={surgeData()?.meta.strongP ?? 0.19}
        note={REGION === '제주' ? '제주 급등 예측은 아직 없습니다.' : '예측이 아직 없습니다.'} />
    </div>
  )
}
