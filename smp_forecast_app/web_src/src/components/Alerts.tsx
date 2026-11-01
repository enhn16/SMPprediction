import { D, LEVEL, REGION, T_STRONG, T_WARN, dropTarget, surgeData, type Why } from '../data'
import { lab } from '../fmt'
import { LV } from '../theme'

type Lv = 'low' | 'warn' | 'strong' | 'na'
const NAME: Record<Lv, string> = { low: '낮음', warn: '주의', strong: '경보', na: '준비 중' }
// 과제3 근거 문장이 붙기 전까지 자리 확인용 임시 문장 (모든 예측에 근거 3줄이 붙을 예정)
const SAMPLE: Record<'drop' | 'surge', Why> = {
  drop: [{ t: '내일은 평일이라 전력 수요가 높은 편입니다', dir: 'down' }, { t: '오후 하늘이 흐릴 것으로 예보돼 태양광 발전이 적습니다', dir: 'down' }, { t: '최근 7일 한낮 최저 가격이 90원대로 높았습니다', dir: 'down' }],
  surge: [{ t: '어제 저녁 가격이 최근 28일 수준보다 높았습니다', dir: 'up' }, { t: '기온이 낮아 난방 수요가 늘 수 있습니다', dir: 'up' }, { t: '주간 최대부하 전망은 평소 수준입니다', dir: 'down' }],
}
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
  const reasons = why ?? SAMPLE[kind]
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
      {lv !== 'na' && (
        <ul className="mt-2 space-y-1">
          {reasons.slice(0, 3).map((r) => (
            <li key={r.t} className="flex gap-2 text-[13px] leading-5 text-ink">
              <span className="mt-[1px] shrink-0 text-[10px]" style={{ color: r.dir === 'up' ? LV.strong.text : LV.low.text }}>{r.dir === 'up' ? '▲' : '▼'}</span>
              {r.t}
            </li>
          ))}
          {!why && <li className="text-[11px] text-mute">(예시 문장)</li>}
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
