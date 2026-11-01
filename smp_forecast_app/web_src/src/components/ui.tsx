import type { ReactNode } from 'react'
import type { Row } from '../data'
import { downloadCsv } from '../fmt'

/** 페이지 머리: 분류(eyebrow) · 제목 · 기준 정보(meta, 세로선으로 구분) · 오른쪽 조작 */
/** 페이지 머리: 분류(파랑 작은 글씨) · 제목 · 기준 정보. meta의 hl = 판단 기준(기준일·판단 시각) → 파랑 꼬리표.
 *  나중에 1차(15시)·2차(21시) 표시도 이 자리 */
export function PageHeader({ title, sub, right, eyebrow, meta }: { title: string; sub?: ReactNode; right?: ReactNode; eyebrow?: ReactNode; meta?: { k: string; v: ReactNode; hl?: boolean }[] }) {
  return (
    <header className="mb-7 flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
      <div className="min-w-0">
        {eyebrow && <div className="eyebrow mb-1.5">{eyebrow}</div>}
        <h1 className="h-page">{title}</h1>
        {sub && <p className="mt-1.5 text-[13.5px] text-sub">{sub}</p>}
        {meta && meta.length > 0 && (
          <dl className="mt-3.5 flex flex-wrap items-center gap-x-5 gap-y-2 text-[12.5px]">
            {meta.map((m) => m.hl ? (
              <div key={m.k} className="flex items-baseline gap-1.5 rounded-ui bg-brand-soft px-2.5 py-1 text-brand-dark">
                <dt className="font-medium">{m.k}</dt><dd className="font-semibold tabular-nums">{m.v}</dd>
              </div>
            ) : (
              <div key={m.k} className="flex items-baseline gap-1.5">
                <dt className="text-mute">{m.k}</dt><dd className="font-semibold text-ink tabular-nums">{m.v}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>
      {right}
    </header>
  )
}

/** 흰 면 하나: 제목 줄(구분선 없음) + 본문. flush = 본문 좌우 여백만 (표를 넓게) */
export function Card({ title, sub, right, children, className = '', flush }: { title?: ReactNode; sub?: ReactNode; right?: ReactNode; children?: ReactNode; className?: string; flush?: boolean }) {
  const body = children != null && children !== false
  return (
    <section className={`card ${className} ${body ? '' : 'pb-5'}`}>
      {(title || right) && (
        <div className="panel-head">
          <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-0.5">
            {title && <h2 className="h-sec">{title}</h2>}
            {sub && <span className="muted">{sub}</span>}
          </div>
          {right}
        </div>
      )}
      {body && <div className={flush ? 'px-6 pb-4 pt-3' : 'panel-body'}>{children}</div>}
    </section>
  )
}

/** 숫자와 단위 분리: '4.2원' → 4.2 + 작은 '원' (숫자 하나짜리 문자열만, '14.6% / 5.1%'는 그대로) */
export function Num({ v }: { v: ReactNode }) {
  const m = typeof v === 'string' ? v.match(/^([-+−]?[\d.,]+)\s?(원|%|일|시간)$/) : null
  return m ? <>{m[1]}<span className="unit">{m[2]}</span></> : <>{v}</>
}

export type Tone = 'good' | 'bad' | undefined
// 목표 안 = 초록, 목표 밖 = 기본색 (빨강은 급락·급등 경보에만 — 성능 숫자에는 '목표 미달' 글로 표시)
const toneCls = (t: Tone) => (t === 'good' ? 'text-good' : 'text-ink')
/** 지표 묶음: 흰 면 하나에 여백으로만 나눔 (선 없음) */
export function Kpis({ items }: { items: { k: string; v: ReactNode; s?: ReactNode; tone?: Tone; main?: boolean }[] }) {
  const lead = Math.max(0, items.findIndex((x) => x.main))   // 대표 지표: main 표시, 없으면 첫 칸
  // 한 줄에 다 들어가는 폭(bp)에서만 지표 사이 얇은 세로선
  const [cols, bp] = items.length >= 5 ? ['grid-cols-2 md:grid-cols-3 xl:grid-cols-5', 'xl'] : items.length === 4 ? ['grid-cols-2 lg:grid-cols-4', 'lg']
    : items.length === 3 ? ['grid-cols-1 sm:grid-cols-3', 'sm'] : ['grid-cols-2', '']
  const line = { xl: 'xl:border-l xl:border-line xl:pl-6', lg: 'lg:border-l lg:border-line lg:pl-6', sm: 'sm:border-l sm:border-line sm:pl-6', '': 'border-l border-line pl-6' }[bp]
  return (
    <div className={`card grid gap-x-6 gap-y-5 px-6 py-5 ${cols}`}>
      {items.map((it, i) => (
        <div key={it.k} className={`min-w-0 ${i ? line : ''}`}>
          {/* 라벨 → 핵심 숫자 → 보조 설명. 대표 지표만 진한 파랑, 나머지는 기본색(좋음·나쁨은 색) */}
          <div className="t-label">{it.k}</div>
          <div className={`v-kpi mt-1.5 ${it.tone ? toneCls(it.tone) : i === lead ? 'text-brand-dark' : 'text-ink'}`}><Num v={it.v} /></div>
          {it.s && <div className="mt-1 text-[11.5px] leading-4 text-mute">{it.s}</div>}
        </div>
      ))}
    </div>
  )
}

/** 구분 버튼: 옅은 트랙 위, 선택 = 진한 파랑 채움 */
export function Seg<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { v: T; t: string }[] }) {
  return (
    <div className="seg" role="group">
      {options.map((o) => <button key={o.v} onClick={() => onChange(o.v)} aria-pressed={value === o.v}>{o.t}</button>)}
    </div>
  )
}

export function Select({ value, onChange, options, label }: { value: string; onChange: (v: string) => void; options: { v: string; t: string }[]; label?: string }) {
  return (
    <label className="flex items-center gap-2 text-[12.5px] font-medium text-sub">
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)} className="field">
        {options.map((o) => <option key={o.v} value={o.v}>{o.t}</option>)}
      </select>
    </label>
  )
}

export type Col = { k: string; t: string; l?: boolean; w?: boolean; f?: (v: any, r: Row) => ReactNode; cls?: (r: Row) => string }
/** CSV 내려받기 버튼 (작게, 패널·표 오른쪽 위) */
export function CsvButton({ onClick }: { onClick: () => void }) {
  return (
    <button onClick={onClick} className="btn h-7 px-2.5 text-[12px]">
      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 4v11M7 10l5 5 5-5M5 20h14" /></svg>CSV
    </button>
  )
}

/** caption·csv가 있으면 표 바로 위에 붙은 머리줄(왼쪽 제목, 오른쪽 CSV)로 */
export function Table({ rows, cols, sel, csv, caption }: { rows: Row[]; cols: Col[]; sel?: (r: Row) => boolean; csv?: string; caption?: ReactNode }) {
  // CSV는 표시 형식이 아닌 원래 값으로
  const save = () => downloadCsv(csv!, cols.map((c) => c.t), rows.map((r) => cols.map((c) => r[c.k])))
  const bar = (caption || csv) && (
    <div className="flex min-h-8 items-center justify-between gap-3 pb-2">
      <span className="text-[13px] font-semibold text-ink">{caption}</span>
      {csv && <CsvButton onClick={save} />}
    </div>
  )
  return (
    <div>
      {bar}
      <div className="tbl-wrap"><table className="tbl">
        <thead><tr>{cols.map((c) => <th key={c.k} className={`${c.l ? 'l' : ''} ${c.w ? 'w' : ''}`}>{c.t}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={sel?.(r) ? 'sel' : ''}>
              {cols.map((c) => <td key={c.k} className={`${c.l ? 'l' : ''} ${c.w ? 'w' : ''} ${c.cls ? c.cls(r) : ''}`}>{c.f ? c.f(r[c.k], r) : (r[c.k] ?? '-')}</td>)}
            </tr>
          ))}
        </tbody>
      </table></div>
    </div>
  )
}

/** 작은 꼬리표 */
export function Pill({ children, tone = 'brand' }: { children: ReactNode; tone?: 'brand' | 'gray' | 'red' }) {
  const c = tone === 'gray' ? 'bg-track text-sub' : tone === 'red' ? 'bg-bad/10 text-bad' : 'bg-brand-soft text-brand-dark'
  return <span className={`inline-block rounded-[4px] px-1.5 py-px align-middle text-[11.5px] font-semibold ${c}`}>{children}</span>
}

export function Legend({ items }: { items: { t: string; c: 'band' | 'pred' | 'act' | 'out' | 'shade' | string }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-sub">
      {items.map((it) => (
        <span key={it.t} className="inline-flex items-center gap-1.5">
          {it.c === 'band' && <i className="inline-block h-2.5 w-4 rounded-[2px] bg-band" />}
          {it.c === 'pred' && <i className="inline-block h-[2px] w-4 bg-brand" />}
          {it.c === 'act' && <i className="inline-block h-[2px] w-4 bg-ink" />}
          {it.c === 'out' && <i className="inline-block h-2 w-2 rounded-full bg-bad" />}
          {it.c === 'shade' && <i className="inline-block h-2.5 w-4 rounded-[2px] bg-disabled" />}
          {it.c.startsWith('#') && <i className="inline-block h-[2px] w-4" style={{ background: it.c }} />}
          {it.t}
        </span>
      ))}
    </div>
  )
}
