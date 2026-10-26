import type { ReactNode } from 'react'
import type { Row } from '../data'
import { downloadCsv } from '../fmt'

export function PageHeader({ title, sub, right }: { title: string; sub?: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-[22px] font-bold tracking-tight text-brand-ink">{title}</h1>
        {sub && <p className="mt-1 text-[13px] text-sub">{sub}</p>}
      </div>
      {right}
    </div>
  )
}

export function Card({ title, sub, right, children, className = '' }: { title?: ReactNode; sub?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card card-pad ${className}`}>
      {(title || right) && (
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-baseline gap-2">
            {title && <h2 className="h-sec">{title}</h2>}
            {sub && <span className="muted">{sub}</span>}
          </div>
          {right}
        </div>
      )}
      {children}
    </section>
  )
}

export type Tone = 'good' | 'bad' | undefined
export function Kpis({ items }: { items: { k: string; v: ReactNode; s?: ReactNode; tone?: Tone }[] }) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
      {items.map((it) => (
        <div key={it.k} className="card px-4 py-3.5">
          <div className="text-[12px] font-medium text-sub">{it.k}</div>
          <div className={`mt-1 text-[22px] font-bold tabular-nums ${it.tone === 'good' ? 'text-good' : it.tone === 'bad' ? 'text-bad' : 'text-brand-dark'}`}>{it.v}</div>
          {it.s && <div className="mt-0.5 text-[12px] text-mute">{it.s}</div>}
        </div>
      ))}
    </div>
  )
}

export function Seg<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { v: T; t: string }[] }) {
  return (
    <div className="inline-flex rounded-lg bg-[#e9eef3] p-1">
      {options.map((o) => (
        <button key={o.v} onClick={() => onChange(o.v)}
          className={`rounded-md px-3.5 py-1.5 text-[13px] font-semibold transition ${value === o.v ? 'bg-white text-brand-dark shadow-sm' : 'text-sub hover:text-brand-dark'}`}>
          {o.t}
        </button>
      ))}
    </div>
  )
}

export function Select({ value, onChange, options, label }: { value: string; onChange: (v: string) => void; options: { v: string; t: string }[]; label?: string }) {
  return (
    <label className="flex items-center gap-2 text-[12.5px] text-sub">
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="h-9 rounded-lg border border-line bg-white px-3 text-[13.5px] text-ink outline-none focus:border-brand">
        {options.map((o) => <option key={o.v} value={o.v}>{o.t}</option>)}
      </select>
    </label>
  )
}

export type Col = { k: string; t: string; l?: boolean; w?: boolean; f?: (v: any, r: Row) => ReactNode; cls?: (r: Row) => string }
/** CSV 내려받기 버튼 (작게, 카드·표 오른쪽 위) */
export function CsvButton({ onClick }: { onClick: () => void }) {
  return (
    <button onClick={onClick} className="inline-flex items-center gap-1 rounded-md border border-line bg-white px-2.5 py-1 text-[12px] font-semibold text-sub hover:border-brand/60 hover:text-brand-dark">
      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><path d="M12 4v11M7 10l5 5 5-5M5 20h14" /></svg>CSV
    </button>
  )
}

export function Table({ rows, cols, sel, csv }: { rows: Row[]; cols: Col[]; sel?: (r: Row) => boolean; csv?: string }) {
  const table = (
    <div className="overflow-x-auto rounded-lg border border-line">
      <table className="tbl">
        <thead><tr>{cols.map((c) => <th key={c.k} className={`${c.l ? 'l' : ''} ${c.w ? 'w' : ''}`}>{c.t}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={sel?.(r) ? 'sel' : ''}>
              {cols.map((c) => <td key={c.k} className={`${c.l ? 'l' : ''} ${c.w ? 'w' : ''} ${c.cls ? c.cls(r) : ''}`}>{c.f ? c.f(r[c.k], r) : (r[c.k] ?? '-')}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
  if (!csv) return table
  // CSV는 표시 형식이 아닌 원래 값으로
  const save = () => downloadCsv(csv, cols.map((c) => c.t), rows.map((r) => cols.map((c) => r[c.k])))
  return <div className="space-y-1.5"><div className="flex justify-end"><CsvButton onClick={save} /></div>{table}</div>
}

export function Pill({ children, tone = 'brand' }: { children: ReactNode; tone?: 'brand' | 'gray' | 'red' }) {
  const c = tone === 'gray' ? 'bg-[#eef1f4] text-sub' : tone === 'red' ? 'bg-[#fde8ea] text-bad' : 'bg-brand-soft text-brand-dark'
  return <span className={`inline-block rounded-full px-2.5 py-0.5 text-[12px] font-semibold ${c}`}>{children}</span>
}

export function Legend({ items }: { items: { t: string; c: 'band' | 'pred' | 'act' | 'out' | 'shade' | string }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-sub">
      {items.map((it) => (
        <span key={it.t} className="inline-flex items-center gap-1.5">
          {it.c === 'band' && <i className="inline-block h-2.5 w-4 rounded-sm border border-brand/50 bg-band" />}
          {it.c === 'pred' && <i className="inline-block h-0.5 w-4 bg-brand" />}
          {it.c === 'act' && <i className="inline-block h-0.5 w-4 bg-ink" />}
          {it.c === 'out' && <i className="inline-block h-2 w-2 rounded-full bg-bad" />}
          {it.c === 'shade' && <i className="inline-block h-2.5 w-4 rounded-sm bg-[#eef1f4]" />}
          {it.c.startsWith('#') && <i className="inline-block h-0.5 w-4" style={{ background: it.c }} />}
          {it.t}
        </span>
      ))}
    </div>
  )
}
