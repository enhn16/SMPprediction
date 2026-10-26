export type N = number | null | undefined

const bad = (v: N): v is null | undefined => v == null || Number.isNaN(v)
export const fmt = (v: N, d = 1) => (bad(v) ? '-' : v.toFixed(d))
export const pct = (v: N, d = 1) => (bad(v) ? '-' : (v * 100).toFixed(d) + '%')
export const sgn = (v: N, d = 1) => (bad(v) ? '-' : (v > 0 ? '+' : '') + v.toFixed(d))
export const avg = (a: N[]) => {
  const v = a.filter((x): x is number => !bad(x))
  return v.length ? v.reduce((p, c) => p + c, 0) / v.length : null
}
const WD = ['일', '월', '화', '수', '목', '금', '토']
export const wd = (d: string) => WD[new Date(d + 'T00:00:00').getDay()]
export const md = (d: string) => `${+d.slice(5, 7)}/${+d.slice(8, 10)}`
export const lab = (d: string) => `${md(d)}(${wd(d)})`
export const addDays = (d: string, k: number) => {
  const t = new Date(d + 'T00:00:00Z')
  t.setUTCDate(t.getUTCDate() + k)
  return t.toISOString().slice(0, 10)
}
export const dayDiff = (a: string, b: string) => Math.round((Date.parse(b + 'T00:00:00Z') - Date.parse(a + 'T00:00:00Z')) / 864e5)
export const toIso = (d: Date) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
export const fromIso = (s: string) => new Date(s + 'T00:00:00')
export const inside = (y: N, lo: N, hi: N) => !bad(y) && !bad(lo) && !bad(hi) && y >= lo && y <= hi
export const range = (a: string, b: string) => {
  const out: string[] = []
  for (let d = a; d <= b; d = addDays(d, 1)) out.push(d)
  return out
}

/** 표를 CSV로 내려받기 (엑셀에서 한글이 깨지지 않게 BOM 포함) */
export function downloadCsv(name: string, header: string[], rows: unknown[][]) {
  const cell = (v: unknown) => {
    const s = v == null || (typeof v === 'number' && !isFinite(v)) ? '' : String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const text = '﻿' + [header, ...rows].map((r) => r.map(cell).join(',')).join('\r\n')
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }))
  a.download = name.endsWith('.csv') ? name : name + '.csv'
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}
