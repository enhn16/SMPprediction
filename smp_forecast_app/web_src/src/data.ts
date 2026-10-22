// 데이터: ../web/data/forecast_data.js (window.SMP_DATA) — Python export_web.py가 생성
export type Row = Record<string, any>
export type Drop = { grade: string; min: number | null; prob: number | null }
export type Day = { date: string; H: number; dt?: string; off?: boolean; p: (number | null)[]; lo: (number | null)[]; hi: (number | null)[]; y: (number | null)[] }
export type SmpData = {
  generated: string
  lastPublished: string
  lastRun: string | null
  week: { base: string; days: Day[]; drop?: Drop }
  month: { base: string; d: string[]; dt: (string | null)[]; [k: string]: any }
  past: {
    hourly: Record<string, Record<string, [(number | null)[], (number | null)[], (number | null)[]]>>
    actual: Record<string, (number | null)[]>
    daily: Record<string, Record<string, Record<string, [number | null, number | null, number | null]>>>
    dailyActual: Record<string, Record<string, number | null>>
    drop: Record<string, Drop>
  }
  dayinfo: Record<string, { dt: string; off: boolean }>
  perf: { week: { models: Row[]; ens: Row[]; test: Row[]; choice: Row }; month: { models: Row[]; ens: Row[]; test: Row[]; choice: Row }; dropCal: Row[]; dropChoice: Row }
  docs: { final: Row; metrics: Row[]; nextday: Row[]; dropTest: Row[]; dropZoo: Row[] }
}

declare global {
  interface Window { SMP_DATA?: SmpData }
}

export const D = window.SMP_DATA as SmpData

export const STATS = ['mean', 'mid', 'eve', 'midmin'] as const
export type Stat = (typeof STATS)[number]
export const STAT_NAME: Record<Stat, string> = { mean: '일평균', mid: '한낮 11~15시', eve: '저녁 18~22시', midmin: '한낮 최저' }

// 급락 등급 (과제1): 확률 문턱은 D_choice.json의 등급 문턱 확률
const thr = (D?.perf?.dropChoice?.['등급 문턱 확률'] ?? {}) as Record<string, number>
export const T_WARN = thr['주의(90.1원)'] ?? 0.106
export const T_STRONG = thr['강한 경보(75.3원)'] ?? 0.326
export const LEVEL: Record<string, { key: 'low' | 'warn' | 'strong'; name: string }> = {
  '없음': { key: 'low', name: '낮음' }, '주의': { key: 'warn', name: '주의' }, '강한 경보': { key: 'strong', name: '강한 경보' },
}
export const info = (d: string) => D.dayinfo[d] ?? { dt: '', off: false }
