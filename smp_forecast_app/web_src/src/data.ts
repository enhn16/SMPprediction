// 데이터: ../web/data/forecast_data.js (window.SMP_DATA) — Python export_web.py가 생성
export type Row = Record<string, any>
export type Drop = { grade: string; min: number | null; prob: number | null }
export type Day = { date: string; H: number; dt?: string; off?: boolean; p: (number | null)[]; lo: (number | null)[]; hi: (number | null)[]; y: (number | null)[] }
export type DropMeta = { warn: number | null; strong: number | null; model: string; real: Record<string, { rate: number; k: number; n: number }> }
// 계통별로 바뀌는 부분 (최상위 = 육지, jeju = 제주)
type RegionData = {
  lastPublished: string
  week: { base: string; days: Day[]; drop?: Drop }
  month: { base: string; d: string[]; dt: (string | null)[]; [k: string]: any }
  past: {
    hourly: Record<string, Record<string, [(number | null)[], (number | null)[], (number | null)[]]>>
    actual: Record<string, (number | null)[]>
    daily: Record<string, Record<string, Record<string, [number | null, number | null, number | null]>>>
    dailyActual: Record<string, Record<string, number | null>>
    drop: Record<string, Drop>
  }
  dropMeta: DropMeta
}
export type SmpData = RegionData & {
  generated: string
  lastRun: string | null
  jeju: (RegionData & { jdocs: Record<string, any> }) | null
  dayinfo: Record<string, { dt: string; off: boolean }>
  perf: { week: { models: Row[]; ens: Row[]; test: Row[]; choice: Row }; month: { models: Row[]; ens: Row[]; test: Row[]; choice: Row }; dropCal: Row[]; dropChoice: Row }
  docs: { final: Row; metrics: Row[]; nextday: Row[]; dropTest: Row[]; dropZoo: Row[] }
}

declare global {
  interface Window { SMP_DATA?: SmpData }
}

const ROOT = window.SMP_DATA as SmpData
export type Region = '육지' | '제주'
export const HAS_JEJU = !!ROOT?.jeju

// 현재 계통의 데이터. setRegion 뒤 화면을 다시 그리면(App의 key) 모든 페이지가 새 값을 읽는다
export let D: SmpData = ROOT
export let REGION: Region = '육지'
export let T_WARN = 0.106, T_STRONG = 0.326 // 급락 등급 문턱 확률
export let DROP_REAL: DropMeta['real'] = {} // 2026년 등급별 실제 급락 비율

export function setRegion(r: Region) {
  REGION = r === '제주' && ROOT.jeju ? '제주' : '육지'
  D = REGION === '제주' ? ({ ...ROOT, ...ROOT.jeju } as SmpData) : ROOT
  T_WARN = D.dropMeta?.warn ?? T_WARN
  T_STRONG = D.dropMeta?.strong ?? T_STRONG
  DROP_REAL = D.dropMeta?.real ?? {}
}
if (ROOT) setRegion('육지')

export const STATS = ['mean', 'mid', 'eve', 'midmin'] as const
export type Stat = (typeof STATS)[number]
export const STAT_NAME: Record<Stat, string> = { mean: '일평균', mid: '한낮 11~15시', eve: '저녁 18~22시', midmin: '한낮 최저' }

// 급락 등급 (과제1): 확률 문턱은 계통별 dropMeta (육지 D_choice.json, 제주 JD_choice.json의 등급 문턱 확률)
export const LEVEL: Record<string, { key: 'low' | 'warn' | 'strong'; name: string }> = {
  '없음': { key: 'low', name: '낮음' }, '주의': { key: 'warn', name: '주의' }, '강한 경보': { key: 'strong', name: '강한 경보' },
}
export const info = (d: string) => D.dayinfo[d] ?? { dt: '', off: false }
