// 데이터: ../web/data/forecast_data.js (window.SMP_DATA, 육지·공통) + forecast_data_jeju.js (window.SMP_JEJU, 제주를 고를 때 읽음) — Python export_web.py가 생성
export type Row = Record<string, any>
export type Drop = { grade: string; min: number | null; prob: number | null }
export type Day = { date: string; H: number; dt?: string; off?: boolean; p: (number | null)[]; lo: (number | null)[]; hi: (number | null)[]; y: (number | null)[] }
export type DropMeta = { warn: number | null; strong: number | null; model: string; real: Record<string, { rate: number; k: number; n: number }>; lines?: { 주의?: number; 경보?: number } }
// 계통별로 바뀌는 부분 (SMP_DATA 최상위 = 육지, SMP_JEJU = 제주)
export type RegionData = {
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
/** 예측 근거 (과제3): 위험을 높이는 쪽 up / 낮추는 쪽 down, 쉬운 문장 */
export type Why = { t: string; dir: 'up' | 'down' | 'none'; v?: string }[] // none = 확실한 근거 없음 (화살표 없이 회색 문장), v = 근거별 영향 (급락 원·급등 %p)
export type SurgeLevel = { n: number; k: number; rate: number | null; perMonth: number; lift: number | null }
export type SurgeData = { past: Record<string, { grade: string; prob: number | null }>; meta: { base: number | null; days: number; warnP: number; strongP: number; warn: SurgeLevel; strong: SurgeLevel }; model: string }
export type SmpData = RegionData & {
  surge?: SurgeData | null // 육지만
  why?: { drop?: Record<string, Why>; surge?: Record<string, Why> } // 경보 근거 문장 (대상일별, 육지만)
  generated: string
  lastRun: string | null
  hasJeju?: boolean
  jdocs?: Record<string, any> // 제주일 때만
  dayinfo: Record<string, { dt: string; off: boolean }>
  perf: { week: { models: Row[]; ens: Row[]; test: Row[]; choice: Row; k3?: Row; k3Score?: Row[]; rl?: Row; rlSelect?: Row[]; rlTest?: Row[] }; month: { models: Row[]; ens: Row[]; test: Row[]; choice: Row; mm?: Row; mmScore?: Row[] }; dropCal: Row[]; dropChoice: Row }
  docs: { final: Row; metrics: Row[]; nextday: Row[]; dropTest: Row[]; dropZoo: Row[]; team?: Row }
}

declare global {
  interface Window { SMP_DATA?: SmpData; SMP_JEJU?: RegionData & { jdocs: Record<string, any> } }
}

const ROOT = window.SMP_DATA as SmpData
export type Region = '육지' | '제주'
export const HAS_JEJU = !!ROOT?.hasJeju

// 현재 계통의 데이터. setRegion 뒤 화면을 다시 그리면(App의 key) 모든 페이지가 새 값을 읽는다
export let D: SmpData = ROOT
export let REGION: Region = '육지'
export let T_WARN = 0.106, T_STRONG = 0.326 // 급락 등급 문턱 확률
export let DROP_REAL: DropMeta['real'] = {} // 2026년 등급별 실제 급락 비율

/** 제주 데이터 파일을 처음 고를 때 한 번 읽음 (파일로 열어도 script 태그는 동작) */
let jejuLoad: Promise<boolean> | null = null
export function loadJeju(): Promise<boolean> {
  if (window.SMP_JEJU) return Promise.resolve(true)
  jejuLoad ??= new Promise((ok) => {
    const s = document.createElement('script')
    s.src = 'data/forecast_data_jeju.js'
    s.onload = () => ok(!!window.SMP_JEJU)
    s.onerror = () => { jejuLoad = null; ok(false) }
    document.head.appendChild(s)
  })
  return jejuLoad
}

export function setRegion(r: Region) {
  const J = window.SMP_JEJU
  REGION = r === '제주' && J ? '제주' : '육지'
  D = REGION === '제주' ? ({ ...ROOT, ...J } as SmpData) : ROOT
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
  '없음': { key: 'low', name: '낮음' }, '주의': { key: 'warn', name: '주의' }, '강한 경보': { key: 'strong', name: '경보' }, '경보': { key: 'strong', name: '경보' },
}
export const info = (d: string) => D.dayinfo[d] ?? { dt: '', off: false }
// 급락 경보 대상일: 육지는 모든 날(팀 규칙 3, 10/28). 제주는 전체 날 모델(R2)이 나오기 전까지 휴일·주말만
export const dropTarget = (_d: string) => true   // 급락 = 모든 날 (제주도 2026-11-03부터)
/** 내일 급등 경보 (R4): 육지만 — 제주를 볼 때는 없음 */
export const surgeData = (): SurgeData | null => (REGION === '육지' ? ROOT?.surge ?? null : null)
