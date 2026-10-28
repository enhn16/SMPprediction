// 디자인 팔레트 — 값은 index.css의 :root(사용자 지정 팔레트)가 유일한 원본. 여기서는 그 변수를 읽어 그래프(ECharts) 등 JS에서 쓴다.
// 팔레트에 없는 색(연한 배경·진한 글자)은 팔레트 색을 흰색·검정과 섞어 만든다 → 팔레트를 바꾸면 함께 바뀐다.
const css = (name: string, fallback: string) =>
  (typeof document !== 'undefined' && getComputedStyle(document.documentElement).getPropertyValue(name).trim()) || fallback

export const P = {
  primary: css('--primary', '#1A8CDC'),
  primaryDark: css('--primary-dark', '#005082'),
  primaryLight: css('--primary-light', '#96F0FA'),
  red: css('--accent-red', '#D83F4B'),
  yellow: css('--accent-yellow', '#F8D400'),
  green: css('--accent-green', '#8CDBB4'),
  cyan: css('--accent-cyan', '#8CDDE8'),
  orange: css('--accent-orange', '#E09A46'),
  background: css('--background', '#ECF2F5'),
  surface: css('--surface', '#FFFFFF'),
  text: css('--text-primary', '#2A262A'),
  textSub: css('--text-secondary', '#666666'),
  textMuted: css('--text-muted', '#8B8B8B'),
  border: css('--border', '#E5E5E5'),
  disabled: css('--disabled', '#F2F2F2'),
}

const rgb = (hex: string) => { const h = hex.replace('#', ''); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) }
/** a를 b 쪽으로 t만큼 섞음 (t = 0 → a, 1 → b) */
export const mix = (a: string, b: string, t: number) => {
  const x = rgb(a), y = rgb(b)
  return '#' + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, '0')).join('')
}
export const alpha = (hex: string, a: number) => { const [r, g, b] = rgb(hex); return `rgba(${r},${g},${b},${a})` }

/** 그래프 공통 색 */
export const C = {
  brand: P.primary, dark: P.primaryDark, ink: P.text, sub: P.textSub, mute: P.textMuted, line: P.border,
  bad: P.red, band: alpha(P.primary, 0.16), bandEdge: alpha(P.primary, 0.4), zoom: alpha(P.primary, 0.12), shade: alpha(P.textMuted, 0.1),
  divider: mix(P.border, '#000000', 0.05), tooltip: P.text, tooltipBad: mix(P.red, '#ffffff', 0.4),
}

/** 급락·경보 단계 색: fill = 막대·점, soft = 칩 배경, text = 흰 바탕 글자 (밝은 초록·노랑은 어둡게) */
export const LV = {
  low: { fill: P.green, soft: mix(P.green, '#ffffff', 0.65), text: mix(P.green, '#000000', 0.38) },
  warn: { fill: P.yellow, soft: mix(P.yellow, '#ffffff', 0.72), text: mix(P.yellow, '#000000', 0.58) },
  strong: { fill: P.red, soft: mix(P.red, '#ffffff', 0.75), text: mix(P.red, '#000000', 0.3) },
}

/** 제주 표시 (로고·계통 이름) */
export const JEJU = { main: P.orange, soft: mix(P.orange, '#ffffff', 0.45) }
