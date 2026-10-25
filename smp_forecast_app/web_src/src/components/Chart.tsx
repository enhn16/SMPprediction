import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { BarChart, LineChart, ScatterChart } from 'echarts/charts'
import { DataZoomComponent, GridComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { fmt, inside, type N } from '../fmt'

echarts.use([LineChart, BarChart, ScatterChart, GridComponent, TooltipComponent, MarkAreaComponent, MarkLineComponent, DataZoomComponent, CanvasRenderer])

export function EChart({ option, height = 340 }: { option: echarts.EChartsCoreOption; height?: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const inst = useRef<echarts.ECharts | null>(null)
  useEffect(() => {
    if (!ref.current) return
    inst.current = echarts.init(ref.current, undefined, { renderer: 'canvas' })
    const ro = new ResizeObserver(() => inst.current?.resize())
    ro.observe(ref.current)
    return () => { ro.disconnect(); inst.current?.dispose(); inst.current = null }
  }, [])
  useEffect(() => { inst.current?.setOption(option, true) }, [option])
  return <div ref={ref} style={{ width: '100%', height }} />
}

const C = { brand: '#1a8cdc', dark: '#005082', ink: '#1f2933', bad: '#d83f4b', line: '#e3e8ee', mute: '#8a97a4', band: 'rgba(26,140,220,0.16)' }

/** 예측 + 80% 범위 + 공표 SMP. x = 라벨 배열, shade = [시작, 끝] 인덱스 구간(회색), dividers = 세로선 인덱스 */
export function bandOption(o: {
  x: string[]; p: N[]; lo: N[]; hi: N[]; y: N[]; tip: (i: number) => string
  shade?: [number, number][]; dividers?: number[]; xLabel?: (i: number) => string; zoom?: boolean; symbols?: boolean; daily?: boolean
}): echarts.EChartsCoreOption {
  const n = o.x.length
  const lo = o.lo.map((v) => v ?? null), diff = o.hi.map((v, i) => (v != null && o.lo[i] != null ? (v as number) - (o.lo[i] as number) : null))
  const outPts = o.y.map((v, i) => (v != null && o.lo[i] != null && !inside(v, o.lo[i], o.hi[i]) ? [i, v] : null)).filter(Boolean)
  const vals = [...o.lo, ...o.hi, ...o.y, ...o.p].filter((v): v is number => v != null)
  const vmin = Math.min(...vals)
  // 제주는 음수 가격이 있어 0 아래도 그림 (그때는 눈금이 겹치지 않게 ECharts가 하한을 고름)
  const ymin0 = vmin >= 0 ? Math.max(0, Math.floor((vmin - 8) / 10) * 10) : undefined
  return {
    animation: false,
    grid: { left: 44, right: 16, top: 16, bottom: o.zoom ? 62 : 30 },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: C.dark, type: 'dashed' } },
      backgroundColor: '#0b2a42', borderWidth: 0, textStyle: { color: '#fff', fontSize: 12 },
      formatter: (ps: any) => {
        const i = Array.isArray(ps) ? ps[0].dataIndex : ps.dataIndex
        const y = o.y[i]
        return `<b>${o.tip(i)}</b><br/>예측 <b>${fmt(o.p[i])}</b> · 범위 ${fmt(o.lo[i])} ~ ${fmt(o.hi[i])}`
          + (y != null ? `<br/>공표 <b>${fmt(y)}</b> (${y - (o.p[i] ?? 0) > 0 ? '+' : ''}${fmt(y - (o.p[i] ?? 0))})${o.lo[i] != null && !inside(y, o.lo[i], o.hi[i]) ? ' <span style="color:#ff8a93">범위 밖</span>' : ''}` : '')
      },
    },
    xAxis: {
      type: 'category', data: o.x, boundaryGap: !!o.daily, // 일별은 날짜 칸 단위 (휴무일 음영이 한 칸 폭)
      axisLine: { lineStyle: { color: C.line } }, axisTick: { show: false },
      axisLabel: { color: C.mute, fontSize: 11, interval: 0, formatter: (_: string, i: number) => (o.xLabel ? o.xLabel(i) : _) },
    },
    yAxis: { type: 'value', min: ymin0,splitLine: { lineStyle: { color: C.line } }, axisLabel: { color: C.mute, fontSize: 11 }, name: '원', nameTextStyle: { color: C.mute, fontSize: 11, padding: [0, 24, 0, 0] } },
    dataZoom: o.zoom ? [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8, borderColor: C.line, fillerColor: 'rgba(26,140,220,0.12)', handleSize: '80%', showDetail: false }] : [],
    series: [
      { name: 'lo', type: 'line', data: lo, stack: 'band', symbol: 'none', lineStyle: { opacity: 0 }, silent: true,
        markArea: o.shade?.length ? { silent: true, itemStyle: { color: 'rgba(120,140,160,0.08)' }, data: o.shade.map(([a, b]) => [{ xAxis: Math.max(0, a) }, { xAxis: Math.min(n - 1, b) }]) } : undefined,
        markLine: o.dividers?.length ? { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: '#c9d4de', type: 'solid', width: 1 }, data: o.dividers.map((d) => ({ xAxis: d })) } : undefined },
      { name: 'band', type: 'line', data: diff, stack: 'band', symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: C.band }, silent: true },
      { name: '예측', type: 'line', data: o.p, symbol: 'none', lineStyle: { color: C.brand, width: 2.2 }, z: 3 },
      { name: '공표', type: 'line', data: o.y, symbol: o.symbols ? 'circle' : 'none', symbolSize: 4, itemStyle: { color: C.ink }, lineStyle: { color: C.ink, width: 1.4 }, z: 4 },
      { name: '범위 밖', type: 'scatter', data: outPts, symbolSize: 6, itemStyle: { color: C.bad }, z: 5, silent: true },
    ],
  }
}
