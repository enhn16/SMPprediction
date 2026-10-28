import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { BarChart, LineChart, ScatterChart } from 'echarts/charts'
import { DataZoomComponent, GridComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { fmt, inside, type N } from '../fmt'
import { C } from '../theme'

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


/** 툴팁 한 줄: 이름(흐리게) · 값(오른쪽 정렬) */
const tipRow = (k: string, v: string, color?: string) =>
  `<tr><td style="padding:1px 14px 1px 0;opacity:.7">${k}</td><td style="text-align:right;font-weight:600${color ? `;color:${color}` : ''}">${v}</td></tr>`

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
      backgroundColor: C.tooltip, borderWidth: 0, borderRadius: 2, padding: [6, 9], textStyle: { color: '#fff', fontSize: 12 },
      formatter: (ps: any) => {
        const i = Array.isArray(ps) ? ps[0].dataIndex : ps.dataIndex
        const y = o.y[i], d = y != null ? y - (o.p[i] ?? 0) : null
        const out = y != null && o.lo[i] != null && !inside(y, o.lo[i], o.hi[i])
        return `<div style="font-weight:700;margin-bottom:4px">${o.tip(i)}</div><table>`
          + tipRow('예측', fmt(o.p[i])) + tipRow('80% 범위', `${fmt(o.lo[i])} ~ ${fmt(o.hi[i])}`)
          + (y != null ? tipRow('공표', `${fmt(y)} <span style="opacity:.75">(${(d as number) > 0 ? '+' : ''}${fmt(d)})</span>`) : '')
          + (out ? tipRow('', '범위 밖', C.tooltipBad) : '') + '</table>'
      },
    },
    xAxis: {
      type: 'category', data: o.x, boundaryGap: !!o.daily, // 일별은 날짜 칸 단위 (휴무일 음영이 한 칸 폭)
      axisLine: { lineStyle: { color: C.divider } }, axisTick: { show: false },
      axisLabel: { color: C.sub, fontSize: 11, interval: 0, formatter: (_: string, i: number) => (o.xLabel ? o.xLabel(i) : _) },
    },
    yAxis: [{ type: 'value', min: ymin0,splitLine: { lineStyle: { color: C.line, type: 'dashed' } }, axisLabel: { color: C.sub, fontSize: 11 }, name: '원', nameTextStyle: { color: C.mute, fontSize: 11, padding: [0, 24, 0, 0] } },
      { type: 'value', min: 0, max: 1, show: false }],
    dataZoom: o.zoom ? [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8, borderColor: C.line, fillerColor: C.zoom, handleSize: '80%', showDetail: false }] : [],
    series: [
      { name: 'lo', type: 'line', data: lo, stack: 'band', symbol: 'none', lineStyle: { color: C.bandEdge, width: 1 }, silent: true,
        markArea: o.shade?.length && !o.daily ? { silent: true, itemStyle: { color: C.shade }, data: o.shade.map(([a, b]) => [{ xAxis: Math.max(0, a) }, { xAxis: Math.min(n - 1, b) }]) } : undefined,
        markLine: o.dividers?.length ? { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: C.divider, type: 'solid', width: 1 }, data: o.dividers.map((d) => ({ xAxis: d })) } : undefined },
      { name: 'band', type: 'line', data: diff, stack: 'band', symbol: 'none', lineStyle: { color: C.bandEdge, width: 1 }, areaStyle: { color: C.band }, silent: true },
      { name: '예측', type: 'line', data: o.p, symbol: 'none', lineStyle: { color: C.brand, width: 2.2 }, z: 3 },
      { name: '공표', type: 'line', data: o.y, symbol: o.symbols ? 'circle' : 'none', symbolSize: 4, itemStyle: { color: C.ink }, lineStyle: { color: C.ink, width: 1.8 }, z: 4 },
      ...(o.daily && o.shade?.length ? [{ name: 'shade', type: 'bar', yAxisIndex: 1, silent: true, barWidth: '100%', z: 0, itemStyle: { color: C.shade },
        data: o.x.map((_, i) => (o.shade!.some(([a, b]) => i >= a && i <= b) ? 1 : null)) }] : []),
      { name: '범위 밖', type: 'scatter', data: outPts, symbolSize: 6, itemStyle: { color: C.bad }, z: 5, silent: true },
    ],
  }
}
