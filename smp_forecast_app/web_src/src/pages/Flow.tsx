import { useMemo, useState } from 'react'
import { D, LEVEL, REGION, STAT_NAME, dropTarget, info, type Stat } from '../data'
import { addDays, avg, downloadCsv, fmt, inside, lab, md, pct, range, sgn } from '../fmt'
import { bandOption, EChart } from '../components/Chart'
import { C, LV, P, alpha } from '../theme'
import { DropChip } from '../components/Drop'
import { RangePicker } from '../components/RangePicker'
import { Card, CsvButton, Kpis, Legend, PageHeader, Seg, Select } from '../components/ui'

type Kind = 'wk' | 'drop' | 'mo'
const KINDS: { v: Kind; t: string }[] = [{ v: 'wk', t: '가격 예측 (시간별)' }, { v: 'drop', t: '한낮 최저 · 급락 경보' }, { v: 'mo', t: '한 달 예측 (일별)' }]

export function Flow() {
  const days = Object.keys(D.past.actual).sort()
  const MIN = days[0], MAX = days[days.length - 1]
  const [from, setFrom] = useState(addDays(MAX, -13))
  const [to, setTo] = useState(MAX)
  const [kind, setKind] = useState<Kind>('wk') // 기본 = 시간별 가격 예측과 80% 범위
  const [hW, setHW] = useState('1')
  const [hM, setHM] = useState('7')
  const [stat, setStat] = useState<Stat>('mean')
  const dates = range(from, to)

  return (
    <div className="space-y-5">
      <PageHeader eyebrow={`검증 · ${REGION}`} title="이전 흐름 확인" sub="지나간 날짜에 그때 냈던 예측과 공표 SMP를 비교합니다 · 최대 2주"
        meta={[{ k: '기간', v: <>{lab(from)} ~ {lab(to)}</>, hl: true }, { k: '일수', v: `${dates.length}일` }, { k: '선택 가능', v: <>{MIN} ~ {MAX}</> }]} />
      {/* 조작 한 면: 볼 예측 종류 · 기간 · 예측 시점 */}
      <section className="card space-y-4 px-6 py-5">
        <Seg value={kind} onChange={setKind} options={KINDS} />
        <div className="flex flex-wrap items-center justify-between gap-4">
        <RangePicker from={from} to={to} min={MIN} max={MAX} onChange={(a, b) => { setFrom(a); setTo(b) }} />
        <div className="flex flex-wrap items-center gap-3">
            {kind === 'wk' && <Select label="예측 시점" value={hW} onChange={setHW} options={Object.keys(D.past.hourly).sort().map((h) => ({ v: h, t: h === '1' ? '하루 전 (D+1)' : `${h}일 전 (D+${h})` }))} />}
            {kind === 'mo' && <Select label="예측 시점" value={hM} onChange={setHM} options={Object.keys(D.past.daily).sort((a, b) => +a - +b).map((h) => ({ v: h, t: `${h}일 전 예측` }))} />}
            {kind === 'mo' && <Select value={stat} onChange={(v) => setStat(v as Stat)} options={(Object.keys(STAT_NAME) as Stat[]).map((v) => ({ v, t: STAT_NAME[v] }))} />}
        </div>
        </div>
      </section>
      {kind === 'drop' ? <DropFlow dates={dates} /> : kind === 'wk' ? <HourlyFlow key={dates.join()} dates={dates} H={hW} /> : <DailyFlow dates={dates} H={hM} stat={stat} />}
    </div>
  )
}

function DropFlow({ dates }: { dates: string[] }) {
  const hasMin = REGION === '육지' // 제주 급락 모델은 분류(KNN)라 예상 한낮 최저가 없음
  const WARN = D.dropMeta.lines?.['주의'] ?? 89.8, STRONG = D.dropMeta.lines?.['경보'] ?? 66.7 // 육지 예상 한낮 최저 문턱 (drop_choice_육지.json)
  const rows = dates.map((d) => {
    const on = dropTarget(d)   // 육지 = 모든 날, 제주 = 전체 날 모델 전까지 휴일·주말만
    const dr = on ? D.past.drop[d] : undefined, y = D.past.dailyActual[d]?.midmin ?? null
    const alarm = dr ? dr.grade !== '없음' : null, drop = y != null ? y < 30 : null
    const res = !on ? '평일' : alarm == null || drop == null ? '-' : alarm && drop ? '적중' : !alarm && drop ? '놓침' : alarm ? '헛경보' : '정상'
    return { d, dr, y, res }
  })
  const opt = useMemo(() => ({
    animation: false,
    grid: { left: 44, right: 44, top: 24, bottom: 30 },
    tooltip: { trigger: 'axis', backgroundColor: C.tooltip, borderWidth: 0, textStyle: { color: '#fff', fontSize: 12 },
      formatter: (ps: any) => { const r = rows[ps[0].dataIndex]; return `<b>${lab(r.d)}</b><br/>${hasMin ? `예상 한낮 최저 <b>${fmt(r.dr?.min)}</b>원 · ` : ''}공표 한낮 최저 <b>${fmt(r.y)}</b>원<br/>급락 확률 ${r.dr?.prob != null ? Math.round(r.dr.prob * 100) + '%' : '-'} · ${r.dr ? LEVEL[r.dr.grade]?.name : '-'} · ${r.res}` } },
    xAxis: { type: 'category', data: dates.map(lab), axisLabel: { color: C.mute, fontSize: 11 }, axisTick: { show: false }, axisLine: { lineStyle: { color: C.line } } },
    yAxis: [{ type: 'value', min: hasMin ? 0 : undefined, name: '원', nameTextStyle: { color: C.mute, fontSize: 11 }, splitLine: { lineStyle: { color: C.line } }, axisLabel: { color: C.mute, fontSize: 11 } },
      { type: 'value', min: 0, max: 100, name: '확률 %', nameTextStyle: { color: C.mute, fontSize: 11 }, splitLine: { show: false }, axisLabel: { color: C.mute, fontSize: 11 } }],
    series: [
      { name: '급락 확률', type: 'bar', yAxisIndex: 1, barWidth: '45%', data: rows.map((r) => ({ value: r.dr?.prob != null ? +(r.dr.prob * 100).toFixed(1) : null,
        itemStyle: { color: (r.dr?.grade === '강한 경보' || r.dr?.grade === '경보') ? alpha(LV.strong.fill, 0.4) : r.dr?.grade === '주의' ? alpha(LV.warn.fill, 0.55) : alpha(LV.low.fill, 0.55), borderRadius: [3, 3, 0, 0] } })) },
      ...(!hasMin ? [] : [{ name: '예상 한낮 최저', type: 'line', data: rows.map((r) => r.dr?.min ?? null), symbol: 'circle', symbolSize: 7, lineStyle: { color: C.brand, width: 2.2 }, itemStyle: { color: C.brand },
        markLine: { silent: true, symbol: 'none', data: [
          { yAxis: WARN, lineStyle: { color: LV.warn.text, type: 'dashed' }, label: { formatter: `주의 ${WARN}`, color: LV.warn.text, fontSize: 11, position: 'insideEndTop' } },
          { yAxis: STRONG, lineStyle: { color: C.bad, type: 'dashed' }, label: { formatter: `경보 ${STRONG}`, color: C.bad, fontSize: 11, position: 'insideEndTop' } },
          { yAxis: 30, lineStyle: { color: C.ink, type: 'dotted' }, label: { formatter: '급락 30원', color: C.ink, fontSize: 11, position: 'insideEndTop' } }] } }]),
      { name: '공표 한낮 최저', type: 'line', data: rows.map((r) => r.y), symbol: 'rect', symbolSize: 7, lineStyle: { color: C.ink, width: 1.4 },
        itemStyle: { color: (p: any) => (rows[p.dataIndex].y != null && rows[p.dataIndex].y! < 30 ? C.bad : C.ink) },
        markLine: hasMin ? undefined : { silent: true, symbol: 'none', data: [{ yAxis: 30, lineStyle: { color: C.ink, type: 'dotted' }, label: { formatter: '급락 30원', color: C.ink, fontSize: 11, position: 'insideEndTop' } }] } },
    ],
  }), [dates.join()])
  const n = rows.filter((r) => r.res !== '-' && r.res !== '평일')
  const cnt = (k: string) => n.filter((r) => r.res === k).length
  return (
    <>
      <Kpis items={[
        { k: '비교한 날', v: n.length, s: `${lab(dates[0])} ~ ${lab(dates[dates.length - 1])}` },
        { k: '실제 급락일', v: cnt('적중') + cnt('놓침'), s: '한낮 최저 30원 미만' },
        { k: '적중 / 놓침', main: true, v: `${cnt('적중')} / ${cnt('놓침')}`, s: '주의 이상 경보 기준' },
        { k: '헛경보', v: cnt('헛경보'), s: '경보했지만 급락 없음' },
        { k: '평균 급락 확률', v: pct(avg(rows.map((r) => r.dr?.prob ?? null)), 0), s: `휴일·주말 ${dates.filter((d) => info(d).off).length}일 (평일은 대상 아님)` },
      ]} />
      <Card title={hasMin ? '예상 한낮 최저 vs 공표 한낮 최저' : '급락 확률 vs 공표 한낮 최저'} sub={hasMin ? '막대 = 급락 확률(오른쪽 축) · 점선 = 등급 문턱' : '막대 = 급락 확률(오른쪽 축) · 점선 = 급락 기준 30원'}
        right={<Legend items={[...(hasMin ? [{ t: '예상 한낮 최저', c: 'pred' }] : []), { t: '공표 한낮 최저', c: 'act' }]} />}>
        <EChart option={opt} height={340} />
      </Card>
      <Card flush title="날짜별" right={<CsvButton onClick={() => downloadCsv(`${REGION}_급락판별_${dates[0]}_${dates[dates.length - 1]}`,
        ['날짜', '등급', '급락 확률', ...(hasMin ? ['예상 한낮 최저'] : []), '공표 한낮 최저', '결과'],
        rows.map((r) => [r.d, r.dr ? LEVEL[r.dr.grade]?.name : null, r.dr?.prob, ...(hasMin ? [r.dr?.min] : []), r.y, r.res]))} />}>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr><th className="l">날짜</th>{hasMin && <th>예상 한낮 최저</th>}<th className="l">급락 확률 · 등급</th><th>공표 한낮 최저</th><th className="l">결과</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.d}>
                  <td className={`l ${info(r.d).off ? 'text-bad' : ''}`}>{lab(r.d)} {info(r.d).dt !== '평일' ? info(r.d).dt : ''}</td>
                  {hasMin && <td>{fmt(r.dr?.min)}</td>}<td className="l"><DropChip d={r.dr} date={r.d} /></td>
                  <td className={`font-semibold ${r.y != null && r.y < 30 ? 'text-bad' : ''}`}>{fmt(r.y)}</td>
                  <td className={`l font-semibold ${r.res === '적중' ? 'text-good' : r.res === '놓침' || r.res === '헛경보' ? 'text-ink' : 'text-mute'}`}>{r.res}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <p className="muted">내일 급락 경보 = 대상일 하루 전 15시에 {D.dropMeta.model}이 낸 판정. 급락 = 11~15시 최저 SMP 30원 미만 (육지는 모든 날, 제주는 현재 휴일·주말). 결과는 주의 이상을 경보로 봅니다.</p>
    </>
  )
}

function Summary({ s, unit }: { s: { p: any[]; lo: any[]; hi: any[]; y: any[] }; unit: string }) {
  const idx = s.p.map((p, i) => (p != null && s.y[i] != null ? i : -1)).filter((i) => i >= 0)
  const cov = idx.length ? idx.filter((i) => inside(s.y[i], s.lo[i], s.hi[i])).length / idx.length : null
  const below = idx.length ? idx.filter((i) => s.y[i] < s.lo[i]).length / idx.length : null
  return <Kpis items={[
    { k: `비교한 ${unit}`, v: idx.length },
    { k: '평균 오차 (MAE)', main: true, v: fmt(avg(idx.map((i) => Math.abs(s.y[i] - s.p[i])))) + '원', s: `편향(공표−예측) ${sgn(avg(idx.map((i) => s.y[i] - s.p[i])))}원` },
    { k: '80% 범위 포함률', v: pct(cov), s: '목표 75~85%', tone: cov == null ? undefined : cov >= 0.75 && cov <= 0.85 ? 'good' : 'bad' },
    { k: '하한 아래 / 상한 위', v: `${pct(below)} / ${pct(cov == null || below == null ? null : 1 - cov - below)}`, s: '각각 약 10%가 균형' },
    { k: '평균 범위 폭', v: fmt(avg(idx.map((i) => s.hi[i] - s.lo[i]))) + '원' },
  ]} />
}

function HourlyFlow({ dates, H }: { dates: string[]; H: string }) {
  const src = D.past.hourly[H] ?? {}
  const [day, setDay] = useState<string | null>(null) // null = 기간 전체, 날짜 = 그날 24시간
  const shown = day ? [day] : dates
  const s = { p: [] as any[], lo: [] as any[], hi: [] as any[], y: [] as any[] }
  shown.forEach((d) => { const r = src[d], y = D.past.actual[d] ?? Array(24).fill(null); for (let i = 0; i < 24; i++) { s.p.push(r?.[0][i] ?? null); s.lo.push(r?.[1][i] ?? null); s.hi.push(r?.[2][i] ?? null); s.y.push(y[i]) } })
  const all = { p: [] as any[], lo: [] as any[], hi: [] as any[], y: [] as any[] }
  dates.forEach((d) => { const r = src[d], y = D.past.actual[d] ?? Array(24).fill(null); for (let i = 0; i < 24; i++) { all.p.push(r?.[0][i] ?? null); all.lo.push(r?.[1][i] ?? null); all.hi.push(r?.[2][i] ?? null); all.y.push(y[i]) } })
  const opt = useMemo(() => day
    ? bandOption({ x: [...Array(24).keys()].map((i) => `${i + 1}시`), ...s, tip: (i) => `${lab(day)} ${i + 1}시`,
      xLabel: (i) => (i % 2 === 0 ? `${i + 1}시` : ''), shade: [[10, 14]], symbols: true })
    : bandOption({ x: dates.flatMap((d) => [...Array(24).keys()].map((i) => `${d} ${i}`)), ...s,
      tip: (i) => `${lab(dates[Math.floor(i / 24)])} ${(i % 24) + 1}시`, xLabel: (i) => (i % 24 === 11 ? (dates.length > 7 ? md(dates[Math.floor(i / 24)]) : lab(dates[Math.floor(i / 24)])) : ''),
      shade: dates.map((_, k) => [k * 24 + 10, k * 24 + 14] as [number, number]), dividers: dates.map((_, k) => k * 24).slice(1), zoom: dates.length > 7 }), [dates.join(), H, day])
  const when = H === '1' ? '하루 전' : H + '일 전'
  return (
    <>
      <Summary s={all} unit="시간" />
      <Card title={day ? `${lab(day)} 시간별 · ${when} 예측 vs 공표` : `시간별 · ${when} 예측 vs 공표`}
        sub={day ? '회색 = 한낮 11~15시' : '회색 = 한낮 11~15시 · 아래 표에서 날짜를 누르면 그날만 크게'}
        right={<div className="flex flex-wrap items-center gap-3">
          <Legend items={[{ t: '80% 범위 (당시)', c: 'band' }, { t: '당시 예측', c: 'pred' }, { t: '공표 SMP', c: 'act' }, { t: '범위 밖', c: 'out' }]} />
          {day && <button onClick={() => setDay(null)} className="btn">← 기간 전체</button>}
        </div>}>
        <EChart option={opt} height={380} />
      </Card>
      <Card flush title="날짜별" sub="범위 = 그날 24시간 하한·상한 평균 · 포함률 = 공표 SMP가 범위 안에 든 시간 비율"
        right={<CsvButton onClick={() => downloadCsv(`${REGION}_시간별_D+${H}_${dates[0]}_${dates[dates.length - 1]}`,
          ['날짜', '시', '예측', '하한', '상한', '공표 SMP', '범위 안'],
          dates.flatMap((d) => [...Array(24).keys()].map((i) => {
            const r = src[d], y = D.past.actual[d]?.[i] ?? null
            return [d, i + 1, r?.[0][i], r?.[1][i], r?.[2][i], y, y == null || !r ? null : inside(y, r[1][i], r[2][i]) ? 1 : 0]
          })))} />}>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr><th className="l">날짜</th><th>예측 일평균</th><th>80% 범위</th><th>공표 일평균</th><th>MAE</th><th>포함률</th><th>하한 아래 / 상한 위</th></tr></thead>
            <tbody>
              {dates.map((d) => {
                const r = src[d], y = D.past.actual[d] ?? []
                const ok = r ? [...Array(24).keys()].filter((i) => r[0][i] != null && y[i] != null) : []
                const cov = ok.length ? ok.filter((i) => inside(y[i], r![1][i], r![2][i])).length / ok.length : null
                const below = ok.filter((i) => r![1][i] != null && (y[i] as number) < (r![1][i] as number)).length
                const above = ok.filter((i) => r![2][i] != null && (y[i] as number) > (r![2][i] as number)).length
                return (
                  <tr key={d} onClick={() => r && setDay(d)} className={`${r ? 'cursor-pointer hover:bg-brand-soft/60' : ''} ${day === d ? 'sel' : ''}`}>
                    <td className={`l ${info(d).off ? 'text-bad' : ''}`}>{lab(d)} {info(d).dt !== '평일' ? info(d).dt : ''}</td>
                    <td className="font-semibold text-brand-dark">{fmt(r ? avg(r[0]) : null)}</td>
                    <td className="text-sub">{r ? `${fmt(avg(r[1]))} ~ ${fmt(avg(r[2]))}` : '-'}</td>
                    <td className="font-semibold">{fmt(avg(y))}</td>
                    <td>{fmt(ok.length ? avg(ok.map((i) => Math.abs((y[i] as number) - (r![0][i] as number)))) : null)}</td>
                    <td className={cov != null && cov >= 0.75 ? 'text-good' : ''}>{pct(cov)}</td>
                    <td className="text-sub">{ok.length ? `${below} / ${above}시간` : '-'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
      <CoverageTrend H={H} end={dates[dates.length - 1]} />
      <p className="muted">{when} 예측 = 대상일보다 {H}일 앞선 기준일 15시에 일주일 모델({REGION === '제주' ? '육지 일주일 예측 + 제주 잔차 보정' : 'Chronos-2 small + 잔차 보정'})이 낸 예측과 80% 범위입니다. 한낮 최저 가격과 급락 경보은 '한낮 최저 · 급락 경보' 탭에서 봅니다.</p>
    </>
  )
}

function DailyFlow({ dates, H, stat }: { dates: string[]; H: string; stat: Stat }) {
  const src = D.past.daily[H] ?? {}
  const s = { p: [] as any[], lo: [] as any[], hi: [] as any[], y: [] as any[] }
  dates.forEach((d) => { const r = src[d]?.[stat]; s.p.push(r?.[0] ?? null); s.lo.push(r?.[1] ?? null); s.hi.push(r?.[2] ?? null); s.y.push(D.past.dailyActual[d]?.[stat] ?? null) })
  const opt = useMemo(() => bandOption({ x: dates.map(lab), ...s, tip: (i) => `${lab(dates[i])} · ${STAT_NAME[stat]}`, symbols: true, daily: true,
    shade: dates.map((d, i) => i).filter((i) => info(dates[i]).off).map((i) => [i, i] as [number, number]) }), [dates.join(), H, stat])
  return (
    <>
      <Summary s={s} unit="날" />
      <Card title={`일별 · ${STAT_NAME[stat]} · ${H}일 전 예측 vs 공표`} sub="회색 = 휴무일"
        right={<Legend items={[{ t: '80% 범위', c: 'band' }, { t: '당시 예측', c: 'pred' }, { t: '공표', c: 'act' }, { t: '범위 밖', c: 'out' }]} />}>
        <EChart option={opt} height={340} />
      </Card>
      <Card flush title="날짜별" right={<CsvButton onClick={() => downloadCsv(`${REGION}_한달_${STAT_NAME[stat]}_${H}일전_${dates[0]}_${dates[dates.length - 1]}`,
        ['날짜', '하한', '예측', '상한', '공표'], dates.map((d, i) => [d, s.lo[i], s.p[i], s.hi[i], s.y[i]]))} />}>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr><th className="l">날짜</th><th>하한</th><th>예측</th><th>상한</th><th>공표</th><th>오차 (공표−예측)</th><th className="l">범위</th></tr></thead>
            <tbody>
              {dates.map((d, i) => (
                <tr key={d}>
                  <td className={`l ${info(d).off ? 'text-bad' : ''}`}>{lab(d)} {info(d).dt !== '평일' ? info(d).dt : ''}</td>
                  <td>{fmt(s.lo[i])}</td><td className="font-semibold text-brand-dark">{fmt(s.p[i])}</td><td>{fmt(s.hi[i])}</td><td className="font-semibold">{fmt(s.y[i])}</td>
                  <td>{s.y[i] != null && s.p[i] != null ? sgn(s.y[i] - s.p[i]) : '-'}</td>
                  <td className={`l font-semibold ${s.y[i] == null ? 'text-mute' : inside(s.y[i], s.lo[i], s.hi[i]) ? 'text-good' : 'text-ink'}`}>{s.y[i] == null ? '-' : inside(s.y[i], s.lo[i], s.hi[i]) ? '안' : '밖'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  )
}

/** 포함률 추이: 최근 60일, 7일 이동(그 7일의 모든 시간을 합쳐 계산). 점검용 — 이 값을 보고 모델을 고치지 않음 (2026 튜닝 금지) */
function CoverageTrend({ H, end }: { H: string; end: string }) {
  const src = D.past.hourly[H] ?? {}
  const all = range(addDays(end, -65), end)
  const daily = all.map((d) => {
    const r = src[d], y = D.past.actual[d] ?? []
    let n = 0, inn = 0, lo = 0
    if (r) for (let i = 0; i < 24; i++) {
      const v = y[i], l = r[1][i], h = r[2][i]
      if (v == null || l == null || h == null) continue
      n++; if (v < l) lo++; else if (v <= h) inn++
    }
    return { d, n, inn, lo }
  })
  const pts = daily.slice(6).map((_, k) => {
    const w = daily.slice(k, k + 7), n = w.reduce((a, x) => a + x.n, 0)
    if (n < 24 * 4) return { d: w[6].d, cov: null, lo: null, hi: null }
    const inn = w.reduce((a, x) => a + x.inn, 0), lo = w.reduce((a, x) => a + x.lo, 0)
    return { d: w[6].d, cov: (inn / n) * 100, lo: (lo / n) * 100, hi: ((n - inn - lo) / n) * 100 }
  }).slice(-60)
  const r1 = (v: number | null) => (v == null ? null : +v.toFixed(1))
  const opt = useMemo(() => ({
    animation: false,
    grid: { left: 44, right: 16, top: 20, bottom: 30 },
    tooltip: { trigger: 'axis', backgroundColor: C.tooltip, borderWidth: 0, textStyle: { color: '#fff', fontSize: 12 },
      formatter: (ps: any) => { const p = pts[ps[0].dataIndex]; return `<b>${lab(p.d)}까지 7일</b><br/>포함률 <b>${fmt(p.cov)}%</b><br/>하한 아래 ${fmt(p.lo)}% · 상한 위 ${fmt(p.hi)}%` } },
    xAxis: { type: 'category', data: pts.map((p) => md(p.d)), axisLabel: { color: C.mute, fontSize: 11 }, axisTick: { show: false }, axisLine: { lineStyle: { color: C.line } } },
    yAxis: { type: 'value', min: 0, max: 100, name: '%', nameTextStyle: { color: C.mute, fontSize: 11 }, splitLine: { lineStyle: { color: C.line } }, axisLabel: { color: C.mute, fontSize: 11 } },
    series: [
      { name: '포함률', type: 'line', data: pts.map((p) => r1(p.cov)), symbol: 'none', lineStyle: { color: C.brand, width: 2.2 }, z: 3,
        markArea: { silent: true, itemStyle: { color: alpha(LV.low.fill, 0.22) }, data: [[{ yAxis: 75 }, { yAxis: 85 }]] } },
      { name: '하한 아래', type: 'line', data: pts.map((p) => r1(p.lo)), symbol: 'none', lineStyle: { color: C.bad, width: 1.4, type: 'dashed' },
        markArea: { silent: true, itemStyle: { color: C.shade }, data: [[{ yAxis: 0 }, { yAxis: 10 }]] } },
      { name: '상한 위', type: 'line', data: pts.map((p) => r1(p.hi)), symbol: 'none', lineStyle: { color: P.orange, width: 1.4, type: 'dashed' } },
    ],
  }), [H, end])
  return (
    <Card title="80% 범위 포함률 추이" sub={`최근 60일 · 7일 이동 · ${H === '1' ? '하루 전' : H + '일 전'} 예측 · 초록 띠 = 목표 75~85%`}
      right={<Legend items={[{ t: '포함률', c: C.brand }, { t: '하한 아래', c: C.bad }, { t: '상한 위', c: P.orange }]} />}>
      <EChart option={opt} height={260} />
      <p className="muted mt-2">하한 아래·상한 위가 각각 10% 근처면 균형입니다. 한쪽이 계속 크면 범위가 그쪽으로 좁다는 뜻입니다 (점검용, 이 화면을 보고 모델을 고치지 않습니다).</p>
    </Card>
  )
}
