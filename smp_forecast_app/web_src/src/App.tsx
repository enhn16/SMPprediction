import { useEffect, useState } from 'react'
import { D, HAS_JEJU, REGION, loadJeju, setRegion, type Region } from './data'
import { dayDiff, lab, toIso } from './fmt'
import { Week } from './pages/Week'
import { Month } from './pages/Month'
import { Flow } from './pages/Flow'
import { Docs, type DocPage } from './pages/Docs'
import { Help } from './pages/Help'

export type Route = 'week' | 'month' | 'flow' | 'docs' | 'help'
// 좌측 메뉴: 비슷한 것끼리 묶음 (예측 / 검증 / 안내)
const NAV: { group: string; items: { r: Route; t: string; icon: string }[] }[] = [
  { group: '예측', items: [{ r: 'week', t: '일주일 예측', icon: 'M3 5h18M3 10h18M3 15h12M3 20h8' }, { r: 'month', t: '한 달 예측', icon: 'M4 5h16v15H4zM4 9h16M9 3v4M15 3v4' }] },
  { group: '검증', items: [{ r: 'flow', t: '이전 흐름 확인', icon: 'M3 12a9 9 0 1 0 3-6.7M3 4v5h5M12 7v5l3 2' }] },
  { group: '안내', items: [{ r: 'docs', t: '참고 문서', icon: 'M6 3h9l5 5v13H6zM14 3v6h6M9 13h8M9 17h6' }, { r: 'help', t: '설명', icon: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 16v-1M12 8a2.2 2.2 0 0 1 1.3 4c-.8.5-1.3 1-1.3 2' }] },
]
const DOC_PAGES: DocPage[] = ['ov', 'drop', 'wk', 'mo']

function parse(): { r: Route; doc: DocPage } {
  const [a, b] = location.hash.slice(1).split('/')
  const r = (['week', 'month', 'flow', 'docs', 'help'] as Route[]).includes(a as Route) ? (a as Route) : 'help' // 첫 화면 = 설명
  return { r, doc: DOC_PAGES.includes(b as DocPage) ? (b as DocPage) : 'ov' }
}

function Icon({ d }: { d: string }) {
  return <svg viewBox="0 0 24 24" className="h-[18px] w-[18px] shrink-0" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round"><path d={d} /></svg>
}

// 계통: 주소의 ?region=제주 → 지난번 선택 → 육지
function savedRegion(): Region {
  const q = new URLSearchParams(location.search).get('region')
  if (q === '제주' || q === '육지') return q
  try { return localStorage.getItem('smp-region') === '제주' ? '제주' : '육지' } catch { return '육지' }
}

/** 지금 있어야 할 마지막 기준일: 매일 15:30 예측 (여유 15분) 전이면 어제 */
function dueBase() {
  const now = new Date()
  const d = new Date(now)
  if (now.getHours() * 60 + now.getMinutes() < 15 * 60 + 45) d.setDate(d.getDate() - 1)
  return toIso(d)
}

/** 예측이 늦어지면 맨 위에 경고 (운영 PC가 꺼져 있던 경우 등) */
function StaleBanner() {
  const due = dueBase(), base = D.week.base
  const late = dayDiff(base, due)
  if (late <= 0) return null
  return (
    <div className="mb-5 rounded-lg border border-[#f2c200] bg-[#fdf6d8] px-4 py-3 text-[13.5px] text-[#6b5300]">
      <b>예측이 {late}일 늦어지고 있습니다.</b> 마지막 기준일 {lab(base)}, 지금쯤 있어야 할 기준일 {lab(due)}.
      운영 PC가 켜지면 밀린 날을 자동으로 따라잡습니다 (마지막 산출 {D.lastRun ?? '-'}).
    </div>
  )
}

export default function App() {
  const [st, setSt] = useState(parse)
  const [region, setReg] = useState<Region>('육지')
  const [loading, setLoading] = useState(false)
  const pick = async (r: Region) => {
    if (r === '제주') {
      setLoading(true)
      const ok = await loadJeju() // 제주 데이터는 처음 고를 때 읽음
      setLoading(false)
      if (!ok) { alert('제주 데이터 파일(data/forecast_data_jeju.js)을 읽지 못했습니다.'); return }
    }
    setRegion(r)
    setReg(REGION)
    try { localStorage.setItem('smp-region', REGION) } catch { /* 저장 못 해도 동작 */ }
  }
  useEffect(() => { if (savedRegion() === '제주' && HAS_JEJU) pick('제주') }, [])
  const [menu, setMenu] = useState(false)
  useEffect(() => {
    const h = () => { setSt(parse()); window.scrollTo(0, 0) }
    addEventListener('hashchange', h)
    return () => removeEventListener('hashchange', h)
  }, [])
  const go = (r: Route, doc?: DocPage) => { location.hash = r === 'docs' ? `docs/${doc ?? st.doc}` : r; setMenu(false) }

  const side = (
    <nav className="flex h-full flex-col">
      <div className="px-5 pb-5 pt-6">
        <div className="flex items-center gap-2.5">
          <span className="grid h-9 w-9 place-items-center rounded-lg bg-brand text-[13px] font-extrabold text-white">SMP</span>
          <div>
            <div className="text-[16px] font-extrabold leading-tight text-white">SMP 예측</div>
            <div className="text-[11.5px] text-[#9fc7e6]">{region} 계통한계가격</div>
          </div>
        </div>
        {HAS_JEJU && (
          <div className="mt-4 grid grid-cols-2 gap-1 rounded-lg bg-white/10 p-1" role="group" aria-label="계통 선택">
            {(['육지', '제주'] as Region[]).map((r) => (
              <button key={r} onClick={() => { pick(r); setMenu(false) }} aria-pressed={region === r} disabled={loading}
                className={`rounded-md py-1.5 text-[13px] font-bold transition ${region === r ? 'bg-white text-brand-dark shadow-sm' : 'text-[#d5e6f3] hover:bg-white/10'}`}>
                {r === '제주' && loading ? '읽는 중…' : r}</button>
            ))}
          </div>
        )}
      </div>
      <div className="flex-1 space-y-1 px-3">
        {NAV.map((g, gi) => (
          <div key={g.group}>
            {gi > 0 && <div className="mx-2 my-3 border-t border-white/10" />}
            <div className="px-3 pb-1.5 text-[11px] font-semibold tracking-wider text-[#7fa9c8]">{g.group}</div>
            {g.items.map((it) => {
              const on = st.r === it.r
              return (
                <button key={it.r} onClick={() => go(it.r)}
                  className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-[14px] font-semibold transition ${on ? 'bg-white text-brand-dark shadow-sm' : 'text-[#d5e6f3] hover:bg-white/10 hover:text-white'}`}>
                  <Icon d={it.icon} />{it.t}
                </button>
              )
            })}
          </div>
        ))}
      </div>
      <div className="space-y-1 border-t border-white/10 px-5 py-4 text-[11.5px] leading-5 text-[#9fc7e6]">
        <div>최신 공표 <b className="text-white">{lab(D.lastPublished)}</b></div>
        {D.lastRun && <div>예측 산출 <b className="text-white">{D.lastRun}</b></div>}
        <div>데이터 생성 {D.generated}</div>
      </div>
    </nav>
  )

  return (
    <div className="min-h-screen">
      <aside className="fixed inset-y-0 left-0 z-20 hidden w-[232px] bg-brand-ink lg:block">{side}</aside>
      {/* 모바일 상단 바 */}
      <header className="sticky top-0 z-20 flex h-14 items-center justify-between bg-brand-ink px-4 lg:hidden">
        <span className="text-[16px] font-extrabold text-white">SMP 예측 <span className="text-[13px] font-semibold text-[#9fc7e6]">· {region}</span></span>
        <button onClick={() => setMenu(!menu)} className="rounded-lg p-2 text-white hover:bg-white/10" aria-label="메뉴">
          <Icon d={menu ? 'M6 6l12 12M18 6L6 18' : 'M4 7h16M4 12h16M4 17h16'} />
        </button>
      </header>
      {menu && (
        <div className="fixed inset-0 z-30 lg:hidden" onClick={() => setMenu(false)}>
          <div className="absolute inset-0 bg-black/30" />
          <aside className="absolute inset-y-0 left-0 w-[260px] bg-brand-ink" onClick={(e) => e.stopPropagation()}>{side}</aside>
        </div>
      )}
      <main className="px-4 py-6 lg:ml-[232px] lg:px-8 lg:py-8">
        <div key={region} className="mx-auto max-w-[1240px]">
          {st.r !== 'docs' && <StaleBanner />}
          {st.r === 'week' && <Week />}
          {st.r === 'month' && <Month />}
          {st.r === 'flow' && <Flow />}
          {st.r === 'docs' && <Docs page={st.doc} onPage={(p) => go('docs', p)} />}
          {st.r === 'help' && <Help go={go} />}
          <footer className="mt-10 border-t border-line pt-4 text-[12px] text-mute">
            출처: 전력거래소·기상청·한국천문연구원·EPSIS 공공데이터 · 숙명여대 P4 산학 프로젝트 (한국중부발전 동반성장부)
          </footer>
        </div>
      </main>
    </div>
  )
}
