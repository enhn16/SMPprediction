import { D, LEVEL, REGION, dropTarget, surgeData } from '../data'
import { avg, fmt, lab } from '../fmt'
import { Card, Num, PageHeader } from '../components/ui'
import type { Route } from '../App'

export function Help({ go }: { go: (r: Route) => void }) {
  const W = D.week, M = D.month, dr = W.drop
  const tiles: { r: Route; t: string; v: string; s: string }[] = [
    { r: 'week', t: '내일 급락 경보', v: !dropTarget(W.days[0].date) ? '준비 중' : dr ? `${LEVEL[dr.grade]?.name ?? dr.grade}` : '-', s: !dropTarget(W.days[0].date) ? '제주 평일 급락 예측을 준비하고 있습니다' : `${lab(W.days[0].date)} 한낮 최저 30원 미만 여부` },
    { r: 'week', t: '내일 급등 경보', v: !surgeData() ? '준비 중' : ({ 경보: '경보', 주의: '주의' } as Record<string, string>)[surgeData()!.past[W.days[0].date]?.grade ?? ''] ?? '낮음',
      s: surgeData() ? `${lab(W.days[0].date)} 하루 최고 SMP 급등 여부` : '제주 급등 예측은 아직 없습니다' },
    { r: 'week', t: '일주일 예측', v: `${fmt(avg(W.days.flatMap((d) => d.p)))}원`, s: `${lab(W.days[0].date)} ~ ${lab(W.days[W.days.length - 1].date)} 평균` },
    { r: 'month', t: '한 달 예측', v: `${fmt(avg(M.p_mean))}원`, s: `${lab(M.d[0])} ~ ${lab(M.d[M.d.length - 1])} 일평균 (참고)` },
  ]
  return (
    <div className="space-y-5">
      <PageHeader eyebrow={`안내 · ${REGION}`} title="SMP 예측 웹앱" sub={`${REGION} 계통한계가격(SMP)을 기준일 15시에 예측하고, 실제 가격이 들어올 80% 범위를 함께 보여 줍니다 · 왼쪽 메뉴에서 육지/제주 전환`}
        meta={[{ k: '기준일', v: <>{W.base} 15시 판단</>, hl: true }, { k: '최신 공표', v: lab(D.lastPublished) }]} />
      <div className="card grid gap-1 p-2 md:grid-cols-2 xl:grid-cols-4">
        {tiles.map((t) => (
          <button key={t.t} onClick={() => go(t.r)} className="group rounded-ui px-5 py-4 text-left transition hover:bg-track">
            <div className="flex items-center justify-between">
              <span className="t-label">{t.t}</span><span className="text-[12px] font-semibold text-brand opacity-0 transition group-hover:opacity-100">열기 →</span>
            </div>
            <div className="v-hero mt-2 text-ink"><Num v={t.v} /></div>
            <div className="mt-1 text-[12px] text-mute">{t.s}</div>
          </button>
        ))}
      </div>
      <Card title="메뉴 안내">
        <div className="grid gap-x-12 gap-y-6 md:grid-cols-2">
          {[
            ['일주일 예측', '내일부터 7일 동안 시간별 SMP와 80% 범위. 맨 위에 내일 급락·급등 경보와 그렇게 본 근거가 붙습니다. 예측값은 그대로 두고, 내일 급락 경보가 나면 내일 한낮 범위만 아래로 넓힙니다.'],
            ['한 달 예측', M.mode === 'next_month' ? '매월 말에 다음 달 모든 날의 일평균을 80% 범위와 함께 예측합니다. 지난 날은 공표값과 나란히. 정확도가 낮아 참고 전망입니다.' : '7~34일 뒤를 일 단위(일평균·한낮·저녁·한낮 최저)로. 주차·월별 평균도 함께. 정확도가 낮아 참고 전망입니다.'],
            ['이전 흐름 확인', '달력에서 기간(최대 2주)을 골라, 그때 냈던 예측(급락 경보·일주일·한 달)을 공표 SMP와 비교합니다.'],
            ['예측 달력', '달력에서 날짜마다 그날 예측이 80% 범위를 벗어난 시간 수를 색으로 봅니다 (초록 0~5시간 · 주황 6~10시간 · 빨강 11시간 이상). 월을 넘겨 가며 빗나간 시기를 찾습니다.'],
            ['참고 문서', '예측별 최종 모델, 비교한 모델과 성능, 선택 이유, 사용 데이터, 성능 평가표.'],
          ].map(([t, s]) => (
            <div key={t}>
              <div className="text-[14px] font-semibold text-brand-dark">{t}</div>
              <p className="mt-1 text-[13.5px] leading-6 text-sub">{s}</p>
            </div>
          ))}
        </div>
      </Card>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="80% 범위란?">
          <div className="prose-doc space-y-2">
            <p>「실제 가격이 이 사이에 들어올 것」이라는 약속의 폭입니다. 10번 중 8번은 공표 SMP가 범위 안에 들어와야 하고, 벗어날 때는 아래·위로 반반씩이 균형입니다.</p>
            <p className="note">활쏘기 비유 — 예측값은 과녁 중심, 범위는 둘레의 원, 공표 SMP는 날아오는 화살.</p>
          </div>
        </Card>
        <Card title="급락·급등 경보란?">
          <div className="prose-doc space-y-2">
            <p>한낮에 태양광이 많고 수요가 적으면 SMP가 0원 근처로 떨어집니다 (주로 휴일, 평일에도 생김). 내일 <b>11~15시 최저 SMP가 30원 미만</b>일지를 판별해 <b>낮음 · 주의 · 경보</b>와 확률, 그렇게 본 근거를 함께 보여 줍니다.</p>
            <p><b>급등</b>은 내일 하루 최고 SMP가 최근 4주 평소 최고의 1.25배 이상으로 뛰는 날입니다. 같은 방식으로 <b>낮음 · 주의 · 경보</b>와 확률, 근거를 보여 주며, 휴일은 수요가 적어 경보를 내지 않습니다. 급등 '주의'는 헛경보가 많아 참고로 보세요.</p>
          </div>
        </Card>
        <Card title="어떻게 골랐나">
          <div className="prose-doc space-y-2">
            <p>여러 모델(부스팅·선형·트리·신경망·시계열 기초모델 Chronos 등)과 앙상블을 같은 조건(기준일 15시에 알 수 있는 정보)으로 비교해 <b>2025년 성적만으로</b> 고르고, 2026년은 마지막에 채점했습니다.</p>
          </div>
        </Card>
        <Card title="읽을 때 주의">
          <div className="prose-doc"><ul>
            <li>2일 뒤부터의 휴일 한낮 급락(0원 근처)은 범위가 잘 담지 못합니다 — 내일은 급락 경보에 맞춰 범위를 넓히고, 그 뒤는 급락 경보가 없습니다.</li>
            <li>멀리 볼수록 범위가 넓어집니다. 한 달 예측은 가격 수준이 급변하는 달에 범위를 벗어나기 쉽습니다.</li>
          </ul></div>
        </Card>
      </div>
    </div>
  )
}
