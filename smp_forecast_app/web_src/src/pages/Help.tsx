import { D, LEVEL } from '../data'
import { avg, fmt, lab } from '../fmt'
import { Card, PageHeader } from '../components/ui'
import type { Route } from '../App'

export function Help({ go }: { go: (r: Route) => void }) {
  const W = D.week, M = D.month, dr = W.drop
  const tiles: { r: Route; t: string; v: string; s: string }[] = [
    { r: 'week', t: '다음 날 급락 판별', v: dr?.prob != null ? `${Math.round(dr.prob * 100)}% · ${LEVEL[dr.grade]?.name ?? dr.grade}` : '-', s: `${lab(W.days[0].date)} 한낮 최저 30원 미만 확률` },
    { r: 'week', t: '일주일 예측', v: `${fmt(avg(W.days.flatMap((d) => d.p)))}원`, s: `${lab(W.days[0].date)} ~ ${lab(W.days[W.days.length - 1].date)} 평균` },
    { r: 'month', t: '한 달 예측', v: `${fmt(avg(M.p_mean))}원`, s: `${lab(M.d[0])} ~ ${lab(M.d[M.d.length - 1])} 일평균 (참고)` },
  ]
  return (
    <div className="space-y-5">
      <PageHeader title="SMP 예측 웹앱" sub="육지 계통한계가격(SMP)을 기준일 15시에 예측하고, 실제 가격이 들어올 80% 범위를 함께 보여 줍니다" />
      <div className="grid gap-3 md:grid-cols-3">
        {tiles.map((t) => (
          <button key={t.t} onClick={() => go(t.r)} className="card group p-5 text-left transition hover:border-brand/60 hover:shadow-md">
            <div className="text-[12.5px] font-semibold text-sub">{t.t}</div>
            <div className="mt-1 text-[26px] font-extrabold tracking-tight text-brand-dark tabular-nums">{t.v}</div>
            <div className="mt-0.5 text-[12.5px] text-mute">{t.s}</div>
            <div className="mt-3 text-[12.5px] font-semibold text-brand group-hover:underline">자세히 보기 →</div>
          </button>
        ))}
      </div>
      <Card title="메뉴 안내">
        <div className="grid gap-4 md:grid-cols-2">
          {[
            ['일주일 예측', '내일부터 7일 동안 시간별 SMP와 80% 범위. 맨 위에 과제1 모델의 다음 날 급락 판별(확률·등급)이 붙습니다. 예측값은 그대로 두고 경보만 더합니다.'],
            ['한 달 예측', '7~34일 뒤를 일 단위(일평균·한낮·저녁·한낮 최저)로. 주차·월별 평균도 함께. 정확도가 낮아 참고 전망입니다.'],
            ['이전 흐름 확인', '달력에서 기간(최대 2주)을 골라, 그때 냈던 예측(급락 판별·일주일·한 달)을 공표 SMP와 비교합니다.'],
            ['참고 문서', '예측별 최종 모델, 비교한 모델과 성능, 선택 이유, 사용 데이터, 팀 비교용 성적표.'],
          ].map(([t, s]) => (
            <div key={t} className="rounded-lg bg-[#f7f9fb] p-4">
              <div className="text-[14px] font-bold text-brand-ink">{t}</div>
              <p className="mt-1 text-[13.5px] leading-6 text-sub">{s}</p>
            </div>
          ))}
        </div>
      </Card>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="80% 범위란?">
          <div className="prose-doc space-y-2">
            <p>「실제 가격이 이 사이에 들어올 것」이라는 약속의 폭입니다. 10번 중 8번은 공표 SMP가 범위 안에 들어와야 하고, 벗어날 때는 아래·위로 반반씩이 균형입니다.</p>
            <p className="rounded-lg border-l-4 border-brand bg-brand-soft px-4 py-3">활쏘기 비유 — 예측값은 과녁 중심, 범위는 둘레의 원, 공표 SMP는 날아오는 화살.</p>
          </div>
        </Card>
        <Card title="급락 판별이란?">
          <div className="prose-doc space-y-2">
            <p>휴일 한낮에 태양광이 많고 수요가 적으면 SMP가 0원 근처로 떨어집니다. 내일 <b>11~15시 최저 SMP가 30원 미만</b>일지를 과제1 모델(Ridge + EBM + 랜덤포레스트)이 판별해 <b>낮음 · 주의 · 강한 경보</b>와 확률로 보여 줍니다.</p>
          </div>
        </Card>
        <Card title="어떻게 골랐나">
          <div className="prose-doc space-y-2">
            <p>여러 모델(부스팅·선형·트리·신경망·시계열 기초모델 Chronos 등)과 앙상블을 같은 조건(기준일 15시에 알 수 있는 정보)으로 비교해 <b>2025년 성적만으로</b> 고르고, 2026년은 마지막에 채점했습니다.</p>
          </div>
        </Card>
        <Card title="읽을 때 주의">
          <div className="prose-doc"><ul>
            <li>휴일 한낮 급락(0원 근처)은 범위가 잘 담지 못합니다 — 급락 판별을 함께 보세요.</li>
            <li>멀리 볼수록 범위가 넓어집니다. 한 달 예측은 가격 수준이 급변하는 달에 범위를 벗어나기 쉽습니다.</li>
          </ul></div>
        </Card>
      </div>
    </div>
  )
}
