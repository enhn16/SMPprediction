import type { ReactNode } from 'react'
import { D, REGION, STAT_NAME, T_STRONG, T_WARN, type Row, type Stat } from '../data'
import { fmt, pct, sgn } from '../fmt'
import { Card, Kpis, PageHeader, Seg, Table, type Col } from '../components/ui'

export type DocPage = 'ov' | 'drop' | 'wk' | 'mo'
// 등급 표시 이름: 데이터 값 '강한 경보'를 현업 표현 '경보'로 (2026-10-26)
const gname = (v: any) => (v == null ? '-' : String(v).replace('강한 경보', '경보'))
const n2 = (v: any) => fmt(v, 2), n3 = (v: any) => fmt(v, 3)
const MET = D.docs?.metrics ?? [], FIN = D.docs?.final ?? {}
const WZC = D.perf.week?.choice ?? {}, MZC = D.perf.month?.choice ?? {}
const testRange = FIN['시험 기간'] ?? '2026-01-01 ~'
const met = (pred: string, per: string, seg: string): Row => MET.find((r) => r['예측'] === pred && String(r['기간']).startsWith(per) && r['구분'] === seg) ?? {}
const covCls = (r: Row) => (r['포함률'] == null ? '' : r['포함률'] >= 0.75 && r['포함률'] <= 0.85 ? 'text-good' : 'text-bad font-semibold')
const METCOLS: Col[] = [{ k: '구분', t: '구분', l: true }, { k: '표본수', t: '표본 수' }, { k: 'MAE', t: 'MAE', f: n2 }, { k: 'RMSE', t: 'RMSE', f: n2 },
  { k: 'MAPE(20원 이상)', t: 'MAPE', f: (v) => pct(v) }, { k: 'rMAE', t: 'rMAE', f: n3 }, { k: '편향(공표−예측)', t: '편향', f: (v) => sgn(v) },
  { k: '포함률', t: '포함률', f: (v) => pct(v), cls: covCls }, { k: '하한 아래', t: '하한 아래', f: (v) => pct(v) }, { k: '상한 위', t: '상한 위', f: (v) => pct(v) },
  { k: '평균 폭', t: '평균 폭', f: (v) => fmt(v) }, { k: '구간점수', t: '구간점수', f: (v) => fmt(v) }]
const metRows = (pred: string, per: string, keep?: (s: string) => boolean) => MET.filter((r) => r['예측'] === pred && String(r['기간']).startsWith(per) && (!keep || keep(r['구분'])))
const tone = (v: any) => (v == null ? undefined : v >= 0.75 && v <= 0.85 ? 'good' : 'bad') as 'good' | 'bad' | undefined

function Sec({ title, sub, children }: { title: string; sub?: ReactNode; children: ReactNode }) {
  return <Card title={title} sub={sub}><div className="prose-doc space-y-3">{children}</div></Card>
}
function DataTable({ rows }: { rows: string[][] }) {
  return <Table rows={rows.map(([a, b, c]) => ({ a, b, c }))} cols={[{ k: 'a', t: '자료', l: true, w: true }, { k: 'b', t: '쓰는 방식', l: true, w: true }, { k: 'c', t: '확보 시점 (판단: 기준일 15시)', l: true, w: true }]} />
}
const NotUsed = () => (
  <ul>
    <li><b>대상일 하루전 수요예측·SMP</b>: 평일 19~20시, 주말 14시 이전 공표라 15시 판단에 쓸 수 없습니다 (공표경계 판정서). D-1까지의 실적만 씁니다.</li>
    <li><b>실측 날씨</b>(일사량·운량 실측): 판단 시각에 알 수 없는 값이라 쓰지 않았습니다.</li>
    <li><b>담당자 운영 모델의 예측값</b>: 웹앱은 자체 모델만 씁니다 (재배포 금지 조건).</li>
    <li><b>전력수급예보 API</b>: 2026-10-22 승인. 내일 1건만 제공하고 과거 조회가 안 돼 매일 쌓는 중이며, 아직 피처로 쓰지 않습니다 (향후 후보).</li>
  </ul>
)

const DATA_HOURLY = [
  ['시간별 SMP (육지)', 'Chronos 입력: 최근 2,048시간(약 85일). 잔차 보정: 어제·같은 유형일·28일 평균 대비 차이', '기준일 SMP는 전날 저녁 공표 → 기준일 15시에 확보'],
  ['달력 (휴무일·요일유형·명절·근로자의날, 앞뒤 휴무)', 'Chronos 공변량(과거·미래), 잔차 보정 피처', '사전 확정 (천문연구원 특일정보)'],
  ['전력거래소 주간 전력수요 전망', '대상일 최대부하, 전망 내 비율, 전날 대비', '게시 다음 날부터 (게시판 이미지 OCR + 전력정보앱)'],
  ['기상청 단기 육상예보 (하늘·강수확률·기온)', 'D+1~D+3 대상일 오전·오후', '기준일 11시(없으면 05시) 발표분'],
  ['기상청 중기 육상예보 (하늘·강수확률)', 'D+4~D+7 (기온 없음)', '기준일 06시 발표분'],
  ['Open-Meteo 일사·운량 예보 (ECMWF·GFS·JMA, 7개 도시)', '한낮 11~15시 일사 합·운량 평균, D+1만', '대상일 2일 전 실행분'],
  ['EPSIS 연료비 단가 (LNG·유연탄)', '월 값', 'M월 값은 M+2월 1일부터'],
  ['시장참여 태양광 설비용량', '월 값', 'M월 값은 M+2월 1일부터'],
]

/** 성적 지표 읽는 법 (육지·제주 공용) */
function Glossary({ base }: { base: string }) {
  return (
    <Sec title="성적 지표 읽는 법">
        <dl className="grid grid-cols-1 gap-x-5 gap-y-2 text-[14px] md:grid-cols-[max-content_1fr]">
          {[['MAE (원)', '공표 SMP와 예측의 차이(절댓값) 평균. 작을수록 좋음. 협의된 주 지표.'], ['RMSE (원)', '차이 제곱 평균의 제곱근. 크게 틀린 시간이 많을수록 커짐.'],
            ['MAPE', '차이 ÷ 공표 SMP의 평균. 0원 근처에서 값이 터지므로 공표 20원 이상 시간만 계산 (현업 운영 기준과 같음).'],
            ['rMAE', `MAE ÷ 단순 기준 MAE. 1보다 작으면 단순 기준보다 나음. 단순 기준: ${base}.`],
            ['편향 (원)', '평균(공표 − 예측). +면 예측이 낮게, −면 높게 나오는 경향.'], ['포함률', '80% 범위 안에 공표 SMP가 든 비율. 성공 기준 75~85%.'],
            ['하한 아래 / 상한 위', '범위를 벗어난 방향. 각각 약 10%면 균형.'], ['평균 폭 (원)', '상한 − 하한 평균. 포함률이 같으면 좁을수록 좋음.'],
            ['구간점수 (Winkler)', '폭 + 벗어난 만큼의 벌점(× 10). 포함률과 폭을 한 숫자로 묶은 값, 낮을수록 좋음. 범위 방법 선택 규칙.'],
            ['재현율 / 정밀도', '급락 경보: 실제 급락일 중 경보한 비율 / 경보한 날 중 실제 급락 비율.'], ['헛경보율', '급락이 없던 날 중 경보한 비율.'],
            ['ROC-AUC / PR-AUC', '경보 문턱과 무관한 판별력. 1에 가까울수록 좋음 (PR-AUC는 드문 급락에 민감).']].map(([k, v]) => (
            <div key={k} className="contents"><dt className="font-semibold text-brand-dark">{k}</dt><dd className="text-ink">{v}</dd></div>
          ))}
        </dl>
      </Sec>
  )
}

function Overview() {
  const w = met('일주일', '2026', '전체'), w1 = met('일주일 D+1', '2026', '전체'), m = met('한 달', '2026', '일평균')
  const final = [
    { a: '내일 급락 경보', m: '과제1 최종 모델: Ridge + EBM + 랜덤포레스트 평균 → 내일 한낮 최저 SMP → 주의(≤90.1원)·경보(≤75.3원), 급락 확률', s: '급락 20일 중 주의 이상 18일 적중 (헛경보 72) · 경보 16일 (헛경보 27) · ROC-AUC 0.93' },
    { a: '일주일 예측 (D+1~D+7, 시간별)', m: `Chronos-2 small + ${WZC['잔차 보정 λ'] ?? 0.5} × 잔차 보정 (외부 정보 LightGBM)`, s: `MAE ${n2(w.MAE)}원 · 포함률 ${pct(w['포함률'])} · 폭 ${fmt(w['평균 폭'])}원 (내일 시간별 MAE ${n2(w1.MAE)}원)` },
    { a: '한 달 예측 (7~34일 뒤, 일별)', m: '기준(최근 같은 유형일 4일 평균) + λ × (LSTM·GRU 평균 − 기준) · λ 일평균 0.25, 한낮·저녁·한낮 최저 0.5', s: `일평균 MAE ${n2(m.MAE)}원 · 포함률 ${pct(m['포함률'])} (참고 전망)` },
  ]
  const head = [{ ...w, 예측: '일주일 (전체)' }, { ...w1, 예측: '일주일 · 내일 (D+1)' },
    ...(['일평균', '한낮 11~15시', '저녁 18~22시', '한낮 최저'] as const).map((s) => ({ ...met('한 달', '2026', s), 예측: '한 달 · ' + s }))]
  const dt = (D.docs?.dropTest ?? []).filter((r) => r['구간'] === '2026 전체' && r['경보'] !== '주의만')
  return (
    <>
      <Sec title="최종 모델" sub={`${FIN['확정일'] ?? ''} 확정`}>
        <p>모든 모델은 <b>기준일 15시에 알 수 있는 정보만</b> 쓰고, <b>2025년 성적만으로</b> 골랐습니다. 2026년({testRange})은 고른 뒤 채점만 했습니다.</p>
        <Table rows={final} cols={[{ k: 'a', t: '예측', l: true }, { k: 'm', t: '최종 모델', l: true, w: true }, { k: 's', t: '2026 시험 성적', l: true, w: true }]} />
      </Sec>
      <Sec title="팀 비교용 핵심 지표" sub="2026 시험 · 같은 정의로 다시 계산하면 비교할 수 있습니다">
        <p className="!text-[13px] font-semibold">가격 예측 (일주일·한 달)</p>
        <Table rows={head} csv="육지_최종성적_2026" cols={[{ k: '예측', t: '예측', l: true }, ...METCOLS.slice(1)]} />
        <p className="!text-[13px] font-semibold">내일 급락 경보</p>
        <Table rows={dt} cols={[{ k: '경보', t: '경보 기준', l: true, f: gname }, { k: '일수', t: '일수' }, { k: '급락일', t: '급락일' }, { k: '적중', t: '적중' }, { k: '놓침', t: '놓침' }, { k: '헛경보', t: '헛경보' },
          { k: '재현율', t: '재현율', f: (v) => pct(v) }, { k: '정밀도', t: '정밀도', f: (v) => pct(v) }, { k: '헛경보율', t: '헛경보율', f: (v) => pct(v) }, { k: 'ROC-AUC', t: 'ROC-AUC', f: n3 }, { k: 'PR-AUC', t: 'PR-AUC', f: n3 }]} />
      </Sec>
      <Glossary base="일주일 = 지난주 같은 시간, 내일(D+1) = 전날 같은 시간, 한 달 = 같은 유형일 4일 평균" />
      <Sec title="평가 규칙">
        <ul>
          <li><b>판단 시각 = 기준일 15시</b> (현업 주간 예측과 같음). 대상일 하루전 수요예측·SMP는 저녁 공표라 쓰지 않습니다.</li>
          <li><b>시간 순서만</b>: 매월 1일 재학습하며 그 전까지의 자료로만 학습 (랜덤 분할 없음). 2025~2026 모든 예측이 '그 시점에 낼 수 있었던 예측'입니다.</li>
          <li><b>선택</b>: 2025 상반기로 순위·가중치·보정 강도를 맞추고 → 2025 하반기 성적으로 비교 (0.05원 이내면 단순한 쪽) → 2025 전체로 다시 맞춤. 급락 경보은 2025 교차검증(급락 90%를 잡을 때 헛경보 수).</li>
          <li><b>시험</b>: 2026-01-01 이후 한 번 채점. 모델을 바꿔 다시 채점한 경우 '2차·3차 시험'으로 표기합니다 (일주일 3차, 한 달 3차).</li>
        </ul>
      </Sec>
      <Sec title="쓰지 않은 정보"><NotUsed /></Sec>
    </>
  )
}

function DropDoc() {
  const dt = D.docs?.dropTest ?? [], dz = D.docs?.dropZoo ?? []
  const all = dt.find((r) => r['경보'] === '주의 이상' && r['구간'] === '2026 전체') ?? {}
  const st = dt.find((r) => r['경보'] === '강한 경보' && r['구간'] === '2026 전체') ?? {}
  return (
    <>
      <Kpis items={[
        { k: '재현율 (주의 이상)', v: pct(all['재현율'], 0), s: `급락 ${all['급락일']}일 중 ${all['적중']}일 적중` },
        { k: '헛경보 (주의 이상)', v: `${all['헛경보']}일`, s: `정밀도 ${pct(all['정밀도'], 0)}` },
        { k: '재현율 (경보)', v: pct(st['재현율'], 0), s: `${st['적중']}일 적중 · 헛경보 ${st['헛경보']}일` },
        { k: 'ROC-AUC', v: n3(all['ROC-AUC']), s: `PR-AUC ${n3(all['PR-AUC'])}` },
        { k: '시험 기간', v: `${all['일수'] ?? '-'}일`, s: '2026-01-01 ~ 10-18 (과제1 1회 채점)' },
      ]} />
      <Sec title="무엇을 판별하나">
        <p>기준일 15시에 <b>내일 한낮(11~15시) 최저 SMP가 30원 미만으로 떨어질지</b>를 판별합니다. 휴일 한낮에 태양광이 많고 수요가 적으면 SMP가 0원 근처로 급락하는데, 가격 예측(일주일 모델)은 이런 날을 잘 맞히지 못하므로 따로 경보합니다. 웹에서는 <b>가격 예측값은 그대로 두고 급락 확률만 붙이는</b> 투트랙 방식입니다 (학생질의 Q1-3).</p>
      </Sec>
      <Sec title="최종 모델: Ridge + EBM + 랜덤포레스트">
        <ol>
          <li>세 모델이 각각 <b>내일 한낮 최저 SMP(원)</b>를 예측하고 평균합니다. Ridge(선형, 안정적), EBM(설명 가능한 부스팅, 변수별 효과를 그래프로 확인 가능), 랜덤포레스트(비선형·상호작용)가 서로 다른 오차를 냅니다.</li>
          <li><b>등급</b>: 예상 한낮 최저 ≤ 90.1원 = <b>주의</b>, ≤ 75.3원 = <b>경보</b>. 문턱은 2025년 급락의 90%를 잡는 값입니다 (회사 의견: 헛경보가 많아도 급락을 놓치지 않는 쪽).</li>
          <li><b>급락 확률</b>: 예상 한낮 최저를 로지스틱식으로 확률로 바꿉니다 (2025로 맞춤). 주의 ≈ {pct(T_WARN, 0)}, 경보 ≈ {pct(T_STRONG, 0)}부터입니다.</li>
          <li>매일 판단일까지의 자료로 다시 학습합니다 (운영).</li>
        </ol>
      </Sec>
      <Sec title="선택 이유" sub="2025 교차검증 (2024 → 2025 상반기, 2024~2025 상반기 → 하반기)">
        <ul>
          <li>단일 모델 31종 + 앙상블 25종(Chronos·TabPFN·LSTM·LightGBM 포함) 중 <b>급락 90%를 잡을 때 헛경보 25일로 1위</b> (이전 Ridge+EBM 31일). 사전 규칙(15% 이상 개선)을 충족해 교체했습니다.</li>
          <li>순위 규칙은 2026 채점 전에 고정: ① 급락 90% 잡을 때 헛경보 수 → ② 100% 잡을 때 헛경보 → ③ PR-AUC.</li>
          <li>2026에서는 이전 모델과 성적이 비슷했습니다 (부트스트랩 차이 95% 구간 −22~+20일, 우연 범위). 사전 규칙대로 유지했습니다.</li>
        </ul>
        <Table rows={dz} sel={(r) => r === dz[0]} cols={[{ k: '구분', t: '구분', l: true }, { k: '모델', t: '모델', l: true }, { k: '헛경보@80%', t: '헛경보 (급락 80% 잡을 때)' },
          { k: '헛경보@90%', t: '헛경보 (90%)' }, { k: '헛경보@100%', t: '헛경보 (100%)' }, { k: 'PR-AUC', t: 'PR-AUC', f: n3 }, { k: 'ROC-AUC', t: 'ROC-AUC', f: n3 }]} />
      </Sec>
      <Sec title="사용 데이터" sub="피처 28개 · 모두 기준일 15시에 확보">
        <DataTable rows={[
          ['달력 (13)', '휴무일·주말만·단일 공휴일·긴 연휴, 연휴 길이·며칠째, 명절, 근로자의날, 징검다리, 휴무 다음 날, 월(계절), 추세', '사전 확정'],
          ['최근 SMP (4)', '직전 같은 유형일 한낮 최저, 어제 한낮 최저, 최근 14일 급락 횟수, 최근 7일 최저', '기준일분까지 (전날 저녁 공표)'],
          ['수요 (2)', '어제 하루전 수요예측, 최근 28일 대비 비율', '기준일분까지 (내일분은 저녁 공표라 미사용)'],
          ['날씨 예보 (6)', '내일 오전·오후 하늘상태, 오후 맑음 여부·강수확률·비 여부·기온', '기준일 11시(없으면 05시) 단기 육상예보'],
          ['주간 전력수요 전망 (2)', '내일 최대부하, 전망 내 비율', '게시 다음 날부터'],
          ['태양광 설비용량 (1)', '시장참여 태양광 설비(GW)', 'M월 값은 M+2월 1일부터']]} />
      </Sec>
      <Sec title="2026 시험 성적표" sub="2026-01-01 ~ 10-18, 결정 후 1회 채점">
        <Table rows={dt.filter((r) => r['경보'] !== '주의만')} sel={(r) => r['구간'] === '2026 전체' && r['경보'] !== '-'} cols={[{ k: '방법', t: '방법', l: true }, { k: '경보', t: '경보', l: true, f: gname }, { k: '구간', t: '구간', l: true },
          { k: '일수', t: '일수' }, { k: '급락일', t: '급락일' }, { k: '적중', t: '적중' }, { k: '놓침', t: '놓침' }, { k: '헛경보', t: '헛경보' },
          { k: '재현율', t: '재현율', f: (v) => pct(v) }, { k: '정밀도', t: '정밀도', f: (v) => pct(v) }, { k: 'ROC-AUC', t: 'ROC-AUC', f: n3 }]} />
        <p>단순 규칙 '휴무일이면 급락'도 재현율은 90%지만 헛경보가 81일로 더 많고 판별력(ROC-AUC 0.80)이 낮습니다.</p>
      </Sec>
      <Sec title="급락 확률은 믿을 만한가">
        <Table rows={(D.perf.dropCal ?? []).filter((r) => r['구분'] !== '전체 Brier')} cols={[{ k: '연도', t: '연도', l: true }, { k: '구분', t: '구분', l: true, f: gname }, { k: '일수', t: '일수' },
          { k: '평균 확률', t: '표시한 평균 확률', f: (v) => pct(v) }, { k: '실제 급락 비율', t: '실제 급락 비율', f: (v) => pct(v) }, { k: '급락일', t: '급락일' }]} />
        <p>2026년에는 높은 확률 구간에서 실제보다 높게 표시되는 경향이 있어, 급락 표시에 등급별 2026 실제 급락 비율을 함께 적습니다.</p>
      </Sec>
      <Sec title="한계">
        <ul>
          <li>헛경보가 많습니다 (주의 이상 정밀도 20%). 급락을 놓치지 않는 쪽으로 문턱을 정한 결과입니다. 경보만 보면 정밀도 37%입니다.</li>
          <li>평일 급락은 2026년 2일뿐이라 평일 성적은 참고 수준입니다.</li>
        </ul>
      </Sec>
    </>
  )
}

function WeekDoc() {
  const W = D.perf.week ?? ({} as any)
  const t = met('일주일', '2026', '전체'), t1 = met('일주일 D+1', '2026', '전체')
  const hs = [1, 2, 3, 4, 5, 6, 7].map((h) => met('일주일', '2026', 'D+' + h))
  const nd = D.docs?.nextday ?? []
  return (
    <>
      <Kpis items={[
        { k: 'MAE (2026)', v: n2(t.MAE) + '원', s: `D+1 ${n2(hs[0].MAE)} → D+7 ${n2(hs[6].MAE)}` },
        { k: 'rMAE', v: n3(t.rMAE), s: `지난주 같은 시간 ${n2(t['단순 기준 MAE'])}원 대비` },
        { k: '80% 범위 포함률', v: pct(t['포함률']), s: '목표 75~85%', tone: tone(t['포함률']) },
        { k: '평균 폭', v: fmt(t['평균 폭']) + '원', s: `D+1 ${fmt(hs[0]['평균 폭'])} → D+7 ${fmt(hs[6]['평균 폭'])}` },
        { k: '내일 시간별 (D+1)', v: n2(t1.MAE) + '원', s: `포함률 ${pct(t1['포함률'])} · rMAE ${n3(t1.rMAE)}` },
      ]} />
      <Sec title={`최종 모델: Chronos-2 small + ${WZC['잔차 보정 λ'] ?? 0.5} × 잔차 보정`}>
        <ol>
          <li><b>Chronos-2 small</b> (Amazon 공개 시계열 기초모델, 우리 자료로 추가 학습 없음): 최근 85일 시간별 SMP와 휴무일 달력을 넣으면 앞으로 168시간의 중앙값·분위수를 냅니다. 매일 최근 흐름에 바로 맞춥니다.</li>
          <li><b>잔차 보정</b> (LightGBM, 며칠 뒤마다 따로): Chronos는 SMP 과거값과 달력만 봅니다. 그래서 <b>Chronos가 틀린 만큼(공표 − Chronos)</b>을 수요전망·날씨예보·연료비·태양광 설비로 예측해 그 절반(λ = 0.5)을 더합니다. 매월(운영은 매주) 재학습.</li>
          <li><b>80% 범위</b>: D+1은 Chronos-2 분위수(10%·90%)를 최근 90일 실제 이탈 폭으로 보정(시간대 묶음 × 휴무일), D+2~D+7은 Chronos-2 분위수 그대로 (2025 하반기 구간점수 최소).</li>
        </ol>
      </Sec>
      <Sec title="선택 이유">
        <ul>
          <li>단일 모델 18종 중 Chronos 계열 3종이 1~3위. 학습형 모델은 2025 하반기 수준 변화에 크게 나빠졌습니다 (LightGBM 9.72 → 10.94, CatBoost 9.81 → 14.23).</li>
          <li>Chronos에 LSTM·GRU·LightGBM을 <b>그대로 섞으면 오히려 나빠짐</b> (9.49~9.59 vs 단독 9.41): 성능 차이가 2원 가까이 나고 오차 상관이 0.79로 높아 서로 보완하지 못합니다.</li>
          <li>대신 <b>Chronos가 모르는 외부 정보로 Chronos의 오차를 고치는</b> 잔차 보정이 2025 하반기 9.21원으로 1위 (−2.2%). 모든 지평·휴무일·저녁에서 개선, 2026 시험에서도 9.81 → {n2(t.MAE)}원.</li>
        </ul>
      </Sec>
      <Sec title="비교한 단일 모델" sub="2025년 MAE (원), 모두 같은 조건">
        <Table rows={W.models ?? []} sel={(r) => r === (W.models ?? [])[0]} cols={[{ k: '모델', t: '모델', l: true }, { k: '2025 상반기', t: '상반기', f: n2 }, { k: '2025 하반기', t: '하반기', f: n2 }, { k: '2025 전체', t: '전체', f: n2 },
          ...[1, 2, 3, 4, 5, 6, 7].map((x) => ({ k: `2025 D+${x}`, t: `D+${x}`, f: n2 }))]} />
        <p className="!text-[12.5px] !text-mute">재학습: LightGBM·XGBoost·Ridge 매월, CatBoost·랜덤포레스트·엑스트라트리·EBM·MLP·TabPFN 분기. LSTM·GRU·1D-CNN·Transformer는 과거 7일 입력 소형(CPU 시간 제약).</p>
      </Sec>
      <Sec title="앙상블·보정 후보" sub="2025 상반기로 맞추고 하반기 MAE로 비교">
        <Table rows={W.ens ?? []} sel={(r) => String(r['앙상블']).startsWith(WZC['선택 앙상블'] ?? '@')} cols={[{ k: '앙상블', t: '후보', l: true }, { k: '구성 수', t: '모델 수' }, { k: '2025 하반기', t: '2025 하반기 MAE', f: n3 }, { k: '구성', t: '구성', l: true, w: true }]} />
      </Sec>
      <Sec title="내일 시간별 가격 (D+1)" sub="같은 날짜·시간끼리 MAE (원)">
        <p>내일 24시간 가격은 이 모델의 D+1입니다. 내일만 따로 만든 모델(LightGBM·CatBoost 앙상블 + 급락 결합)보다 2025·2026 모두 낮은 오차였습니다.</p>
        <Table rows={nd.map((r) => ({ ...r, 모델: String(r['모델']).replace(' (최종)', ' (일주일 모델 D+1)') }))} sel={(r) => String(r['모델']).includes('일주일 모델')}
          cols={[{ k: '모델', t: '모델', l: true }, { k: '2025 상반기', t: '2025 상반기', f: n2 }, { k: '2025 하반기', t: '2025 하반기', f: n2 }, { k: '2025 전체', t: '2025 전체', f: n2 }, { k: '2026 (시험)', t: '2026 (시험)', f: n2 }]} />
      </Sec>
      <Sec title="사용 데이터"><DataTable rows={DATA_HOURLY} /><p className="!text-[13px] font-semibold">쓰지 않은 정보</p><NotUsed /></Sec>
      <Sec title="2026 시험 성적표 (일주일 3차 시험)" sub={testRange}>
        <Table rows={metRows('일주일', '2026', (s) => !s.startsWith('D+'))} csv="육지_일주일_2026_구간별" cols={METCOLS} />
        <p className="!text-[13px] font-semibold">며칠 뒤별</p>
        <Table rows={metRows('일주일', '2026', (s) => s.startsWith('D+'))} csv="육지_일주일_2026_지평별" cols={METCOLS} />
        <p className="!text-[13px] font-semibold">2025 하반기 (선택 구간)</p>
        <Table rows={metRows('일주일', '2025', (s) => !s.startsWith('D+'))} csv="육지_일주일_2025하반기" cols={METCOLS} />
      </Sec>
      <Sec title="한계">
        <ul>
          <li>멀리 볼수록 상한 위 이탈이 늘어납니다 (D+7 약 15%) — 갑작스런 가격 상승을 범위가 늦게 따라갑니다.</li>
          <li>휴무일 한낮·급락 시간은 오차가 큽니다. 급락은 내일 급락 경보을 함께 보세요.</li>
          <li>'3차 시험'은 2026을 앞서 두 번 채점한 뒤의 결과라 1차 시험보다 낙관적일 수 있습니다.</li>
        </ul>
      </Sec>
    </>
  )
}

function MonthDoc() {
  const M = D.perf.month ?? ({} as any)
  const ch = MZC['지표별'] ?? {}
  const piv: Record<string, Row> = {}
  ;(M.models ?? []).forEach((r: Row) => { (piv[r['모델']] ||= { 모델: r['모델'] })[r['지표']] = r['2025 전체'] })
  const prow = Object.values(piv).sort((a, b) => a.mean - b.mean)
  const keys = Object.keys(STAT_NAME) as Stat[]
  const ens = keys.flatMap((s) => (M.ens ?? []).filter((r: Row) => r['지표'] === s).sort((a: Row, b: Row) => a['2025 하반기'] - b['2025 하반기']).slice(0, 4).map((r: Row) => ({ ...r, 지표: STAT_NAME[s], _s: s })))
  const lam = (s: Stat) => (String(ch[s]?.['점예측'] ?? '').match(/\+ ([0-9.]+)/) ?? [])[1] ?? '-'
  return (
    <>
      <Kpis items={keys.map((s) => { const r = met('한 달', '2026', STAT_NAME[s]); return { k: STAT_NAME[s], v: n2(r.MAE) + '원', s: `포함률 ${pct(r['포함률'])} · rMAE ${n3(r.rMAE)}`, tone: tone(r['포함률']) } })
        .concat([{ k: '비교한 후보', v: '15 + 앙상블', s: '지표마다 따로 선택', tone: undefined }])} />
      <Sec title="최종 모델: 기준 + λ × (LSTM·GRU 평균 − 기준)">
        <ul>
          <li><b>기준</b>: 대상일과 같은 유형(평일/휴무일)인 최근 4일의 지표 평균.</li>
          <li><b>LSTM·GRU</b>: 최근 56일 4개 지표 + 앞으로 34일 달력을 넣어 28일 × 4개 지표를 한 번에 예측 (시드 3개씩 평균, 운영은 매주 재학습).</li>
          <li><b>λ</b> (기준을 신경망 쪽으로 당기는 정도): {keys.map((s) => `${STAT_NAME[s]} ${lam(s)}`).join(' · ')}.</li>
          <li><b>80% 범위</b>: 과거 오차의 주차 × 휴무일별 분위수 — {keys.map((s) => `${STAT_NAME[s]} ${ch[s]?.['범위'] ?? '-'}`).join(' · ')}.</li>
        </ul>
      </Sec>
      <Sec title="선택 이유">
        <ul>
          <li>1~4주 앞은 연료비·수급 국면 변화가 가격을 좌우해, <b>기준을 확실히 이기는 단일 모델이 없었습니다</b> (일평균 2025: 기준 9.09 vs Chronos-2 9.12, 엑스트라트리 9.30).</li>
          <li>기준을 LSTM·GRU 쪽으로 조금 당기는 조합이 2025 하반기 네 지표 모두 1위권 (일평균 8.56 vs 기준 8.79). 일주일과 달리 한 달에서는 LSTM 계열이 실제로 보탬이 됐습니다.</li>
          <li>2026 시험: 한낮(+4.7%)·한낮 최저(+8.8%)는 기준보다 좋았지만 일평균(−2.9%)·저녁(−14.9%)은 나빴습니다. 2026을 보고 다시 고치지 않고 <b>참고 전망</b>으로 표시합니다.</li>
        </ul>
      </Sec>
      <Sec title="비교한 단일 모델" sub="2025년 MAE (원), 일평균 순">
        <Table rows={prow} sel={(r) => r === prow[0]} cols={[{ k: '모델', t: '모델', l: true }, ...keys.map((k) => ({ k, t: STAT_NAME[k], f: n2 }))]} />
      </Sec>
      <Sec title="앙상블 후보 상위" sub="지표별 2025 하반기 MAE · 1위와 0.05원 이내면 구성 모델이 적은 쪽">
        <Table rows={ens} sel={(r) => r['앙상블'] === ch[r._s]?.['점예측']} cols={[{ k: '지표', t: '지표', l: true }, { k: '앙상블', t: '후보', l: true, w: true }, { k: '구성 수', t: '모델 수' }, { k: '2025 하반기', t: '2025 하반기 MAE', f: n3 }]} />
      </Sec>
      <Sec title="사용 데이터">
        <DataTable rows={[['일별 SMP 지표 (일평균·한낮·저녁·한낮 최저)', '최근 7·28일 평균, 같은 유형일 4일 평균, 최근 56일 시계열(LSTM·GRU)', '기준일까지 공표분'],
          ['달력', '대상일 요일·월·휴무·연휴·명절, 앞뒤 휴무', '사전 확정'], ['작년 계절 비율', '작년 대상일 ±3일 평균 ÷ 작년 기준일 직전 28일 (SMP·수요예측)', '작년 실적이라 확정'],
          ['하루전 수요예측 (실적)', '최근 7일 일평균', '기준일분까지 (전날 저녁 공표)'], ['연료비 단가 (LNG·유연탄)·태양광 설비', '월 값과 3개월 전 대비 변화', 'M월 값은 M+2월 1일부터']]} />
      </Sec>
      <Sec title="2026 시험 성적표 (한 달 3차 시험)" sub={testRange}>
        <Table rows={metRows('한 달', '2026', (s) => !s.includes('주차'))} csv="육지_한달_2026_지표별" cols={METCOLS} />
        <p className="!text-[13px] font-semibold">주차별</p>
        <Table rows={metRows('한 달', '2026', (s) => s.includes('주차'))} csv="육지_한달_2026_주차별" cols={METCOLS} />
        <p className="!text-[13px] font-semibold">2025 하반기 (선택 구간)</p>
        <Table rows={metRows('한 달', '2025', (s) => !s.includes('주차'))} csv="육지_한달_2025하반기" cols={METCOLS} />
      </Sec>
      <Sec title="한계">
        <ul>
          <li>한낮·한낮 최저의 포함률이 70% 안팎으로 목표(75%)에 못 미칩니다. 멀수록(4주차) 더 낮습니다.</li>
          <li>개선에는 연료비 전망·수급 국면 정보가 필요하며, 공개 자료만으로는 한계가 있습니다.</li>
        </ul>
      </Sec>
    </>
  )
}

const PAGES: Record<DocPage, { t: string; sub: string; el: () => ReactNode }> = {
  ov: { t: '개요', sub: '최종 모델 · 팀 비교용 지표 · 평가 규칙', el: Overview },
  drop: { t: '내일 급락 경보', sub: '과제1 모델 · 내일 한낮 최저 SMP 30원 미만 판별', el: DropDoc },
  wk: { t: '일주일 예측', sub: '기준일 15시 → 1~7일 뒤 시간별 SMP + 80% 범위', el: WeekDoc },
  mo: { t: '한 달 예측', sub: '기준일 15시 → 7~34일 뒤 일별 4개 지표 + 80% 범위 · 참고 전망', el: MonthDoc },
}

// 제주 참고 문서: 육지 참고 문서와 같은 꼴(핵심 지표 → 최종 모델 → 선택 이유 → 비교 후보 → 사용 데이터 → 2026 성적표 → 한계)
// 내용은 제주 채팅 결과 (docs/JEJU.md 0·3~6절), 성적은 JF1·J3·JMZ2 표, 모델 설정은 실제 운영 설정 파일 JWZ·JMZ·JD_choice.json
const JD = () => (D as any).jdocs ?? {}
const jmet = (pred: string, per: string, seg: string): Row => (JD().metrics ?? []).find((r: Row) => r['예측'] === pred && String(r['기간']).startsWith(per) && r['구분'] === seg) ?? {}
const jmetRows = (pred: string, per: string, keep: (s: string) => boolean): Row[] => (JD().metrics ?? []).filter((r: Row) => r['예측'] === pred && String(r['기간']).startsWith(per) && keep(r['구분']))
const JTEST = '2026-01-01 ~ 10-23'
// 제주 한 달 2026 표(JMZ2)를 표준 성적표 열 이름으로
const jmon = (k: Stat): Row => {
  const r: Row = (JD().monthTest ?? []).find((x: Row) => x['지표'] === k) ?? {}
  return { ...r, 구분: STAT_NAME[k], 표본수: r['일수'], '단순 기준 MAE': r['기준 MAE'], '하한 아래': r['하한아래'], '상한 위': r['상한위'], '평균 폭': r['평균폭'] }
}
// 급락 2026 표(J3)에 재현율·정밀도·헛경보율을 붙임
const jdropRows = (): Row[] => {
  const t: Row[] = JD().dropTest ?? [], all = t.find((r) => r['구분'] === '전체') ?? {}
  const n = all['일수'], k = all['급락일']
  return t.filter((r) => r['구분'] !== '전체').map((r) => ({
    ...r, 경보: r['구분'], 일수: n, 급락일: k, 'ROC-AUC': all['ROC-AUC'], 'PR-AUC': all['PR-AUC'],
    재현율: r['적중'] / k, 정밀도: r['적중'] / (r['적중'] + r['헛경보']), 헛경보율: r['헛경보'] / (n - k),
  }))
}
const DROPCOLS: Col[] = [{ k: '경보', t: '경보 기준', l: true, f: gname }, { k: '일수', t: '일수' }, { k: '급락일', t: '급락일' }, { k: '적중', t: '적중' }, { k: '놓침', t: '놓침' }, { k: '헛경보', t: '헛경보' },
  { k: '재현율', t: '재현율', f: (v) => pct(v) }, { k: '정밀도', t: '정밀도', f: (v) => pct(v) }, { k: '헛경보율', t: '헛경보율', f: (v) => pct(v) }, { k: 'ROC-AUC', t: 'ROC-AUC', f: n3 }, { k: 'PR-AUC', t: 'PR-AUC', f: n3 }]

const J_NOT_USED = (
  <ul>
    <li><b>대상일 하루전 수요예측·SMP</b>: 저녁에 공표돼 15시 판단에 쓸 수 없습니다. 기준일까지의 실적만 씁니다.</li>
    <li><b>동네예보 제주(184)</b>: 기준일 17시 발표분이라 못 씁니다. 구역예보(11시 발표)를 씁니다. 구역예보에는 풍속이 없어 풍속은 Open-Meteo에서만 가져옵니다.</li>
    <li><b>제주 15분 SMP·수요예측(시범사업 API)</b>: 공개가 약 2일 늦고 기존 자료와 같은 값입니다.</li>
    <li><b>제주 출력제어·지역별 발전 실적</b>: 분기·연 단위로만 공개돼 해석용으로만 봤습니다.</li>
    <li><b>실측 날씨</b>: 판단 시각에 알 수 없어 쓰지 않았습니다.</li>
  </ul>
)

function JOverview() {
  const w = jmet('일주일', '2026', '전체'), w1 = jmet('일주일 D+1', '2026', '전체'), m = jmon('mean')
  const dr = jdropRows(), dW = dr.find((r) => r['경보'] === '주의 이상') ?? {}, dS = dr.find((r) => r['경보'] === '강한 경보') ?? {}
  const lam = String(Object.values(JD().weekChoice?.['잔차 보정 λ(2025 전체)'] ?? {})[0] ?? 0.25)
  const final = [
    { a: '내일 급락 경보', m: `과제1 제주 모델: 피처 ${JD().dropFeatures ?? 63}개 + KNN 분류 (과거 비슷한 날 25일의 급락 비율) → 주의·경보, 급락 확률`,
      s: `급락 ${dW['급락일'] ?? '-'}일 중 주의 이상 ${dW['적중'] ?? '-'}일 적중 (헛경보 ${dW['헛경보'] ?? '-'}) · 경보 ${dS['적중'] ?? '-'}일 (헛경보 ${dS['헛경보'] ?? '-'}) · ROC-AUC ${n2(dW['ROC-AUC'])}` },
    { a: '일주일 예측 (D+1~D+7, 시간별)', m: `육지 일주일 최종 예측 + ${lam} × 제주 잔차 보정 (LightGBM)`,
      s: `MAE ${n2(w.MAE)}원 · 포함률 ${pct(w['포함률'])} · 폭 ${fmt(w['평균 폭'])}원 (내일 시간별 MAE ${n2(w1.MAE)}원)` },
    { a: '한 달 예측 (7~34일 뒤, 일별)', m: '육지 한 달 최종 예측 + 지표별 최근 제주−육지 차이', s: `일평균 MAE ${n2(m.MAE)}원 · 포함률 ${pct(m['포함률'])} (참고 전망)` },
  ]
  const head = [{ ...w, 예측: '일주일 (전체)' }, { ...w1, 예측: '일주일 · 내일 (D+1)' }, ...(Object.keys(STAT_NAME) as Stat[]).map((k) => ({ ...jmon(k), 예측: '한 달 · ' + STAT_NAME[k] }))]
  return (
    <>
      <Sec title="최종 모델" sub="2026-10-25 확정">
        <p>모든 모델은 <b>기준일 15시에 알 수 있는 정보만</b> 쓰고, <b>2025년 성적만으로</b> 골랐습니다. 2026년({JTEST})은 고른 뒤 채점만 했습니다.</p>
        <p>제주 SMP는 육지와 같은 가격인 시간이 2022년 31% → 2026년 91%(3분기 99%)라 <b>가격 예측은 육지 예측에서 출발</b>합니다. 대신 한낮 급락이 육지보다 3배 잦고(약 14%) 음수 가격이 있어, 급락 경보은 제주 전용 모델을 쓰고 범위 하한은 0원에서 자르지 않습니다.</p>
        <Table rows={final} cols={[{ k: 'a', t: '예측', l: true }, { k: 'm', t: '최종 모델', l: true, w: true }, { k: 's', t: '2026 시험 성적', l: true, w: true }]} />
      </Sec>
      <Sec title="팀 비교용 핵심 지표" sub="2026 시험 · 같은 정의로 다시 계산하면 비교할 수 있습니다">
        <p className="!text-[13px] font-semibold">가격 예측 (일주일·한 달)</p>
        <Table rows={head} csv="제주_최종성적_2026" cols={[{ k: '예측', t: '예측', l: true }, ...METCOLS.slice(1)]} />
        <p className="!text-[13px] font-semibold">내일 급락 경보</p>
        <Table rows={dr} csv="제주_급락_2026" cols={DROPCOLS} />
      </Sec>
      <Glossary base="일주일 = 지난주 같은 시간, 내일(D+1) = 전날 같은 시간, 한 달 = 제주 최근 같은 유형일 4일 평균" />
      <Sec title="평가 규칙">
        <ul>
          <li><b>판단 시각 = 기준일 15시</b>. 대상일 하루전 수요예측·SMP는 저녁 공표라 쓰지 않습니다.</li>
          <li><b>시간 순서만</b>: 잔차 보정은 매월 1일 재학습하며 그 전까지의 자료로만 학습 (랜덤 분할 없음).</li>
          <li><b>선택</b>: 일주일·한 달은 2025 상반기로 맞추고 2025 하반기로 비교 (0.05원 이내면 단순한 쪽). 급락은 2025 교차검증(급락 90%를 잡을 때 헛경보 수)에 안정성 기준을 더함.</li>
          <li><b>시험</b>: 2026-01-01 ~ 10-23에 한 번 채점 (제주 1차 시험). 이후 2026 성적으로 모델을 바꾸지 않습니다.</li>
        </ul>
      </Sec>
      <Sec title="쓰지 않은 정보">{J_NOT_USED}</Sec>
    </>
  )
}

function JDrop() {
  const dr = jdropRows(), all = dr.find((r) => r['경보'] === '주의 이상') ?? {}, st = dr.find((r) => r['경보'] === '강한 경보') ?? {}
  const dc = JD().dropChoice ?? {}, o = dc['2025 표본 밖'] ?? {}, dz: Row[] = JD().dropZoo ?? []
  return (
    <>
      <Kpis items={[
        { k: '재현율 (주의 이상)', v: pct(all['재현율'], 0), s: `급락 ${all['급락일'] ?? '-'}일 중 ${all['적중'] ?? '-'}일 적중` },
        { k: '헛경보 (주의 이상)', v: `${all['헛경보'] ?? '-'}일`, s: `정밀도 ${pct(all['정밀도'], 0)}` },
        { k: '재현율 (경보)', v: pct(st['재현율'], 0), s: `${st['적중'] ?? '-'}일 적중 · 헛경보 ${st['헛경보'] ?? '-'}일` },
        { k: 'ROC-AUC', v: n3(all['ROC-AUC']), s: `PR-AUC ${n3(all['PR-AUC'])}` },
        { k: '시험 기간', v: `${all['일수'] ?? '-'}일`, s: `${JTEST} (제주 1회 채점)` },
      ]} />
      <Sec title="무엇을 판별하나">
        <p>기준일 15시에 <b>내일 한낮(11~15시) 제주 최저 SMP가 30원 미만으로 떨어질지</b>를 판별합니다. 제주는 급락이 잦고(2024년 20일, 2025년 60일, 2026년 54일), 육지가 급락한 날은 제주도 모두 급락했지만 <b>제주만 급락한 날</b>(평일·낮은 수요·맑음·강풍)이 훨씬 많습니다. 육지 예측의 한낮 최저만으로는 이런 날을 거의 못 가려(헛경보 265일) 제주 전용 모델을 만들었습니다. 웹에서는 가격 예측값은 그대로 두고 경보만 붙입니다.</p>
      </Sec>
      <Sec title="최종 모델: KNN 분류">
        <ol>
          <li>내일의 조건(피처 {JD().dropFeatures ?? 63}개)을 표준화한 뒤, 학습 기간(2024-01 ~ 2025-12, 731일)에서 <b>가장 비슷한 25일</b>을 찾아 거리 가중 급락 비율을 점수로 냅니다. "비슷했던 과거 날 중 몇 일이 급락했나"로 설명할 수 있습니다.</li>
          <li><b>등급</b>: 점수 {dc['주의 문턱'] ?? '-'} 이상 = <b>주의</b> (2025 급락 90%를 잡는 값), {dc['강한 경보 문턱'] ?? '-'} 이상 = <b>경보</b> (2025 정밀도 약 50%인 가장 느슨한 값).</li>
          <li><b>급락 확률</b>: 점수를 로지스틱식으로 확률로 바꿉니다 (2025로 맞춤). 주의 ≈ {pct(T_WARN, 0)}, 경보 ≈ {pct(T_STRONG, 0)}부터입니다.</li>
          <li>2026 시험과 같은 <b>고정 학습</b> 모델을 운영에 씁니다 (매월 재학습은 2025에서 더 나빴음).</li>
        </ol>
      </Sec>
      <Sec title="선택 이유" sub="2025 교차검증 (2024 → 2025 상반기, 2024~2025 상반기 → 하반기)">
        <ul>
          <li>피처 묶음 3종 × (단일 모델 41개 + 사전 고정 앙상블 23개) = <b>195개</b>를 비교했습니다 (부스팅·TabPFN·LSTM·Chronos 포함). 순위: ① 급락 90%를 잡을 때 헛경보 수 → ② 100% → ③ PR-AUC.</li>
          <li>1위 L1 로지스틱은 '2024만 학습 → 2025'(문턱을 정하는 방식)에서 헛경보 129일로 크게 흔들렸습니다. 그래서 상위 20개를 이 방식으로도 채점해 <b>두 검증 중 나쁜 쪽</b>으로 다시 순위를 매겼고, <b>KNN 분류가 1위</b>(72일)였습니다. 이 기준은 결과를 본 뒤 추가했습니다.</li>
          <li>부스팅·TabPFN·신경망·앙상블은 상위권 밖 — 급락일이 80일뿐이라 복잡한 모델이 과적합했습니다. 2025 표본 밖: 헛경보@90% {o['헛경보@90%'] ?? '-'}일, ROC-AUC {n3(o['ROC-AUC'])}.</li>
        </ul>
        <Table rows={dz} sel={(r) => r === dz[0]} cols={[{ k: '피처', t: '피처 묶음', l: true }, { k: '구분', t: '구분', l: true }, { k: '모델', t: '모델', l: true },
          { k: '헛경보@90%', t: '헛경보 (급락 90%, 교차검증)' }, { k: '2024 학습→2025 헛경보@90%', t: '헛경보 (2024 학습→2025)' }, { k: '나쁜 쪽', t: '나쁜 쪽 (순위 기준)' },
          { k: '헛경보@100%', t: '헛경보 (100%)' }, { k: 'PR-AUC', t: 'PR-AUC', f: n3 }, { k: 'ROC-AUC', t: 'ROC-AUC', f: n3 }]} />
      </Sec>
      <Sec title="사용 데이터" sub={`피처 ${JD().dropFeatures ?? 63}개 · 모두 기준일 15시에 확보`}>
        <DataTable rows={[
          ['달력 (12)', '휴무일·요일 유형·연휴 길이와 순서·명절·전날·다음날 휴무·계절', '사전 확정'],
          ['제주 SMP 지난 값 (11)', '전날 한낮 최저·같은 유형일 한낮 최저·최근 급락 수·육지와 같은 가격 비율', '기준일분까지 (전날 저녁 공표)'],
          ['제주 수요예측 (3)', '전날 한낮·평균, 28일 중앙값 대비', '기준일분까지 (내일분은 저녁 공표라 미사용)'],
          ['기상청 구역예보 제주·서귀포 (6)', '오전·오후 하늘, 강수확률·강수, 기온, 북풍 성분', '기준일 11시 발표분'],
          ['Open-Meteo 제주 4지점 (6)', '한낮 일사·운량, 10m·100m 풍속', '대상일 2일 전 실행분'],
          ['육지 정보 (11)', '육지 전날 한낮 최저, 날씨 예보, 주간 수요 전망, 연료비, 태양광 설비, 육지 예측의 내일 한낮 최저', '기준일 15시까지'],
          ['제주 예측 (2)', '제주 Chronos-2 small의 내일 한낮 최저·하위 10%', '기준일 15시 계산'],
          ['제주 5분 계통 실적 (12)', '기준일 오전 풍력·한낮 태양광·수요·순부하·신재생 비중, 최근 7일 값', '기준일 14:25까지']]} />
      </Sec>
      <Sec title="2026 시험 성적표" sub={`${JTEST}, 결정 후 1회 채점`}>
        <Table rows={dr} csv="제주_급락_2026" cols={DROPCOLS} />
        <p>휴무일 급락 30일 중 25일을 잡았지만 <b>평일 급락은 24일 중 14일</b>만 잡았습니다.</p>
      </Sec>
      <Sec title="급락 확률은 믿을 만한가">
        <Table rows={JD().dropCal ?? []} cols={[{ k: '연도', t: '연도', l: true }, { k: '구분', t: '구분', l: true, f: gname }, { k: '일수', t: '일수' },
          { k: '평균 확률', t: '표시한 평균 확률', f: (v) => pct(v) }, { k: '실제 급락 비율', t: '실제 급락 비율', f: (v) => pct(v) }, { k: '급락일', t: '급락일' }]} />
        <p>2026년에는 '낮음'·'주의' 날의 실제 급락이 표시 확률보다 높았습니다(낮음 3% → 6.7%). 급락 표시에 등급별 2026 실제 급락 비율을 함께 적습니다.</p>
      </Sec>
      <Sec title="한계">
        <ul>
          <li>목표 재현율 90%에 못 미칩니다(72%). 평일에 생기는 '제주만 급락'이 약점입니다.</li>
          <li>결정 뒤 참고로 비교하면 1차 후보(랜덤포레스트)는 2026년에 49일을 잡고 헛경보 76일이었습니다. <b>같은 적중 수로 맞추면 1차 후보의 헛경보가 같거나 적어</b>, 2025의 개선이 2026에 재현되지 않았습니다. 급락일이 적어 모델 간 우열이 해마다 뒤집히므로 "헛경보가 줄었다"고 말하지 않습니다.</li>
          <li>개선 후보(새 시험 구간이 생긴 뒤에만 비교 가능): 제주 출력제어 실시간 정보, 제주 태양광·풍력 발전 예보, HVDC 연계선 운전 정보.</li>
        </ul>
      </Sec>
    </>
  )
}

const J_DATA_HOURLY = [
  ['육지 일주일 최종 예측', '제주 예측의 출발점 (Chronos-2 small + 잔차 보정)', '같은 기준일 15시 산출'],
  ['시간별 SMP (제주)', '잔차 피처: 어제·같은 유형일·28일 평균 대비 차이, 제주−육지 같은 가격 비율·차이·최근 급락 수', '기준일 SMP는 전날 저녁 공표'],
  ['달력 (휴무일·요일유형·명절, 앞뒤 휴무)', '잔차 보정 피처', '사전 확정'],
  ['제주 하루전 수요예측 (실적)', '지난 값', '기준일분까지 (내일분은 저녁 공표라 미사용)'],
  ['기상청 구역예보 제주·서귀포 (하늘·강수·기온·풍향)', 'D+1~D+3', '기준일 11시 발표분'],
  ['Open-Meteo 제주 4지점 (일사·운량·10m/100m 풍속)', 'D+1~D+6', 'D+1 = 대상일 2일 전 실행분, D+2~ = (지평+1)일 전 실행분'],
  ['제주 Chronos-2 small 분위수', '80% 범위 (D+1·2·4~7)', '기준일 15시 계산'],
]

function JWeek() {
  const wc = JD().weekChoice ?? {}
  const lam = String(Object.values(wc['잔차 보정 λ(2025 전체)'] ?? {})[0] ?? 0.25)
  const t = jmet('일주일', '2026', '전체'), t1 = jmet('일주일 D+1', '2026', '전체')
  const hs = [1, 2, 3, 4, 5, 6, 7].map((h) => jmet('일주일', '2026', 'D+' + h))
  const rng = Object.entries(wc['범위'] ?? {}).map(([h, m]) => `D+${h} ${m}`)
  const wm: Row[] = JD().weekModels ?? [], we: Row[] = JD().weekEns ?? []
  const d1 = ['2025', '2026'].flatMap((p) => jmetRows('일주일 D+1', p, () => true).map((r) => ({ ...r, 구분: `${p === '2025' ? '2025 하반기' : '2026 시험'} · ${r['구분']}` })))
  return (
    <>
      <Kpis items={[
        { k: 'MAE (2026)', v: n2(t.MAE) + '원', s: `D+1 ${n2(hs[0].MAE)} → D+7 ${n2(hs[6].MAE)}` },
        { k: 'rMAE', v: n3(t.rMAE), s: `지난주 같은 시간 ${n2(t['단순 기준 MAE'])}원 대비` },
        { k: '80% 범위 포함률', v: pct(t['포함률']), s: '목표 75~85%', tone: tone(t['포함률']) },
        { k: '평균 폭', v: fmt(t['평균 폭']) + '원', s: `D+1 ${fmt(hs[0]['평균 폭'])} → D+7 ${fmt(hs[6]['평균 폭'])}` },
        { k: '내일 시간별 (D+1)', v: n2(t1.MAE) + '원', s: `포함률 ${pct(t1['포함률'])} · rMAE ${n3(t1.rMAE)}` },
      ]} />
      <Sec title={`최종 모델: 육지 일주일 예측 + ${lam} × 제주 잔차 보정`}>
        <ol>
          <li><b>육지 일주일 최종 예측</b> (Chronos-2 small + 0.5 × 잔차 보정): 육지 웹의 일주일 예측을 그대로 출발점으로 씁니다.</li>
          <li><b>제주 잔차 보정</b> (LightGBM 절대오차, 며칠 뒤마다 따로): <b>육지 예측이 제주에서 틀린 만큼(제주 공표 − 육지 예측)</b>을 제주 수요·구역예보·풍속·일사, 제주−육지 관계로 예측해 그 {lam}배를 더합니다. 매월 재학습, 학습 시작 2023-01.</li>
          <li><b>80% 범위</b> (2025 하반기 구간점수 최소, 하한 0원 자르기 없음): {rng.join(' · ')}.</li>
        </ol>
      </Sec>
      <Sec title="선택 이유">
        <ul>
          <li>제주 가격은 대부분 육지와 같아 <b>'육지 예측 그대로'가 이미 상위권</b>(2025 하반기 9.74원)이었습니다. 제주 SMP로 직접 돌린 Chronos(9.75~10.25), 시계열 신경망(10.29~10.63), 표 모델(11.30~22.26)은 이보다 나빴습니다.</li>
          <li>육지 예측에 제주 잔차 보정을 더한 24종 중 <b>제주외부 피처·절대오차·보정 0.25배</b>가 9.68원으로 1위. 앙상블 최고(9.645, 13개 혼합)와 0.05원 이내라 단순한 단일 후보를 골랐습니다.</li>
          <li>보정을 4분의 1만 반영할 때 가장 좋았다는 것은 제주 고유 정보의 설명력이 작다는 뜻입니다. 2026에서도 '육지 예측 그대로'(11.16원)보다 0.08원(0.7%) 나았습니다.</li>
        </ul>
      </Sec>
      <Sec title="비교한 후보" sub="2025년 MAE (원), 상위 20개 · 모두 같은 조건">
        <Table rows={wm} sel={(r) => r === wm[0]} cols={[{ k: '모델', t: '모델', l: true }, { k: '2025 상반기', t: '상반기', f: n2 }, { k: '2025 하반기', t: '하반기', f: n2 }, { k: '2025 전체', t: '전체', f: n2 },
          ...[1, 2, 3, 4, 5, 6, 7].map((x) => ({ k: `하반기 D+${x}`, t: `D+${x}`, f: n2 }))]} />
        <p className="!text-[12.5px] !text-mute">D+1~D+7 열은 2025 하반기 MAE. TabPFN은 CPU에서 지평 하나 83분이라 제외했습니다.</p>
      </Sec>
      <Sec title="앙상블·보정 후보" sub="2025 상반기로 맞추고 하반기 MAE로 비교">
        <Table rows={we} sel={(r) => String(r['앙상블']).startsWith('단일 최고')} cols={[{ k: '앙상블', t: '후보', l: true }, { k: '구성 수', t: '모델 수' }, { k: '2025 하반기', t: '2025 하반기 MAE', f: n3 }, { k: '구성', t: '구성', l: true, w: true }]} />
      </Sec>
      <Sec title="내일 시간별 가격 (D+1)" sub="일주일 모델의 D+1 · 표준 지표">
        <p>내일 24시간 제주 가격은 이 모델의 D+1입니다.</p>
        <Table rows={d1} csv="제주_내일시간별" cols={METCOLS} />
      </Sec>
      <Sec title="사용 데이터"><DataTable rows={J_DATA_HOURLY} /><p className="!text-[13px] font-semibold">쓰지 않은 정보</p>{J_NOT_USED}</Sec>
      <Sec title="2026 시험 성적표 (제주 1차 시험)" sub={JTEST}>
        <Table rows={jmetRows('일주일', '2026', (s) => !s.startsWith('D+'))} csv="제주_일주일_2026_구간별" cols={METCOLS} />
        <p className="!text-[13px] font-semibold">며칠 뒤별</p>
        <Table rows={jmetRows('일주일', '2026', (s) => s.startsWith('D+'))} csv="제주_일주일_2026_지평별" cols={METCOLS} />
        <p className="!text-[13px] font-semibold">2025 하반기 (선택 구간)</p>
        <Table rows={jmetRows('일주일', '2025', (s) => !s.startsWith('D+'))} csv="제주_일주일_2025하반기" cols={METCOLS} />
      </Sec>
      <Sec title="한계">
        <ul>
          <li>제주 성적은 사실상 육지 예측 품질이 결정합니다 (제주 보정 효과 0.7%).</li>
          <li>MAE가 2025보다 큰 것은 2026 가격 변동이 커졌기 때문입니다 (단순 기준도 12.19 → 13.95원). rMAE는 0.79로 그대로입니다.</li>
          <li>휴무일 한낮은 하한 아래 이탈이 약 19%, 급락 시간은 범위가 거의 담지 못합니다 — 내일 급락 경보을 함께 보세요.</li>
        </ul>
      </Sec>
    </>
  )
}

function JMonth() {
  const mc = JD().monthChoice?.['지표별'] ?? {}
  const keys = Object.keys(STAT_NAME) as Stat[]
  const cmp: Row[] = (JD().monthCompare ?? []).filter((r: Row) => r['구분'] === '점예측').map((r: Row) => ({ ...r, 지표: STAT_NAME[r['지표'] as Stat] ?? r['지표'], _s: r['지표'] }))
  const rng: Row[] = (JD().monthCompare ?? []).filter((r: Row) => r['구분'] !== '점예측').map((r: Row) => ({ ...r, 지표: STAT_NAME[r['지표'] as Stat] ?? r['지표'], _s: r['지표'] }))
  return (
    <>
      <Kpis items={keys.map((s) => { const r = jmon(s); return { k: STAT_NAME[s], v: n2(r.MAE) + '원', s: `포함률 ${pct(r['포함률'])} · rMAE ${n3(r.rMAE)}`, tone: tone(r['포함률']) } })
        .concat([{ k: '비교한 후보', v: `${cmp.length / 4 || 6}종 × 4지표`, s: '지표마다 따로 선택', tone: undefined }])} />
      <Sec title="최종 모델: 육지 한 달 예측 + 최근 제주−육지 차이">
        <ul>
          <li><b>육지 한 달 최종 예측</b>: 최근 같은 유형일 4일 평균 + λ × (LSTM·GRU 평균 − 기준) — 육지 웹의 한 달 예측 그대로.</li>
          <li><b>제주−육지 차이</b>: 기준일 이전 실적으로만 계산해 지표별로 더합니다 — {keys.map((s) => `${STAT_NAME[s]} ${mc[s]?.['점예측'] ?? '-'}`).join(' · ')}.</li>
          <li><b>80% 범위</b>: 육지 범위를 제주 예측값으로 옮기고 폭을 배수로 넓힘 — {keys.map((s) => `${STAT_NAME[s]} ${mc[s]?.['범위'] ?? '-'}`).join(' · ')}. 하한 0원 자르기 없음.</li>
        </ul>
      </Sec>
      <Sec title="선택 이유">
        <ul>
          <li>후보: 육지 그대로 / + 최근 7·28·56일 차이 / + 0.5 × 28일 차이 / + 같은 유형일 8일 차이, 범위 폭 ×1~2. 2025 하반기 MAE로 고르고(0.05원 이내면 단순한 쪽), 범위는 포함률 75~85% 중 구간점수가 가장 작은 것.</li>
          <li>육지 한 달 기록이 기준일 2025-05-28부터라 상반기 맞춤 단계는 없습니다 (후보에 학습할 모수가 없음).</li>
          <li>2026 시험: 한낮·한낮 최저는 기준보다 5~12% 좋았지만 저녁은 18% 나빴습니다. 2026을 보고 다시 고치지 않고 <b>참고 전망</b>으로 표시합니다.</li>
        </ul>
      </Sec>
      <Sec title="비교한 후보 (점예측)" sub="2025 하반기 MAE (원) · 지표별">
        <Table rows={cmp} sel={(r) => r['후보'] === mc[r._s]?.['점예측']} cols={[{ k: '지표', t: '지표', l: true }, { k: '후보', t: '후보', l: true, w: true }, { k: 'MAE', t: 'MAE', f: n2 }, { k: 'rMAE', t: 'rMAE', f: n3 }]} />
      </Sec>
      <Sec title="범위 후보" sub="2025 하반기 · 포함률 75~85% 중 구간점수 최소">
        <Table rows={rng} sel={(r) => r['후보'] === mc[r._s]?.['범위']} cols={[{ k: '지표', t: '지표', l: true }, { k: '후보', t: '후보', l: true, w: true }, { k: '포함률', t: '포함률', f: (v) => pct(v) },
          { k: '하한아래', t: '하한 아래', f: (v) => pct(v) }, { k: '상한위', t: '상한 위', f: (v) => pct(v) }, { k: '평균폭', t: '평균 폭', f: (v) => fmt(v) }, { k: '구간점수', t: '구간점수', f: (v) => fmt(v) }]} />
      </Sec>
      <Sec title="사용 데이터">
        <DataTable rows={[['육지 한 달 최종 예측 (일평균·한낮·저녁·한낮 최저)', '제주 예측의 출발점, 범위의 기준', '같은 기준일 15시 산출'],
          ['제주·육지 일별 SMP 지표', '최근 7·28·56일, 같은 유형일 8일의 제주−육지 차이', '기준일까지 공표분'], ['달력', '대상일 휴무 여부(같은 유형일 계산)', '사전 확정']]} />
      </Sec>
      <Sec title="2026 시험 성적표 (제주 1차 시험)" sub={`${JTEST} · 기준 = 제주 최근 같은 유형일 4일 평균`}>
        <Table rows={keys.map(jmon)} csv="제주_한달_2026_지표별" cols={[...METCOLS.filter((c) => !['RMSE', 'MAPE(20원 이상)', '편향(공표−예측)'].includes(c.k)),
          { k: '육지 그대로 MAE', t: '육지 그대로 MAE', f: n2 }]} />
        <p>주차별 일평균 MAE: 1주 뒤 약 9.7원 → 4주 뒤 약 15.1원 (포함률 80% → 76%).</p>
      </Sec>
      <Sec title="한계">
        <ul>
          <li>저녁(18~22시)은 기준보다 18% 나쁩니다 — 육지 한 달 모델의 2026 저녁 약점이 그대로 옮겨왔습니다.</li>
          <li>2025에 고른 '최근 차이 더하기'가 2026 한낮·한낮 최저에서는 '육지 그대로'보다 1~1.7원 손해였습니다.</li>
          <li>한낮 최저 범위는 평균 폭이 약 146원이라 실용성이 낮습니다 (급락 때문에 아래가 크게 열림).</li>
        </ul>
      </Sec>
    </>
  )
}

const JPAGES: Record<DocPage, { t: string; sub: string; el: () => ReactNode }> = {
  ov: { t: '개요', sub: '최종 모델 · 팀 비교용 지표 · 평가 규칙', el: JOverview },
  drop: { t: '내일 급락 경보', sub: '과제1 제주 모델 · 내일 한낮 최저 SMP 30원 미만 판별', el: JDrop },
  wk: { t: '일주일 예측', sub: '기준일 15시 → 1~7일 뒤 시간별 SMP + 80% 범위', el: JWeek },
  mo: { t: '한 달 예측', sub: '기준일 15시 → 7~34일 뒤 일별 4개 지표 + 80% 범위 · 참고 전망', el: JMonth },
}

export function Docs({ page, onPage }: { page: DocPage; onPage: (p: DocPage) => void }) {
  const PG = REGION === '제주' ? JPAGES : PAGES
  const P = PG[page]
  return (
    <div className="space-y-5">
      <PageHeader title={`참고 문서 · ${REGION} · ${P.t}`} sub={P.sub}
        right={<Seg value={page} onChange={onPage} options={(Object.keys(PG) as DocPage[]).map((v) => ({ v, t: PG[v].t }))} />} />
      <P.el />
    </div>
  )
}
