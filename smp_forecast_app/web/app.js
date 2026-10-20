/* SMP 예측 범위 웹앱 — 데이터는 data/forecast_data.js (window.SMP_DATA) */
(function () {
  const D = window.SMP_DATA;
  if (!D) { document.body.innerHTML = '<p style="padding:20px">데이터 파일(data/forecast_data.js)이 없습니다. python -m smp_forecast_app.src.export_web 을 먼저 실행하세요.</p>'; return; }
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const fmt = (v, d = 1) => (v == null || isNaN(v) ? '-' : Number(v).toFixed(d));
  const pct = (v) => (v == null ? '-' : (v * 100).toFixed(1) + '%');
  const SRC_NAME = { B: '자체 모델 (전날 15시)', A: '담당자 모델 (참고)' };
  const WD = ['일', '월', '화', '수', '목', '금', '토'];
  const state = { src: 'B', perf: 'B', date: null };

  const allDates = Object.keys(D.days).sort();
  const datesFor = (src) => allDates.filter((d) => D.days[d][src]);

  // ---------- 탭 ----------
  $$('.tab').forEach((b) => b.addEventListener('click', () => {
    $$('.tab').forEach((x) => x.classList.toggle('on', x === b));
    $$('.panel').forEach((p) => p.classList.toggle('on', p.id === 'tab-' + b.dataset.tab));
    if (b.dataset.tab === 'perf') renderPerf();
    if (b.dataset.tab === 'week') setWDate(state.wdate);
  }));

  // ---------- 하루 보기 ----------
  function setDate(d) {
    const list = datesFor(state.src);
    if (!list.length) return;
    if (!d || !D.days[d] || !D.days[d][state.src]) {
      // 가장 가까운 날짜로
      d = list.reduce((a, b) => (Math.abs(new Date(b) - new Date(d || list[list.length - 1])) < Math.abs(new Date(a) - new Date(d || list[list.length - 1])) ? b : a));
    }
    state.date = d;
    $('#date').value = d;
    $('#date').min = list[0]; $('#date').max = list[list.length - 1];
    try { history.replaceState(null, '', '#' + d + ',' + state.src); } catch (e) { /* 파일로 열 때 무시 */ }
    renderDay();
  }
  function step(k) {
    const list = datesFor(state.src);
    const i = list.indexOf(state.date);
    const j = Math.min(list.length - 1, Math.max(0, i + k));
    setDate(list[j]);
  }
  $('#prev').onclick = () => step(-1);
  $('#next').onclick = () => step(1);
  $('#latest').onclick = () => { const l = datesFor(state.src); setDate(l[l.length - 1]); };
  $('#date').onchange = (e) => setDate(e.target.value);
  $$('.seg-b').forEach((b) => b.addEventListener('click', () => {
    state.src = b.dataset.src;
    $$('.seg-b').forEach((x) => x.classList.toggle('on', x === b));
    setDate(state.date);
  }));
  document.addEventListener('keydown', (e) => {
    if (!$('#tab-day').classList.contains('on') || e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowLeft') step(-1);
    if (e.key === 'ArrowRight') step(1);
  });

  function renderDay() {
    const day = D.days[state.date];
    const s = day[state.src];
    const dt = new Date(state.date + 'T00:00:00');
    const hasY = s.y.some((v) => v != null);

    // 배지: 과제1 급락 등급 (예측값은 그대로, 배지만 — 투트랙)
    const bd = [];
    bd.push(`<span class="badge info">${state.date} (${WD[dt.getDay()]}) · ${day.dt || ''}</span>`);
    if (day.drop && day.drop.grade && day.drop.grade !== '없음') {
      const cls = day.drop.grade === '강한 경보' ? 'strong' : 'warn';
      bd.push(`<span class="badge ${cls}">한낮 급락 ${day.drop.grade} · 과제1 예상 한낮 최저 ${fmt(day.drop.min)}원 — 범위 하한보다 더 내려갈 수 있음</span>`);
    }
    if (!hasY) bd.push('<span class="badge info">실적 미확정 (예측만 표시)</span>');
    $('#badges').innerHTML = bd.join('');

    // 카드
    const idx = [...Array(24).keys()];
    const mean = avg(s.p), width = avg(idx.map((i) => (s.hi[i] != null && s.lo[i] != null ? s.hi[i] - s.lo[i] : null)));
    $('#c-mean').textContent = fmt(mean) + '원';
    $('#c-mean-s').textContent = `범위 ${fmt(avg(s.lo))} ~ ${fmt(avg(s.hi))}원`;
    $('#c-width').textContent = fmt(width) + '원';
    const mids = [10, 11, 12, 13, 14];
    const midMin = Math.min(...mids.map((i) => s.p[i]).filter((v) => v != null));
    const midI = mids.find((i) => s.p[i] === midMin);
    $('#c-mid').textContent = fmt(midMin) + '원';
    $('#c-mid-s').textContent = midI != null ? `${midI + 1}시 · 하한 ${fmt(s.lo[midI])}원` : '';
    if (hasY) {
      const known = idx.filter((i) => s.y[i] != null && s.lo[i] != null);
      const inn = known.filter((i) => s.y[i] >= s.lo[i] && s.y[i] <= s.hi[i]).length;
      const r = inn / known.length;
      $('#c-in').textContent = `${inn}/${known.length}시간`;
      $('#c-in').className = 'v ' + (r >= 0.75 ? 'good' : r < 0.6 ? 'bad' : '');
      const mae = avg(known.map((i) => Math.abs(s.y[i] - s.p[i])));
      $('#c-in-s').textContent = `예측 평균 오차 ${fmt(mae)}원`;
    } else {
      $('#c-in').textContent = '-'; $('#c-in').className = 'v'; $('#c-in-s').textContent = '실적 공표 후 확인';
    }

    drawDayChart(s);
    // 표
    let h = '<tr><th>시간</th><th>하한</th><th>예측</th><th>상한</th><th>폭</th><th>실적</th><th>범위</th></tr>';
    idx.forEach((i) => {
      const y = s.y[i];
      const inn = y != null && s.lo[i] != null ? (y < s.lo[i] ? '아래 이탈' : y > s.hi[i] ? '위 이탈' : '안') : '';
      const cls = inn === '안' ? 'in' : inn ? 'out' : '';
      h += `<tr><td>${i + 1}시</td><td>${fmt(s.lo[i])}</td><td>${fmt(s.p[i])}</td><td>${fmt(s.hi[i])}</td><td>${fmt(s.hi[i] - s.lo[i])}</td><td>${fmt(y)}</td><td class="${cls}">${inn}</td></tr>`;
    });
    $('#htable').innerHTML = h;
    $('#day-note').textContent = state.src === 'A'
      ? '담당자 모델: 동반성장부 시간별 모델 예측값 + 오차 분위수 범위. 내일 수요예측(전날 19~20시 공표)을 쓰는 23:30 산출 기준이라 전날 15시 판단과 조건이 다릅니다. 시간 표기 1~24시(12시 = 11:00~12:00).'
      : '자체 모델: 전날 15시까지 알 수 있는 정보만 사용. 2025년까지 자료로 범위 방식을 정하고 2026년은 한 번 채점. 시간 표기 1~24시(12시 = 11:00~12:00).';
  }

  function avg(a) { const v = a.filter((x) => x != null && !isNaN(x)); return v.length ? v.reduce((p, c) => p + c, 0) / v.length : null; }

  // ---------- SVG 그래프 ----------
  const NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
  function niceTicks(lo, hi, n) {
    const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const st = [1, 2, 5, 10].map((m) => m * mag).find((m) => span / m <= n) || 10 * mag;
    const out = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) out.push(v); return out;
  }

  function drawDayChart(s) {
    const box = $('#chart'); box.innerHTML = '';
    const W = box.clientWidth || 600, H = box.clientHeight || 300;
    const m = { l: 40, r: 10, t: 10, b: 26 };
    const vals = [...s.lo, ...s.hi, ...s.y, ...s.p].filter((v) => v != null);
    let y0 = Math.max(0, Math.min(...vals) - 8), y1 = Math.max(...vals) + 8;
    const X = (h) => m.l + ((h - 1) / 23) * (W - m.l - m.r);
    const Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '시간별 예측과 80% 범위' }, box);
    const g = el('g', { class: 'grid' }, svg), ax = el('g', { class: 'axis' }, svg);
    niceTicks(y0, y1, 5).forEach((v) => {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g);
      el('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, ax).textContent = v;
    });
    for (let h = 1; h <= 24; h++) {
      if (W < 480 && h % 3 !== 0 && h !== 1) continue;
      el('text', { x: X(h), y: H - 8, 'text-anchor': 'middle' }, ax).textContent = h;
    }
    el('text', { x: m.l - 34, y: m.t + 2, 'text-anchor': 'start' }, ax).textContent = '원/kWh';
    // 한낮 표시
    el('rect', { x: X(10.5), y: m.t, width: X(15.5) - X(10.5), height: H - m.t - m.b, fill: 'var(--target)' }, svg);
    // 범위
    const hs = [...Array(24).keys()].map((i) => i + 1).filter((h) => s.lo[h - 1] != null);
    const top = hs.map((h) => `${X(h)},${Y(s.hi[h - 1])}`), bot = hs.slice().reverse().map((h) => `${X(h)},${Y(s.lo[h - 1])}`);
    el('polygon', { points: top.concat(bot).join(' '), fill: 'var(--band)', stroke: 'var(--band-edge)', 'stroke-width': 1 }, svg);
    // 예측선
    el('polyline', { points: hs.map((h) => `${X(h)},${Y(s.p[h - 1])}`).join(' '), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.2 }, svg);
    // 실적 점
    hs.forEach((h) => {
      const y = s.y[h - 1]; if (y == null) return;
      const out = y < s.lo[h - 1] || y > s.hi[h - 1];
      el('circle', { cx: X(h), cy: Y(y), r: 3.6, fill: out ? 'var(--bad)' : 'var(--ink)' }, svg);
    });
    // 툴팁
    const cursor = el('line', { y1: m.t, y2: H - m.b, stroke: 'var(--muted)', 'stroke-dasharray': '3 3', visibility: 'hidden' }, svg);
    const tip = $('#tip');
    const move = (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left) * (W / r.width);
      const h = Math.min(24, Math.max(1, Math.round(1 + ((px - m.l) / (W - m.l - m.r)) * 23)));
      const i = h - 1;
      cursor.setAttribute('x1', X(h)); cursor.setAttribute('x2', X(h)); cursor.setAttribute('visibility', 'visible');
      tip.hidden = false;
      tip.innerHTML = `<b>${h}시</b> 예측 ${fmt(s.p[i])} · 범위 ${fmt(s.lo[i])}~${fmt(s.hi[i])}` + (s.y[i] != null ? ` · 실적 ${fmt(s.y[i])}` : '');
      const bx = box.getBoundingClientRect(), cb = $('.chartbox').getBoundingClientRect();
      let left = (X(h) / W) * bx.width + (bx.left - cb.left) + 10;
      if (left + tip.offsetWidth > cb.width - 4) left -= tip.offsetWidth + 20;
      tip.style.left = left + 'px'; tip.style.top = (bx.top - cb.top + 8) + 'px';
    };
    svg.addEventListener('mousemove', move); svg.addEventListener('touchstart', move, { passive: true }); svg.addEventListener('touchmove', move, { passive: true });
    svg.addEventListener('mouseleave', () => { tip.hidden = true; cursor.setAttribute('visibility', 'hidden'); });
  }

  // ---------- 주간 보기 ----------
  const wDates = Object.keys(D.weekly || {}).sort();
  state.wdate = wDates[wDates.length - 1];
  const addDays = (d, k) => { const t = new Date(d + 'T00:00:00Z'); t.setUTCDate(t.getUTCDate() + k); return t.toISOString().slice(0, 10); };
  function setWDate(d) {
    if (!wDates.length) { $('#tab-week').innerHTML = '<p class="note">주간 예측 데이터가 없습니다.</p>'; return; }
    if (!D.weekly[d]) d = wDates.reduce((a, b) => (Math.abs(new Date(b) - new Date(d)) < Math.abs(new Date(a) - new Date(d)) ? b : a));
    state.wdate = d; $('#wdate').value = d; $('#wdate').min = wDates[0]; $('#wdate').max = wDates[wDates.length - 1];
    renderWeek();
  }
  $('#wprev').onclick = () => setWDate(wDates[Math.max(0, wDates.indexOf(state.wdate) - 1)]);
  $('#wnext').onclick = () => setWDate(wDates[Math.min(wDates.length - 1, wDates.indexOf(state.wdate) + 1)]);
  $('#wlatest').onclick = () => setWDate(wDates[wDates.length - 1]);
  $('#wdate').onchange = (e) => setWDate(e.target.value);

  function renderWeek() {
    const W = D.weekly[state.wdate];
    const cat = { p: [], lo: [], hi: [], y: [] };
    W.forEach((s) => ['p', 'lo', 'hi', 'y'].forEach((k) => cat[k].push(...s[k])));
    const labels = W.map((_, i) => addDays(state.wdate, i + 1));
    drawSeries('#wchart', '#wtip', cat, 144, (i) => `${labels[Math.floor(i / 24)].slice(5)} ${i % 24 + 1}시`, labels);
    let h = '<tr><th>대상일</th><th>지평</th><th>일평균 예측</th><th>일평균 범위</th><th>한낮 최저 예측</th><th>저녁 최고 예측</th><th>실적 일평균</th><th>범위 안</th><th>급락 배지</th></tr>';
    W.forEach((s, i) => {
      const d = labels[i], info = D.days[d] || {}, dt = new Date(d + 'T00:00:00');
      const mid = Math.min(...[10, 11, 12, 13, 14].map((j) => s.p[j])), eve = Math.max(...[17, 18, 19, 20, 21].map((j) => s.p[j]));
      const known = [...Array(24).keys()].filter((j) => s.y[j] != null);
      const inn = known.filter((j) => s.y[j] >= s.lo[j] && s.y[j] <= s.hi[j]).length;
      const badge = i === 0 && info.drop && info.drop.grade !== '없음' ? info.drop.grade : '';
      h += `<tr><td>${d.slice(5)} (${WD[dt.getDay()]})${info.dt ? ' · ' + info.dt : ''}</td><td>D+${i + 1}</td><td>${fmt(avg(s.p))}</td><td>${fmt(avg(s.lo))} ~ ${fmt(avg(s.hi))}</td><td>${fmt(mid)}</td><td>${fmt(eve)}</td><td>${known.length ? fmt(avg(known.map((j) => s.y[j]))) : '-'}</td><td class="${known.length ? (inn / known.length >= 0.75 ? 'in' : 'out') : ''}">${known.length ? inn + '/' + known.length : '-'}</td><td class="${badge ? 'out' : ''}">${badge}</td></tr>`;
    });
    $('#wtable').innerHTML = h;
  }

  // 범용 시계열 그래프 (주간 144시간)
  function drawSeries(boxSel, tipSel, s, n, tipLabel, dayLabels) {
    const box = $(boxSel); box.innerHTML = '';
    const W = box.clientWidth || 600, H = box.clientHeight || 300, m = { l: 40, r: 10, t: 10, b: 26 };
    const vals = [...s.lo, ...s.hi, ...s.y, ...s.p].filter((v) => v != null);
    const y0 = Math.max(0, Math.min(...vals) - 8), y1 = Math.max(...vals) + 8;
    const X = (i) => m.l + (i / (n - 1)) * (W - m.l - m.r), Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '주간 시간별 예측과 80% 범위' }, box);
    const g = el('g', { class: 'grid' }, svg), ax = el('g', { class: 'axis' }, svg);
    niceTicks(y0, y1, 5).forEach((v) => { el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); el('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, ax).textContent = v; });
    dayLabels.forEach((d, k) => {
      if (k > 0) el('line', { x1: X(k * 24 - 0.5), x2: X(k * 24 - 0.5), y1: m.t, y2: H - m.b, stroke: 'var(--muted)', 'stroke-opacity': 0.4 }, svg);
      el('text', { x: X(k * 24 + 11.5), y: H - 8, 'text-anchor': 'middle' }, ax).textContent = d.slice(5);
      el('rect', { x: X(k * 24 + 10), y: m.t, width: X(k * 24 + 14) - X(k * 24 + 10), height: H - m.t - m.b, fill: 'var(--target)' }, svg);
    });
    const idx = [...Array(n).keys()].filter((i) => s.lo[i] != null);
    el('polygon', { points: idx.map((i) => `${X(i)},${Y(s.hi[i])}`).concat(idx.slice().reverse().map((i) => `${X(i)},${Y(s.lo[i])}`)).join(' '), fill: 'var(--band)', stroke: 'var(--band-edge)', 'stroke-width': 1 }, svg);
    el('polyline', { points: idx.map((i) => `${X(i)},${Y(s.p[i])}`).join(' '), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 1.8 }, svg);
    idx.forEach((i) => { const y = s.y[i]; if (y == null) return; const out = y < s.lo[i] || y > s.hi[i]; el('circle', { cx: X(i), cy: Y(y), r: 2.2, fill: out ? 'var(--bad)' : 'var(--ink)' }, svg); });
    const cursor = el('line', { y1: m.t, y2: H - m.b, stroke: 'var(--muted)', 'stroke-dasharray': '3 3', visibility: 'hidden' }, svg);
    const tip = $(tipSel);
    const move = (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left) * (W / r.width);
      const i = Math.min(n - 1, Math.max(0, Math.round(((px - m.l) / (W - m.l - m.r)) * (n - 1))));
      cursor.setAttribute('x1', X(i)); cursor.setAttribute('x2', X(i)); cursor.setAttribute('visibility', 'visible');
      tip.hidden = false;
      tip.innerHTML = `<b>${tipLabel(i)}</b> 예측 ${fmt(s.p[i])} · 범위 ${fmt(s.lo[i])}~${fmt(s.hi[i])}` + (s.y[i] != null ? ` · 실적 ${fmt(s.y[i])}` : '');
      const cb = box.parentElement.getBoundingClientRect(), bx = box.getBoundingClientRect();
      let left = (X(i) / W) * bx.width + (bx.left - cb.left) + 10;
      if (left + tip.offsetWidth > cb.width - 4) left -= tip.offsetWidth + 20;
      tip.style.left = left + 'px'; tip.style.top = (bx.top - cb.top + 8) + 'px';
    };
    svg.addEventListener('mousemove', move); svg.addEventListener('touchstart', move, { passive: true }); svg.addEventListener('touchmove', move, { passive: true });
    svg.addEventListener('mouseleave', () => { tip.hidden = true; cursor.setAttribute('visibility', 'hidden'); });
  }

  // ---------- 성적 ----------
  $$('.seg-p').forEach((b) => b.addEventListener('click', () => {
    state.perf = b.dataset.src; $$('.seg-p').forEach((x) => x.classList.toggle('on', x === b)); renderPerf();
  }));

  function renderPerf() {
    const P = D[state.perf];
    const chosen = P.choice['범위 방법'] || P.choice['선택 방법'];
    const all = P.test.filter((r) => r['방법'] === chosen && r['구간'] === '전체')[0] || {};
    const ok = all['포함률'] >= 0.75 && all['포함률'] <= 0.85;
    const pt = P.choice['점예측'] ? P.choice['점예측']['2026 시험'] : null;
    $('#perf-head').innerHTML = [
      ['80% 범위 포함률', pct(all['포함률']), ok ? '목표 75~85% 충족' : '목표 75~85% 밖', ok ? 'good' : 'bad'],
      ['범위 평균 폭', fmt(all['평균폭']) + '원', '좁을수록 쓸모 있음', ''],
      ['하한 아래 / 상한 위', pct(all['하한아래']) + ' / ' + pct(all['상한위']), '각각 약 10%가 균형', ''],
      ['예측 평균 오차(MAE)', fmt(all['MAE'], 2) + '원', pt ? `전일 같은 시간 대비 rMAE ${fmt(pt.rMAE, 3)}` : `${all['시간수'] || '-'}시간`, ''],
    ].map(([k, v, s, c]) => `<div class="card"><div class="k">${k}</div><div class="v ${c}">${v}</div><div class="s">${s}</div></div>`).join('');

    let h = '<tr><th>방법</th><th>구간</th><th>시간수</th><th>포함률</th><th>하한 아래</th><th>상한 위</th><th>평균 폭</th><th>구간점수</th><th>MAE</th></tr>';
    P.test.forEach((r) => {
      h += `<tr class="${r['방법'] === chosen ? 'sel' : ''}"><td>${r['방법'] === chosen ? '★ ' : ''}${r['방법']}</td><td>${r['구간']}</td><td>${r['시간수']}</td><td>${pct(r['포함률'])}</td><td>${pct(r['하한아래'])}</td><td>${pct(r['상한위'])}</td><td>${fmt(r['평균폭'])}</td><td>${fmt(r['구간점수'])}</td><td>${fmt(r['MAE'], 2)}</td></tr>`;
    });
    $('#ptable').innerHTML = h;
    drawHourChart(P.byHour);
    let mh = '<tr><th>월</th><th>시간수</th><th>포함률</th><th>하한 아래</th><th>상한 위</th><th>평균 폭</th><th>MAE</th></tr>';
    P.byMonth.forEach((r) => { mh += `<tr><td>${r['월']}</td><td>${r['시간수']}</td><td>${pct(r['포함률'])}</td><td>${pct(r['하한아래'])}</td><td>${pct(r['상한위'])}</td><td>${fmt(r['평균폭'])}</td><td>${fmt(r['MAE'], 2)}</td></tr>`; });
    $('#mtable').innerHTML = mh;

    let ex = `<p class="note">선택 방법 ★: ${chosen} — 2025년 성적(포함률 75~85% 중 구간점수 최소)으로 2026 채점 전에 고정.`;
    if (P.choice['점예측']) { const c = P.choice['점예측']; ex += ` 점예측: LightGBM, 학습 시작 ${c['학습시작']}년, 피처 ${c['피처']}, 목표 ${c['목표'] === 'diff' ? '직전 같은 유형일 대비 차이' : '원값'}, ${c['재학습']} 재학습.`; }
    if (P.choice['예측값']) ex += ` 예측값: ${P.choice['예측값']}.`;
    ex += ' 구간점수 = 폭 + 이탈 벌점(낮을수록 좋음).</p>';
    if (state.perf === 'B' && D.W && D.W.test.length) {
      ex += '<h2>지평별 성적 <span class="muted">주간 보기 · 2026 한 번 채점 · 지평마다 2025로 범위 방식 선택</span></h2><div class="tscroll"><table><tr><th>지평</th><th>범위 방법</th><th>포함률</th><th>하한 아래</th><th>상한 위</th><th>평균 폭</th><th>MAE</th><th>휴무일 한낮 포함률</th></tr>';
      D.W.test.filter((r) => r['구간'] === '전체').forEach((r) => {
        const hm = D.W.test.find((x) => x.H === r.H && x['구간'] === '휴무일 한낮(11~15시)') || {};
        const okk = r['포함률'] >= 0.75 && r['포함률'] <= 0.85;
        ex += `<tr><td>D+${r.H}</td><td>${r['방법']}</td><td class="${okk ? 'in' : 'out'}">${pct(r['포함률'])}</td><td>${pct(r['하한아래'])}</td><td>${pct(r['상한위'])}</td><td>${fmt(r['평균폭'])}</td><td>${fmt(r['MAE'], 2)}</td><td>${pct(hm['포함률'])}</td></tr>`;
      });
      ex += '</table></div>';
      if (D.W.ref && D.W.ref.length) {
        ex += '<h2>운영 주간 예측과 비교 <span class="muted">같은 기준일·지평·시간 (참고)</span></h2><div class="tscroll"><table><tr><th>지평</th><th>기준일 수</th><th>자체 MAE</th><th>운영 MAE</th><th>자체 포함률</th><th>운영 + 같은 폭 포함률</th><th>기간</th></tr>';
        D.W.ref.forEach((r) => { ex += `<tr><td>D+${r.H}</td><td>${r['기준일 수']}</td><td>${fmt(r['자체 MAE'], 2)}</td><td>${fmt(r['운영 MAE'], 2)}</td><td>${pct(r['자체 포함률'])}</td><td>${pct(r['운영+같은 폭 포함률'])}</td><td>${r['기간']}</td></tr>`; });
        ex += '</table></div>';
      }
    }
    const ref = P.weekly || P.operational;
    if (ref && ref.length) {
      ex += `<h2>참고 비교 <span class="muted">${state.perf === 'B' ? '운영 주간 예측 D+1(15시)과 같은 날' : '운영 공표 예측 기록(23:30)에 같은 방법 적용'}</span></h2><div class="tscroll"><table><tr><th>대상</th><th>구간</th><th>시간수</th><th>포함률</th><th>평균 폭</th><th>MAE</th></tr>`;
      ref.forEach((r) => { ex += `<tr><td>${r['대상'] || r['방법']}</td><td>${r['구간'] || r['기간'] || ''}</td><td>${r['시간수']}</td><td>${pct(r['포함률'])}</td><td>${fmt(r['평균폭'])}</td><td>${fmt(r['MAE'], 2)}</td></tr>`; });
      ex += '</table></div>';
    }
    $('#perf-extra').innerHTML = ex;
  }

  function drawHourChart(rows) {
    const box = $('#hourchart'); box.innerHTML = '';
    if (!rows || !rows.length) return;
    const W = box.clientWidth - 16 || 600, H = (box.clientHeight || 220) - 16;
    const m = { l: 36, r: 6, t: 8, b: 22 };
    const X = (h) => m.l + ((h - 0.5) / 24) * (W - m.l - m.r), bw = (W - m.l - m.r) / 24 * 0.7;
    const Y = (v) => H - m.b - (v / 1) * (H - m.t - m.b);
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '시간대별 포함률' }, box);
    el('rect', { x: m.l, y: Y(0.85), width: W - m.l - m.r, height: Y(0.75) - Y(0.85), fill: 'var(--target)' }, svg);
    const ax = el('g', { class: 'axis' }, svg), g = el('g', { class: 'grid' }, svg);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); el('text', { x: m.l - 5, y: Y(v) + 4, 'text-anchor': 'end' }, ax).textContent = (v * 100) + '%'; });
    rows.forEach((r) => {
      const h = r['시간'], v = r['포함률'];
      const okk = v >= 0.75 && v <= 0.85;
      const rect = el('rect', { x: X(h) - bw / 2, y: Y(v), width: bw, height: Y(0) - Y(v), rx: 2, fill: okk ? 'var(--accent)' : 'var(--warn)' }, svg);
      el('title', {}, rect).textContent = `${h}시 포함률 ${pct(v)}, 폭 ${fmt(r['평균폭'])}원`;
      if (W > 480 || h % 3 === 0) el('text', { x: X(h), y: H - 6, 'text-anchor': 'middle' }, ax).textContent = h;
    });
  }

  window.addEventListener('resize', () => { if (state.date) drawDayChart(D.days[state.date][state.src]); if ($('#tab-week').classList.contains('on')) renderWeek(); if ($('#tab-perf').classList.contains('on')) drawHourChart(D[state.perf].byHour); });

  // ---------- 시작 ----------
  $('#foot').textContent = `데이터 생성 ${D.generated} · 출처: 전력거래소·기상청·한국천문연구원 공공데이터, 동반성장부 SMP 예측 프로그램(교육용) · 숙명여대 P4`;
  const hash = (location.hash || '').slice(1).split(',');
  if (hash[1] === 'A' || hash[1] === 'B') { state.src = hash[1]; $$('.seg-b').forEach((x) => x.classList.toggle('on', x.dataset.src === state.src)); }
  const list = datesFor(state.src);
  setDate(hash[0] && D.days[hash[0]] ? hash[0] : list[list.length - 1]);
  if (hash[0] === 'perf' || hash[0] === 'help' || hash[0] === 'week') $(`.tab[data-tab="${hash[0]}"]`).click();
})();
