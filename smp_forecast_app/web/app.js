/* SMP 예측 웹앱 — 데이터: data/forecast_data.js (window.SMP_DATA) */
(function () {
  const D = window.SMP_DATA;
  if (!D) { document.body.innerHTML = '<p style="padding:20px">데이터 파일(data/forecast_data.js)이 없습니다. python -m smp_forecast_app.src.export_web 실행 후 다시 여세요.</p>'; return; }
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const fmt = (v, d = 1) => (v == null || isNaN(v) ? '-' : Number(v).toFixed(d));
  const pct = (v) => (v == null || isNaN(v) ? '-' : (v * 100).toFixed(1) + '%');
  const sgn = (v) => (v == null || isNaN(v) ? '-' : (v > 0 ? '+' : '') + Number(v).toFixed(1));
  const avg = (a) => { const v = a.filter((x) => x != null && !isNaN(x)); return v.length ? v.reduce((p, c) => p + c, 0) / v.length : null; };
  const WD = ['일', '월', '화', '수', '목', '금', '토'];
  const wd = (d) => WD[new Date(d + 'T00:00:00').getDay()];
  const md = (d) => `${+d.slice(5, 7)}/${+d.slice(8, 10)}`;
  const addDays = (d, k) => { const t = new Date(d + 'T00:00:00Z'); t.setUTCDate(t.getUTCDate() + k); return t.toISOString().slice(0, 10); };
  const inside = (y, lo, hi) => y != null && lo != null && y >= lo && y <= hi;
  const info = (d) => (D.dayinfo && D.dayinfo[d]) || {};

  // ---------- 탭 ----------
  const renderers = {};
  $$('.tab').forEach((b) => b.addEventListener('click', () => {
    $$('.tab').forEach((x) => x.classList.toggle('on', x === b));
    $$('.panel').forEach((p) => p.classList.toggle('on', p.id === 'tab-' + b.dataset.tab));
    if (renderers[b.dataset.tab]) renderers[b.dataset.tab]();
  }));

  // ---------- 공용 그래프 ----------
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  function niceTicks(lo, hi, n) {
    const span = Math.max(hi - lo, 1), raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const st = [1, 2, 5, 10].map((m) => m * mag).find((m) => span / m <= n) || 10 * mag;
    const out = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) out.push(Math.round(v * 100) / 100); return out;
  }
  /* s = {p, lo, hi, y} 같은 길이 배열. o = {xTicks:[[i,label]], shade:[[i0,i1]], dividers:[i], tipLabel(i), dot} */
  function chart(boxSel, tipSel, s, o) {
    const box = $(boxSel); box.innerHTML = '';
    const n = s.p.length;
    const W = box.clientWidth || 700, H = box.clientHeight || 320, m = { l: 42, r: 12, t: 10, b: 26 };
    const vals = [...s.lo, ...s.hi, ...s.y, ...s.p].filter((v) => v != null);
    if (!vals.length) { box.innerHTML = '<p class="note" style="padding:20px">표시할 자료가 없습니다.</p>'; return; }
    const y0 = Math.max(0, Math.min(...vals) - 8), y1 = Math.max(...vals) + 8;
    const X = (i) => m.l + (n === 1 ? 0 : (i / (n - 1)) * (W - m.l - m.r));
    const Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' }, box);
    const step = n > 1 ? (W - m.l - m.r) / (n - 1) : 10;
    (o.shade || []).forEach(([a, b]) => el('rect', { x: X(a) - step / 2, y: m.t, width: X(b) - X(a) + step, height: H - m.t - m.b, fill: 'var(--disabled)' }, svg));
    const g = el('g', { class: 'grid' }, svg), ax = el('g', { class: 'axis' }, svg);
    niceTicks(y0, y1, 5).forEach((v) => { el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); el('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, ax).textContent = v; });
    el('text', { x: 2, y: m.t + 4, 'text-anchor': 'start' }, ax).textContent = '원';
    (o.dividers || []).forEach((i) => el('line', { x1: X(i) - step / 2, x2: X(i) - step / 2, y1: m.t, y2: H - m.b, stroke: '#b9c7d1' }, svg));
    (o.xTicks || []).forEach(([i, t]) => { el('text', { x: X(i), y: H - 8, 'text-anchor': 'middle' }, ax).textContent = t; });
    // 범위: 연속 구간별 다각형
    let segIdx = [];
    const flushBand = () => {
      if (segIdx.length > 1) el('polygon', { points: segIdx.map((i) => `${X(i)},${Y(s.hi[i])}`).concat(segIdx.slice().reverse().map((i) => `${X(i)},${Y(s.lo[i])}`)).join(' '), fill: 'var(--band)', stroke: 'var(--band-edge)', 'stroke-width': 1 }, svg);
      segIdx = [];
    };
    for (let i = 0; i < n; i++) { if (s.lo[i] == null || s.hi[i] == null) { flushBand(); continue; } segIdx.push(i); }
    flushBand();
    const line = (arr, attrs) => {
      let seg = [];
      const fl = () => { if (seg.length > 1) el('polyline', { points: seg.join(' '), fill: 'none', ...attrs }, svg); seg = []; };
      for (let i = 0; i < n; i++) { if (arr[i] == null) { fl(); continue; } seg.push(`${X(i)},${Y(arr[i])}`); }
      fl();
    };
    line(s.p, { stroke: 'var(--primary)', 'stroke-width': 2.2 });
    line(s.y, { stroke: 'var(--actual)', 'stroke-width': 1.4 });
    const r = o.dot ?? 3.2;
    for (let i = 0; i < n; i++) {
      if (s.y[i] == null) continue;
      const out = s.lo[i] != null && !inside(s.y[i], s.lo[i], s.hi[i]);
      if (r > 0 || out) el('rect', { x: X(i) - Math.max(r, 2), y: Y(s.y[i]) - Math.max(r, 2), width: 2 * Math.max(r, 2), height: 2 * Math.max(r, 2), fill: out ? 'var(--accent-red)' : 'var(--actual)' }, svg);
    }
    const cursor = el('line', { y1: m.t, y2: H - m.b, stroke: 'var(--primary-dark)', 'stroke-dasharray': '3 3', visibility: 'hidden' }, svg);
    const tip = $(tipSel);
    const move = (ev) => {
      const rc = svg.getBoundingClientRect();
      const px = ((ev.touches ? ev.touches[0].clientX : ev.clientX) - rc.left) * (W / rc.width);
      const i = Math.min(n - 1, Math.max(0, Math.round((px - m.l) / step)));
      cursor.setAttribute('x1', X(i)); cursor.setAttribute('x2', X(i)); cursor.setAttribute('visibility', 'visible');
      tip.hidden = false;
      tip.innerHTML = `<b>${o.tipLabel(i)}</b>&nbsp; 예측 ${fmt(s.p[i])} · 범위 ${fmt(s.lo[i])}~${fmt(s.hi[i])}` + (s.y[i] != null ? ` · <b>공표 ${fmt(s.y[i])}</b> (${sgn(s.y[i] - s.p[i])})` : '');
      const cb = box.parentElement.getBoundingClientRect(), bx = box.getBoundingClientRect();
      let left = (X(i) / W) * bx.width + (bx.left - cb.left) + 12;
      if (left + tip.offsetWidth > cb.width - 4) left -= tip.offsetWidth + 24;
      tip.style.left = left + 'px'; tip.style.top = (bx.top - cb.top + 6) + 'px';
    };
    svg.addEventListener('mousemove', move); svg.addEventListener('touchstart', move, { passive: true }); svg.addEventListener('touchmove', move, { passive: true });
    svg.addEventListener('mouseleave', () => { tip.hidden = true; cursor.setAttribute('visibility', 'hidden'); });
  }

  // ---------- 급락 확률 (과제1 → 확률 보정): 확률 숫자 + 낮음·주의·강한 경보 눈금 막대 ----------
  const DTHR = (D.perf.dropChoice || {})['등급 문턱 확률'] || {};
  const T1 = DTHR['주의(90.1원)'] ?? 0.106, T2 = DTHR['강한 경보(75.3원)'] ?? 0.326;
  const LV = { '강한 경보': ['strong', '강한 경보'], '주의': ['warn', '주의'], '없음': ['low', '낮음'] };
  const realRate = {};
  ((D.perf && D.perf.dropCal) || []).filter((r) => r['연도'] === 2026 && String(r['구분']).startsWith('등급 '))
    .forEach((r) => { realRate[r['구분'].slice(3)] = r; });
  function dropBadge(dr, compact) {
    if (!dr) return '';
    const [cls, name] = LV[dr.grade] || ['low', dr.grade];
    const ptxt = dr.prob == null ? '-' : Math.round(dr.prob * 100) + '%';
    if (compact) return `<span class="dchip ${cls}"><i></i>급락 확률 <b>${ptxt}</b> · ${name}</span>`;
    const rr = realRate[dr.grade], pos = Math.min(100, Math.max(0, (dr.prob || 0) * 100));
    const w1 = T1 * 100, w2 = (T2 - T1) * 100, w3 = 100 - T2 * 100;
    return `<div class="dmeter ${cls}">
      <div class="dm-l"><div class="k">내일 한낮(11~15시) 급락 확률</div><div class="v">${ptxt}</div><div class="lvl"><i></i>${name}</div></div>
      <div class="dm-r"><div class="bar"><span class="z low" style="width:${w1}%"></span><span class="z warn" style="width:${w2}%"></span><span class="z strong" style="width:${w3}%"></span>
        <b class="mk" style="left:${pos}%"></b></div>
        <div class="zl"><span style="width:${w1}%">낮음</span><span style="width:${w2}%">주의 ${Math.round(T1 * 100)}%~</span><span style="width:${w3}%">강한 경보 ${Math.round(T2 * 100)}%~</span></div>
        <div class="sub">급락 = 한낮 최저 SMP 30원 미만 · 과제1 예상 한낮 최저 ${fmt(dr.min)}원${rr ? ` · 2026년 이 등급 날 실제 급락 ${Math.round(rr['실제 급락 비율'] * 100)}% (${rr['급락일']}/${rr['일수']}일)` : ''}</div></div></div>`;
  }

  // ---------- 일주일 예측 (앞으로 7일만) ----------
  let wView = 'all', wDay = 0;
  $$('#w-view button').forEach((b) => b.addEventListener('click', () => {
    wView = b.dataset.v; $$('#w-view button').forEach((x) => x.classList.toggle('on', x === b)); renderWeek();
  }));
  function renderWeek() {
    const Wk = D.week, days = Wk.days;
    $('#w-title').textContent = `${md(days[0].date)}(${wd(days[0].date)}) ~ ${md(days[days.length - 1].date)}(${wd(days[days.length - 1].date)}) 시간별 예측`;
    $('#w-sub').textContent = `기준일 ${Wk.base} 15시 판단`;
    const bd = [dropBadge(Wk.drop, false)];
    const pubDays = days.filter((d) => d.y.some((v) => v != null)).length;
    if (pubDays) bd.push(`<span class="badge pub">공표 SMP ${pubDays}일분 겹쳐 표시</span>`);
    $('#w-badges').innerHTML = bd.join('');
    // 날짜 카드 (누르면 그 날만)
    $('#w-cards').innerHTML = days.map((d, i) => {
      const mn = Math.min(...[10, 11, 12, 13, 14].map((j) => d.p[j]));
      const mx = Math.max(...d.p.filter((v) => v != null));
      return `<button class="dcard ${d.off ? 'off' : ''} ${wView === 'day' && wDay === i ? 'on' : ''}" data-i="${i}">
        <div class="t">${md(d.date)} (${wd(d.date)}) ${d.dt && d.dt !== '평일' ? d.dt : ''}</div>
        <div class="v">${fmt(avg(d.p))}<small>원</small></div>
        <div class="r">${fmt(avg(d.lo))} ~ ${fmt(avg(d.hi))}</div>
        <div class="r">한낮 최저 ${fmt(mn)} · 최고 ${fmt(mx)}</div>
        ${i === 0 && D.week.drop ? dropBadge(D.week.drop, true) : ''}</button>`;
    }).join('');
    $$('#w-cards .dcard').forEach((c) => c.addEventListener('click', () => {
      wDay = +c.dataset.i; wView = 'day'; $$('#w-view button').forEach((x) => x.classList.toggle('on', x.dataset.v === 'day')); renderWeek();
    }));
    let s, o;
    if (wView === 'all') {
      s = { p: [], lo: [], hi: [], y: [] };
      days.forEach((d) => ['p', 'lo', 'hi', 'y'].forEach((k) => s[k].push(...d[k])));
      o = { xTicks: days.map((d, k) => [k * 24 + 11.5, `${md(d.date)}(${wd(d.date)})`]), dividers: days.map((_, k) => k * 24).slice(1),
        shade: days.map((_, k) => [k * 24 + 10, k * 24 + 14]), tipLabel: (i) => `${md(days[Math.floor(i / 24)].date)} ${i % 24 + 1}시`, dot: 0 };
    } else {
      const d = days[wDay];
      s = { p: d.p, lo: d.lo, hi: d.hi, y: d.y };
      o = { xTicks: [...Array(24).keys()].filter((i) => innerWidth > 560 || i % 3 === 2).map((i) => [i, i + 1]), shade: [[10, 14]], tipLabel: (i) => `${md(d.date)} ${i + 1}시` };
    }
    chart('#w-chart', '#w-tip', s, o);
    const show = wView === 'all' ? days : [days[wDay]];
    let h = '<tr><th class="l">날짜</th><th class="l">시간</th><th>하한</th><th>예측</th><th>상한</th><th>공표 SMP</th></tr>';
    show.forEach((d) => d.p.forEach((p, i) => {
      h += `<tr class="${d.off ? 'off' : ''}"><td class="l">${md(d.date)}(${wd(d.date)})</td><td class="l">${i + 1}시</td><td>${fmt(d.lo[i])}</td><td>${fmt(p)}</td><td>${fmt(d.hi[i])}</td><td class="act">${fmt(d.y[i])}</td></tr>`;
    }));
    $('#w-table').innerHTML = h;
  }
  renderers.week = renderWeek;

  // ---------- 한 달 예측 ----------
  const SN = { mean: '일평균', mid: '한낮 11~15시 평균', eve: '저녁 18~22시 평균', midmin: '한낮 최저' };
  let mStat = 'mean';
  $$('#m-stat button').forEach((b) => b.addEventListener('click', () => {
    mStat = b.dataset.s; $$('#m-stat button').forEach((x) => x.classList.toggle('on', x === b)); renderMonth();
  }));
  function renderMonth() {
    const M = D.month, k = mStat, n = M.d.length;
    const s = { p: M['p_' + k], lo: M['lo_' + k], hi: M['hi_' + k], y: M['y_' + k] };
    $('#m-title').textContent = `${md(M.d[0])} ~ ${md(M.d[n - 1])} 일별 예측 (${SN[k]})`;
    $('#m-sub').textContent = `기준일 ${M.base} 15시 판단 · 7~34일 뒤`;
    // 월별 평균 (달력 월)
    const byMonth = {};
    M.d.forEach((d, i) => { const key = d.slice(0, 7); (byMonth[key] = byMonth[key] || []).push(i); });
    $('#m-badges').innerHTML = Object.entries(byMonth).map(([ym, idx]) =>
      `<span class="badge">${+ym.slice(5)}월 (${idx.length}일분) 평균 예상 <b>${fmt(avg(idx.map((i) => s.p[i])))}원</b> · 범위 ${fmt(avg(idx.map((i) => s.lo[i])))}~${fmt(avg(idx.map((i) => s.hi[i])))}</span>`).join('');
    let wk = '';
    for (let w = 0; w < 4; w++) {
      const r = [...Array(7).keys()].map((j) => w * 7 + j).filter((j) => j < n);
      if (!r.length) continue;
      const ys = r.map((j) => s.y[j]).filter((v) => v != null);
      wk += `<div class="wk"><div class="t">${w + 1}주차 · ${md(M.d[r[0]])} ~ ${md(M.d[r[r.length - 1]])}</div>
        <div class="v">${fmt(avg(r.map((j) => s.p[j])))}원</div><div class="r">범위 ${fmt(avg(r.map((j) => s.lo[j])))} ~ ${fmt(avg(r.map((j) => s.hi[j])))}</div>
        <div class="a">${ys.length ? `공표 ${fmt(avg(ys))}원 (${ys.length}일)` : ''}</div></div>`;
    }
    $('#m-weeks').innerHTML = wk;
    const shade = []; M.dt.forEach((t, i) => { if (t && t !== '평일') shade.push([i, i]); });
    chart('#m-chart', '#m-tip', s, { xTicks: M.d.map((d, i) => [i, md(d)]).filter((_, i) => i % (innerWidth > 700 ? 2 : 4) === 0), shade,
      dividers: [7, 14, 21], tipLabel: (i) => `${md(M.d[i])}(${wd(M.d[i])}) ${M.dt[i] || ''} · ${SN[k]}` });
    let h = `<tr><th class="l">날짜</th><th>며칠 뒤</th><th>하한</th><th>예측</th><th>상한</th><th>공표</th></tr>`;
    M.d.forEach((d, i) => { h += `<tr class="${M.dt[i] && M.dt[i] !== '평일' ? 'off' : ''}"><td class="l">${md(d)} (${wd(d)}) ${M.dt[i] || ''}</td><td>${i + 7}</td><td>${fmt(s.lo[i])}</td><td>${fmt(s.p[i])}</td><td>${fmt(s.hi[i])}</td><td class="act">${fmt(s.y[i])}</td></tr>`; });
    $('#m-table').innerHTML = h;
  }
  renderers.month = renderMonth;

  // ---------- 이전 흐름 확인 (달력에서 기간 선택, 최대 14일) ----------
  const P = D.past;
  const MAXD = 14;
  const lastAct = Object.keys(P.actual).sort().pop();
  const firstAct = Object.keys(P.actual).sort()[0];
  const dayDiff = (a, b) => Math.round((new Date(b + 'T00:00:00Z') - new Date(a + 'T00:00:00Z')) / 864e5);
  const lab = (d) => `${md(d)}(${wd(d)})`;
  let fS = null, fE = null, picking = false, calView = null, hoverD = null;

  function fillFlowH() {
    const k = $('#f-kind').value;
    $('#f-h').hidden = k === 'd1'; $('#f-stat').hidden = k !== 'mo';
    if (k === 'wk') $('#f-h').innerHTML = Object.keys(P.hourly).map(Number).sort((a, b) => a - b)
      .map((h) => `<option value="${h}">${h === 1 ? '하루 전 (D+1)' : h + '일 전 (D+' + h + ')'}</option>`).join('');
    if (k === 'mo') $('#f-h').innerHTML = Object.keys(P.daily).map(Number).sort((a, b) => a - b)
      .map((h) => `<option value="${h}">${h}일 전 예측</option>`).join('');
  }
  // 달력: 두 달씩 (휴대폰은 한 달), 시작일 → 종료일 순서로 고름. 종료일은 시작일부터 14일 이내.
  function calMonth(y, m) {
    const first = new Date(Date.UTC(y, m, 1)), days = new Date(Date.UTC(y, m + 1, 0)).getUTCDate();
    let h = `<div class="cal-m"><h4>${y}년 ${m + 1}월</h4><div class="cal-grid">`
      + WD.map((w, i) => `<div class="wdh ${i === 0 ? 'sun' : ''}">${w}</div>`).join('');
    for (let i = 0; i < first.getUTCDay(); i++) h += '<span></span>';
    const pEnd = picking && hoverD && hoverD >= fS ? hoverD : fE;
    for (let d = 1; d <= days; d++) {
      const ds = `${y}-${String(m + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
      let dis = ds < firstAct || ds > lastAct;
      if (picking && fS && ds > fS && dayDiff(fS, ds) >= MAXD) dis = true;
      const cls = ['cal-d', info(ds).off || new Date(ds + 'T00:00:00').getDay() === 0 ? 'off' : '',
        ds === fS ? 's' : '', !picking && ds === fE ? 'e' : '', fS && pEnd && ds > fS && ds < pEnd ? 'in' : '', picking && ds === hoverD && ds > fS ? 'e' : ''].join(' ');
      h += `<button type="button" class="${cls}" data-d="${ds}" ${dis ? 'disabled' : ''}>${d}</button>`;
    }
    return h + '</div></div>';
  }
  function renderCal() {
    const pop = $('#f-cal');
    const [y, m] = calView;
    const n2 = m === 11 ? [y + 1, 0] : [y, m + 1];
    pop.innerHTML = `<div class="cal-head"><span class="msg">${picking ? `<b>종료일</b>을 고르세요 (시작일 포함 최대 ${MAXD}일)` : '<b>시작일</b>을 고르세요'}</span>
      <span class="nav"><button class="btn" data-nav="-1" type="button">‹</button><button class="btn" data-nav="1" type="button">›</button></span></div>
      <div class="cal-months">${calMonth(y, m)}${calMonth(n2[0], n2[1])}</div>
      <div class="cal-foot"><span class="q"><button class="btn" data-q="7" type="button">최근 1주</button><button class="btn" data-q="14" type="button">최근 2주</button></span>
      <span class="note" style="margin:0">선택 가능: ${md(firstAct)}/${firstAct.slice(0, 4)} ~ ${lab(lastAct)}</span></div>`;
    pop.querySelectorAll('[data-nav]').forEach((b) => b.addEventListener('click', (e) => {
      e.stopPropagation(); let [yy, mm] = calView; mm += +b.dataset.nav; if (mm < 0) { mm = 11; yy--; } if (mm > 11) { mm = 0; yy++; } calView = [yy, mm]; renderCal();
    }));
    pop.querySelectorAll('[data-q]').forEach((b) => b.addEventListener('click', (e) => {
      e.stopPropagation(); fE = lastAct; fS = addDays(lastAct, -(+b.dataset.q) + 1); picking = false; closeCal(); renderFlow();
    }));
    pop.querySelectorAll('.cal-d').forEach((b) => {
      b.addEventListener('click', (e) => {
        e.stopPropagation();
        const d = b.dataset.d;
        if (!picking || d < fS) { fS = d; fE = null; picking = true; renderCal(); return; }
        fE = d; picking = false; closeCal(); renderFlow();
      });
      b.addEventListener('mouseenter', () => { if (picking) { hoverD = b.dataset.d; renderCalMarks(); } });
    });
    $('#f-start-box').classList.toggle('on', !picking); $('#f-end-box').classList.toggle('on', picking);
  }
  function renderCalMarks() { // 마우스를 올린 날까지 미리 칠하기 (다시 그리기)
    const sc = $('#f-cal').scrollTop; renderCal(); $('#f-cal').scrollTop = sc;
  }
  function openCal(endSide) {
    const ref = (endSide ? fE : fS) || lastAct;
    const t = new Date(ref + 'T00:00:00Z');
    let y = t.getUTCFullYear(), m = t.getUTCMonth();
    if (innerWidth > 760) { m -= 1; if (m < 0) { m = 11; y--; } }   // 두 달 보기: 고른 날이 오른쪽 달
    calView = [y, m];
    picking = !!endSide && !!fS; hoverD = null;
    $('#f-cal').hidden = false; renderCal();
  }
  function closeCal() {
    $('#f-cal').hidden = true; picking = false; hoverD = null;
    $('#f-start-box').classList.remove('on'); $('#f-end-box').classList.remove('on');
    if (fS && !fE) fE = fS;
  }
  $('#f-start-box').addEventListener('click', (e) => { e.stopPropagation(); fE = fE || null; openCal(false); });
  $('#f-end-box').addEventListener('click', (e) => { e.stopPropagation(); openCal(true); });
  $('#f-cal').addEventListener('click', (e) => e.stopPropagation());
  document.addEventListener('click', () => { if (!$('#f-cal').hidden) { closeCal(); renderFlow(); } });
  $('#f-kind').addEventListener('change', () => { fillFlowH(); renderFlow(); });
  ['#f-h', '#f-stat'].forEach((s) => $(s).addEventListener('change', renderFlow));

  function renderFlow() {
    $('#f-start-txt').textContent = fS ? lab(fS) : '-';
    $('#f-end-txt').textContent = fE ? lab(fE) : '-';
    if (!fS || !fE) return;
    $('#f-days').textContent = `${dayDiff(fS, fE) + 1}일`;
    const kind = $('#f-kind').value, H = kind === 'd1' ? '1' : $('#f-h').value;
    const dates = []; for (let d = fS; d <= fE; d = addDays(d, 1)) dates.push(d);
    const s = { p: [], lo: [], hi: [], y: [] }, labs = [], rows = [];
    if (kind !== 'mo') {
      const src = P.hourly[H] || {};
      dates.forEach((d) => {
        const r = src[d], y = P.actual[d] || Array(24).fill(null);
        for (let i = 0; i < 24; i++) { s.p.push(r ? r[0][i] : null); s.lo.push(r ? r[1][i] : null); s.hi.push(r ? r[2][i] : null); s.y.push(y[i]); labs.push([d, i]); }
        const ok = r ? [...Array(24).keys()].filter((i) => r[0][i] != null && y[i] != null) : [];
        rows.push({ d, p: r ? avg(r[0]) : null, y: avg(y), mae: ok.length ? avg(ok.map((i) => Math.abs(y[i] - r[0][i]))) : null,
          cov: ok.length ? ok.filter((i) => inside(y[i], r[1][i], r[2][i])).length / ok.length : null,
          pmin: r ? Math.min(...[10, 11, 12, 13, 14].map((i) => r[0][i])) : null,
          ymin: y[10] != null ? Math.min(...[10, 11, 12, 13, 14].map((i) => y[i]).filter((v) => v != null)) : null, drop: P.drop[d] });
      });
    } else {
      const st = $('#f-stat').value, src = P.daily[H] || {};
      dates.forEach((d) => {
        const r = src[d] && src[d][st], y = P.dailyActual[d] ? P.dailyActual[d][st] : null;
        s.p.push(r ? r[0] : null); s.lo.push(r ? r[1] : null); s.hi.push(r ? r[2] : null); s.y.push(y); labs.push([d, null]);
        rows.push({ d, p: r ? r[0] : null, lo: r ? r[1] : null, hi: r ? r[2] : null, y });
      });
    }
    const per = kind === 'mo' ? 1 : 24;
    const shade = [];
    if (kind !== 'mo') dates.forEach((d, k) => shade.push([k * 24 + 10, k * 24 + 14]));
    else dates.forEach((d, k) => { if (info(d).off) shade.push([k, k]); });
    const step = innerWidth > 700 ? 1 : 2;
    chart('#f-chart', '#f-tip', s, { xTicks: dates.map((d, k) => [k * per + (per - 1) / 2, lab(d)]).filter((_, k) => k % step === 0),
      shade, dividers: kind !== 'mo' ? dates.map((_, k) => k * 24).slice(1) : [],
      tipLabel: (i) => `${lab(labs[i][0])}${labs[i][1] != null ? ' ' + (labs[i][1] + 1) + '시' : ''}`, dot: kind === 'mo' ? 3.5 : (dates.length <= 7 ? 1.6 : 0) });
    const idx = s.p.map((p, i) => (p != null && s.y[i] != null ? i : -1)).filter((i) => i >= 0);
    const mae = avg(idx.map((i) => Math.abs(s.y[i] - s.p[i])));
    const cov = idx.length ? idx.filter((i) => inside(s.y[i], s.lo[i], s.hi[i])).length / idx.length : null;
    const below = idx.length ? idx.filter((i) => s.y[i] < s.lo[i]).length / idx.length : null;
    const bias = avg(idx.map((i) => s.y[i] - s.p[i]));
    const covOk = cov != null && cov >= 0.75 && cov <= 0.85;
    $('#f-kpis').innerHTML = [[`비교한 ${kind === 'mo' ? '날' : '시간'}`, idx.length, `${lab(fS)} ~ ${lab(fE)}`, ''],
      ['평균 오차 (MAE)', fmt(mae) + '원', `편향(공표−예측) ${sgn(bias)}원`, ''],
      ['80% 범위 포함률', pct(cov), '목표 75~85%', cov == null ? '' : covOk ? 'good' : 'bad'],
      ['하한 아래 / 상한 위', `${pct(below)} / ${pct(cov == null ? null : 1 - cov - below)}`, '각각 약 10%가 균형', ''],
      ['평균 범위 폭', fmt(avg(idx.map((i) => s.hi[i] - s.lo[i]))) + '원', '', '']]
      .map(([k, v, sub, c]) => `<div class="kpi"><div class="k">${k}</div><div class="v ${c}">${v}</div><div class="s">${sub}</div></div>`).join('');
    let h;
    if (kind !== 'mo') {
      h = `<tr><th class="l">날짜</th><th>예측 일평균</th><th>공표 일평균</th><th>MAE</th><th>포함률</th><th>예측 한낮 최저</th><th>공표 한낮 최저</th>${kind === 'd1' ? '<th class="l">급락 경고 (과제1)</th>' : ''}</tr>`;
      rows.forEach((r) => {
        const dr = r.drop, hit = r.ymin != null && r.ymin < 30;
        h += `<tr class="${info(r.d).off ? 'off' : ''}"><td class="l">${lab(r.d)} ${info(r.d).dt && info(r.d).dt !== '평일' ? info(r.d).dt : ''}</td><td>${fmt(r.p)}</td><td class="act">${fmt(r.y)}</td>
          <td>${fmt(r.mae)}</td><td class="${r.cov == null ? '' : r.cov >= 0.75 ? 'in' : 'out'}">${pct(r.cov)}</td><td>${fmt(r.pmin)}</td><td class="act">${fmt(r.ymin)}${hit ? ' <span class="badge strong sm">급락</span>' : ''}</td>
          ${kind === 'd1' ? `<td class="l">${dr ? dropBadge(dr, true) : '-'}</td>` : ''}</tr>`;
      });
    } else {
      h = '<tr><th class="l">날짜</th><th>하한</th><th>예측</th><th>상한</th><th>공표</th><th>오차(공표−예측)</th><th class="l">범위</th></tr>';
      rows.forEach((r) => { h += `<tr class="${info(r.d).off ? 'off' : ''}"><td class="l">${lab(r.d)} ${info(r.d).dt && info(r.d).dt !== '평일' ? info(r.d).dt : ''}</td><td>${fmt(r.lo)}</td><td>${fmt(r.p)}</td><td>${fmt(r.hi)}</td>
        <td class="act">${fmt(r.y)}</td><td>${r.y != null && r.p != null ? sgn(r.y - r.p) : '-'}</td><td class="l ${r.y == null ? '' : inside(r.y, r.lo, r.hi) ? 'in' : 'out'}">${r.y == null ? '-' : inside(r.y, r.lo, r.hi) ? '안' : '밖'}</td></tr>`; });
    }
    $('#f-table').innerHTML = h;
    $('#f-note').textContent = kind === 'd1' ? '다음 날 예측 = 대상일 하루 전 15시에 낸 예측 (일주일 모델의 D+1). 급락 경고는 과제1 모델이 같은 시각에 낸 등급·확률입니다.'
      : kind === 'wk' ? `'${H}일 전' = 대상일보다 ${H}일 앞선 기준일 15시에 낸 예측입니다.`
        : `한 달 예측은 일별 지표(${$('#f-stat').selectedOptions[0].text})로 비교합니다. ${H}일 전 기준일 15시에 낸 예측입니다.`;
  }
  renderers.flow = () => { if (!fS) { fillFlowH(); fE = lastAct; fS = addDays(lastAct, -MAXD + 1); } renderFlow(); };

  // ---------- 참고 문서 (개요 · 다음 날 · 일주일 · 한 달) ----------
  function tbl(rows, cols) {
    let h = '<div class="tscroll"><table><tr>' + cols.map((c) => `<th class="${c.l ? 'l' : ''} ${c.w ? 'wrap' : ''}">${c.t}</th>`).join('') + '</tr>';
    rows.forEach((r) => { h += `<tr class="${r._sel ? 'sel' : ''}">` + cols.map((c) => `<td class="${c.l ? 'l' : ''} ${c.w ? 'wrap' : ''} ${c.cls ? c.cls(r) : ''}">${c.f ? c.f(r[c.k], r) : r[c.k] ?? '-'}</td>`).join('') + '</tr>'; });
    return h + '</table></div>';
  }
  const n2 = (v) => fmt(v, 2), n3 = (v) => fmt(v, 3);
  const covCls = (r) => (r['포함률'] == null ? '' : r['포함률'] >= 0.75 && r['포함률'] <= 0.85 ? 'in' : 'out');
  const kpiRow = (items) => `<div class="kpis">${items.map(([k, v, s, c]) => `<div class="kpi"><div class="k">${k}</div><div class="v ${c || ''}">${v}</div><div class="s">${s || ''}</div></div>`).join('')}</div>`;
  const DOC = D.docs || {}, MET = DOC.metrics || [], FIN = DOC.final || {};
  const met = (pred, per, seg) => MET.find((r) => r['예측'] === pred && r['기간'].startsWith(per) && r['구분'] === seg) || {};
  const METCOLS = [{ k: '구분', t: '구분', l: 1 }, { k: '표본수', t: '표본 수' }, { k: 'MAE', t: 'MAE', f: n2 }, { k: 'RMSE', t: 'RMSE', f: n2 },
    { k: 'MAPE(20원 이상)', t: 'MAPE', f: pct }, { k: 'rMAE', t: 'rMAE', f: n3 }, { k: '편향(공표−예측)', t: '편향', f: sgn },
    { k: '포함률', t: '포함률', f: pct, cls: covCls }, { k: '하한 아래', t: '하한 아래', f: pct }, { k: '상한 위', t: '상한 위', f: pct },
    { k: '평균 폭', t: '평균 폭', f: fmt }, { k: '구간점수', t: '구간점수', f: fmt }];
  const metTable = (pred, per, segs) => tbl(MET.filter((r) => r['예측'] === pred && r['기간'].startsWith(per) && (!segs || segs(r['구분']))), METCOLS);
  const testRange = FIN['시험 기간'] || '2026-01-01 ~';
  const WZC = (D.perf.week || {}).choice || {}, MZC = (D.perf.month || {}).choice || {};

  const DATA_HOURLY = [
    ['시간별 SMP (육지)', 'Chronos 입력: 최근 2,048시간(약 85일). 잔차 보정: 어제·최근 같은 유형일·28일 평균 대비 차이', '기준일 SMP는 전날 저녁 공표 → 기준일 15시에 확보'],
    ['달력 (휴무일·요일유형·명절·근로자의날, 앞뒤 휴무)', 'Chronos 공변량(과거·미래), 잔차 보정 피처', '사전 확정 (천문연구원 특일정보)'],
    ['전력거래소 주간 전력수요 전망', '대상일 최대부하, 전망 내 비율, 전날 대비', '게시 다음 날부터 사용 (게시판 이미지 OCR + 전력정보앱)'],
    ['기상청 단기 육상예보 (하늘·강수확률·기온)', 'D+1~D+3 대상일 오전·오후', '기준일 11시(없으면 05시) 발표분'],
    ['기상청 중기 육상예보 (하늘·강수확률)', 'D+4~D+7 (기온 없음)', '기준일 06시 발표분'],
    ['Open-Meteo 일사·운량 예보 (ECMWF·GFS·JMA, 7개 도시)', '한낮 11~15시 일사 합·운량 평균, D+1만', '대상일 2일 전 실행분'],
    ['EPSIS 연료비 단가 (LNG·유연탄)', '월 값', 'M월 값은 M+2월 1일부터 사용'],
    ['시장참여 태양광 설비용량', '월 값', 'M월 값은 M+2월 1일부터 사용'],
  ];
  const NOT_USED = '<ul><li><b>대상일 하루전 수요예측·SMP</b>: 평일 19~20시, 주말 14시 이전 공표라 15시 판단에 쓸 수 없습니다 (공표경계 판정서). D-1까지의 실적만 씁니다.</li>'
    + '<li><b>실측 날씨</b>(일사량·운량 실측): 판단 시각에 알 수 없는 값이라 쓰지 않았습니다.</li>'
    + '<li><b>담당자 운영 모델의 예측값</b>: 웹앱은 자체 모델만 씁니다 (재배포 금지 조건).</li>'
    + '<li><b>전력수급예보 API</b>: 2026-10-22 승인. 내일 1건만 제공하고 과거 조회가 안 돼 매일 쌓는 중입니다. 피처로는 아직 쓰지 않습니다 (향후 후보).</li></ul>';
  const dataTable = (rows) => tbl(rows.map(([a, b, c]) => ({ a, b, c })), [{ k: 'a', t: '자료', l: 1, w: 1 }, { k: 'b', t: '쓰는 방식', l: 1, w: 1 }, { k: 'c', t: '확보 시점 (판단: 기준일 15시)', l: 1, w: 1 }]);

  function docOverview() {
    const r = (p, s) => met(p, '2026', s);
    const rows = [
      { a: '급락 판별 (내일 한낮)', m: 'Ridge + EBM + 랜덤포레스트 평균 → 예상 한낮 최저 SMP → 문턱 90.1원(주의)·75.3원(강한 경보), 확률 보정', s: '급락 20일 중 주의 이상 18일(90%) 적중, 헛경보 72일 · 강한 경보 16일(80%), 헛경보 27일 · ROC-AUC 0.93' },
      { a: '다음 날 (D+1, 시간별)', m: 'Chronos-2 small + 0.5 × 잔차 보정 (일주일 모델의 D+1)', s: `MAE ${n2(r('다음 날', '전체').MAE)}원 · 포함률 ${pct(r('다음 날', '전체')['포함률'])} · 폭 ${fmt(r('다음 날', '전체')['평균 폭'])}원` },
      { a: '일주일 (D+1~D+7, 시간별)', m: `Chronos-2 small + ${WZC['잔차 보정 λ'] ?? 0.5} × 잔차 보정(외부 정보 LightGBM)`, s: `MAE ${n2(r('일주일', '전체').MAE)}원 · 포함률 ${pct(r('일주일', '전체')['포함률'])} · 폭 ${fmt(r('일주일', '전체')['평균 폭'])}원` },
      { a: '한 달 (7~34일 뒤, 일별)', m: '기준(최근 같은 유형일 4일 평균) + λ × (LSTM·GRU 평균 − 기준), λ = 일평균 0.25 · 한낮·저녁·한낮 최저 0.5', s: `일평균 MAE ${n2(r('한 달', '일평균').MAE)}원 · 포함률 ${pct(r('한 달', '일평균')['포함률'])} (참고 전망)` },
    ];
    const head = ['다음 날', '일주일'].map((p) => ({ ...r(p, '전체'), 예측: p })).concat(['일평균', '한낮 11~15시', '저녁 18~22시', '한낮 최저'].map((s) => ({ ...r('한 달', s), 예측: '한 달 · ' + s })));
    return `<div class="doc">
      <h2>최종 모델 (2026-10-23 확정)</h2>
      <p class="lead">모든 모델은 <b>기준일 15시에 알 수 있는 정보만</b> 쓰고, <b>2025년 성적만으로</b> 골랐습니다. 2026년(${testRange})은 고른 뒤 채점만 했습니다.</p>
      ${tbl(rows, [{ k: 'a', t: '예측', l: 1 }, { k: 'm', t: '최종 모델', l: 1, w: 1 }, { k: 's', t: '2026 시험 성적', l: 1, w: 1 }])}
      <h2>팀 비교용 핵심 지표 <span class="muted">2026 시험 · 같은 정의로 다시 계산하면 비교할 수 있습니다</span></h2>
      ${tbl(head, [{ k: '예측', t: '예측', l: 1 }, ...METCOLS.slice(1)])}
      <p class="note">한 달 예측은 일별 지표(일평균 등)를 7~34일 뒤 전체로 묶은 값입니다. 하루 단위라 표본 수는 '기준일 × 대상일' 수입니다.</p>
      <h2>성적 지표 읽는 법</h2>
      <dl class="defs">
        <dt>MAE (원)</dt><dd>공표 SMP와 예측의 차이(절댓값) 평균. 작을수록 좋음. 협의된 주 지표.</dd>
        <dt>RMSE (원)</dt><dd>차이 제곱 평균의 제곱근. 크게 틀린 시간이 많을수록 커짐.</dd>
        <dt>MAPE</dt><dd>차이 ÷ 공표 SMP의 평균. SMP가 0원 근처일 때 값이 터지므로 <b>공표 20원 이상 시간만</b> 계산 (현업 운영 기준과 같음).</dd>
        <dt>rMAE</dt><dd>MAE ÷ 단순 기준의 MAE. 1보다 작으면 단순 기준보다 낫다는 뜻. 단순 기준: 다음 날 = 전날 같은 시간, 일주일 = 지난주 같은 시간, 한 달 = 같은 유형일 4일 평균.</dd>
        <dt>편향 (원)</dt><dd>평균(공표 − 예측). +면 예측이 낮게, −면 높게 나오는 경향.</dd>
        <dt>포함률</dt><dd>80% 범위 안에 공표 SMP가 든 비율. 성공 기준 <b>75~85%</b>.</dd>
        <dt>하한 아래 / 상한 위</dt><dd>범위를 벗어난 방향. 각각 약 10%면 균형.</dd>
        <dt>평균 폭 (원)</dt><dd>상한 − 하한 평균. 포함률이 같으면 좁을수록 좋음.</dd>
        <dt>구간점수 (Winkler)</dt><dd>폭 + 벗어난 만큼의 벌점(× 10). 포함률과 폭을 한 숫자로 묶은 값, 낮을수록 좋음. 범위 방법 선택 규칙에 사용.</dd>
      </dl>
      <h2>평가 규칙</h2>
      <ul>
        <li><b>판단 시각 = 기준일 15시</b> (현업 주간 예측과 같음). 대상일 하루전 수요예측·SMP는 저녁 공표라 쓰지 않습니다.</li>
        <li><b>시간 순서만</b>: 매월 1일 재학습하며 그 전까지의 자료로만 학습 (랜덤 분할 없음). 2025~2026 모든 예측이 '그 시점에 낼 수 있었던 예측'입니다.</li>
        <li><b>선택</b>: 2025 상반기로 순위·가중치·보정 강도를 맞추고 → 2025 하반기 성적으로 비교 (0.05원 이내면 단순한 쪽) → 2025 전체로 다시 맞춤.</li>
        <li><b>시험</b>: 2026-01-01 이후 한 번 채점. 모델을 바꿔 다시 채점한 경우 '2차·3차 시험'으로 표기합니다 (일주일 3차, 한 달 3차).</li>
      </ul>
      <h2>쓰지 않은 정보</h2>${NOT_USED}
    </div>`;
  }

  function docNextDay() {
    const t = met('다음 날', '2026', '전체'), v = met('다음 날', '2025', '전체');
    const dz = DOC.dropZoo || [], dt = DOC.dropTest || [];
    const nd = (DOC.nextday || []).map((r) => ({ ...r, _sel: String(r['모델']).includes('최종') }));
    const thr = (D.perf.dropChoice || {})['등급 문턱 확률'] || {};
    return `<div class="doc">
      <h2>다음 날 예측 <span class="muted">기준일 15시 → 내일 1~24시 SMP + 80% 범위 + 한낮 급락 확률</span></h2>
      ${kpiRow([['MAE (2026)', n2(t.MAE) + '원', `2025 하반기 ${n2(v.MAE)}원`], ['rMAE', n3(t.rMAE), `전날 같은 시간 ${n2(t['단순 기준 MAE'])}원 대비`],
        ['80% 범위 포함률', pct(t['포함률']), '목표 75~85%', t['포함률'] >= 0.75 && t['포함률'] <= 0.85 ? 'good' : 'bad'], ['평균 폭', fmt(t['평균 폭']) + '원', `하한 아래 ${pct(t['하한 아래'])} · 상한 위 ${pct(t['상한 위'])}`],
        ['급락 판별 (2026)', '18/20일', '주의 이상 적중 · 헛경보 72일']])}
      <h2>최종 모델: Chronos-2 small + 잔차 보정</h2>
      <ol>
        <li><b>Chronos-2 small</b> (Amazon 공개 시계열 기초모델, 추가 학습 없음): 최근 85일 시간별 SMP와 휴무일 달력을 넣으면 앞으로 7일의 중앙값·분위수를 냅니다. 다음 날 예측은 그중 첫날입니다.</li>
        <li><b>잔차 보정</b> (LightGBM): Chronos는 SMP 과거값과 달력만 봅니다. 그래서 <b>Chronos가 틀린 만큼(공표 − Chronos)</b>을 수요전망·날씨예보·연료비·태양광 설비로 예측하고, 그 절반(λ = 0.5)을 더합니다. 매월(운영은 매주) 재학습.</li>
        <li><b>80% 범위</b>: Chronos-2 분위수(10%·90%)를 최근 90일 실제 이탈 폭으로 넓히거나 좁힘 (보정 분위수 회귀, 시간대 묶음 × 휴무일별).</li>
      </ol>
      <h2>비교한 모델과 성능 <span class="muted">같은 날짜·시간끼리 MAE (원)</span></h2>
      ${tbl(nd, [{ k: '모델', t: '모델', l: 1 }, { k: '2025 상반기', t: '2025 상반기', f: n2 }, { k: '2025 하반기', t: '2025 하반기', f: n2 }, { k: '2025 전체', t: '2025 전체 (선택)', f: n2 }, { k: '2026 (시험)', t: '2026 (시험)', f: n2 }])}
      <p class="note">일주일 모델 비교(18종 + 앙상블 23종)의 D+1 열도 같은 결론입니다 — 일주일 예측 페이지 참고.</p>
      <h2>선택 이유</h2>
      <ul>
        <li>2025년 MAE 7.80원으로 이전 내일 전용 모델(8.80원)보다 1원(11%) 낮습니다. 2026 시험에서도 8.19원 vs 8.97원.</li>
        <li>내일 전용 모델은 부스팅(LightGBM·CatBoost)이라 2025 하반기 수준 변화에 늦게 따라갔지만(상반기 8.21 → 하반기 9.38), Chronos는 매일 최근 흐름을 읽어 흔들림이 적습니다 (7.78 → 7.82).</li>
        <li>일주일 예측과 같은 모델이라 웹의 내일·일주일 값이 서로 어긋나지 않습니다.</li>
      </ul>
      <h2>사용 데이터</h2>${dataTable(DATA_HOURLY)}
      <h2>급락 판별 (과제1 모델) <span class="muted">내일 11~15시 최저 SMP가 30원 미만인지</span></h2>
      <ul>
        <li><b>모델</b>: Ridge + EBM + 랜덤포레스트가 각각 '내일 한낮 최저 SMP'를 예측하고 평균. 2025 교차검증에서 단일 31종 + 앙상블 25종 중 1위 (급락 90%를 잡을 때 헛경보 25일).</li>
        <li><b>등급</b>: 예상 한낮 최저 90.1원 이하 = <b>주의</b>, 75.3원 이하 = <b>강한 경보</b> (2025 급락의 90%를 잡는 문턱).</li>
        <li><b>급락 확률</b>: 예상 한낮 최저를 로지스틱식으로 확률로 바꿈 (2025로 맞춤). 주의 ≈ ${pct(thr['주의(90.1원)'])}, 강한 경보 ≈ ${pct(thr['강한 경보(75.3원)'])}부터.</li>
        <li><b>쓰는 정보</b>: 최근 SMP 흐름, 휴무일·연휴·명절, D-1까지의 하루전 수요예측, 주간 전력수요 전망, 하늘상태·강수 예보, 태양광 설비용량.</li>
      </ul>
      ${tbl(dt.filter((r) => r['경보'] !== '주의만'), [{ k: '방법', t: '방법', l: 1 }, { k: '경보', t: '경보', l: 1 }, { k: '구간', t: '구간', l: 1 }, { k: '일수', t: '일수' }, { k: '급락일', t: '급락일' },
        { k: '적중', t: '적중' }, { k: '놓침', t: '놓침' }, { k: '헛경보', t: '헛경보' }, { k: '재현율', t: '재현율', f: pct }, { k: '정밀도', t: '정밀도', f: pct }, { k: 'ROC-AUC', t: 'ROC-AUC', f: n3 }])}
      <h3>급락 판별 후보 비교 (2025 교차검증, 상위 15)</h3>
      ${tbl(dz.map((r, i) => ({ ...r, _sel: i === 0 })), [{ k: '구분', t: '구분', l: 1 }, { k: '모델', t: '모델', l: 1 }, { k: '헛경보@80%', t: '헛경보 (급락 80% 잡을 때)' }, { k: '헛경보@90%', t: '헛경보 (90%)' }, { k: '헛경보@100%', t: '헛경보 (100%)' }, { k: 'PR-AUC', t: 'PR-AUC', f: n3 }, { k: 'ROC-AUC', t: 'ROC-AUC', f: n3 }])}
      <h3>급락 확률은 믿을 만한가</h3>
      ${tbl((D.perf.dropCal || []).filter((r) => r['구분'] !== '전체 Brier'), [{ k: '연도', t: '연도', l: 1 }, { k: '구분', t: '구분', l: 1 }, { k: '일수', t: '일수' }, { k: '평균 확률', t: '표시한 평균 확률', f: pct }, { k: '실제 급락 비율', t: '실제 급락 비율', f: pct }, { k: '급락일', t: '급락일' }])}
      <p class="note">2026년에는 높은 확률 구간에서 실제보다 높게 표시되는 경향이 있어, 배지에 등급별 2026 실제 급락 비율을 함께 적습니다.</p>
      <h2>2026 시험 성적표 <span class="muted">${testRange}</span></h2>${metTable('다음 날', '2026')}
      <h3>2025 하반기 (선택 구간)</h3>${metTable('다음 날', '2025')}
      <h2>한계</h2>
      <ul><li>휴무일 한낮·급락 시간은 오차가 큽니다 (위 표). 0원 근처 급락은 범위가 잘 담지 못하므로 급락 확률을 함께 봐야 합니다.</li>
        <li>평일 급락은 2026년 2일뿐이라 평일 성적은 참고 수준입니다.</li></ul>
    </div>`;
  }

  function docWeek() {
    const t = met('일주일', '2026', '전체');
    const W = D.perf.week || {};
    const ens = (W.ens || []).map((r) => ({ ...r, _sel: String(r['앙상블']).startsWith(WZC['선택 앙상블'] || '@') }));
    const hs = [1, 2, 3, 4, 5, 6, 7].map((h) => met('일주일', '2026', 'D+' + h));
    return `<div class="doc">
      <h2>일주일 예측 <span class="muted">기준일 15시 → 1~7일 뒤 시간별 SMP + 80% 범위</span></h2>
      ${kpiRow([['MAE (2026)', n2(t.MAE) + '원', `D+1 ${n2(hs[0].MAE)} → D+7 ${n2(hs[6].MAE)}`], ['rMAE', n3(t.rMAE), `지난주 같은 시간 ${n2(t['단순 기준 MAE'])}원 대비`],
        ['80% 범위 포함률', pct(t['포함률']), '목표 75~85%', t['포함률'] >= 0.75 && t['포함률'] <= 0.85 ? 'good' : 'bad'], ['평균 폭', fmt(t['평균 폭']) + '원', `D+1 ${fmt(hs[0]['평균 폭'])} → D+7 ${fmt(hs[6]['평균 폭'])}`],
        ['비교한 후보', '18 + 23', '단일 모델 + 앙상블·보정']])}
      <h2>최종 모델: Chronos-2 small + ${WZC['잔차 보정 λ'] ?? 0.5} × 잔차 보정 (${WZC['잔차 보정 피처 묶음'] || '외부만'})</h2>
      <ul>
        <li><b>Chronos-2 small</b>: 최근 85일 시간별 SMP + 휴무일 달력 → 168시간 중앙값. 우리 자료로 학습하지 않는 사전학습 모델이라 매일 최근 흐름에 바로 맞춥니다.</li>
        <li><b>잔차 보정</b>: 며칠 뒤(H)마다 따로 학습한 LightGBM(L1)이 'Chronos가 틀린 만큼'을 외부 정보로 예측. 학습 이력은 Chronos의 2023~ 과거 예측.</li>
        <li><b>80% 범위</b>: D+1은 Chronos-2 분위수를 최근 90일 이탈로 보정, D+2~D+7은 Chronos-2 분위수 그대로 (2025 하반기 구간점수 최소).</li>
      </ul>
      <h2>선택 이유</h2>
      <ul>
        <li>단일 모델 18종 중 Chronos 계열 3종이 1~3위. 학습형 모델(부스팅·신경망)은 2025 하반기 수준 변화에 크게 나빠졌습니다 (LightGBM 9.72 → 10.94, CatBoost 9.81 → 14.23).</li>
        <li>Chronos에 LSTM·GRU·LightGBM을 <b>그대로 섞으면 오히려 나빠짐</b> (9.49~9.59 vs 단독 9.41): 성능 차이가 2원 가까이 나고 오차 상관이 0.79로 높아 서로 보완하지 못합니다.</li>
        <li>대신 <b>Chronos가 모르는 외부 정보로 Chronos의 오차를 고치는</b> 잔차 보정이 2025 하반기 9.21원으로 1위 (단독 대비 −2.2%). 모든 지평·휴무일·저녁에서 개선, 2026 시험에서도 9.81 → ${n2(t.MAE)}원.</li>
      </ul>
      <h2>비교한 단일 모델 <span class="muted">2025년 MAE (원), 모두 같은 조건</span></h2>
      ${tbl((W.models || []).map((r, i) => ({ ...r, _sel: i === 0 })), [{ k: '모델', t: '모델', l: 1 }, { k: '2025 상반기', t: '상반기', f: n2 }, { k: '2025 하반기', t: '하반기', f: n2 }, { k: '2025 전체', t: '전체', f: n2 },
        ...[1, 2, 3, 4, 5, 6, 7].map((x) => ({ k: `2025 D+${x}`, t: `D+${x}`, f: n2 }))])}
      <p class="note">재학습: LightGBM·XGBoost·Ridge 매월, CatBoost·랜덤포레스트·엑스트라트리·EBM·MLP·TabPFN 분기, LSTM·GRU·1D-CNN·Transformer는 과거 7일 입력 소형(CPU 시간 제약). 기준 2종은 단순 규칙.</p>
      <h2>앙상블·보정 후보 <span class="muted">2025 상반기로 맞추고 하반기 MAE로 비교</span></h2>
      ${tbl(ens, [{ k: '앙상블', t: '후보', l: 1 }, { k: '구성 수', t: '모델 수' }, { k: '2025 하반기', t: '2025 하반기 MAE', f: n3 }, { k: '구성', t: '구성', l: 1, w: 1 }])}
      <h2>사용 데이터</h2>${dataTable(DATA_HOURLY)}
      <h3>쓰지 않은 정보</h3>${NOT_USED}
      <h2>2026 시험 성적표 (일주일 3차 시험) <span class="muted">${testRange}</span></h2>
      ${metTable('일주일', '2026', (s) => !s.startsWith('D+'))}
      <h3>며칠 뒤별</h3>${metTable('일주일', '2026', (s) => s.startsWith('D+'))}
      <h3>2025 하반기 (선택 구간)</h3>${metTable('일주일', '2025', (s) => !s.startsWith('D+'))}
      <h2>한계</h2>
      <ul><li>멀리 볼수록 상한 위 이탈이 늘어납니다 (D+7 약 15%) — 갑작스런 가격 상승을 범위가 늦게 따라갑니다.</li>
        <li>'3차 시험'은 2026을 앞서 두 번 채점한 뒤의 결과라, 1차 시험보다 낙관적일 수 있습니다.</li></ul>
    </div>`;
  }

  function docMonth() {
    const M = D.perf.month || {};
    const SNM = { mean: '일평균', mid: '한낮 11~15시', eve: '저녁 18~22시', midmin: '한낮 최저' };
    const t = Object.fromEntries(Object.values(SNM).map((s) => [s, met('한 달', '2026', s)]));
    const models = (M.models || []);
    const piv = {};
    models.forEach((r) => { (piv[r['모델']] = piv[r['모델']] || { 모델: r['모델'] })[r['지표']] = r['2025 전체']; });
    const prow = Object.values(piv).sort((a, b) => a.mean - b.mean);
    const ch = MZC['지표별'] || {};
    return `<div class="doc">
      <h2>한 달 예측 <span class="muted">기준일 15시 → 7~34일 뒤 일별 4개 지표 + 80% 범위 · 참고 전망</span></h2>
      ${kpiRow(Object.entries(t).map(([s, r]) => [s, n2(r.MAE) + '원', `포함률 ${pct(r['포함률'])} · rMAE ${n3(r.rMAE)}`, r['포함률'] >= 0.75 ? 'good' : 'bad']).concat([['비교한 후보', '15 + 앙상블', '지표마다 따로 선택']]))}
      <h2>최종 모델: 기준 + λ × (LSTM·GRU 평균 − 기준)</h2>
      <ul>
        <li><b>기준</b>: 대상일과 같은 유형(평일/휴무일)인 최근 4일의 지표 평균.</li>
        <li><b>LSTM·GRU</b>: 최근 56일 4개 지표 + 앞으로 34일 달력을 넣어 28일 × 4개 지표를 한 번에 예측 (시드 3개씩 평균, 운영은 매주 재학습).</li>
        <li><b>λ</b>: 기준을 신경망 쪽으로 얼마나 당길지 — ${Object.entries(ch).map(([s, v]) => `${SNM[s]} ${(String(v['점예측']).match(/\+ ([0-9.]+)/) || [])[1] || '-'}`).join(' · ')}.</li>
        <li><b>80% 범위</b>: 과거 오차의 주차 × 휴무일별 분위수 — ${Object.entries(ch).map(([s, v]) => `${SNM[s]} ${v['범위']}`).join(' · ')}.</li>
      </ul>
      <h2>선택 이유</h2>
      <ul>
        <li>1~4주 앞은 연료비·수급 국면 변화가 가격을 좌우해, <b>기준(같은 유형일 평균)을 확실히 이기는 단일 모델이 없었습니다</b> (일평균 2025: 기준 9.09 vs Chronos-2 9.12, 엑스트라트리 9.30).</li>
        <li>기준을 LSTM·GRU 쪽으로 조금 당기는 조합이 2025 하반기 네 지표 모두 1위 (일평균 8.56 vs 기준 8.79). 일주일과 달리 한 달에서는 LSTM 계열이 실제로 보탬이 됐습니다.</li>
        <li>2026 시험: 한낮(+4.7%)·한낮 최저(+8.8%)는 기준보다 좋았지만 일평균(−2.9%)·저녁(−14.9%)은 나빴습니다. 2026을 보고 다시 고치지 않고 <b>참고 전망</b>으로 표시합니다.</li>
      </ul>
      <h2>비교한 단일 모델 <span class="muted">2025년 MAE (원), 일평균 순</span></h2>
      ${tbl(prow, [{ k: '모델', t: '모델', l: 1 }, ...Object.entries(SNM).map(([k, v]) => ({ k, t: v, f: n2 }))])}
      <h2>앙상블 후보 상위 <span class="muted">지표별 2025 하반기 MAE</span></h2>
      ${tbl(Object.keys(SNM).flatMap((s) => (M.ens || []).filter((r) => r['지표'] === s).sort((a, b) => a['2025 하반기'] - b['2025 하반기']).slice(0, 4)
        .map((r) => ({ ...r, 지표: SNM[s], _sel: ch[s] && r['앙상블'] === ch[s]['점예측'] }))), [{ k: '지표', t: '지표', l: 1 }, { k: '앙상블', t: '후보', l: 1, w: 1 }, { k: '구성 수', t: '모델 수' }, { k: '2025 하반기', t: '2025 하반기 MAE', f: n3 }])}
      <p class="note">선택 규칙: 1위와 0.05원 이내면 구성 모델이 적은 쪽.</p>
      <h2>사용 데이터</h2>
      ${dataTable([['일별 SMP 지표 (일평균·한낮 11~15시·저녁 18~22시·한낮 최저)', '최근 7·28일 평균, 같은 유형일 4일 평균, 최근 56일 시계열(LSTM·GRU)', '기준일까지 공표분'],
        ['달력', '대상일 요일·월·휴무·연휴·명절, 앞뒤 휴무', '사전 확정'],
        ['작년 계절 비율', '작년 대상일 ±3일 평균 ÷ 작년 기준일 직전 28일 (SMP·수요예측)', '작년 실적이라 확정'],
        ['하루전 수요예측 (실적)', '최근 7일 일평균', '기준일분까지 (전날 저녁 공표)'],
        ['연료비 단가 (LNG·유연탄)·태양광 설비', '월 값과 3개월 전 대비 변화', 'M월 값은 M+2월 1일부터']])}
      <h2>2026 시험 성적표 (한 달 3차 시험) <span class="muted">${testRange}</span></h2>
      ${metTable('한 달', '2026', (s) => !s.includes('주차'))}
      <h3>주차별</h3>${metTable('한 달', '2026', (s) => s.includes('주차'))}
      <h3>2025 하반기 (선택 구간)</h3>${metTable('한 달', '2025', (s) => !s.includes('주차'))}
      <h2>한계</h2>
      <ul><li>한낮·한낮 최저의 포함률이 70% 안팎으로 목표(75%)에 못 미칩니다. 멀수록(4주차) 더 낮습니다.</li>
        <li>개선에는 연료비 전망·수급 국면 정보가 필요하며, 공개 자료만으로는 한계가 있습니다.</li></ul>
    </div>`;
  }

  const DOCS = { ov: docOverview, d1: docNextDay, wk: docWeek, mo: docMonth };
  let dPage = 'ov';
  $$('#d-page button').forEach((b) => b.addEventListener('click', () => {
    dPage = b.dataset.p; $$('#d-page button').forEach((x) => x.classList.toggle('on', x === b)); renderers.docs();
  }));
  renderers.docs = () => {
    $('#docs').innerHTML = DOCS[dPage]();
    $('#d-sub').textContent = `확정 ${FIN['확정일'] || '-'} · 시험 ${testRange}`;
  };

  window.addEventListener('resize', () => { const t = $('.tab.on').dataset.tab; if (renderers[t]) renderers[t](); });
  $('#pub').innerHTML = `최신 공표 SMP <b>${D.lastPublished} (${wd(D.lastPublished)})</b>` + (D.lastRun ? ` · 최근 예측 산출 <b>${D.lastRun}</b>` : '') + ` · 갱신 ${D.generated}`;
  $('#foot').textContent = `출처: 전력거래소·기상청·한국천문연구원·EPSIS 공공데이터 · 숙명여대 P4 산학 프로젝트 · 데이터 생성 ${D.generated}`;
  renderWeek();
  const [hash, sub] = (location.hash || '').slice(1).split('-');   // 예: #docs-wk, #flow
  if (sub && DOCS[sub]) { dPage = sub; $$('#d-page button').forEach((x) => x.classList.toggle('on', x.dataset.p === sub)); }
  if (['month', 'flow', 'docs', 'help'].includes(hash)) $(`.tab[data-tab="${hash}"]`).click();
})();
