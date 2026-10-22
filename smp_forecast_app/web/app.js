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

  // ---------- 급락 배지 (과제1 → 확률 보정) ----------
  const GRADE_CLS = { '강한 경보': 'strong', '주의': 'warn', '없음': 'low' };
  const GRADE_TXT = { '강한 경보': '급락 강한 경보', '주의': '급락 주의', '없음': '급락 가능성 낮음' };
  const realRate = {};
  ((D.perf && D.perf.dropCal) || []).filter((r) => r['연도'] === 2026 && String(r['구분']).startsWith('등급 '))
    .forEach((r) => { realRate[r['구분'].slice(3)] = r; });
  function dropBadge(dr, compact) {
    if (!dr) return '';
    const cls = GRADE_CLS[dr.grade] || 'low', p = dr.prob == null ? '-' : Math.round(dr.prob * 100) + '%';
    if (compact) return `<span class="badge ${cls} sm">${GRADE_TXT[dr.grade] || dr.grade} ${p}</span>`;
    const rr = realRate[dr.grade];
    const note = rr ? ` · 2026년 이 등급 날 실제 급락 ${Math.round(rr['실제 급락 비율'] * 100)}% (${rr['급락일']}/${rr['일수']}일)` : '';
    return `<span class="badge ${cls}"><b>내일 ${GRADE_TXT[dr.grade] || dr.grade}</b> · 한낮 급락 확률 <b class="prob">${p}</b>`
      + `<span class="sub">과제1 예상 한낮 최저 ${fmt(dr.min)}원${note}</span></span>`;
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
        ${i === 0 && D.week.drop && D.week.drop.grade !== '없음' ? dropBadge(D.week.drop, true) : ''}</button>`;
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

  // ---------- 지난 예측 ----------
  const P = D.past;
  const lastAct = Object.keys(P.actual).sort().pop();
  const setRange = (days) => { $('#p-to').value = lastAct; $('#p-from').value = addDays(lastAct, -days + 1); renderPast(); };
  $$('#tab-past [data-q]').forEach((b) => b.addEventListener('click', () => setRange(+b.dataset.q)));
  function fillH() {
    const kind = $('#p-kind').value;
    const hs = kind === 'h' ? Object.keys(P.hourly).map(Number).sort((a, b) => a - b) : Object.keys(P.daily).map(Number).sort((a, b) => a - b);
    $('#p-h').innerHTML = hs.map((h) => `<option value="${h}">${kind === 'h' ? 'D+' + h + ' (' + h + '일 전)' : h + '일 전'}</option>`).join('');
    $('#p-stat').hidden = kind !== 'd';
  }
  ['#p-kind'].forEach((s) => $(s).addEventListener('change', () => { fillH(); renderPast(); }));
  ['#p-h', '#p-stat', '#p-from', '#p-to'].forEach((s) => $(s).addEventListener('change', renderPast));
  function renderPast() {
    const from = $('#p-from').value, to = $('#p-to').value, kind = $('#p-kind').value, H = $('#p-h').value;
    const dates = []; for (let d = from; d <= to; d = addDays(d, 1)) dates.push(d);
    const s = { p: [], lo: [], hi: [], y: [] }, lab = [];
    if (kind === 'h') {
      const src = P.hourly[H] || {};
      dates.forEach((d) => { const r = src[d], y = P.actual[d] || Array(24).fill(null);
        for (let i = 0; i < 24; i++) { s.p.push(r ? r[0][i] : null); s.lo.push(r ? r[1][i] : null); s.hi.push(r ? r[2][i] : null); s.y.push(y[i]); lab.push([d, i]); } });
    } else {
      const st = $('#p-stat').value, src = P.daily[H] || {};
      dates.forEach((d) => { const r = src[d] && src[d][st]; s.p.push(r ? r[0] : null); s.lo.push(r ? r[1] : null); s.hi.push(r ? r[2] : null);
        s.y.push(P.dailyActual[d] ? P.dailyActual[d][st] : null); lab.push([d, null]); });
    }
    const per = kind === 'h' ? 24 : 1;
    const step = Math.max(1, Math.ceil(dates.length / (innerWidth > 700 ? 14 : 6)));
    const shade = [];
    if (kind === 'h') dates.forEach((d, k) => shade.push([k * 24 + 10, k * 24 + 14]));
    else dates.forEach((d, k) => { if (info(d).off) shade.push([k, k]); });
    chart('#p-chart', '#p-tip', s, { xTicks: dates.map((d, k) => [k * per + (per - 1) / 2, md(d)]).filter((_, k) => k % step === 0),
      shade: dates.length <= 31 ? shade : [], dividers: kind === 'h' && dates.length <= 14 ? dates.map((_, k) => k * 24).slice(1) : [],
      tipLabel: (i) => `${md(lab[i][0])}(${wd(lab[i][0])})${lab[i][1] != null ? ' ' + (lab[i][1] + 1) + '시' : ''}`, dot: kind === 'h' ? (dates.length <= 7 ? 1.5 : 0) : 3 });
    const ok = s.p.map((p, i) => p != null && s.y[i] != null);
    const idx = ok.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
    const mae = avg(idx.map((i) => Math.abs(s.y[i] - s.p[i]))), cov = idx.length ? idx.filter((i) => inside(s.y[i], s.lo[i], s.hi[i])).length / idx.length : null;
    const below = idx.length ? idx.filter((i) => s.y[i] < s.lo[i]).length / idx.length : null;
    const covOk = cov != null && cov >= 0.75 && cov <= 0.85;
    $('#p-kpis').innerHTML = [['비교한 ' + (kind === 'h' ? '시간' : '일'), idx.length, `${md(from)} ~ ${md(to)}`, ''],
      ['예측 평균 오차(MAE)', fmt(mae) + '원', '공표 − 예측의 절댓값 평균', ''],
      ['80% 범위 포함률', pct(cov), '목표 75~85%', cov == null ? '' : covOk ? 'good' : 'bad'],
      ['하한 아래 / 상한 위', `${pct(below)} / ${pct(cov == null ? null : 1 - cov - below)}`, '각각 약 10%가 균형', ''],
      ['평균 범위 폭', fmt(avg(idx.map((i) => s.hi[i] - s.lo[i]))) + '원', '', '']]
      .map(([k, v, sub, c]) => `<div class="kpi"><div class="k">${k}</div><div class="v ${c}">${v}</div><div class="s">${sub}</div></div>`).join('');
  }
  renderers.past = () => { if (!$('#p-from').value) { fillH(); setRange(14); } else renderPast(); };

  // ---------- 성적 ----------
  function tbl(rows, cols) {
    let h = '<div class="tscroll"><table><tr>' + cols.map((c) => `<th class="${c.l ? 'l' : ''}">${c.t}</th>`).join('') + '</tr>';
    rows.forEach((r) => { h += `<tr class="${r._sel ? 'sel' : ''}">` + cols.map((c) => `<td class="${c.l ? 'l' : ''} ${c.cls ? c.cls(r) : ''}">${c.f ? c.f(r[c.k], r) : r[c.k] ?? '-'}</td>`).join('') + '</tr>'; });
    return h + '</table></div>';
  }
  const okCls = (r) => (r['포함률'] >= 0.75 && r['포함률'] <= 0.85 ? 'in' : 'out');
  const n2 = (v) => fmt(v, 2);
  renderers.perf = function () {
    const P2 = D.perf; let h = '';
    const W = P2.week;
    if (W.test.length) {
      const all = W.test.find((r) => r.H === '전체') || {};
      h += `<h2>일주일 예측 <span class="muted">선택: ${(W.choice || {})['선택 앙상블'] || '-'} · 2026 시험 (2025로 선택)</span></h2>`;
      h += `<div class="kpis">${[['80% 범위 포함률', pct(all['포함률']), '목표 75~85%', okCls(all) === 'in' ? 'good' : 'bad'], ['평균 폭', fmt(all['평균폭']) + '원', ''],
        ['하한 아래 / 상한 위', `${pct(all['하한아래'])} / ${pct(all['상한위'])}`, ''], ['MAE', n2(all['MAE']) + '원', `LightGBM 단독 ${n2(all['LightGBM 단독 MAE'])} · 지난주 그대로 ${n2(all['기준(지난주) MAE'])}`],
        ['구성', '', ((W.choice || {})['구성'] || []).join(', ')]].map(([k, v, s, c]) => `<div class="kpi"><div class="k">${k}</div><div class="v ${c || ''}">${v}</div><div class="s">${s}</div></div>`).join('')}</div>`;
      h += tbl(W.test, [{ k: 'H', t: '며칠 뒤', l: 1, f: (v) => (v === '전체' ? '전체' : 'D+' + v) }, { k: '포함률', t: '포함률', f: pct, cls: okCls }, { k: '하한아래', t: '하한 아래', f: pct },
        { k: '상한위', t: '상한 위', f: pct }, { k: '평균폭', t: '평균 폭', f: fmt }, { k: 'MAE', t: 'MAE', f: n2 }, { k: 'LightGBM 단독 MAE', t: 'LightGBM 단독', f: n2 }, { k: '기준(지난주) MAE', t: '지난주 그대로', f: n2 }]);
      h += '<h2>모델 순위 <span class="muted">2025년 MAE (모두 같은 조건, 기준일 15시 판단)</span></h2>';
      h += tbl(W.models, [{ k: '모델', t: '모델', l: 1 }, { k: '2025 상반기', t: '2025 상반기', f: n2 }, { k: '2025 하반기', t: '2025 하반기', f: n2 }, { k: '2025 전체', t: '2025 전체', f: n2 },
        ...[1, 2, 3, 4, 5, 6, 7].map((x) => ({ k: `2025 D+${x}`, t: `D+${x}`, f: n2 }))]);
      h += '<h2>앙상블 비교 <span class="muted">2025 상반기로 맞추고 하반기 MAE로 비교</span></h2>';
      h += tbl(W.ens, [{ k: '앙상블', t: '앙상블', l: 1 }, { k: '구성 수', t: '모델 수' }, { k: '2025 하반기', t: '2025 하반기 MAE', f: (v) => fmt(v, 3) }, { k: '구성', t: '구성', l: 1 }]);
    }
    const Mo = P2.month;
    if (Mo.test && Mo.test.length) {
      h += `<h2>한 달 예측 <span class="muted">선택: ${(Mo.choice || {})['선택 앙상블'] || '-'} · 2026 시험</span></h2>`;
      h += tbl(Mo.test, Object.keys(Mo.test[0]).map((k) => ({ k, t: k, l: typeof Mo.test[0][k] === 'string', f: (v) => (typeof v === 'number' ? (k.includes('률') || k.includes('아래') || k.includes('위') ? pct(v) : n2(v)) : v) })));
      if (Mo.models.length) h += tbl(Mo.models, Object.keys(Mo.models[0]).filter((k) => k !== '코드').map((k) => ({ k, t: k, l: typeof Mo.models[0][k] === 'string', f: (v) => (typeof v === 'number' ? n2(v) : v) })));
    }
    if (P2.dropCal && P2.dropCal.length) {
      h += '<h2>내일 급락 경고(과제1)의 확률 신뢰도 <span class="muted">보정식 2025, 2026은 확인만</span></h2>';
      h += tbl(P2.dropCal.filter((r) => r['구분'] !== '전체 Brier'), [{ k: '연도', t: '연도', l: 1 }, { k: '구분', t: '구분', l: 1 }, { k: '일수', t: '일수' },
        { k: '평균 확률', t: '표시한 평균 확률', f: pct }, { k: '실제 급락 비율', t: '실제 급락 비율', f: pct }, { k: '급락일', t: '급락일' }]);
    }
    $('#perf').innerHTML = h || '<p class="note">성적 자료가 아직 없습니다.</p>';
  };

  window.addEventListener('resize', () => { const t = $('.tab.on').dataset.tab; if (renderers[t]) renderers[t](); });
  $('#pub').innerHTML = `최신 공표 SMP <b>${D.lastPublished} (${wd(D.lastPublished)})</b>` + (D.lastRun ? ` · 최근 예측 산출 <b>${D.lastRun}</b>` : '') + ` · 갱신 ${D.generated}`;
  $('#foot').textContent = `출처: 전력거래소·기상청·한국천문연구원·EPSIS 공공데이터 · 숙명여대 P4 산학 프로젝트 · 데이터 생성 ${D.generated}`;
  renderWeek();
  const hash = (location.hash || '').slice(1);
  if (['month', 'past', 'perf', 'help'].includes(hash)) $(`.tab[data-tab="${hash}"]`).click();
})();
