(() => {
"use strict";

/* ---------- sizes: all five are distinct on purpose ---------- */
const T = 4, G = 2, R = 3, H = 6, D = 5, W = 720;
const AX = {
  T: { n: T, name: 'token', s: 't' },
  G: { n: G, name: 'key head', s: 'g' },
  H: { n: H, name: 'head', s: 'h' },
  d: { n: D, name: 'feature', s: 'f' }
};

/* ---------- colour: hue = head, lightness = token (OKLCH -> sRGB) ---------- */
const LV = [0.905, 0.81, 0.695, 0.56], CV = [0.085, 0.13, 0.16, 0.17];
const KH = [255, 62];                      // key/value heads: blue, orange
const QH = [150, 210, 290, 340, 30, 100];  // query heads: 3 cool, 3 warm
function lin(L, C, h) {
  const a = C * Math.cos(h * Math.PI / 180), b = C * Math.sin(h * Math.PI / 180);
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
}
function hex(L, C, h) {
  let c = C, rgb = lin(L, c, h);
  for (let i = 0; i < 60 && !rgb.every(v => v >= -0.0005 && v <= 1.0005); i++) { c *= 0.95; rgb = lin(L, c, h); }
  return '#' + rgb.map(v => {
    v = Math.min(1, Math.max(0, v));
    v = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
    return Math.round(v * 255).toString(16).padStart(2, '0');
  }).join('');
}
const memo = {};
const IH = [25, 145, 255];                  // image channels: red, green, blue
function hueOf(e) { return e.kind === 'k' ? KH[e.head] : QH[e.head]; }
function color(e) {
  if (e.kind === 'i') {                     // image pixel: hue = channel, shade = which patch it belongs to
    const u = e.patch / (e.np - 1), k = 'i' + e.c + '_' + e.patch + '_' + e.np, L = 0.93 - 0.364 * u;   // patches run light to dark
    return [memo[k] || (memo[k] = hex(L, 0.055 + 0.112 * u, IH[e.c])), L < 0.64];
  }
  const k = hueOf(e) + '_' + e.tok;
  return [memo[k] || (memo[k] = hex(LV[e.tok], CV[e.tok], hueOf(e))), e.tok >= 3];
}
const key = e => e.kind === 'i' ? `i${e.c}.${e.h}.${e.w}` : `${e.kind}${e.head}.${e.tok}.${e.f}`;
const vkey = e => e.kind === 'i' ? `p${e.patch}` : `${e.kind}${e.head}.${e.tok}`;

/* ---------- a tensor: flat buffer + shape + strides ---------- */
const rm = shape => shape.map((_, k) => shape.slice(k + 1).reduce((a, b) => a * b, 1));
class Tn {
  constructor(buf, shape, strides, names) {
    this.buf = buf; this.shape = shape; this.strides = strides || rm(shape); this.names = names || shape.map(() => null);
  }
  get size() { return this.shape.reduce((a, b) => a * b, 1); }
  off(idx) { return idx.reduce((s, i, k) => s + i * this.strides[k], 0); }
  at(idx) { return this.buf[this.off(idx)]; }
  permute(p) { return new Tn(this.buf, p.map(k => this.shape[k]), p.map(k => this.strides[k]), p.map(k => this.names[k])); }
  order() {                                   // buffer addresses in logical (row-major) order
    const out = [], n = this.shape.length, idx = Array(n).fill(0);
    for (let c = 0; c < this.size; c++) {
      out.push(this.off(idx));
      for (let k = n - 1; k >= 0; k--) { if (++idx[k] < this.shape[k]) break; idx[k] = 0; }
    }
    return out;
  }
  isContig() { const s = rm(this.shape); return this.buf.length === this.size && s.every((v, k) => v === this.strides[k]); }
  contiguous() { return this.isContig() ? this : new Tn(this.order().map(o => this.buf[o]), this.shape.slice(), null, this.names.slice()); }
  reshape(shape, names) { const c = this.contiguous(); return new Tn(c.buf, shape, null, names); }
  unsq(ax) {
    const sh = this.shape.slice(), st = this.strides.slice(), nm = this.names.slice();
    sh.splice(ax, 0, 1); st.splice(ax, 0, 0); nm.splice(ax, 0, null);
    return new Tn(this.buf, sh, st, nm);
  }
  expand(ax, n) { const sh = this.shape.slice(); sh[ax] = n; return new Tn(this.buf, sh, this.strides.slice(), this.names.slice()); }
}
/* build a fresh contiguous tensor; every element remembers what it means and its address */
function mk(kind, names) {
  const shape = names.map(n => AX[n].n), st = rm(shape), size = shape.reduce((a, b) => a * b, 1), buf = [];
  for (let i = 0; i < size; i++) {
    const idx = shape.map((n, k) => Math.floor(i / st[k]) % n);
    const g = n => { const k = names.indexOf(n); return k < 0 ? 0 : idx[k]; };
    buf.push({ kind, tok: g('T'), head: g(kind === 'k' ? 'G' : 'H'), f: g('d'), tag: i });
  }
  return new Tn(buf, shape, st, names.slice());
}

/* ---------- SVG primitives ---------- */
const DEFS = '<defs><marker id="arr" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M1.5,1.2L6.8,4L1.5,6.8" class="ah"/></marker></defs>';
const tx = (x, y, t, cls, a) => `<text x="${x}" y="${y}" class="${cls || 'lbl'}" text-anchor="${a || 'start'}">${t}</text>`;
const at = (g, x, y) => `<g transform="translate(${x},${y})">${g.s !== undefined ? g.s : g}</g>`;
const svg = (h, inner, w, label) => `<svg viewBox="0 0 ${w || W} ${h}" role="img" aria-label="${label || 'figure'}" style="min-width:${Math.min(w || W, 600)}px;max-width:${w || W}px">${inner}</svg>`;

function cell(x, y, w, h, e, o) {
  o = o || {};
  const [fill, dk] = color(e);
  return `<g class="cell${o.ghost ? ' ghost' : ''}" data-el="${key(e)}" data-v="${vkey(e)}" data-tag="${e.tag}">` +
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${Math.min(2.5, w / 4)}" fill="${fill}"/>` +
    (o.tag ? `<text x="${x + w / 2}" y="${y + h / 2 + 0.5}" class="tag ${dk ? 'tw' : 'tb'}">${e.tag}</text>` : '') + '</g>';
}

/* logical view: last axis = columns, next = rows, next = blocks across, next = rows of blocks */
function grid(tn, o) {
  o = o || {};
  const c = o.c || 22, n = tn.shape.length, gap = o.gap === undefined ? Math.round(c * 1.1) : o.gap;
  const [NR, NB, nr, nc] = [1, 1, 1, 1].concat(tn.shape).slice(-4);
  const nm = k => tn.names[k - (4 - n)];
  const th = (n >= 3 && o.titles !== false) ? 15 : 0, bw = nc * c, bh = nr * c, tag = o.tag !== false && c >= 18;
  const tfn = o.title || ((I, J) => {
    if (o.vn) return n === 3 ? `${o.vn}[${J}]` : `${o.vn}[${I}, ${J}]`;   // plain index labels
    if (n === 3) { const a = AX[nm(1)]; return a ? `${a.name} ${J}` : `[${J}]`; }
    const a = AX[nm(0)], b = AX[nm(1)]; return `${a ? a.s : ''}${I}, ${b ? b.s : ''}${J}`;
  });
  const ra = AX[nm(2)];
  let s = ''; const seen = new Set();
  for (let I = 0; I < NR; I++) for (let J = 0; J < NB; J++) {
    const bx = J * (bw + gap), by = th + I * (bh + th + gap);
    if (th) s += tx(bx, by - 5, tfn(I, J), 'lbl ttl');
    for (let k = 0; k < nr; k++) {
      if (J === 0 && n >= 2 && (ra || o.vn) && o.rowLab !== false) s += tx(bx - 6, by + k * c + c / 2 + 0.5, o.vn ? k : ra.s + k, 'lbl sm mid', 'end');
      for (let l = 0; l < nc; l++) {
        const off = tn.off([I, J, k, l].slice(4 - n)), ghost = seen.has(off); seen.add(off);
        s += cell(bx + l * c, by + k * c, c, c, tn.buf[off], { tag, ghost: ghost && o.ghost });
      }
    }
  }
  return { s, w: NB * bw + (NB - 1) * gap, h: NR * (bh + th) + (NR - 1) * gap, bw, bh, th, gap };
}

/* the strip of memory, with optional brackets (outer -> inner chunk sizes) */
function strip(buf, o) {
  o = o || {};
  const c = o.c || 20, h = o.h || c;
  let s = '';
  buf.forEach((e, i) => { s += cell(i * c, 0, c, h, e, { tag: o.tag === true || (o.tag !== false && c >= 18) }); });
  if (o.dom) for (let i = o.dom; i < buf.length; i += o.dom) s += `<path class="sep" d="M${i * c},-1V${h + 1}"/>`;
  let y = h + 4; const br = o.brackets || [];
  for (let L = br.length - 1; L >= 0; L--) {
    for (let a = 0; a < buf.length; a += br[L]) {
      const b = Math.min(buf.length, a + br[L]);
      s += `<path class="brk${L === 0 && br.length > 1 ? ' brk-o' : ''}" d="M${a * c + 2},${y}v5H${b * c - 2}v-5"/>`;
    }
    y += 9;
  }
  return { s, w: buf.length * c, h: y };
}

/* the order in which a view reads the strip. Each run of consecutive cells is a solid line that ends in a dot;
   from the dot a dashed hop leaves, and its arrowhead lands on the exact cell where the reading continues. */
function readPath(order, c) {
  const runs = []; let a = order[0], p = order[0];
  for (let i = 1; i < order.length; i++) { if (order[i] === p + 1) { p = order[i]; continue; } runs.push([a, p]); a = p = order[i]; }
  runs.push([a, p]);
  const y = -9, cx = i => (i + 0.5) * c;
  if (runs.length === 1) return { s: `<path class="rp" d="M${a * c + 4},${y - 1}H${(p + 1) * c - 5}" marker-end="url(#arr)"/>`, up: 20, runs: 1 };
  const hops = runs.slice(0, -1).map((r, i) => ({ from: cx(r[1]), to: cx(runs[i + 1][0]) }));
  // short hops stay low and long ones go higher, so two hops never share a stretch of the same level
  const levels = [];
  hops.map((h, i) => i).sort((i, j) => Math.abs(hops[i].to - hops[i].from) - Math.abs(hops[j].to - hops[j].from)).forEach(i => {
    const lo = Math.min(hops[i].from, hops[i].to) - c * 0.6, hi = Math.max(hops[i].from, hops[i].to) + c * 0.6;
    let L = 0; while (levels[L] && levels[L].some(([u, v]) => lo < v && hi > u)) L++;
    (levels[L] = levels[L] || []).push([lo, hi]); hops[i].L = L;
  });
  const H = L => 15 + L * 12, r = 5; let s = '';
  hops.forEach(h => {
    const d = h.to > h.from ? 1 : -1, t = y - H(h.L);
    s += `<path class="rp rpj" d="M${h.from},${y - 4}V${t + r}Q${h.from},${t} ${h.from + d * r},${t}H${h.to - d * r}Q${h.to},${t} ${h.to},${t + r}V${y - 2.5}" marker-end="url(#arr)"/>`;
  });
  runs.forEach(([a, b], i) => {
    s += `<path class="rp" d="M${cx(a)},${y}H${cx(b)}"/><circle class="dot" cx="${cx(b)}" cy="${y}" r="3"/>`;
    if (i === 0) s += `<circle class="ring" cx="${cx(a)}" cy="${y}" r="3"/>`;
  });
  return { s, up: H(levels.length - 1) + 12, runs: runs.length };
}

/* ribbons from where each vector sits in strip A to where it sits in strip B */
function braid(A, B, c, y1, y2) {
  const pos = new Map(); B.forEach((e, i) => pos.set(key(e), i));
  let s = '', i = 0;
  while (i < A.length) {
    let j = i;
    while (j + 1 < A.length && vkey(A[j + 1]) === vkey(A[i]) && pos.get(key(A[j + 1])) === pos.get(key(A[j])) + 1) j++;
    const a0 = i * c + 0.6, a1 = (j + 1) * c - 0.6, b0 = pos.get(key(A[i])) * c + 0.6, b1 = b0 + (a1 - a0), m = (y1 + y2) / 2;
    s += `<path class="rib" data-v="${vkey(A[i])}" fill="${color(A[i])[0]}" d="M${a0},${y1}H${a1}C${a1},${m} ${b1},${m} ${b1},${y2}H${b0}C${b0},${m} ${a0},${m} ${a0},${y1}Z"/>`;
    i = j + 1;
  }
  return s;
}


/* ---------- figure plumbing ---------- */
const $ = (s, r) => (r || document).querySelector(s);
const figs = {};
const blogCode = h => (h || '').replace(/<code>/g, '<code class="language-plaintext highlighter-rouge">');
function mount(id, build, o) {
  const fig = document.getElementById(id); if (!fig) return;
  o = o || {};
  const api = { fig, stage: $('.tl-stage', fig), ro: $('.tl-readout', fig), state: Object.assign({}, o.state), o };
  api.rest = () => { if (api.ro) api.ro.innerHTML = blogCode(o.note ? o.note(api) : api.ro.dataset.rest); };
  api.redraw = () => { api.stage.innerHTML = build(api); api.rest(); };
  if (api.ro) api.ro.dataset.rest = api.ro.innerHTML;
  fig.querySelectorAll('button[data-k]').forEach(b => b.addEventListener('click', () => {
    api.state[b.dataset.k] = b.dataset.v;
    fig.querySelectorAll(`button[data-k="${b.dataset.k}"]`).forEach(x => x.setAttribute('aria-pressed', x === b));
    api.redraw();
  }));
  linkHover(api);
  api.redraw();
  figs[id] = api; return api;
}
function describe(t, o) {
  const d = t.dataset;
  if (d.s) {
    const [h, i, j] = d.s.split(',').map(Number);
    return `<code>att[${h}, ${i}, ${j}]</code> = <code>q[${h}, ${i}, :]</code> · <code>k[${h}, ${j}, :]</code>: in query head ${h}, token ${i} looks at token ${j} through key head ${Math.floor(h / R)}.`;
  }
  const m = /^([kq])(\d+)\.(\d+)(?:\.(\d+))?$/.exec(d.el || d.v || ''); if (!m) return null;
  if (o.plain) {   // part 1: no names on the axes, just indices
    const idx = o.plain === 'A' ? `A[${m[3]}, ${m[4]}]` : `x[${m[3]}, ${m[2]}, ${m[4] !== undefined ? m[4] : ':'}]`;
    return `<code>${idx}</code>` + (m[4] !== undefined && d.tag !== undefined ? `, address ${d.tag}` : ', one run of five numbers');
  }
  const head = m[1] === 'q' ? 'query head' : (o.khead || 'head');
  return `token ${m[3]}, ${head} ${m[2]}` + (m[4] !== undefined ? `, feature ${m[4]}` : '') + (o.addr && d.tag !== undefined ? `, address ${d.tag}` : '');
}
function linkHover(api) {
  const fig = api.fig;
  const clear = () => { fig.classList.remove('hov'); fig.querySelectorAll('.on,.von').forEach(n => n.classList.remove('on', 'von')); };
  fig.addEventListener('pointerover', ev => {
    const t = ev.target.closest('.cell,.rib'); clear();
    if (!t || !fig.contains(t)) { api.rest(); return; }
    fig.classList.add('hov');
    if (t.dataset.links) {
      const ls = t.dataset.links.split(' ');
      fig.querySelectorAll('[data-v]').forEach(n => { if (ls.includes(n.dataset.v)) n.classList.add('on'); });
      t.classList.add('on');
    } else {
      fig.querySelectorAll(`[data-v="${t.dataset.v}"]`).forEach(n => n.classList.add('von'));
      if (t.dataset.el) fig.querySelectorAll(`[data-el="${t.dataset.el}"]`).forEach(n => n.classList.add('on'));
      else t.classList.add('on');
    }
    if (api.ro) { const h = (api.o.hover || describe)(t, api.o); if (h) api.ro.innerHTML = blogCode(h); }
  });
  fig.addEventListener('pointerleave', () => { clear(); api.rest(); });
}

/* ---------- 1. the hook: same shape, different tensor ---------- */
function figHook() {
  const x = mk('k', ['T', 'G', 'd']), c = 22;
  const ok = x.permute([1, 0, 2]).reshape([G, T * D]), bad = x.reshape([G, T * D]);
  const g0 = grid(x, { c, vn: 'x' }), g1 = grid(ok, { c, vn: 'out' }), g2 = grid(bad, { c, vn: 'out' });
  const y0 = 30, x1 = (W - g1.w) / 2 + 6, p1 = y0 + g0.h + 56, p2 = p1 + g1.h + 66;
  let s = tx(W / 2, 16, 'x, shape (4, 2, 5)', 'lbl code', 'middle') + at(g0, (W - g0.w) / 2 + 6, y0);
  s += tx(x1, p1 - 10, 'x.transpose(0, 1).reshape(2, 20)', 'lbl code') + at(g1, x1, p1) +
    tx(x1, p1 + g1.h + 18, '✓ every row is one hue, with its four shades in order', 'lbl ok');
  s += tx(x1, p2 - 10, 'x.reshape(2, 20)', 'lbl code') + at(g2, x1, p2) +
    tx(x1, p2 + g2.h + 18, '✗ every row is half of the strip: both hues, two shades of each', 'lbl bad');
  return svg(p2 + g2.h + 30, s, W, 'The same tensor regrouped two ways');
}

/* ---------- 2. strip + odometer ---------- */
function figStrip() {
  const x = mk('k', ['T', 'G', 'd']), g = grid(x, { c: 22, vn: 'x' }), st = strip(x.buf, { c: 18, brackets: [10, 5] });
  const sy = 8 + g.h + 46;
  let s = DEFS + at(g, (W - g.w) / 2, 8) + at({ s: readPath(x.order(), 18).s + st.s }, 0, sy);
  return svg(sy + st.h + 6, s, W, 'A (4, 2, 5) tensor and its strip of memory');
}
function hoverStrip(t) {
  const m = /^k(\d+)\.(\d+)\.(\d+)$/.exec(t.dataset.el || ''); if (!m) return null;
  const [g, tok, f] = [+m[1], +m[2], +m[3]];
  return `<code>x[${tok}, ${g}, ${f}]</code> is at address <code>${tok}·10 + ${g}·5 + ${f}·1 = ${tok * 10 + g * 5 + f}</code>`;
}

/* ---------- 3. reshape = brackets ---------- */
const RS = {
  '40': 'No brackets: the logical view is the strip itself.',
  '4,10': 'One bracket of 10 for every index on axis 0. The last two axes are fused into one.',
  '4,2,5': 'Each bracket of 10 holds two inner brackets of 5. This is the shape we started with.',
  '8,5': 'The first two axes fused into one of size 8. Every row is still one of the original runs of five.',
  '5,8': '5 × 8 is 40, so this is legal. It is also nonsense: brackets of 8 cut straight through the runs of five.'
};
function figReshape(api) {
  const x = mk('k', ['T', 'G', 'd']), k = api.state.shape, shape = k.split(',').map(Number);
  const r = x.reshape(shape), c = shape.length === 1 ? 18 : 22, g = grid(r, { c, vn: 'x' });
  const st = strip(x.buf, { c: 18, brackets: r.strides.slice(0, -1) }), sy = 218;
  let s = at(g, (W - g.w) / 2, 6) + tx(0, sy - 10, `x.reshape(${shape.join(', ')})${shape.length === 1 ? '' : ', strides (' + r.strides.join(', ') + ')'}`, 'lbl code') + at(st, 0, sy);
  return svg(sy + 18 + 26, s, W, 'The strip with different brackets');
}

/* ---------- 4a. two axes: mirror ---------- */
function figMirror() {
  const A = mk('k', ['T', 'd']), B = A.permute([1, 0]), c = 26, sc = 20, ga = grid(A, { c, vn: 'A' }), gb = grid(B, { c, vn: 'A' });
  const lcx = 215, rcx = 505, y0 = 40, ax = lcx - ga.w / 2, bx = rcx - gb.w / 2, sy = y0 + 5 * c + 50;
  const st = strip(A.buf, { c: sc });   // one strip of memory, shared by both views
  let s = DEFS + tx(lcx, y0 - 16, 'A, shape (4, 5), strides (5, 1)', 'lbl code', 'middle') + at(ga, ax, y0) +
    `<path class="mir" d="M${ax - 10},${y0 - 10}L${ax + 4 * c + 14},${y0 + 4 * c + 14}"/>` +
    tx(rcx, y0 - 16, 'A.T, shape (5, 4), strides (1, 5)', 'lbl code', 'middle') + at(gb, bx, y0) +
    `<path class="mir" d="M${bx - 10},${y0 - 10}L${bx + 4 * c + 14},${y0 + 4 * c + 14}"/>` +
    `<path class="rp" d="M${ax + ga.w + 28},${y0 + 52}H${bx - 44}" marker-end="url(#arr)"/>` +
    tx(W / 2, sy - 10, 'the memory, shared by A and A.T', 'lbl code', 'middle') + at(st, (W - st.w) / 2, sy);
  return svg(sy + sc + 10, s, W, 'A matrix and its transpose over the same strip of memory');
}

/* ---------- 4b. three axes: every cell travels from its place in x to its place in y ---------- */
function slots(tn, c, gap, vn) {            // where each cell sits in the blocks-of-rows drawing, plus the empty frame
  const [nb, nr, nc] = tn.shape, th = 15, bw = nc * c, pos = new Map(); let frame = '';
  for (let J = 0; J < nb; J++) {
    const bx = J * (bw + gap);
    frame += tx(bx, th - 5, `${vn}[${J}]`, 'lbl ttl');
    for (let k = 0; k < nr; k++) {
      if (J === 0) frame += tx(-6, th + k * c + c / 2 + 0.5, k, 'lbl sm mid', 'end');
      for (let l = 0; l < nc; l++) {
        pos.set(tn.off([J, k, l]), [bx + l * c, th + k * c, J, k]);
        frame += `<rect class="slot" x="${bx + l * c + 1.5}" y="${th + k * c + 1.5}" width="${c - 3}" height="${c - 3}" rx="2.5"/>`;
      }
    }
  }
  return { pos, frame, w: nb * bw + (nb - 1) * gap, h: th + nr * c };
}
function initPerm3() {
  const fig = document.getElementById('fig-perm3'); if (!fig) return;
  const x = mk('k', ['T', 'G', 'd']), c = 22, gap = 24, stage = $('.tl-stage', fig), ro = $('.tl-readout', fig);
  const src = slots(x, c, gap, 'x'), st = strip(x.buf, { c: 18 }), xTop = 22, yTop = 136, sy = 300, x0 = Math.round((W - src.w) / 2) + 6;
  let cells = '';
  x.buf.forEach((e, i) => { const q = src.pos.get(i); cells += `<g class="mv" data-i="${i}" transform="translate(${x0 + q[0]},${xTop + q[1]})">${cell(0, 0, c, c, e, { tag: true })}</g>`; });
  stage.innerHTML = svg(sy + 18 + 8,
    tx(W / 2, 14, 'x, shape (4, 2, 5), strides (10, 5, 1)', 'lbl code', 'middle') + at(grid(x, { c, gap, vn: 'x' }), x0, xTop) +
    `<text class="lbl code ylab" x="${W / 2}" y="${yTop - 10}" text-anchor="middle"></text><g class="fr"></g><g class="mvs">${cells}</g>` +
    tx(W / 2, sy - 10, 'the memory, shared by x and y', 'lbl code', 'middle') + at(st, 0, sy), W, 'x and a permuted view of it, drawn as blocks of rows');
  const mvs = [...fig.querySelectorAll('.mv')], fr = $('.fr', fig), ylab = $('.ylab', fig);
  const HINT = ['the shade', 'the hue', 'the place inside a run of five'];
  let p = [0, 1, 2], raf = 0;
  const rest = () => {
    ro.innerHTML = blogCode(p.join('') === '012'
      ? 'Pick a permutation and every cell of <code>x</code> travels to its place in <code>y</code>.'
      : `The blocks of <code>y</code> follow axis ${p[0]} of <code>x</code> (${HINT[p[0]]}), its rows follow axis ${p[1]} (${HINT[p[1]]}) and its columns follow axis ${p[2]} (${HINT[p[2]]}).`);
  };
  function play(np, animate) {
    p = np; cancelAnimationFrame(raf);
    const y = x.permute(p), dst = slots(y, c, gap, 'y'), y0 = Math.round((W - dst.w) / 2) + 6;
    ylab.textContent = `y = x.permute(${p.join(', ')}), shape (${y.shape.join(', ')}), strides (${y.strides.join(', ')})`;
    fr.setAttribute('transform', `translate(${y0},${yTop})`); fr.innerHTML = dst.frame;
    // cells leave x one destination block at a time, so each block of y is assembled in turn
    const unit = Math.min(90, 1300 / (y.shape[0] * y.shape[1]));   // one row of y after another, a short pause between blocks
    const plan = mvs.map(g => { const i = +g.dataset.i, a = src.pos.get(i), b = dst.pos.get(i); return { g, ax: x0 + a[0], ay: xTop + a[1], bx: y0 + b[0], by: yTop + b[1], delay: (b[2] * y.shape[1] + b[3]) * unit + b[2] * 100 }; });
    const still = !animate || (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    const dur = 700, total = Math.max(...plan.map(q => q.delay)) + dur, t0 = performance.now();
    const frame = now => {
      const el = still ? total : now - t0;
      for (const q of plan) {
        const t = Math.min(1, Math.max(0, (el - q.delay) / dur)), e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        q.g.setAttribute('transform', `translate(${(q.ax + (q.bx - q.ax) * e).toFixed(2)},${(q.ay + (q.by - q.ay) * e).toFixed(2)})`);
      }
      if (el < total) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame); rest();
  }
  fig.querySelectorAll('button[data-p]').forEach(b => b.addEventListener('click', () => {
    fig.querySelectorAll('button[data-p]').forEach(q => q.setAttribute('aria-pressed', q === b));
    play(b.dataset.p.split('').map(Number), true);
  }));
  const rp = fig.querySelector('button[data-replay]'); if (rp) rp.addEventListener('click', () => play(p, true));
  linkHover({ fig, ro, rest, o: { hover: t => {
    const m = /^k(\d+)\.(\d+)\.(\d+)$/.exec(t.dataset.el || ''); if (!m) return null;
    const xi = [+m[2], +m[1], +m[3]], yi = p.map(k => xi[k]);
    return `<code>x[${xi.join(', ')}]</code> is <code>y[${yi.join(', ')}]</code>, address ${t.dataset.tag}`;
  } } });
  play([0, 1, 2], false);
}

/* ---------- 4c. more axes: an image cut into patches, one magnitude per axis ---------- */
// five axes drawn by position: dirs tells every axis to go across ('h') or down ('v'), and within each direction
// the axes nest from big steps (the first one) to small steps (the last one); gaps[k] separates two steps of axis k
function lay5(tn, c, dirs, gaps) {
  const sh = tn.shape;
  const steps = d => {    // per level: how far one step moves, and how much one full run of that level covers
    const ks = sh.map((n, k) => k).filter(k => dirs[k] === d), ext = [], step = [];
    let inner = c;
    for (let j = ks.length - 1; j >= 0; j--) {
      const n = sh[ks[j]], gap = inner > c ? gaps[ks[j]] : 0;   // no gaps between single cells
      step[j] = inner + gap; ext[j] = n * inner + (n - 1) * gap; inner = ext[j];
    }
    return { ks, ext, step };
  };
  const Hz = steps('h'), V = steps('v'), pos = new Map(), items = [], idx = sh.map(() => 0);
  const along = D => D.ks.reduce((s, k, j) => s + idx[k] * D.step[j], 0);
  for (let n = 0; n < tn.size; n++) {
    const e = tn.at(idx), x = along(Hz), y = along(V);
    pos.set(e, [x, y]); items.push({ e, x, y, idx: idx.slice() });
    for (let k = sh.length - 1; k >= 0; k--) { if (++idx[k] < sh[k]) break; idx[k] = 0; }
  }
  return { pos, items, w: Hz.ext[0], bh: V.ext[0], H: Hz, V };
}
// how many arrow rows sit above the innermost across arrow
const rows5 = (L, ax) => L.H.ks.length - 1 - L.H.ks.findIndex(k => ax[k]);
function axes5(L, ax, avail) {
  // every arrow starts with a badge carrying its axis number, the same number its label starts with
  const num = t => (/axis (\d)/.exec(t) || [])[1];
  const A = (x1, y1, x2, y2, t) => {
    const v = x1 === x2, bx = v ? x1 : x1 + 7, by = v ? y1 + 7 : y1;
    return `<path class="rp" d="M${x1},${y1}L${x2},${y2}" marker-end="url(#arr)"/><g class="rb"><circle cx="${bx}" cy="${by}" r="7"/><text x="${bx}" y="${by + 0.5}">${num(t)}</text></g>`;
  };
  // a label goes to the right of its arrow, or above it when there is no room left on that side
  const side = (len, y, t) => len + 8 + t.length * 6.8 <= avail ? tx(len + 8, y + 3.5, t, 'lbl') : tx(18, y - 9, t, 'lbl');
  let s = '';
  // across: the outermost axis on top, each arrow as long as one full run of its axis
  L.H.ks.forEach((k, j) => {
    if (!ax[k]) return;
    const y = -14 - 16 * (L.H.ks.length - 1 - j);
    s += A(0, y, L.H.ext[j], y, ax[k]) + side(L.H.ext[j], y, ax[k]);
  });
  // down: at most two labelled axes, the outermost furthest from the cells
  const vs = L.V.ks.map((k, j) => [ax[k], L.V.ext[j]]).filter(v => v[0]);
  if (vs.length === 2) {
    // the short arrow's label sits level with it; the long arrow's label sits lower down, where it is the only arrow left
    const [[ta, bh], [ti, gh]] = vs;
    s += A(-34, 0, -34, bh, ta) + tx(-46, gh + (bh - gh) / 2 + 4, ta, 'lbl', 'end');
    s += A(-14, 0, -14, gh, ti) + tx(-46, gh / 2 + 4, ti, 'lbl', 'end');
  } else if (vs.length === 1) {
    s += A(-14, 0, -14, vs[0][1], vs[0][0]) + tx(-26, 18, vs[0][0], 'lbl', 'end');
  }
  return s;
}
function initPatch(prefix, IH_, IW, P, c) {
  if (!document.getElementById(prefix + '-0')) return;
  const NC = 3, GH = IH_ / P, GW = IW / P, NP = GH * GW, NV = NC * P * P, N = NC * IH_ * IW, mg = 6, bgap = 26, Lm = 150, buf = [];
  for (let i = 0; i < N; i++) {
    const ch = Math.floor(i / (IH_ * IW)), h = Math.floor(i / IW) % IH_, w = i % IW;
    buf.push({ kind: 'i', c: ch, h, w, gh: Math.floor(h / P), ph: h % P, gw: Math.floor(w / P), pw: w % P, patch: Math.floor(h / P) * GW + Math.floor(w / P), np: NP, tag: i });
  }
  const img = new Tn(buf, [NC, IH_, IW]), cut = img.reshape([NC, GH, P, GW, P]), moved = cut.permute([1, 3, 0, 2, 4]), copy = moved.contiguous();
  // default drawing: axis 0 as blocks side by side, axes 1 and 2 going down, axes 3 and 4 going across
  const D5 = 'hvvhh', G5 = [bgap, mg, 0, mg, 0];
  const S = [
    { lay: new Tn(buf, [NC, 1, IH_, 1, IW]), mem: buf, dom: IH_ * IW, ml: `the memory: ${N} numbers, channel after channel`,
      code: `img, shape (${NC}, ${IH_}, ${IW})`, names: 'img[channel, row, col]',
      ax: ['axis 0: channel', null, 'axis 1: row', null, 'axis 2: col'], idx: e => `img[${e.c}, ${e.h}, ${e.w}]` },
    { lay: cut, mem: buf, dom: IH_ * IW, ml: 'the memory: untouched',
      code: `cut = img.reshape(${cut.shape.join(', ')})`, names: 'cut[channel, grid row, patch row, grid col, patch col]',
      ax: ['axis 0: channel', 'axis 1: grid row', 'axis 2: patch row', 'axis 3: grid col', 'axis 4: patch col'], idx: e => `cut[${e.c}, ${e.gh}, ${e.ph}, ${e.gw}, ${e.pw}]` },
    // grid row goes down and grid col across, so every patch stays where it was in the image
    { lay: moved, dirs: 'vhvhh', gaps: [bgap, bgap, 0, mg, 0], mem: buf, dom: IH_ * IW, ml: 'the memory: still untouched',
      code: `moved = cut.permute(1, 3, 0, 2, 4), shape (${moved.shape.join(', ')})`, names: 'moved[grid row, grid col, channel, patch row, patch col]',
      ax: ['axis 0: grid row', 'axis 1: grid col', 'axis 2: channel', 'axis 3: patch row', 'axis 4: patch col'], idx: e => `moved[${e.gh}, ${e.gw}, ${e.c}, ${e.ph}, ${e.pw}]` },
    { lay: new Tn(copy.buf, [1, NP, 1, 1, NV]), mem: copy.buf, dom: NV, ml: 'a new strip, patch after patch: this reshape had to copy',
      code: `patches = moved.reshape(${NP}, ${NV})`, names: 'patches[patch, value]',
      ax: [null, 'axis 0: patch', null, null, 'axis 1: value'], idx: e => `patches[${e.patch}, ${e.c * P * P + e.ph * P + e.pw}]` }
  ];
  // one static figure per stage; hovering a pixel in any of them lights it up in all four
  const group = S.map((st, k) => {
    const fig = document.getElementById(prefix + '-' + k), ro = $('.tl-readout', fig);
    const L = lay5(st.lay, c, st.dirs || D5, st.gaps || G5);
    const top = 28 + [34, 44, 60][rows5(L, st.ax)];   // room for the arrows this stage actually has
    const x0 = Math.round(Lm + Math.max(0, (W - Lm - L.w) / 2)), sy = top + L.bh + 46;
    let cells = ''; L.pos.forEach((q, e) => { cells += cell(q[0], q[1], c, c, e, { tag: true }); });
    $('.tl-stage', fig).innerHTML = svg(sy + 16 + 8, DEFS + tx(W / 2, 14, st.code, 'lbl code', 'middle') +
      at(axes5(L, st.ax, W - x0) + cells, x0, top) +
      tx(W / 2, sy - 10, st.ml, 'lbl code', 'middle') + at(strip(st.mem, { c: W / N, h: 16, tag: false, dom: st.dom }), 0, sy), W, st.code);
    return { fig, ro, st, rest: `<code>${st.names}</code>` };
  });
  const clear = () => group.forEach(g => {
    g.fig.classList.remove('hov'); g.fig.querySelectorAll('.on,.von').forEach(n => n.classList.remove('on', 'von')); g.ro.innerHTML = blogCode(g.rest);
  });
  group.forEach(g => {
    g.fig.addEventListener('pointerover', ev => {
      const t = ev.target.closest('.cell'); clear();
      const m = t && g.fig.contains(t) && /^i(\d+)\.(\d+)\.(\d+)$/.exec(t.dataset.el || ''); if (!m) return;
      const e = buf[+m[1] * IH_ * IW + +m[2] * IW + +m[3]];
      group.forEach(h => {
        h.fig.classList.add('hov');
        h.fig.querySelectorAll(`[data-v="${t.dataset.v}"]`).forEach(n => n.classList.add('von'));
        h.fig.querySelectorAll(`[data-el="${t.dataset.el}"]`).forEach(n => n.classList.add('on'));
        h.ro.innerHTML = blogCode(`<code>${h.st.idx(e)}</code>, patch ${e.patch}, address ${e.tag} in the image`);
      });
    });
    g.fig.addEventListener('pointerleave', clear);
  });
  clear();
}

/* ---------- 5a. a permuted view reads the strip in hops ---------- */
function figRead() {
  const x = mk('k', ['T', 'G', 'd']), y = x.permute([1, 0, 2]), g = grid(y, { c: 22, vn: 'y' }), rp = readPath(y.order(), 18), st = strip(x.buf, { c: 18 });
  const gy = 24, sy = gy + g.h + 14 + rp.up;
  const s = DEFS + tx(W / 2, 14, `y = x.permute(1, 0, 2), shape (${y.shape.join(', ')}), strides (${y.strides.join(', ')})`, 'lbl code', 'middle') +
    at(g, (W - g.w) / 2 + 6, gy) + at({ s: rp.s + st.s }, 0, sy) + tx(W / 2, sy + 18 + 16, 'the memory, and the path y takes over it', 'lbl code', 'middle');
  return svg(sy + 18 + 26, s, W, 'The path a permuted view takes over the strip of memory');
}

/* ---------- 5b. contiguous ---------- */
function figContig() {
  const x = mk('k', ['T', 'G', 'd']), y = x.permute([1, 0, 2]), yc = y.contiguous(), c = 18;
  const s1 = strip(x.buf, { c }), s2 = strip(yc.buf, { c, brackets: [20, 5] }), y1 = 24, y2 = y1 + 18 + 74;
  const s = tx(W / 2, 14, 'y, strides (5, 10, 1): the old strip', 'lbl code', 'middle') + at(s1, 0, y1) +
    at(braid(x.buf, yc.buf, c, y1 + 19, y2 - 1), 0, 0) + at(s2, 0, y2) +
    tx(W / 2, y2 + s2.h + 14, 'y.contiguous(), strides (20, 5, 1): a new strip, written in the order y reads', 'lbl code', 'middle');
  return svg(y2 + s2.h + 24, s, W, 'contiguous copies the strip into reading order');
}

/* ---------- 6a. split heads, heads to the front ---------- */
function figHeads() {
  const c = 11, q = mk('q', ['T', 'H', 'd']), k = mk('k', ['T', 'G', 'd']), qx = 22, kx = 462;
  const rows = [
    [q.reshape([T, H * D], ['T', null]), k.reshape([T, G * D], ['T', null]), 'wq(x)', 'wk(x)'],
    [q, k, '.view(T, n_heads, d_head)', '.view(T, n_kv_heads, d_head)'],
    [q.permute([1, 0, 2]), k.permute([1, 0, 2]), '.transpose(0, 1)', '.transpose(0, 1)']
  ];
  let y = 16, s = '';
  rows.forEach(([a, b, la, lb]) => {
    const ga = grid(a, { c, gap: 10 }), gb = grid(b, { c, gap: b.shape[0] === G ? 18 : 10 });
    // the code on one line, the shape it produces right under it
    s += tx(qx, y, la, 'lbl code') + tx(kx, y, lb, 'lbl code') + tx(qx, y + 13, `(${a.shape.join(', ')})`, 'lbl sm') + tx(kx, y + 13, `(${b.shape.join(', ')})`, 'lbl sm');
    s += at(ga, qx, y + 24) + at(gb, kx, y + 24);
    y += 24 + Math.max(ga.h, gb.h) + 36;
  });
  return svg(y - 20, s, W, 'Queries and keys: projection, split into heads, heads moved to the front');
}

/* ---------- 6b. share each key head with r query heads ---------- */
function figShare(api) {
  const tile = api.state.mode === 'tile', c = 14, gap = 16;
  const k = mk('k', ['G', 'T', 'd']), q = mk('q', ['H', 'T', 'd']);
  const k4 = tile ? k.unsq(0).expand(0, R) : k.unsq(1).expand(1, R), kx = k4.reshape([H, T, D], ['H', 'T', 'd']);
  const gk = grid(k, { c, gap }), nb = k4.shape[1], g4 = grid(k4, { c, gap, ghost: true, title: (I, J) => `→ h${I * nb + J}` });
  const g6 = grid(kx, { c, gap, title: (I, J) => `h${J} gets g${kx.at([J, 0, 0]).head}` }), gq = grid(q, { c, gap, title: (I, J) => `query head ${J}` });
  const x4 = 272, y0 = 54, y1 = y0 + g4.h + 76, x6 = (W - g6.w) / 2 + 10, k0 = 26;
  let s = DEFS + tx(k0, y0 - 30, 'k', 'lbl code') + tx(k0, y0 - 17, '(2, 4, 5)', 'lbl sm') + at(gk, k0, y0) +
    `<path class="rp" d="M${k0 + gk.w + 12},${y0 + 44}H${x4 - 34}" marker-end="url(#arr)"/>` +
    tx(x4, y0 - 30, tile ? 'k[None].expand(q_per_kv_head, n_kv_heads, T, d_head)' : 'k[:, None].expand(n_kv_heads, q_per_kv_head, T, d_head)', 'lbl code') + tx(x4, y0 - 17, `(${k4.shape.join(', ')})`, 'lbl sm') + at(g4, x4, y0) +
    tx(x4 + g4.w + 20, y0 + 32, 'faded cells are stride-0', 'lbl') + tx(x4 + g4.w + 20, y0 + 47, 'copies: the same memory,', 'lbl') + tx(x4 + g4.w + 20, y0 + 62, 'read again', 'lbl') +
    tx(x6, y1 - 30, '.reshape(n_heads, T, d_head): the blocks above, read row by row', 'lbl code') + tx(x6, y1 - 17, '(6, 4, 5)', 'lbl sm') + at(g6, x6, y1) +
    at(gq, x6, y1 + g6.h + 36);
  for (let h = 0; h < H; h++) {
    const good = kx.at([h, 0, 0]).head === Math.floor(h / R);
    s += tx(x6 + h * (g6.bw + gap) + g6.bw / 2, y1 + g6.h + 16, good ? '✓' : '✗', 'lbl mk ' + (good ? 'ok' : 'bad'), 'middle');
  }
  return svg(y1 + g6.h + 36 + gq.h + 12, s, W, 'Expanding two key heads to six');
}

/* ---------- 6c. scores ---------- */
function figScores() {
  const c = 14, slot = 5 * c, gap = 16, x0 = 190;
  const q = mk('q', ['H', 'T', 'd']), kx = mk('k', ['G', 'T', 'd']).unsq(1).expand(1, R).reshape([H, T, D], ['H', 'T', 'd']), kT = kx.permute([0, 2, 1]);
  const gq = grid(q, { c, gap, title: (I, J) => `head ${J}` }), gk = grid(kT, { c, gap: gap + c, titles: false, rowLab: false });
  const yq = 10, yk = yq + gq.h + 30, ys = yk + 5 * c + 30;
  let s = at(gq, x0, yq) + at(gk, x0 + c / 2, yk);
  for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
    const eq = q.at([h, i, 0]), ek = kx.at([h, j, 0]), x = x0 + h * (slot + gap) + c / 2 + j * c, y = ys + i * c;
    s += `<g class="cell sc" data-s="${h},${i},${j}" data-links="${vkey(eq)} ${vkey(ek)}"><path d="M${x},${y}h${c}L${x},${y + c}Z" fill="${color(eq)[0]}"/><path d="M${x + c},${y}v${c}h${-c}Z" fill="${color(ek)[0]}"/><rect x="${x}" y="${y}" width="${c}" height="${c}" fill="none"/></g>`;
    if (h === 0 && j === 0) s += tx(x - 6, y + c / 2 + 0.5, 't' + i, 'lbl sm mid', 'end');
  }
  s += tx(x0 - 40, yq + 15 + 2 * c + 4, 'q', 'lbl code', 'end') + tx(x0 - 40, yq + 15 + 2 * c + 19, '(6, 4, 5)', 'lbl sm', 'end');
  s += tx(x0 - 40, yk + 2.5 * c, 'k.transpose(-2, -1)', 'lbl code', 'end') + tx(x0 - 40, yk + 2.5 * c + 15, '(6, 5, 4)', 'lbl sm', 'end');
  s += tx(x0 - 40, ys + 2 * c, 'q @ kᵀ', 'lbl code', 'end') + tx(x0 - 40, ys + 2 * c + 15, '(6, 4, 4)', 'lbl sm', 'end');
  for (let h = 0; h < H; h++) {
    const mx = x0 + h * (slot + gap) + slot / 2;
    s += tx(mx, yk - 9, '@', 'lbl op', 'middle') + tx(mx, ys - 9, '=', 'lbl op', 'middle');
  }
  return svg(ys + 4 * c + 12, s, W, 'Attention scores, head by head');
}

/* ---------- 6d. merge heads back ---------- */
function figMerge(api) {
  const bug = api.state.mode === 'bug', c = 5.8, out = mk('q', ['H', 'T', 'd']);
  const moved = bug ? out : out.permute([1, 0, 2]).contiguous(), fin = moved.reshape([T, H * D], ['T', null]);
  const s1 = strip(out.buf, { c, h: 18, dom: 5 }), s2 = strip(moved.buf, { c, h: 18, dom: 5, brackets: [30] }), g = grid(fin, { c: 16 });
  const x0 = (W - 120 * c) / 2, y1 = 26, y2 = y1 + 18 + 96, yg = y2 + 18 + 64;
  let s = tx(x0, 14, 'out, shape (6, 4, 5): memory is head after head', 'lbl code') + at(s1, x0, y1) +
    at(braid(out.buf, moved.buf, c, y1 + 19, y2 - 1), x0, 0) + at(s2, x0, y2) +
    tx(x0, y2 + 46, bug ? 'out.reshape(4, 30): same strip, new brackets every 30' : 'out.transpose(0, 1).contiguous(): memory is now token after token', 'lbl code') +
    tx((W - g.w) / 2, yg - 8, bug ? 'the four rows it produces' : '.view(4, 30): a bracket every 30', 'lbl code') + at(g, (W - g.w) / 2, yg);
  return svg(yg + g.h + 10, s, W, bug ? 'Reshaping without the transpose scrambles the rows' : 'Merging six heads back into one row per token');
}

/* ---------- Part 2, new layout: GQA drawn like the patches, one magnitude per axis (axis 0, the batch, is not drawn) ---------- */
// general version of the patch drawing: after the batch axis, every axis is told to go down ('v') or across ('h');
// within each direction the axes nest from big steps (the first one) to small steps (the last one)
function layG(tn, c, dirs, mg, padH) {
  const sh = tn.shape.slice(1), pick = d => sh.map((n, k) => k).filter(k => dirs[k] === d);
  const steps = (ks, pad) => {    // per level: how far one step moves, and how much one full run of that level covers
    const ext = [], step = []; let inner = c, depth = 0, off = 0, run = 0;
    for (let j = ks.length - 1; j >= 0; j--) {
      const n = sh[ks[j]], gap = depth >= 2 ? 20 : depth * mg; step[j] = inner + gap; ext[j] = n * inner + (n - 1) * gap;
      if (j === ks.length - 1) { run = ext[j]; if (pad > run) ext[j] = pad; }   // innermost run, left-aligned in its slot so arrows and badges line up
      inner = ext[j]; if (n > 1) depth++;
    }
    return { ks, ext, step, off, run };
  };
  const V = steps(pick('v'), 0), Hz = steps(pick('h'), padH || 0), items = [], idx = Array(sh.length).fill(0);
  const along = D => D.off + D.ks.reduce((s, k, j) => s + idx[k] * D.step[j], 0);
  for (let n = 0; n < tn.size; n++) {
    items.push({ e: tn.at([0].concat(idx)), idx: [0].concat(idx), x: along(Hz), y: along(V) });
    for (let k = sh.length - 1; k >= 0; k--) { if (++idx[k] < sh[k]) break; idx[k] = 0; }
  }
  return { items, w: Hz.ext[0], bh: V.ext[0], H: Hz, V };
}
function axesG(L, labels, avail) {
  const num = t => (/axis (\d)/.exec(t) || [])[1];
  const A = (x1, y1, x2, y2, t) => {
    const v = x1 === x2, bx = v ? x1 : x1 + 7, by = v ? y1 + 7 : y1;
    return `<path class="rp" d="M${x1},${y1}L${x2},${y2}" marker-end="url(#arr)"/><g class="rb"><circle cx="${bx}" cy="${by}" r="7"/><text x="${bx}" y="${by + 0.5}">${num(t)}</text></g>`;
  };
  let s = '';
  // across: the outermost axis on top, each arrow as long as one full run of its axis
  L.H.ks.forEach((k, j) => {
    const last = j === L.H.ks.length - 1, x1 = last ? L.H.off : 0, len = Math.max(last ? L.H.run : L.H.ext[j], 22), x2 = x1 + len;
    const t = labels[k], y = -14 - 16 * (L.H.ks.length - 1 - j);
    s += A(x1, y, x2, y, t) + (x2 + 8 + t.length * 6.8 <= avail ? tx(x2 + 8, y + 3.5, t, 'lbl') : tx(x1 + 18, y - 9, t, 'lbl'));
  });
  // down: one arrow per axis, the outermost furthest from the cells
  L.V.ks.forEach((k, j) => {
    const t = labels[k], x = -14 - 20 * (L.V.ks.length - 1 - j), len = Math.max(L.V.ext[j], 24);
    s += A(x, 0, x, len, t) + tx(-26 - 20 * (L.V.ks.length - 1), 18 + 16 * j, t, 'lbl', 'end');
  });
  return s;
}
function initGQA() {
  if (!document.getElementById('fig-gqa-0')) return;
  const c = 12, mg = 6, Lm = 172, NQ = T * H * D;
  const AXN = { T: 'T', kv: 'n_kv_heads', qp: 'q_per_kv_head', d: 'd_head' };
  // queries: memory as the projection writes it (token, head, feature); head h belongs to kv head h // R, at place h % R
  const q0 = mk('q', ['T', 'H', 'd']), k0 = mk('k', ['T', 'G', 'd']);
  const qs = new Tn(q0.buf, [1, T, G, R, D]), ks = new Tn(k0.buf, [1, T, G, 1, D]);          // after the reshape
  const qp = qs.permute([0, 2, 3, 1, 4]), kp = ks.permute([0, 2, 3, 1, 4]);                  // after the permute
  const kT = kp.permute([0, 1, 2, 4, 3]);                                                    // k.transpose(-2, -1)
  const kb = new Tn(k0.buf, [1, G, R, D, T], kT.strides.map((s, i) => i === 2 ? 0 : s));     // and broadcast along axis 2 (stride 0)
  const att = []; for (let g = 0; g < G; g++) for (let j = 0; j < R; j++) for (let i = 0; i < T; i++) for (let s = 0; s < T; s++) att.push({ g, j, i, s });
  // output: fresh from the matmul, so its memory is (kv head, place in group, token, feature) = head after head
  const o0 = mk('q', ['H', 'T', 'd']), ob = new Tn(o0.buf, [1, G, R, T, D]), om = ob.permute([0, 3, 1, 2, 4]), oc = om.contiguous();
  const ax4 = (a, b, cc, d) => ['axis 1: ' + a, 'axis 2: ' + b, 'axis 3: ' + cc, 'axis 4: ' + d];
  const ax2 = v => ['axis 1: T', 'axis 2: ' + v];
  const scell = it => {
    const a = it.e, eq = qp.at([0, a.g, a.j, a.i, 0]), ek = kp.at([0, a.g, 0, a.s, 0]), x = it.x, y = it.y;
    return `<g class="cell sc" data-s="${a.g},${a.j},${a.i},${a.s}" data-links="${vkey(eq)} ${vkey(ek)}"><path d="M${x},${y}h${c}L${x},${y + c}Z" fill="${color(eq)[0]}"/><path d="M${x + c},${y}v${c}h${-c}Z" fill="${color(ek)[0]}"/><rect x="${x}" y="${y}" width="${c}" height="${c}" fill="none"/></g>`;
  };
  const F = [
    { names: 'q[B, T, n_heads * d_head]', panels: [
      { lay: new Tn(q0.buf, [1, T, H * D]), dirs: 'vh', ax: ax2('n_heads * d_head'), code: 'q = wq(x)', shape: '(B, 4, 30)' },
      { lay: new Tn(k0.buf, [1, T, G * D]), dirs: 'vh', ax: ax2('n_kv_heads * d_head'), code: 'k = wk(x)', shape: '(B, 4, 10)' }] },
    { names: 'q[B, T, n_kv_heads, q_per_kv_head, d_head]', panels: [
      { lay: qs, dirs: 'vhhh', ax: ax4(AXN.T, AXN.kv, AXN.qp, AXN.d), code: 'q = q.reshape(B, T, n_kv_heads, q_per_kv_head, d_head)', shape: '(B, 4, 2, 3, 5)' },
      { lay: ks, dirs: 'vhhh', ax: ax4(AXN.T, AXN.kv, '1', AXN.d), code: 'k = k.reshape(B, T, n_kv_heads, 1, d_head)', shape: '(B, 4, 2, 1, 5)' }] },
    { names: 'q[B, n_kv_heads, q_per_kv_head, T, d_head]', panels: [
      { lay: qp, dirs: 'hhvh', ax: ax4(AXN.kv, AXN.qp, AXN.T, AXN.d), code: 'q = q.permute(0, 2, 3, 1, 4)', shape: '(B, 2, 3, 4, 5)' },
      { lay: kp, dirs: 'hhvh', ax: ax4(AXN.kv, '1', AXN.T, AXN.d), code: 'k = k.permute(0, 2, 3, 1, 4)', shape: '(B, 2, 1, 4, 5)' }] },
    { names: 'att[B, n_kv_heads, q_per_kv_head, T, T]', panels: [
      { lay: qp, dirs: 'hhvh', ax: ax4(AXN.kv, AXN.qp, AXN.T, AXN.d), code: 'q', shape: '(B, 2, 3, 4, 5)' },
      { lay: kb, dirs: 'hhvh', pad: 5 * c, ghost: it => it.idx[2] > 0, ax: ax4(AXN.kv, '1, read 3×', AXN.d, AXN.T), code: 'k.transpose(-2, -1): the dotted copies are the same matrix read again', shape: '(B, 2, 1, 5, 4), used as (B, 2, 3, 5, 4)' },
      { lay: new Tn(att, [1, G, R, T, T]), dirs: 'hhvh', pad: 5 * c, cellFn: scell, ax: ax4(AXN.kv, AXN.qp, 'T (asking)', 'T (looked at)'), code: 'att = q @ k.transpose(-2, -1)', shape: '(B, 2, 3, 4, 4)' }] },
    { names: 'out[B, T, n_kv_heads, q_per_kv_head, d_head]', strip: [o0.buf, 'the memory of out: head after head, untouched by the permute'], panels: [
      { lay: om, dirs: 'vhhh', ax: ax4(AXN.T, AXN.kv, AXN.qp, AXN.d), code: 'out = out.permute(0, 3, 1, 2, 4)', shape: '(B, 4, 2, 3, 5)' }] },
    { ribbons: true },
    { names: 'out[B, T, n_heads * d_head]', strip: [oc.buf, 'a new strip, token after token: this reshape had to copy'], panels: [
      { lay: new Tn(oc.buf, [1, T, H * D]), dirs: 'vh', ax: ax2('n_heads * d_head'), code: 'out = out.reshape(B, T, n_heads * d_head)', shape: '(B, 4, 30)' }] },
    { names: 'out[B, T, n_heads * d_head]', strip: [o0.buf, 'the same strip as before, cut every 30 numbers'], panels: [
      { lay: new Tn(o0.buf, [1, T, H * D]), dirs: 'vh', ax: ax2('n_heads * d_head'), code: 'out.reshape(B, T, n_heads * d_head), without the permute', shape: '(B, 4, 30)' }] }
  ];
  const group = F.map((fg, k) => {
    const fig = document.getElementById('fig-gqa-' + k), ro = $('.tl-readout', fig);
    if (fg.ribbons) {     // the copy made by the last reshape: every vector of five numbers goes from the old strip to the new one
      const cs = W / NQ, s1 = strip(o0.buf, { c: cs, h: 18, tag: false, dom: D }), s2 = strip(oc.buf, { c: cs, h: 18, tag: false, dom: D, brackets: [H * D] }), y1 = 26, y2 = y1 + 18 + 96;
      $('.tl-stage', fig).innerHTML = svg(y2 + s2.h + 24, tx(0, 14, 'out as the matmul left it: memory is head after head', 'lbl code') + at(s1, 0, y1) +
        at(braid(o0.buf, oc.buf, cs, y1 + 19, y2 - 1), 0, 0) + at(s2, 0, y2) + tx(0, y2 + s2.h + 14, 'what the reshape writes down: memory is now token after token', 'lbl code'), W, 'The copy made by the last reshape');
      return { fig, ro, rest: '' };
    }
    const Ls = fg.panels.map(p => layG(p.lay, c, p.dirs, mg, p.pad)), x0 = Lm;   // every block starts at the same x, so queries and keys line up
    let s = '', y = 0;
    fg.panels.forEach((p, i) => {
      const L = Ls[i], by = y + 34 + 14 + 16 * L.H.ks.length;
      let cells = ''; L.items.forEach(it => { cells += p.cellFn ? p.cellFn(it) : cell(it.x, it.y, c, c, it.e, { ghost: p.ghost && p.ghost(it) }); });
      s += tx(x0, y + 12, p.code, 'lbl code') + tx(x0, y + 25, p.shape, 'lbl sm') + at(axesG(L, p.ax, W - x0) + cells, x0, by);
      y = by + Math.max(L.bh, 24) + 22;
    });
    s = DEFS + `<g class="ctr">${s}</g>`;
    if (fg.strip) { s += tx(W / 2, y + 12, fg.strip[1], 'lbl code', 'middle') + at(strip(fg.strip[0], { c: W / NQ, h: 16, tag: false, dom: D }), 0, y + 22); y += 22 + 16 + 8; }
    $('.tl-stage', fig).innerHTML = svg(y, s, W, fg.panels.map(p => p.code).join('; '));
    // labels on the left have different widths in every figure, so measure what was drawn and centre it
    try { const g = fig.querySelector('.ctr'), bb = g.getBBox(); g.setAttribute('transform', `translate(${((W - bb.width) / 2 - bb.x).toFixed(1)},0)`); } catch (err) { /* not rendered: keep the default position */ }
    return { fig, ro, rest: `<code>${fg.names}</code>` };
  });
  const say = t => {
    const d = t.dataset;
    if (d.s) { const [g, j, i, s] = d.s.split(',').map(Number); return `<code>att[b, ${g}, ${j}, ${i}, ${s}]</code>: in query head ${g * R + j}, token ${i} looks at token ${s} through kv head ${g}`; }
    const m = /^([kq])(\d+)\.(\d+)\.(\d+)$/.exec(d.el || ''); if (!m) return null;
    return m[1] === 'q' ? `token ${m[3]}, query head ${m[2]} (kv head ${Math.floor(m[2] / R)}, number ${m[2] % R} in its group), feature ${m[4]}` : `token ${m[3]}, kv head ${m[2]}, feature ${m[4]}`;
  };
  const clear = () => group.forEach(g => { g.fig.classList.remove('hov'); g.fig.querySelectorAll('.on,.von').forEach(n => n.classList.remove('on', 'von')); if (g.ro) g.ro.innerHTML = blogCode(g.rest); });
  group.forEach(g => {
    g.fig.addEventListener('pointerover', ev => {
      const t = ev.target.closest('.cell,.rib'); clear(); if (!t || !g.fig.contains(t)) return;
      const links = t.dataset.links ? t.dataset.links.split(' ') : null, txt = say(t);
      group.forEach(h => {
        h.fig.classList.add('hov');
        if (links) h.fig.querySelectorAll('[data-v]').forEach(n => { if (links.includes(n.dataset.v)) n.classList.add('on'); });
        else {
          h.fig.querySelectorAll(`[data-v="${t.dataset.v}"]`).forEach(n => n.classList.add('von'));
          if (t.dataset.el) h.fig.querySelectorAll(`[data-el="${t.dataset.el}"]`).forEach(n => n.classList.add('on'));
        }
        if (txt && h.ro) h.ro.innerHTML = blogCode(txt);
      });
      t.classList.add('on');
    });
    g.fig.addEventListener('pointerleave', clear);
  });
  clear();
}

/* ---------- legend + hero ---------- */
function legend() {
  // one swatch per head (its hue, at a middle shade); the grey row is the only place that shows the token shades
  const sw = (kind, head) => `<span class="sw" style="background:${color({ kind, head, tok: 2 })[0]}"></span>`;
  const el = document.getElementById('tl-legend'); if (!el) return;
  el.innerHTML =
    `<div><span class="lg">key heads</span>${[0, 1].map(g => `<span class="grp">${sw('k', g)}<em>g${g}</em></span>`).join('')}</div>` +
    `<div><span class="lg">query heads</span>${[0, 1, 2, 3, 4, 5].map(h => `<span class="grp">${sw('q', h)}<em>h${h}</em></span>`).join('')}</div>` +
    `<div><span class="lg">tokens</span><span class="grp">${[0, 1, 2, 3].map(t => `<span class="sw" style="background:${hex(LV[t], 0, 0)}"></span>`).join('')}<em>t0 to t3, light to dark</em></span></div>`;
}
function hero() {
  const el = document.getElementById('hero-strip'); if (!el) return;
  const x = mk('k', ['T', 'G', 'd']);
  el.innerHTML = `<svg viewBox="0 0 800 20" preserveAspectRatio="none" aria-hidden="true">${strip(x.buf, { c: 20, tag: false }).s}</svg>`;
}

/* ---------- boot ---------- */
function boot() {
  legend();
  mount('fig-hook', figHook, { plain: 'x' });
  mount('fig-strip', figStrip, { hover: hoverStrip });
  mount('fig-reshape', figReshape, { state: { shape: '4,2,5' }, plain: 'x', note: api => RS[api.state.shape] });
  mount('fig-mirror', figMirror, { plain: 'A' });
  initPerm3();
  initPatch('fig-patch', 6, 6, 3, 20);
  initGQA();
  mount('fig-read', figRead, { plain: 'x' });
  mount('fig-contig', figContig, { plain: 'x' });
  mount('fig-heads', figHeads, { khead: 'key head' });
  mount('fig-share', figShare, { state: { mode: 'interleave' }, khead: 'key head' });
  mount('fig-scores', figScores, { khead: 'key head' });
  mount('fig-merge', figMerge, { state: { mode: 'ok' } });
  mount('fig-merge-bad', figMerge, { state: { mode: 'bug' } });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();

})();
