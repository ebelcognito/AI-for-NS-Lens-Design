"""Render catalogue.json as a single self-contained HTML page.

Design decisions worth keeping:

  * ray paths are drawn as INLINE SVG, not PNG. A rendered PNG of one design is ~150 kB;
    the same design as SVG geometry is ~1.5 kB, so the whole catalogue fits in one page
    with room to spare, needs no image decode, and stays sharp at any zoom.
  * the geometry is embedded as JSON and drawn client-side, so sorting and filtering do
    not require re-rendering anything server-side.

    python ns_probe/render_catalogue.py --catalogue runs/catalogue.json --out runs/catalogue.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAGE = """<title>Ray Path Catalogue</title>
<style>
:root {
  --ground:#F2F5F4; --panel:#FFFFFF; --panel-2:#E9EEED; --line:#D2DBD9;
  --ink:#141C1E; --muted:#5C6F71; --faint:#8CA0A1;
  --ray:#C97A12; --glass:#2E8B99; --glass-fill:rgba(46,139,153,.12);
  --ok:#2E7D5B; --warn:#9A6B1F; --bad:#B4503F; --focus:#2E8B99;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#0D1315; --panel:#151E20; --panel-2:#1D282A; --line:#293638;
    --ink:#E2EAEA; --muted:#8A9C9E; --faint:#617476;
    --ray:#F0A830; --glass:#5CC3D0; --glass-fill:rgba(92,195,208,.13);
    --ok:#5FBF8F; --warn:#D9A441; --bad:#D2685A; --focus:#5CC3D0;
  }
}
:root[data-theme="dark"] {
  --ground:#0D1315; --panel:#151E20; --panel-2:#1D282A; --line:#293638;
  --ink:#E2EAEA; --muted:#8A9C9E; --faint:#617476;
  --ray:#F0A830; --glass:#5CC3D0; --glass-fill:rgba(92,195,208,.13);
  --ok:#5FBF8F; --warn:#D9A441; --bad:#D2685A; --focus:#5CC3D0;
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--ground); color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.mono { font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace; }
.wrap { max-width:1240px; margin:0 auto; padding:40px 24px 80px; }

header { border-bottom:1px solid var(--line); padding-bottom:26px; margin-bottom:8px; }
h1 { margin:0 0 6px; font-size:29px; letter-spacing:-.02em; font-weight:620;
     text-wrap:balance; }
.sub { margin:0; color:var(--muted); max-width:66ch; }
.eyebrow { font-size:11px; letter-spacing:.13em; text-transform:uppercase;
           color:var(--faint); margin:0 0 10px; }

.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(132px,1fr));
         gap:1px; background:var(--line); border:1px solid var(--line);
         margin:26px 0 8px; }
.stat { background:var(--panel); padding:13px 15px; }
.stat b { display:block; font-size:22px; font-weight:620; letter-spacing:-.01em;
          font-variant-numeric:tabular-nums; }
.stat span { font-size:11px; letter-spacing:.07em; text-transform:uppercase;
             color:var(--faint); }

.note { background:var(--panel); border:1px solid var(--line);
        border-left:3px solid var(--warn); padding:13px 16px; margin:22px 0;
        color:var(--muted); font-size:13.5px; }
.note b { color:var(--ink); font-weight:600; }

.bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center;
       margin:26px 0 12px; }
button.f, select {
  font:inherit; font-size:13px; padding:6px 12px; border:1px solid var(--line);
  background:var(--panel); color:var(--muted); cursor:pointer;
}
button.f[aria-pressed="true"] { background:var(--panel-2); color:var(--ink);
                                border-color:var(--focus); }
button.f:focus-visible, select:focus-visible, tr:focus-visible {
  outline:2px solid var(--focus); outline-offset:1px; }
.count { margin-left:auto; color:var(--faint); font-size:12.5px;
         font-variant-numeric:tabular-nums; }

.tablewrap { overflow-x:auto; border:1px solid var(--line); background:var(--panel); }
table { border-collapse:collapse; width:100%; min-width:920px; }
th { text-align:left; font-size:10.5px; letter-spacing:.09em; text-transform:uppercase;
     color:var(--faint); font-weight:600; padding:11px 14px;
     border-bottom:1px solid var(--line); white-space:nowrap; background:var(--panel); }
th.num, td.num { text-align:right; font-variant-numeric:tabular-nums; }
td { padding:11px 14px; border-bottom:1px solid var(--line); vertical-align:middle; }
tbody tr { cursor:pointer; }
tbody tr:hover { background:var(--panel-2); }
tbody tr.open { background:var(--panel-2); }
.sig { font-size:12.5px; letter-spacing:-.01em; white-space:nowrap; }
.sig .tir { color:var(--ray); font-weight:600; }
.sig .arrow { color:var(--faint); }

.pill { display:inline-block; font-size:10.5px; letter-spacing:.05em; padding:2px 7px;
        border:1px solid currentColor; text-transform:uppercase; white-space:nowrap; }
.pill.ok { color:var(--ok); } .pill.bad { color:var(--bad); }
.pill.warn { color:var(--warn); }

.thumb { width:132px; height:44px; display:block; }
.spark { width:78px; height:22px; display:block; }

.detail td { background:var(--ground); padding:20px 14px 26px; }
.dgrid { display:grid; grid-template-columns:minmax(0,1.55fr) minmax(0,1fr); gap:26px; }
@media (max-width:820px) { .dgrid { grid-template-columns:1fr; } }
.big { width:100%; height:auto; border:1px solid var(--line); background:var(--panel); }
.kv { display:grid; grid-template-columns:auto 1fr; gap:5px 16px; font-size:13px;
      align-content:start; }
.kv dt { color:var(--faint); } .kv dd { margin:0; font-variant-numeric:tabular-nums; }
.legend { display:flex; gap:16px; font-size:11.5px; color:var(--faint);
          margin-top:9px; flex-wrap:wrap; }
.swatch { display:inline-block; width:14px; height:2px; vertical-align:middle;
          margin-right:5px; }
footer { margin-top:44px; padding-top:18px; border-top:1px solid var(--line);
         color:var(--faint); font-size:12.5px; }
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Non-sequential fold search &middot; __ROWS__ scored evaluations</p>
  <h1>Ray Path Catalogue</h1>
  <p class="sub">Every interaction topology the search found, with the best design ever
  produced for it &mdash; including the __LOST__ that the quality-diversity archive
  discarded because a different ray path occupied the same cell.</p>
</header>

<div class="stats">
  <div class="stat"><b>__NSIG__</b><span>topologies found</span></div>
  <div class="stat"><b>__NCAT__</b><span>in catalogue</span></div>
  <div class="stat"><b>__NPASS__</b><span>pass hardened</span></div>
  <div class="stat"><b>__NSTABLE__</b><span>topology stable</span></div>
  <div class="stat"><b>__BEST__</b><span>best worst-field</span></div>
</div>

<p class="note"><b>Hardened score is the only comparable number here.</b> The trace scores
came from runs with different field sets and different gates, and several of the highest
were exploits &mdash; a body intersecting the detector plane, a bulb-shaped face, light
focused on axis and blind at the edge of field. Every entry was re-traced under one
config: worst-of-nine-fields, sag ratio &le; 0.268, 12&nbsp;mm detector clearance,
clean-packing required at every field. <b>Rejected rows are kept deliberately</b> &mdash;
a topology that fails the gates is still a real route through the glass, and the failure
is part of the record.</p>

<div class="bar">
  <button class="f" data-filter="all" aria-pressed="true">All</button>
  <button class="f" data-filter="pass" aria-pressed="false">Passes hardened</button>
  <button class="f" data-filter="stable" aria-pressed="false">Topology stable</button>
  <button class="f" data-filter="lost" aria-pressed="false">Never archived</button>
  <select id="folds">
    <option value="">Any fold count</option>
  </select>
  <select id="sort">
    <option value="hard">Sort: hardened score</option>
    <option value="trace">Sort: trace score</option>
    <option value="seen">Sort: times encountered</option>
    <option value="folds">Sort: fold count</option>
  </select>
  <span class="count" id="count"></span>
</div>

<div class="tablewrap">
<table>
  <thead><tr>
    <th>Ray path</th><th>Light path</th><th class="num">Hardened</th>
    <th class="num">Trace</th><th>Across field</th><th class="num">Folds</th>
    <th class="num">Seen</th><th>Status</th>
  </tr></thead>
  <tbody id="rows"></tbody>
</table>
</div>

<div class="legend">
  <span><i class="swatch" style="background:var(--glass)"></i>element body</span>
  <span><i class="swatch" style="background:var(--ray)"></i>dominant ray bundle</span>
  <span><i class="swatch" style="background:var(--faint)"></i>detector plane</span>
  <span>T = transmit &middot; TIR = total internal reflection &middot; PR = partial
  reflection &middot; A B T D = faces (A left, B right, T bottom edge, D top edge)</span>
</div>

<footer>Built __BUILT__ from __ROWS__ scored rows across the fold-search traces.
Geometry re-traced through the ray-optics engine; nothing here is a stored image.</footer>
</div>

<script>
const DATA = __DATA__;
const XS = 120;

function bounds(e) {
  let x0=0, x1=XS, y0=-32, y1=32;
  for (const v of (e.outline||[])) { y0=Math.min(y0,v.y-4); y1=Math.max(y1,v.y+4);
                                     x0=Math.min(x0,v.x-4); x1=Math.max(x1,v.x+4); }
  for (const r of (e.rays||[])) for (const p of r) { y0=Math.min(y0,p[1]); y1=Math.max(y1,p[1]); }
  return [x0,x1,y0,y1];
}
function outlinePath(o) {
  if (!o || !o.length) return "";
  let d = "", prev = null, pending = null;
  const pts = o.concat([o[0]]);
  for (const v of pts) {
    if (v.arc) { pending = v; continue; }
    if (prev === null) { d += `M ${v.x} ${v.y}`; }
    else if (pending) {
      // quadratic through the on-arc control point at t=0.5
      const cx = 2*pending.x - (prev.x + v.x)/2, cy = 2*pending.y - (prev.y + v.y)/2;
      d += ` Q ${cx.toFixed(2)} ${cy.toFixed(2)} ${v.x} ${v.y}`;
    } else { d += ` L ${v.x} ${v.y}`; }
    prev = v; pending = null;
  }
  return d + " Z";
}
function svg(e, w, h, maxRays) {
  const [x0,x1,y0,y1] = bounds(e);
  const rays = (e.rays||[]).filter((_,i,a) => a.length<=maxRays || i % Math.ceil(a.length/maxRays) === 0);
  const paths = rays.map(r => `<polyline points="${r.map(p=>p[0]+','+p[1]).join(' ')}"
      fill="none" stroke="var(--ray)" stroke-width="0.5" stroke-linejoin="round"
      opacity="0.75"/>`).join('');
  return `<svg class="${w>200?'big':'thumb'}" viewBox="${x0} ${y0} ${x1-x0} ${y1-y0}"
    preserveAspectRatio="xMidYMid meet" role="img"
    aria-label="ray path ${e.signature}" ${w>200?`style="aspect-ratio:${(x1-x0)}/${(y1-y0)}"`:''}>
    <g transform="scale(1,-1) translate(0,${-(y0+y1)})">
    <line x1="${XS}" y1="${y0}" x2="${XS}" y2="${y1}" stroke="var(--faint)"
          stroke-width="0.7" stroke-dasharray="2 2"/>
    <path d="${outlinePath(e.outline)}" fill="var(--glass-fill)" stroke="var(--glass)"
          stroke-width="0.8"/>${paths}</g></svg>`;
}
function spark(e) {
  const pf = e.per_field; if (!pf || !pf.length) return '<span style="color:var(--faint)">&mdash;</span>';
  const n = pf.length, mx = Math.max(...pf, 0.01);
  const bars = pf.map((v,i) => {
    const bw = 100/n, x = i*bw, hh = Math.max(1.5, 24*v/mx);
    const c = v <= 0.001 ? 'var(--bad)' : 'var(--ray)';
    return `<rect x="${x+bw*0.15}" y="${24-hh}" width="${bw*0.7}" height="${hh}" fill="${c}"/>`;
  }).join('');
  return `<svg class="spark" viewBox="0 0 100 24" preserveAspectRatio="none" role="img"
    aria-label="throughput at each of ${n} fields">${bars}</svg>`;
}
function fmtSig(s) {
  return s.split(' -> ').map(t => {
    const cls = t.startsWith('TIR') ? 'tir' : '';
    return `<span class="${cls}">${t}</span>`;
  }).join('<span class="arrow"> &rarr; </span>');
}
function statusOf(e) {
  if (e.verdict !== 'PASS') return ['bad','rejected'];
  if (e.hardened_signature !== e.signature) return ['warn','path shifted'];
  if ((e.hardened_fitness||0) <= 0.001) return ['warn','blind at a field'];
  return ['ok','holds'];
}
const tbody = document.getElementById('rows');
const countEl = document.getElementById('count');
let filter = 'all';

const folds = [...new Set(DATA.map(e=>e.tir_count).filter(v=>v!=null))].sort((a,b)=>a-b);
const fsel = document.getElementById('folds');
for (const f of folds) fsel.insertAdjacentHTML('beforeend', `<option value="${f}">${f} folds</option>`);

function visible() {
  let v = DATA.slice();
  if (filter === 'pass') v = v.filter(e => e.verdict==='PASS' && (e.hardened_fitness||0) > 0.001);
  if (filter === 'stable') v = v.filter(e => e.hardened_signature === e.signature);
  if (filter === 'lost') v = v.filter(e => e.never_archived);
  const f = fsel.value;
  if (f !== '') v = v.filter(e => String(e.tir_count) === f);
  const s = document.getElementById('sort').value;
  const key = {hard:e=>-(e.hardened_fitness ?? -1), trace:e=>-e.trace_fitness,
               seen:e=>-e.times_seen, folds:e=>e.tir_count ?? 99}[s];
  return v.sort((a,b)=>key(a)-key(b));
}
function render() {
  const v = visible();
  countEl.textContent = `${v.length} of ${DATA.length} topologies`;
  tbody.innerHTML = v.map((e,i) => {
    const [cls,label] = statusOf(e);
    const hf = e.hardened_fitness == null ? '&mdash;' : e.hardened_fitness.toFixed(3);
    return `<tr tabindex="0" data-i="${DATA.indexOf(e)}">
      <td>${svg(e,132,44,7)}</td>
      <td class="sig mono">${fmtSig(e.signature)}</td>
      <td class="num mono">${hf}</td>
      <td class="num mono">${e.trace_fitness.toFixed(3)}</td>
      <td>${spark(e)}</td>
      <td class="num mono">${e.tir_count ?? '&mdash;'}</td>
      <td class="num mono">${e.times_seen}</td>
      <td><span class="pill ${cls}">${label}</span></td>
    </tr>`;
  }).join('');
}
function detailHTML(e) {
  const p = e.params || {};
  const g = (k,d=2) => p[k] == null ? '&mdash;' : p[k].toFixed(d);
  const num = (v,d=3) => v == null ? '&mdash;' : (+v).toFixed(d);
  return `<tr class="detail"><td colspan="8"><div class="dgrid">
    <div>${svg(e,900,300,99)}
      <div class="legend"><span>central field, dominant bundle
      (${(e.rays||[]).length} rays)</span></div></div>
    <dl class="kv">
      <dt>hardened</dt><dd>${num(e.hardened_fitness)} worst of nine fields</dd>
      <dt>as traced</dt><dd>${num(e.trace_fitness)} <span style="color:var(--faint)">(mixed config)</span></dd>
      <dt>path now</dt><dd class="mono" style="font-size:12px">${e.hardened_signature || '&mdash;'}</dd>
      <dt>spot</dt><dd>${num(e.spot,3)} mm</dd>
      <dt>thinness</dt><dd>${num(e.thinness,1)} mm</dd>
      <dt>crit margin</dt><dd>${num(e.crit_margin,2)}&deg;</dd>
      <dt>NS fraction</dt><dd>${num(e.ns_frac,3)}</dd>
      <dt>index n</dt><dd>${g('n',4)}</dd>
      <dt>thickness</dt><dd>${g('thick',2)} mm</dd>
      <dt>tilts</dt><dd>${g('tilt_a')}&deg; / ${g('tilt_b')}&deg;</dd>
      <dt>sags f/b</dt><dd>${g('fsag')} / ${g('bsag')} mm</dd>
      <dt>found in</dt><dd class="mono" style="font-size:12px">${e.source_run}/${e.source_seed}</dd>
      <dt>encountered</dt><dd>${e.times_seen}&times; across the search</dd>
    </dl></div></td></tr>`;
}
tbody.addEventListener('click', ev => toggle(ev.target.closest('tr')));
tbody.addEventListener('keydown', ev => {
  if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); toggle(ev.target.closest('tr')); }
});
function toggle(tr) {
  if (!tr || tr.classList.contains('detail')) return;
  const nxt = tr.nextElementSibling;
  if (nxt && nxt.classList.contains('detail')) { nxt.remove(); tr.classList.remove('open'); return; }
  document.querySelectorAll('tr.detail').forEach(n=>n.remove());
  document.querySelectorAll('tr.open').forEach(n=>n.classList.remove('open'));
  tr.classList.add('open');
  tr.insertAdjacentHTML('afterend', detailHTML(DATA[+tr.dataset.i]));
}
document.querySelectorAll('button.f').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('button.f').forEach(o=>o.setAttribute('aria-pressed','false'));
  b.setAttribute('aria-pressed','true'); filter = b.dataset.filter; render();
}));
fsel.addEventListener('change', render);
document.getElementById('sort').addEventListener('change', render);
render();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default="runs/catalogue.json")
    ap.add_argument("--out", default="runs/catalogue.html")
    a = ap.parse_args()

    cat = json.loads(Path(a.catalogue).read_text())
    meta, entries = cat["meta"], cat["entries"]

    # which topologies would the (tir_count, thinness) archive have discarded?
    winner: dict = {}
    for e in entries:
        key = (e.get("tir_count"), round((e.get("trace_thinness") or 0) / 6.875))
        cur = winner.get(key)
        if cur is None or e["trace_fitness"] > cur[1]:
            winner[key] = (e["signature"], e["trace_fitness"])
    won = {v[0] for v in winner.values()}

    slim = []
    for e in entries:
        e["never_archived"] = e["signature"] not in won
        slim.append({k: e.get(k) for k in (
            "signature", "hardened_signature", "hardened_fitness", "trace_fitness",
            "verdict", "tir_count", "times_seen", "per_field", "spot", "thinness",
            "crit_margin", "ns_frac", "params", "outline", "rays",
            "source_run", "source_seed", "never_archived")})

    passing = [e for e in slim if e["verdict"] == "PASS" and (e["hardened_fitness"] or 0) > 0.001]
    best = max((e["hardened_fitness"] or 0) for e in slim) if slim else 0.0

    html = (PAGE
            .replace("__DATA__", json.dumps(slim, separators=(",", ":")))
            .replace("__ROWS__", f"{meta['scored_trace_rows']:,}")
            .replace("__NSIG__", str(meta["distinct_signatures_all"]))
            .replace("__NCAT__", str(len(slim)))
            .replace("__NPASS__", str(len(passing)))
            .replace("__NSTABLE__", str(meta["topology_stable"]))
            .replace("__BEST__", f"{best:.3f}")
            .replace("__LOST__", f"{sum(1 for e in slim if e['never_archived'])} never archived")
            .replace("__BUILT__", meta["built"]))
    Path(a.out).write_text(html)
    print(f"wrote {a.out}  ({Path(a.out).stat().st_size/1e6:.2f} MB, {len(slim)} entries)")


if __name__ == "__main__":
    main()
