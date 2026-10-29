"""Export the ray-path catalogue as a plain directory of files.

Not a single self-contained page and not an artifact: one SVG per design on disk, plus
CSV/JSON indices and an index.html that references them. Everything is local and
readable without a server -- open catalogue/index.html with file:// and it works.

SVG rather than PNG for three reasons that all matter here:
  * .gitignore ignores *.png/*.jpg but NOT *.svg, so these files are version-controlled
  * ~2 kB each against ~150 kB for a render, so 171 designs is under half a megabyte
  * they are the traced ray geometry itself, not a picture of it -- zoom is lossless
    and the coordinates can be read straight out of the file

Each SVG is STANDALONE: literal colours, no CSS variables. An SVG loaded through an
<img> tag is an isolated document -- no parent stylesheet reaches it -- so the palette
is chosen to read on both light and dark grounds and the background stays transparent.

    python ns_probe/export_catalogue_dir.py --catalogue runs/catalogue.json --out catalogue
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# mid-tone palette: legible on a white page and on a dark one, since these files are
# viewed both inside index.html and on their own
C_GLASS = "#3E9CAA"
C_GLASS_FILL = "rgba(62,156,170,0.15)"
C_RAY = "#D98A1F"
C_PLANE = "#8A9A9C"
XS = 120.0


def slug(signature: str) -> str:
    """`T@A -> TIR@D -> TIR@B -> T@B` -> `A-Dt-Bt-B`  (lowercase t marks a TIR)."""
    parts = []
    for tok in signature.split(" -> "):
        if "@" not in tok:
            continue
        kind, face = tok.split("@")
        kind = kind.strip().upper()
        parts.append(face.strip() + {"TIR": "t", "PR": "p"}.get(kind, ""))
    return re.sub(r"[^A-Za-z0-9-]", "", "-".join(parts))


def outline_path(o) -> str:
    if not o:
        return ""
    d, prev, pending = "", None, None
    for v in list(o) + [o[0]]:
        if v.get("arc"):
            pending = v
            continue
        if prev is None:
            d += f"M {v['x']:.3f} {v['y']:.3f}"
        elif pending:
            cx = 2 * pending["x"] - (prev["x"] + v["x"]) / 2
            cy = 2 * pending["y"] - (prev["y"] + v["y"]) / 2
            d += f" Q {cx:.3f} {cy:.3f} {v['x']:.3f} {v['y']:.3f}"
        else:
            d += f" L {v['x']:.3f} {v['y']:.3f}"
        prev, pending = v, None
    return d + " Z"


def make_svg(e) -> str:
    x0, x1, y0, y1 = 0.0, XS, -32.0, 32.0
    for v in e.get("outline") or []:
        x0, x1 = min(x0, v["x"] - 4), max(x1, v["x"] + 4)
        y0, y1 = min(y0, v["y"] - 4), max(y1, v["y"] + 4)
    for r in e.get("rays") or []:
        for p in r:
            y0, y1 = min(y0, p[1]), max(y1, p[1])
    rays = "".join(
        f'<polyline points="{" ".join(f"{p[0]},{p[1]}" for p in r)}" fill="none" '
        f'stroke="{C_RAY}" stroke-width="0.5" stroke-linejoin="round" opacity="0.8"/>'
        for r in e.get("rays") or [])
    fit = e.get("hardened_fitness")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {y0:.2f} '
        f'{x1-x0:.2f} {y1-y0:.2f}" width="{(x1-x0)*4:.0f}" height="{(y1-y0)*4:.0f}">\n'
        f'  <title>{e["signature"]}</title>\n'
        f'  <desc>hardened worst-field throughput '
        f'{"n/a" if fit is None else f"{fit:.4f}"}; '
        f'{len(e.get("rays") or [])} rays of the dominant bundle at the central field. '
        f'Faces: A left, B right, T bottom edge, D top edge.</desc>\n'
        f'  <g transform="scale(1,-1) translate(0,{-(y0+y1):.3f})">\n'
        f'    <line x1="{XS}" y1="{y0:.2f}" x2="{XS}" y2="{y1:.2f}" stroke="{C_PLANE}" '
        f'stroke-width="0.7" stroke-dasharray="2 2"/>\n'
        f'    <path d="{outline_path(e.get("outline"))}" fill="{C_GLASS_FILL}" '
        f'stroke="{C_GLASS}" stroke-width="0.8"/>\n'
        f'    {rays}\n  </g>\n</svg>\n')


FIELDS = ["file", "rank", "signature", "hardened_fitness", "trace_fitness", "verdict",
          "topology_stable", "folds", "thinness_mm", "spot_mm", "crit_margin_deg",
          "ns_frac", "times_seen", "never_archived", "relative_sag",
          "source_run", "source_seed", "n_rays_drawn"]


def row_of(e, fname, rank):
    p = e.get("params") or {}
    return {
        "file": fname, "rank": rank, "signature": e["signature"],
        "hardened_fitness": e.get("hardened_fitness"),
        "trace_fitness": round(e["trace_fitness"], 4),
        "verdict": e["verdict"],
        "topology_stable": e.get("hardened_signature") == e["signature"],
        "folds": e.get("tir_count"),
        "thinness_mm": e.get("thinness"), "spot_mm": e.get("spot"),
        "crit_margin_deg": e.get("crit_margin"), "ns_frac": e.get("ns_frac"),
        "times_seen": e.get("times_seen"), "never_archived": e.get("never_archived"),
        "relative_sag": e.get("relative_sag"),
        "source_run": e.get("source_run"), "source_seed": e.get("source_seed"),
        "n_rays_drawn": len(e.get("rays") or []),
        "_params": {k: round(v, 6) for k, v in p.items()},
        "_genotype": e.get("genotype"),
        "_per_field": e.get("per_field"),
        "_hardened_signature": e.get("hardened_signature"),
    }


def build_index_html(rows, meta) -> str:
    cards = []
    for r in rows:
        fit = r["hardened_fitness"]
        badge = ("rejected" if r["verdict"] != "PASS"
                 else "blind at a field" if (fit or 0) <= 0.001
                 else "path shifted" if not r["topology_stable"] else "holds")
        cls = {"rejected": "bad", "blind at a field": "warn",
               "path shifted": "warn", "holds": "ok"}[badge]
        cards.append(
            f'<figure class="card" id="{Path(r["file"]).stem}">\n'
            f'  <a href="svg/{r["file"]}"><img src="svg/{r["file"]}" loading="lazy"\n'
            f'       alt="ray path {r["signature"]}"></a>\n'
            f'  <figcaption>\n'
            f'    <code>{r["signature"]}</code>\n'
            f'    <div class="meta"><b>{"—" if fit is None else f"{fit:.3f}"}</b>'
            f'<span class="pill {cls}">{badge}</span>'
            f'<span>{r["folds"]} folds</span>'
            f'<span>{r["times_seen"]}&times; seen</span></div>\n'
            f'  </figcaption>\n</figure>')
    passing = sum(1 for r in rows if r["verdict"] == "PASS" and (r["hardened_fitness"] or 0) > 0.001)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ray Path Catalogue</title>
<style>
:root {{ --bg:#F3F5F5; --card:#fff; --line:#D3DBDA; --ink:#151D1F; --muted:#5D7072;
        --ok:#2E7D5B; --warn:#946515; --bad:#B04E3E; }}
@media (prefers-color-scheme:dark) {{
  :root {{ --bg:#0E1416; --card:#161F21; --line:#2A3739; --ink:#E1E9E9; --muted:#899B9D;
           --ok:#5FBF8F; --warn:#D9A441; --bad:#D2685A; }} }}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1300px;margin:0 auto;padding:36px 22px 70px}}
h1{{font-size:27px;margin:0 0 6px;letter-spacing:-.02em}}
p.sub{{color:var(--muted);margin:0 0 4px;max-width:70ch}}
.stats{{display:flex;flex-wrap:wrap;gap:22px;margin:22px 0 6px;padding:14px 0;
 border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}
.stats div b{{display:block;font-size:20px;font-variant-numeric:tabular-nums}}
.stats div span{{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}}
.files{{font-size:13.5px;color:var(--muted);margin:18px 0}}
.files code{{color:var(--ink)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px;margin-top:22px}}
.card{{margin:0;background:var(--card);border:1px solid var(--line);padding:10px}}
.card img{{width:100%;height:auto;display:block}}
figcaption code{{font-size:11.5px;word-break:break-all;display:block;margin:8px 0 6px;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.meta{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:12px;
 color:var(--muted);font-variant-numeric:tabular-nums}}
.meta b{{font-size:15px;color:var(--ink)}}
.pill{{font-size:10px;letter-spacing:.04em;text-transform:uppercase;padding:1px 6px;
 border:1px solid currentColor}}
.pill.ok{{color:var(--ok)}} .pill.warn{{color:var(--warn)}} .pill.bad{{color:var(--bad)}}
footer{{margin-top:36px;padding-top:16px;border-top:1px solid var(--line);
 color:var(--muted);font-size:12.5px}}
</style></head><body><div class="wrap">
<h1>Ray Path Catalogue</h1>
<p class="sub">Every interaction topology the non-sequential fold search found, with the
best design produced for it, re-scored under one hardened config.</p>
<div class="stats">
  <div><b>{meta.get('distinct_signatures_all')}</b><span>topologies found</span></div>
  <div><b>{len(rows)}</b><span>in catalogue</span></div>
  <div><b>{passing}</b><span>pass hardened</span></div>
  <div><b>{meta.get('scored_trace_rows'):,}</b><span>scored evaluations</span></div>
</div>
<p class="files">Files in this directory —
<code>svg/</code> one standalone SVG per design (the traced geometry itself) ·
<code>index.csv</code> flat table for spreadsheets ·
<code>index.json</code> same with parameters, genotype and per-field profile ·
<code>catalogue.json</code> the full record including ray polylines.</p>
<p class="files">Signature notation: <code>T</code> transmit, <code>TIR</code> total
internal reflection, <code>PR</code> partial reflection; faces <code>A</code> left,
<code>B</code> right, <code>T</code> bottom edge, <code>D</code> top edge. Cards with no
rays drawn are designs whose dominant path never reaches the detector.</p>
<div class="grid">
{chr(10).join(cards)}
</div>
<footer>Built {meta.get('built')} · hardened config: worst of nine fields,
max_sag_ratio 0.268, 12 mm detector clearance, clean-packing at every field.
Rejected topologies are kept deliberately — a path that fails the gates is still a real
route through the glass.</footer>
</div></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default="runs/catalogue.json")
    ap.add_argument("--out", default="catalogue")
    a = ap.parse_args()

    cat = json.loads(Path(a.catalogue).read_text())
    meta, entries = cat["meta"], cat["entries"]

    # Would the (tir_count, thinness) archive have kept this topology, or deleted it?
    # Computed here rather than read from catalogue.json -- the builder does not record
    # it, and reading a key that is never written gives a silently empty column.
    winner = {}
    for e in entries:
        key = (e.get("tir_count"), round((e.get("trace_thinness") or 0) / 6.875))
        if key not in winner or e["trace_fitness"] > winner[key][1]:
            winner[key] = (e["signature"], e["trace_fitness"])
    won = {v[0] for v in winner.values()}
    for e in entries:
        e["never_archived"] = e["signature"] not in won

    entries.sort(key=lambda e: -(e.get("hardened_fitness") or -1))

    out = Path(a.out)
    svg_dir = out / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    for old in svg_dir.glob("*.svg"):
        old.unlink()

    rows = []
    for i, e in enumerate(entries, 1):
        fit = e.get("hardened_fitness")
        score = "xxxx" if fit is None else f"{int(round(fit*1000)):04d}"
        fname = f"{i:03d}_{score}_{e.get('tir_count','x')}f_{slug(e['signature'])}.svg"
        (svg_dir / fname).write_text(make_svg(e))
        rows.append(row_of(e, fname, i))

    with (out / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "index.json").write_text(json.dumps({"meta": meta, "designs": rows}, indent=1))
    (out / "index.html").write_text(build_index_html(rows, meta))

    total = sum(f.stat().st_size for f in out.rglob("*"))
    print(f"wrote {out}/")
    print(f"  svg/            {len(rows)} files")
    print(f"  index.html      browsable grid")
    print(f"  index.csv       {len(FIELDS)} columns")
    print(f"  index.json      + params, genotype, per-field")
    print(f"  total on disk   {total/1e6:.2f} MB")


if __name__ == "__main__":
    main()
