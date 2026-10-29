"""Render the best designs of a QD run at EVERY field angle it was scored on.

One PNG per (design, field), plus a contact sheet per design that puts the whole field
sweep in a single image -- which is the one to open when you just want to see whether a
design still folds and still lands at the edge of the field.

Reads `archive.json` when the run has finished, and falls back to `trace.jsonl` while it
is still going, so the same command works mid-run.

    PYTHONPATH=.:ns_probe python ns_probe/render_best_designs.py                  # top 6
    PYTHONPATH=.:ns_probe python ns_probe/render_best_designs.py --top 12
    PYTHONPATH=.:ns_probe python ns_probe/render_best_designs.py --per-niche
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
from pathlib import Path

import numpy as np

from nslens.engine import Runner
from nslens.qd import experiments
from nslens.qd.observation import measure
from nslens.scene import CropBox

REPO = Path(__file__).resolve().parents[1]

HQ_RAYS = 61            # denser than the 17 used for scoring: this is for the eye
HQ_W = 1100
HQ_DEPTH = 24
PAD = 8.0


def _layout(problem, p, field_deg, *, rays, crop=None):
    base = problem.layout(p, field_deg=field_deg)
    return type(base)(
        source=type(base.source)(**{**base.source.__dict__, "n_rays": rays}),
        element=base.element, detector=base.detector,
        max_depth=HQ_DEPTH, record=False, crop=crop)


def frame_for(runner, problem, p, fields) -> tuple:
    """One crop containing every ray path at every field, so the panels of a design are
    directly comparable rather than each being auto-framed to itself."""
    ys, xs = [], [0.0, problem.x_sensor]
    for fd in fields:
        lay = problem.layout(p, field_deg=fd)
        lay = type(lay)(source=type(lay.source)(**{**lay.source.__dict__, "n_rays": 9}),
                        element=lay.element, detector=lay.detector,
                        max_depth=HQ_DEPTH, record=True)
        try:
            res, _ = runner.simulate(lay.to_scene_dict())
        except Exception:
            continue
        for r in res.get("raySequences", []) or []:
            for v in r["seq"]:
                xs.append(float(v["x"])); ys.append(float(v["y"]))
            ys.append(float(r["y"]))
    if not ys:
        ys = [-problem.half_h, problem.half_h]
    return ((min(xs) - PAD, min(ys) - PAD), (max(xs) + PAD, max(ys) + PAD))


def render(runner, problem, p, field_deg, frame) -> bytes | None:
    (x0, y0), (x1, y1) = frame
    lay = _layout(problem, p, field_deg, rays=HQ_RAYS,
                  crop=CropBox(p1=(x0, y0), p4=(x1, y1), width=HQ_W))
    try:
        res, _ = runner.simulate(lay.to_scene_dict())
    except Exception:
        return None
    imgs = res.get("images") or []
    if not imgs or not imgs[0].get("dataUrl"):
        return None
    return base64.b64decode(imgs[0]["dataUrl"].split(",", 1)[1])


def contact_sheet(panels, title, sub) -> bytes | None:
    """All fields of one design in a single image, labelled."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    import io
    items = [(lab, Image.open(io.BytesIO(b)).convert("RGB")) for lab, b in panels if b]
    if not items:
        return None
    cols = 3
    rows = (len(items) + cols - 1) // cols
    cw = max(im.width for _, im in items)
    ch = max(im.height for _, im in items)
    scale = min(1.0, 460 / cw)
    cw, ch = int(cw * scale), int(ch * scale)
    pad, lab_h, head = 8, 20, 54
    W = cols * cw + (cols + 1) * pad
    H = head + rows * (ch + lab_h) + (rows + 1) * pad

    def font(sz):
        for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if Path(f).exists():
                return ImageFont.truetype(f, sz)
        return ImageFont.load_default()

    sheet = Image.new("RGB", (W, H), "#0a0a0a")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 10), title, fill="#f0e8c0", font=font(17))
    d.text((pad, 33), sub, fill="#8a8a84", font=font(12))
    for i, (lab, im) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (cw + pad)
        y = head + pad + r * (ch + lab_h + pad)
        sheet.paste(im.resize((cw, ch), Image.LANCZOS), (x, y))
        d.text((x + 2, y + ch + 3), lab, fill="#c9c9c2", font=font(12))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return buf.getvalue()


def load_designs(run_dir: Path, top: int, per_niche: bool) -> list:
    """Elites from archive.json; if a seed has not finished, its trace.jsonl instead."""
    out = []
    for seed_dir in sorted(run_dir.glob("seed*")):
        arch = seed_dir / "archive.json"
        if arch.exists():
            for c in json.loads(arch.read_text())["cells"]:
                out.append(dict(seed=seed_dir.name, src="archive", fitness=c["fitness"],
                                genotype=c["genotype"], tir=c.get("tir_count"),
                                accept=c.get("accept_deg"), info=c))
            continue
        tr = seed_dir / "trace.jsonl"
        if not tr.exists():
            continue
        best = {}
        for line in tr.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("fitness") is None or r["status"] not in ("new", "improved"):
                continue
            key = tuple(r.get("bc") or ())
            if key not in best or r["fitness"] > best[key]["fitness"]:
                best[key] = r
        for k, r in best.items():
            out.append(dict(seed=seed_dir.name, src="trace", fitness=r["fitness"],
                            genotype=r["genotype"],
                            tir=(r.get("info") or {}).get("tir_count"),
                            accept=(k[1] if len(k) > 1 else None), info=r.get("info") or {}))
    out.sort(key=lambda d: -d["fitness"])
    if per_niche:
        seen, keep = set(), []
        for d in out:
            k = (d["tir"], d["accept"])
            if k in seen:
                continue
            seen.add(k); keep.append(d)
        return keep
    return out[:top]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "runs" / "fold24_clean"))
    ap.add_argument("--experiment", default="fold24_clean")
    ap.add_argument("--out", default=None, help="default: <runs>/gallery")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--per-niche", action="store_true",
                    help="best design of every occupied niche instead of the top N")
    ap.add_argument("--fields", default=None,
                    help="comma-separated angles; default = the experiment's own field set")
    args = ap.parse_args()

    cfg = experiments.get(args.experiment)
    problem = cfg.problem
    fields = ([float(v) for v in args.fields.split(",")] if args.fields
              else [float(f) for f in (cfg.fields or (0.0,))])
    run_dir = Path(args.runs)
    out = Path(args.out) if args.out else run_dir / "gallery"
    out.mkdir(parents=True, exist_ok=True)

    designs = load_designs(run_dir, args.top, args.per_niche)
    if not designs:
        print(f"no designs found under {run_dir}")
        return 1
    print(f"{len(designs)} designs x {len(fields)} fields -> {out}")

    manifest = []
    with Runner(timeout_s=30) as runner:
        for i, dsg in enumerate(designs, 1):
            p = problem.vec_to_params(np.asarray(dsg["genotype"], float))
            name = (f"{i:02d}_{dsg['seed']}_{int(dsg['tir'] or 0)}fold"
                    f"_w{dsg['fitness']:.3f}".replace(".", "p"))
            ddir = out / name
            ddir.mkdir(exist_ok=True)

            # caustic verdict at the central field, so the filename tells you what it is
            res, _ = runner.simulate(problem.layout(p, field_deg=0.0).to_scene_dict())
            obs = measure(res, problem.face_of(p), x_sensor=problem.x_sensor,
                          allow_empty=True)
            conv = obs.convergence(n_glass=p.get("n"))
            verdict = ("clean" if conv["clean_pack"]
                       else ("caustic" if conv["crossed"] else "unpacked"))

            frame = frame_for(runner, problem, p, fields)
            panels, files = [], {}
            for fd in fields:
                png = render(runner, problem, p, fd, frame)
                if png is None:
                    print(f"  ! {name} @ {fd:+.0f} failed")
                    continue
                fn = ddir / f"field_{fd:+05.1f}.png".replace("+", "p").replace("-", "m")
                fn.write_bytes(png)
                files[f"{fd:+.1f}"] = fn.name
                panels.append((f"{fd:+.0f} deg", png))

            sub = (f"{dsg['seed']} · worst-field {dsg['fitness']:.4f} · "
                   f"{int(dsg['tir'] or 0)} folds · {verdict} · "
                   f"pack {conv['ratio']:.2f}x" if conv["measured"] else dsg["seed"])
            sheet = contact_sheet(panels, f"{name}  [{verdict}]", sub)
            if sheet:
                (out / f"{name}__sheet.png").write_bytes(sheet)

            manifest.append(dict(
                name=name, seed=dsg["seed"], source=dsg["src"],
                worst_field=dsg["fitness"], folds=dsg["tir"], accept_deg=dsg["accept"],
                verdict=verdict, crossings=conv["n_crossings"],
                pack_ratio=conv["ratio"], exit_spread_deg=conv["exit_spread"],
                params={k: round(float(v), 5) for k, v in p.items()},
                genotype=[float(v) for v in dsg["genotype"]],
                fields=files))
            print(f"  {name}  {verdict}  ({len(files)} fields)")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} designs -> {out}")
    print(f"contact sheets: {out}/*__sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
