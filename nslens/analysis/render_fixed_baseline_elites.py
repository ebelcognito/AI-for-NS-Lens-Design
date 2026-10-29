"""Render every elite from a fixed-environment baseline audit.

The audit records store only the six searched geometry genes.  This renderer restores
the source position, source angle, material index, and detector from the matching fixed
environment before asking the native ray-optics engine for a PNG.  It therefore renders
the audited physical setup rather than silently falling back to the legacy nine-gene
problem.

Example::

    python3 -m nslens.analysis.render_fixed_baseline_elites \
        runs/fixed_baseline_audit_20260903_v1
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from nslens.engine import Runner
from nslens.qd.fixed_environments import (
    GEOMETRY_PARAMETERS,
    problem_from_environment_spec,
)
from nslens.scene import CropBox


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load(audit_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict], set[str]]:
    manifest = json.loads((audit_dir / "manifest.json").read_text())
    specs_payload = json.loads(Path(manifest["environment_specs"]).read_text())
    specs = {
        str(row["environment_id"]): row
        for row in specs_payload["environments"]
        if row.get("status") == "accepted"
    }
    records: list[dict[str, Any]] = []
    for path in sorted((audit_dir / "audits").glob("*/records.jsonl")):
        records.extend(_read_jsonl(path))
    strict = {
        str(row["design_id"])
        for row in _read_jsonl(audit_dir / "strict_intentional.jsonl")
    }
    if len(records) != int(manifest["candidates_total"]):
        raise ValueError(
            f"loaded {len(records)} records, expected {manifest['candidates_total']}"
        )
    return records, specs, strict


def _source_summary(record: dict[str, Any]) -> tuple[str, float]:
    sources = record.get("source_provenance", {}).get("sources", [])
    algorithms = "+".join(sorted({str(row.get("algorithm", "unknown")) for row in sources}))
    fitness = max(
        (float(row["fitness"]) for row in sources if row.get("fitness") is not None),
        default=float(record.get("source_hardened_verdict", {}).get("fitness", 0.0)),
    )
    return algorithms, fitness


def _safe_float(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def _render_one(
    runner: Runner,
    problem,
    params: dict[str, float],
    *,
    field_deg: float,
    rays: int,
    width: int,
) -> bytes:
    # A fixed frame makes different solutions directly comparable and keeps energetic
    # ghost branches from forcing an enormous auto-crop.  It contains the complete
    # element, source bundle, and detector plane for every accepted fixed environment.
    element_y = [float(vertex["y"]) for vertex in problem.element(params).path()]
    source_half = float(problem.beam_w) / 2.0
    y0 = min(-40.0, float(problem.fixed_ray_y) - source_half, *element_y) - 5.0
    y1 = max(40.0, float(problem.fixed_ray_y) + source_half, *element_y) + 5.0
    crop = CropBox(p1=(-6.0, y0), p4=(float(problem.x_sensor) + 6.0, y1), width=width)
    layout = problem.layout(params, field_deg=field_deg)
    layout = replace(
        layout,
        source=replace(layout.source, n_rays=rays),
        max_depth=24,
        record=False,
        crop=crop,
    )
    result, _ = runner.simulate(layout.to_scene_dict())
    images = result.get("images") or []
    if not images or not images[0].get("dataUrl"):
        raise RuntimeError("native engine returned no rendered image")
    return base64.b64decode(images[0]["dataUrl"].split(",", 1)[1])


def _font(size: int):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _contact_sheet(
    entries: Iterable[dict[str, Any]],
    output: Path,
    *,
    title: str,
    columns: int,
) -> None:
    from PIL import Image, ImageDraw

    loaded = []
    for entry in entries:
        image = Image.open(entry["path"]).convert("RGB")
        loaded.append((entry, image))
    if not loaded:
        return

    thumb_w = 360
    sample = loaded[0][1]
    thumb_h = max(1, round(sample.height * thumb_w / sample.width))
    pad, label_h, header_h = 12, 50, 54
    rows = (len(loaded) + columns - 1) // columns
    sheet_w = columns * thumb_w + (columns + 1) * pad
    sheet_h = header_h + rows * (thumb_h + label_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "#0a0a0a")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 12), title, fill="#f0e8c0", font=_font(20))

    for index, (entry, image) in enumerate(loaded):
        row, column = divmod(index, columns)
        x = pad + column * (thumb_w + pad)
        y = header_h + row * (thumb_h + label_h + pad)
        image = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        border = "#d84a4a" if not entry["strict"] else "#4b8771"
        draw.rectangle((x - 2, y - 2, x + thumb_w + 1, y + thumb_h + 1), outline=border, width=3)
        sheet.paste(image, (x, y))
        label = (
            f"{entry['number']:02d}  {entry['short_id']}  {entry['algorithm']}  "
            f"fit {entry['fitness']:.4f}\n"
            f"{entry['environment_id']}  "
            f"{'STRICT PASS' if entry['strict'] else 'MARGIN EXCLUDED'}"
        )
        draw.multiline_text((x, y + thumb_h + 5), label, fill="#d5d5cf", font=_font(12), spacing=3)

    sheet.save(output, "PNG")


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit_dir = Path(args.audit_dir).resolve()
    output = Path(args.out).resolve() if args.out else audit_dir / "renders"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output}; use --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)

    records, specs, strict_ids = _load(audit_dir)
    records.sort(key=lambda row: (str(row["fixed_environment_id"]), -_source_summary(row)[1]))
    manifest_entries: list[dict[str, Any]] = []

    with Runner(timeout_s=30) as runner:
        for number, record in enumerate(records, 1):
            environment_id = str(record["fixed_environment_id"])
            problem = problem_from_environment_spec(specs[environment_id], n_rays=args.rays)
            vector = [float(record["design"][name]) for name in GEOMETRY_PARAMETERS]
            params = problem.vec_to_params(vector)
            if not problem.is_valid(params):
                raise ValueError(f"audited design is no longer geometrically valid: {record['design_id']}")

            algorithm, fitness = _source_summary(record)
            short_id = str(record["design_id"]).rsplit(":", 1)[-1][:8]
            is_strict = str(record["design_id"]) in strict_ids
            status = "strict" if is_strict else "margin-excluded"
            filename = (
                f"{number:02d}_{environment_id}_{short_id}_{algorithm}_"
                f"fit{_safe_float(fitness)}_{status}.png"
            )
            path = output / filename
            png = _render_one(
                runner,
                problem,
                params,
                field_deg=float(args.field),
                rays=int(args.rays),
                width=int(args.width),
            )
            path.write_bytes(png)
            manifest_entries.append(
                {
                    "number": number,
                    "design_id": record["design_id"],
                    "short_id": short_id,
                    "environment_id": environment_id,
                    "algorithm": algorithm,
                    "fitness": fitness,
                    "strict": is_strict,
                    "field_deg": float(args.field),
                    "intended_sequence": record.get("historical_sequence"),
                    "geometry_genes": record["design"],
                    "fixed_environment": specs[environment_id]["fixed"],
                    "png": filename,
                    "path": path,
                }
            )
            print(f"[{number:02d}/{len(records)}] {filename}", flush=True)

    for environment_id in sorted(specs):
        group = [entry for entry in manifest_entries if entry["environment_id"] == environment_id]
        if group:
            _contact_sheet(
                group,
                output / f"contact_{environment_id}.png",
                title=f"{environment_id} - {len(group)} audited elites",
                columns=min(4, len(group)),
            )
    _contact_sheet(
        manifest_entries,
        output / "contact_all_53.png",
        title="All 53 dense-audited fixed-environment elites (central field)",
        columns=4,
    )

    serializable = []
    for entry in manifest_entries:
        row = dict(entry)
        row.pop("path")
        serializable.append(row)
    render_manifest = {
        "source_audit": str(audit_dir),
        "field_deg": float(args.field),
        "rays_per_setup": int(args.rays),
        "renders": len(serializable),
        "strict_pass": sum(bool(row["strict"]) for row in serializable),
        "margin_excluded": sum(not bool(row["strict"]) for row in serializable),
        "entries": serializable,
    }
    (output / "manifest.json").write_text(json.dumps(render_manifest, indent=2) + "\n")
    return render_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dir", help="fixed baseline audit directory")
    parser.add_argument("--out", default=None, help="default: <audit_dir>/renders")
    parser.add_argument("--field", type=float, default=0.0, help="field angle to render")
    parser.add_argument("--rays", type=int, default=61, help="rays in each rendered bundle")
    parser.add_argument("--width", type=int, default=1100, help="individual PNG width")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(
        f"wrote {result['renders']} individual PNGs and contact sheets to "
        f"{Path(args.out).resolve() if args.out else Path(args.audit_dir).resolve() / 'renders'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
