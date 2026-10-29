"""Render recorded portfolio generation winners, with post-run verified highlights."""
from __future__ import annotations
import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image
from .folded import RoutingBrief, evaluate
from .polygon import PolygonCodec
from .portfolio import constraint_rank
from nslens.freeform_body.trace import trace_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--no-text", action="store_true", help="Only emit the text-free video and GIF")
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("max-frames must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run/'summary.json').read_text())['config']
    if config['arm'] != 'portfolio':
        raise ValueError('This renderer uses portfolio constraint ranking.')
    codec, brief = PolygonCodec(**config['codec']), RoutingBrief(**config['brief'])
    rows = [json.loads(line) for line in (args.run/'candidates.jsonl').read_text().splitlines()]
    # Sobol initialization is not a CMA generation. All later populations have
    # popsize members except the final budget-truncated population.
    rows = rows[config['policy']['initial']:]
    popsize = config['popsize']
    groups = [rows[i:i+popsize] for i in range(0, len(rows), popsize)]
    winners = [max(group, key=constraint_rank) for group in groups]
    # Bounded chronological sampling keeps long searches renderable. Always include
    # the strongest coarse winner and the first/last generation.
    if not winners:
        raise ValueError("No CMA generations recorded; use a longer search job")
    indices = np.unique(np.linspace(0, len(winners)-1, min(args.max_frames, len(winners)), dtype=int)).tolist()
    strongest = max(range(len(winners)), key=lambda i: constraint_rank(winners[i]))
    if strongest not in indices:
        indices[len(indices)//2] = strongest
    indices = sorted(set(indices))
    groups = [groups[i] for i in indices]
    winners = [winners[i] for i in indices]
    boundaries = [codec.decode(row['genes']).sample(800)[1] for row in winners]
    axis, position, center, half, sign = brief.detector()
    detector = np.array([[position, center-half], [position, center+half]])
    if axis == 1:
        detector = detector[:, ::-1]
    sources = np.array([ray.origin for ray in brief.rays()])
    scene = np.vstack([*boundaries, detector, sources])
    lo, hi = scene.min(axis=0)-9, scene.max(axis=0)+9
    dense = replace(brief, rays_per_field=101,
                    fields_deg=tuple(np.linspace(min(brief.fields_deg), max(brief.fields_deg), 9)))
    records, frames, labeled_frames, durations = [], [], [], []
    best_routes = {}
    for index, (row, boundary, group) in enumerate(zip(winners, boundaries, groups)):
        highlight, checks = False, []
        if row['feasible']:
            for cap in (200000, 400000):
                with patch('nslens.discovery.folded.trace_bundle', partial(trace_bundle, max_branches=cap)):
                    checks.append(evaluate(codec, row['genes'], dense, geometry_samples=4096))
            verified = checks[0] == checks[1] and checks[-1]['feasible']
            quality = checks[-1]['quality']
            route = checks[-1]['route']
            if verified and (route not in best_routes or quality >= best_routes[route]+.05):
                highlight = True
                best_routes[route] = quality
        result = evaluate(codec, row['genes'], brief, return_trace=True)
        fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=100, facecolor='#0b1220')
        fig.subplots_adjust(left=.035, right=.965, bottom=.04, top=.90)
        ax.set_facecolor('#0b1220')
        closed = np.vstack([boundary, boundary[0]])
        ax.fill(closed[:,0], closed[:,1], color='#78bfe8', alpha=.18)
        ax.plot(closed[:,0], closed[:,1], color='#a2d9f5', lw=2.2)
        if isinstance(result, tuple):
            _, body, trace = result
            rays = brief.rays()
            mid = len(brief.fields_deg)//2
            ids = np.linspace(mid*brief.rays_per_field, (mid+1)*brief.rays_per_field-1, 7, dtype=int)
            for sid in ids:
                choices = [r for r in trace.exits if r.source_id == sid and r.events]
                if not choices:
                    continue
                # Show strongest physical exit branch, avoiding cherry-picked weak folds.
                ray = max(choices, key=lambda r: r.energy)
                origin, direction = np.asarray(ray.origin), np.asarray(ray.direction)
                distance = (position-origin[axis])/direction[axis] if direction[axis]*sign > 1e-9 else 20.
                path = np.asarray([rays[sid].origin, *(e.point for e in ray.events), origin+max(0.,distance)*direction])
                ax.plot(path[:,0], path[:,1], color='#ffcc70', lw=1.15, alpha=.82)
        ax.plot(sources[:brief.rays_per_field,0], sources[:brief.rays_per_field,1], color='#5bd9ed', lw=5, solid_capstyle='round')
        ax.plot(detector[:,0], detector[:,1], color='#c19aff', lw=5, solid_capstyle='round')
        ax.set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), aspect='equal')
        ax.axis('off')
        if highlight:
            # Draw the tick as geometry: no font-dependent glyph or text.
            badge = fig.add_axes([.865,.79,.085,.113], aspect='equal')
            badge.set(xlim=(0,1), ylim=(0,1)); badge.axis('off')
            badge.add_patch(Circle((.5,.5),.46,facecolor='#19b97a'))
            badge.plot([.25,.43,.76],[.50,.32,.68],color='white',lw=4,solid_capstyle='round')
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
        if not args.no_text:
            fig.text(.5,.955,'Best candidate from each generation', ha='center', va='center', color='#e4edf8', fontsize=17)
            fig.canvas.draw()
            labeled_frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
        plt.close(fig)
        durations.append(1000 if highlight else 200)
        records.append(dict(generation=indices[index]+1,evaluation=row['evaluation'],population_size=len(group),highlight=highlight,duration_ms=durations[-1],dense_checks=checks))
        print(f'frame {index+1}/{len(winners)} highlight={highlight}',flush=True)
    outputs = [('evolution', frames)]
    if not args.no_text:
        outputs.append(('evolution_labeled', labeled_frames))
    for name, images in outputs:
        images[0].save(args.out/f'{name}.gif', save_all=True, append_images=images[1:], duration=durations, loop=0, disposal=2)
        command = ['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','25','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(args.out/f'{name}.mp4')]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        for im, duration in zip(images,durations):
            data = im.tobytes()
            for _ in range(duration//40):
                process.stdin.write(data)
        process.stdin.close()
        if process.wait():
            raise RuntimeError('ffmpeg failed')
    frames[next((i for i,r in enumerate(records) if r['highlight']),0)].save(args.out/'preview.png')
    (args.out/'manifest.json').write_text(json.dumps(dict(source=str(args.run),selection='Chronologically sampled generation winners by portfolio constraint rank, plus strongest winner; Sobol excluded; final partial population included',highlight_rule='909-ray checks agree at 200k/400k branch caps and pass all gates; new route or >= 0.05 absolute quality improvement',frames=records),indent=2))
    print(f'Wrote {args.out}; {sum(durations)/1000:.1f}s; {sum(r["highlight"] for r in records)} highlights',flush=True)


if __name__ == '__main__':
    main()
