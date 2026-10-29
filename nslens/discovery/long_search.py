"""Wall-time-limited portfolio exploration with replayable, text-free videos."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from .benchmark import POLICIES, run_job
from .folded import RoutingBrief
from .polygon import PolygonCodec
from .run import write


def task_grid(n=1.60, max_detector_beam_ratio=None, spot_weight=0.):
    """216 fixed task settings; size ranges are priors, not certified package sizes."""
    tasks = []
    for corners, beam, layout, field, size in itertools.product(
            (3, 4, 5), (4., 6.), ('broad', 'right_1p5', 'right_2p5', 'bottom_2p5'),
            (.5, 2., 5.), ('standard', 'small', 'thin')):
        brief = RoutingBrief(beam_width_mm=beam, fields_deg=(-field, 0., field),
                             spot_weight=spot_weight)
        data = asdict(brief)
        if layout.startswith('right'):
            ratio = 1.5 if layout == 'right_1p5' else 2.5
            data.update(detector_center_y=-20., detector_half_height=ratio*beam/2)
        elif layout == 'bottom_2p5':
            data.update(detector_axis=1, detector_travel_sign=-1,
                        detector_position=-55., detector_center=60., detector_half_length=2.5*beam/2)
        if max_detector_beam_ratio is not None:
            capped_half_length = max_detector_beam_ratio*beam/2
            if data.get('detector_half_length') is not None:
                data['detector_half_length'] = min(data['detector_half_length'], capped_half_length)
            else:
                data['detector_half_height'] = min(data['detector_half_height'], capped_half_length)
        bounds = dict(standard=((12.,30.), (12.,30.)),
                      small=((6.,15.), (6.,15.)), thin=((12.,30.), (5.,12.)))
        width, height = bounds[size]
        codec = PolygonCodec(corners, n=n, half_width=width, half_height=height)
        name = f'{layout}_beam{beam:g}_field{field:g}_{size}_k{corners}'
        tasks.append(dict(brief_id=name, codec=asdict(codec), brief=data))
    return tasks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hours', type=float, default=3.5)
    p.add_argument('--ri', type=float, default=1.60,
                   help='Fixed refractive index for every task (1.50–1.70)')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--calls-per-job', type=int, default=4000)
    p.add_argument('--seed', type=int, default=910000)
    p.add_argument('--videos', type=int, default=3)
    p.add_argument('--video-frames', type=int, default=150)
    p.add_argument('--render-timeout-minutes', type=float, default=20.)
    p.add_argument('--max-detector-beam-ratio', type=float,
                   help='Cap full detector length / input beam width for every task')
    p.add_argument('--spot-weight', type=float, default=0.,
                   help='RMS-spot penalty in quality; 1 means normalized by input beam width')
    a = p.parse_args(argv)
    if not 1.50 <= a.ri <= 1.70:
        p.error('--ri must be within [1.50, 1.70]')
    if (not 0 < a.hours <= 24 or min(a.workers, a.calls_per_job, a.video_frames) < 1
            or a.videos < 0 or not 0 < a.render_timeout_minutes <= 120
            or a.max_detector_beam_ratio is not None and a.max_detector_beam_ratio <= 0
            or a.spot_weight < 0):
        p.error('invalid time/count option')
    if a.videos and shutil.which('ffmpeg') is None:
        p.error('ffmpeg is required for automatic videos')
    a.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root/'nslens').rglob('*.py'))
    hashes = {}
    for source in sources:
        relative = source.relative_to(root)
        target = a.out/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    tasks = task_grid(n=a.ri, max_detector_beam_ratio=a.max_detector_beam_ratio,
                      spot_weight=a.spot_weight)
    random.Random(a.seed).shuffle(tasks)
    started = time.time()
    deadline = started + a.hours*3600
    write(a.out/'protocol.json', dict(hours=a.hours, workers=a.workers, ri=a.ri,
          calls_per_job=a.calls_per_job, seed=a.seed, tasks=tasks, sources=hashes,
          deadline_unix=deadline, policy=POLICIES['portfolio'],
          objective=dict(metric='min_field_throughput/(1+spot_weight*RMS_spot/input_beam_width)',
                         spot_weight=a.spot_weight,
                         max_detector_beam_ratio=a.max_detector_beam_ratio),
          scope='Exploratory cold-start portfolio CMA, not learned LQD/ConfigX; no comparative claims',
          limits=f'Fixed n={a.ri:.2f}, 2D circular-face polygons; detector spot is a 2D tangential RMS metric. Package ranges are search priors. No compaction performed.',
          timing='Search deadline checked between populations; final audits and video rendering add time',
          rendering=dict(videos=a.videos, max_frames=a.video_frames, timeout_minutes=a.render_timeout_minutes)))
    summaries, submitted = [], 0
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
        pending = {}
        while time.time() < deadline or pending:
            while time.time() < deadline and len(pending) < a.workers:
                task = tasks[submitted % len(tasks)]
                seed = a.seed + submitted
                unit = f'{task["brief_id"]}_seed{seed}'
                job = dict(**task, unit=unit, arm='portfolio', seed=seed,
                           policy=POLICIES['portfolio'], search_calls=a.calls_per_job,
                           popsize=10, audit_rays=25, audit_limit=8, deadline_unix=deadline,
                           out=str(a.out/unit))
                # Persist the job before execution, including jobs interrupted by errors.
                with (a.out/'jobs.jsonl').open('a') as f:
                    f.write(json.dumps(job)+'\n')
                pending[pool.submit(run_job, job)] = unit
                submitted += 1
            if not pending:
                break
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                unit = pending.pop(future)
                row = future.result()
                summaries.append(row)
                write(a.out/'summary.json', summaries)
                print(f'{len(summaries)} completed | {unit} | verified={row["audited_designs"]} | calls={row["search_calls"]} | remaining={max(0, deadline-time.time())/60:.1f} min', flush=True)
    # Videos follow one real task each; never imply unrelated tasks form one lineage.
    ranked = sorted(summaries, key=lambda r: (r['audited_success'], r['audited_routes'], r['best_audited_quality']), reverse=True)
    chosen, families = [], set()
    for row in ranked:
        if row['proposals'] <= row['config']['policy']['initial']:
            continue
        family = row['brief_id']
        if family not in families:
            chosen.append(row)
            families.add(family)
        if len(chosen) >= a.videos:
            break
    renders = []
    for i, row in enumerate(chosen if a.videos else []):
        output = a.out/f'video_{i+1}'
        command = [sys.executable, '-m', 'nslens.discovery.animate_search',
                   row['config']['out'], '--out', str(output), '--no-text',
                   '--max-frames', str(a.video_frames)]
        print('Rendering', row['unit'], flush=True)
        try:
            with (a.out/f'video_{i+1}.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=a.render_timeout_minutes*60)
            status = 'complete' if result.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            status = 'timeout; source history retained for rerender'
        renders.append(dict(source=row['config']['out'], output=str(output), status=status))
        write(a.out/'videos.json', renders)
    write(a.out/'completion.json', dict(completed_jobs=len(summaries),
          task_settings_completed=len({r['brief_id'] for r in summaries}), task_settings_total=len(tasks),
          successful_jobs=sum(r['audited_success'] for r in summaries),
          wall_seconds=time.time()-started, videos=renders))
    print(f'Finished: {a.out}/completion.json', flush=True)


if __name__ == '__main__':
    main()
