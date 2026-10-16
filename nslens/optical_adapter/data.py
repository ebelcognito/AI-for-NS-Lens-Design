"""Small simulator-verified correction dataset, split by independent CMA ancestry."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from nslens.search.cma_me import CMAES
from nslens.spectrometer import model
from .state import SCHEMA, observe, apply_action


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_dataset(data):
    if data['schema'] != SCHEMA:
        raise ValueError('Unknown feature schema')
    groups = {split: {r['family'] for r in data['rows'] if r['split'] == split}
              for split in ('train', 'validation')}
    if not all(groups.values()) or groups['train'] & groups['validation']:
        raise ValueError('Need disjoint nonempty train/validation ancestry groups')
    if any(r['split'] not in groups for r in data['rows']):
        raise ValueError('Unknown split')
    for row in data['rows']:
        apply_action(row['genes'], row['action'], row['radius'])
        c, s = np.asarray(row['cells']), np.asarray(row['state'])
        if c.shape != (9, 10) or s.shape != (37,) or not np.isfinite(c).all() or not np.isfinite(s).all():
            raise ValueError('Invalid optical features')
        if not np.allclose(s[:36], row['genes']) or not np.isclose(s[-1], row['radius']):
            raise ValueError('Observation/action state mismatch')


def generate(out, families=8, steps=6, probes=12, warmup=64, seed=2909, radius=.05):
    if families < 4 or steps < 1 or probes < 1 or warmup < 8 or warmup % 8:
        raise ValueError('Need >=4 families, positive steps/probes, warmup a positive multiple of 8')
    apply_action(np.zeros(36), np.zeros(36), radius)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); rows = []; calls = dict(objective=0, observation_trace=0)
    def evaluate(g):
        calls['objective'] += 1
        return model.evaluate(g, fields=3, waves=3, pupils=5)['loss']
    held_out = max(1, families // 4)
    for family in range(families):
        family_seed = seed + family
        rng = np.random.default_rng(family_seed)
        # Group assignment precedes optimization and label selection.
        split = 'validation' if family >= families - held_out else 'train'
        cma = CMAES(rng.uniform(-.5, .5, 15), .15, popsize=8, seed=family_seed,
                    bounds=(-np.ones(15), np.ones(15)), boundary_handling='reflect')
        best = None; best_loss = float('inf')
        for _ in range(warmup // 8):
            batch = cma.ask(); losses = []
            for x in batch:
                g = np.r_[x, np.zeros(21)]; loss = evaluate(g); losses.append(loss)
                if loss < best_loss:
                    best, best_loss = g.copy(), loss
            cma.tell(np.argsort(losses))
        for step in range(steps):
            # Label is the best measured local correction INCLUDING no-op.
            # It is a stochastic teacher, not a globally optimal direction.
            cells, state = observe(best, radius); calls['observation_trace'] += 1
            target = best.copy(); target_loss = best_loss
            for _ in range(probes):
                action = rng.uniform(-1, 1, 36)
                # Mix layout/curvature/detector/freeform blocks and full actions.
                blocks = [(0, 7), (7, 13), (13, 15), (15, 36), (0, 36)]
                lo, hi = blocks[int(rng.integers(len(blocks)))]; action[:lo] = 0; action[hi:] = 0
                candidate = apply_action(best, action, radius); loss = evaluate(candidate)
                if loss < target_loss:
                    target, target_loss = candidate, loss
            rows.append(dict(family=f'cma-seed-{family_seed}', split=split, step=step,
                genes=best.tolist(), radius=radius, cells=cells.tolist(), state=state.tolist(),
                action=((target-best)/radius).tolist(), before_loss=best_loss, teacher_loss=target_loss))
            best, best_loss = target, target_loss
        print(json.dumps(dict(family=family+1, families=families, calls=calls)), flush=True)
    dataset = dict(schema=SCHEMA, rows=rows, protocol=dict(seed=seed, families=families,
        steps=steps, probes=probes, warmup=warmup, radius=radius, sampling=[3, 3, 5],
        physical_apertures_mm=[18, 18, 18], split='last quarter of independent seed families held out',
        teacher='best of bounded random block proposals, including unchanged incumbent',
        objective='unmodified nslens.spectrometer.model.evaluate loss',
        model_sha256=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
        calls=calls, total_forward_traces=sum(calls.values()), seconds=time.monotonic()-started,
        interpretation='Small single-target development dataset; not a paper-reproduction benchmark'))
    validate_dataset(dataset); write(out/'dataset.json', dataset)
    return dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True); p.add_argument('--families', type=int, default=8)
    p.add_argument('--steps', type=int, default=6); p.add_argument('--probes', type=int, default=12)
    p.add_argument('--warmup', type=int, default=64); p.add_argument('--seed', type=int, default=2909)
    p.add_argument('--radius', type=float, default=.05)
    generate(**vars(p.parse_args()))


if __name__ == '__main__': main()
