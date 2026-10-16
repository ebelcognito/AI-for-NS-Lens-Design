"""Dense audit of every cached three-mirror freeform endpoint, without selection."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from nslens.analysis.optical_refinement import audit


def run(path):
    data = json.loads(Path(path).read_text())
    row = data['refinements']['3']['coarse']
    result = audit(row['genes'], 3, row['freeform_coefficients_mm'], data['task'])
    return dict(source=path, source_job=data['source_job'], audit=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out)
    files = sorted(Path('experiments/shared_freeform_comparison_20261006').glob('*__last_freeform.json'))
    protocol = dict(scope='Dense audits of all 144 cached three-mirror endpoints; no optimization or selection.',
                    fields=15, pupils=101, crossing_samples=513,
                    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with (out / 'audit_protocol.json').open('x') as handle:
        json.dump(protocol, handle, indent=2)
    with (out / 'dense_audits.jsonl').open('x') as handle:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, row in enumerate(pool.map(run, map(str, files)), 1):
                handle.write(json.dumps(row) + '\n')
                handle.flush()
                if index % 24 == 0:
                    print(f'Audited {index}/{len(files)}', flush=True)


if __name__ == '__main__':
    main()
