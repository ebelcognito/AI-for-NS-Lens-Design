"""Assemble verified camera replay files without interpolating prescriptions."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inputs',nargs='+');p.add_argument('--out',required=True)
    a=p.parse_args();records=[];sources=[]
    for filename in a.inputs:
        source=Path(filename);payload=json.loads(source.read_text())
        rows=payload['records'] if isinstance(payload,dict) else payload
        sources.append(str(source.resolve()))
        for row in rows:
            if 'genes' not in row:raise ValueError('Every state must have a prescription')
            records.append(row)
    Path(a.out).write_text(json.dumps(dict(sources=sources,records=records,
         note='Chronological genuine states. Evaluation counts are local to each run; no geometry interpolation.'),indent=2)+'\n')


if __name__=='__main__':main()
