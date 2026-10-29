"""Re-certify archive candidates under the current closed-boundary verifier."""
import argparse
import json
from pathlib import Path
from .folded_domain import evaluate
from .tools import canonical_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=12)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name in ['freeform','multifold']:
        path=Path(f'runs/ns_autodiff__results_search_prism_{name}/archive_prism_{name}.json')
        archive=json.loads(path.read_text())
        for rank,cell in enumerate(sorted(archive['cells'],key=lambda c:-c['fitness'])[:a.limit]):
            x=cell['genotype']
            row=dict(archive=str(path),archive_hash=canonical_hash(archive),rank=rank,
                     family_id=f'{name}-{cell["i0"]}-{cell["i1"]}',base=x[:9],witness=x[9:])
            row['coarse']=evaluate(x[:9],x[9:],'coarse')
            if row['coarse']['success']:
                row['audit']=evaluate(x[:9],x[9:],'audit')
                if row['audit']['success']:
                    row['convergence']=evaluate(x[:9],x[9:],'convergence')
            row['certified']=bool(row.get('convergence',{}).get('success'))
            rows.append(row)
            (a.out/'screen.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(name,rank,'certified',row['certified'],flush=True)
    (a.out/'summary.json').write_text(json.dumps(dict(screened=len(rows),certified=sum(r['certified'] for r in rows),
        note='Archive cells may share ancestry; these identifiers are not independent train/test families.'),indent=2)+'\n')

if __name__=='__main__':main()
