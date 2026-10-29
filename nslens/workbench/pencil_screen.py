"""Narrow-field folded routing screen, explicitly distinct from imaging certification."""
import argparse,json,hashlib
from pathlib import Path
from .folded_domain import evaluate,LAYOUT,REQUIREMENTS


def pencil_config(width_mm=1.,fan_half_angle_deg=.25,min_folds=2):
    if not 0<width_mm<=1:raise ValueError('field width must be in (0,1] mm')
    if not 0<fan_half_angle_deg<=1:raise ValueError('finite angular half-fan must be in (0,1] degrees')
    if type(min_folds) is not int or min_folds<1:raise ValueError('positive fold count required')
    layout=dict(LAYOUT,source_points=[[55-width_mm/2,50.],[55.,50.],[55+width_mm/2,50.]],
                fan_half_angle_deg=fan_half_angle_deg)
    requirements=dict(REQUIREMENTS,min_tir_bounces=min_folds,min_fov_deg=0.,require_ordering=False)
    return layout,requirements


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=12)
    p.add_argument('--min-folds',type=int,default=2)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    layout,requirements=pencil_config(min_folds=a.min_folds)
    config=dict(layout=layout,requirements=requirements,field_width_mm=1.,angular_half_fan_deg=.25,
        note='Routing-only development screen. No wide-field FOV or imaging-ordering requirement. Geometry/material unchanged.')
    (a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=[]
    for name in ['freeform','multifold']:
        path=Path(f'runs/ns_autodiff__results_search_prism_{name}/archive_prism_{name}.json')
        archive=json.loads(path.read_text())
        for rank,cell in enumerate(sorted(archive['cells'],key=lambda c:-c['fitness'])[:a.limit]):
            x=cell['genotype'];row=dict(archive=str(path),cell=[cell['i0'],cell['i1']],rank=rank,
                base=x[:9],coefficients=x[9:],archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            row['coarse']=evaluate(x[:9],x[9:],'coarse',layout=layout,requirements=requirements)
            if row['coarse']['success']:
                row['audit']=evaluate(x[:9],x[9:],'audit',layout=layout,requirements=requirements)
                if row['audit']['success']:
                    row['convergence']=evaluate(x[:9],x[9:],'convergence',layout=layout,requirements=requirements)
            row['passed']=bool(row.get('convergence',{}).get('success'))
            rows.append(row);(a.out/'screen.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(name,rank,'passed',row['passed'],flush=True)
    (a.out/'summary.json').write_text(json.dumps(dict(screened=len(rows),passed=sum(r['passed'] for r in rows)),indent=2)+'\n')

if __name__=='__main__':main()
