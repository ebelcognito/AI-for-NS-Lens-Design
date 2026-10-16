"""Post-search geometry screen and physical prescriptions for the pilot gallery."""
import argparse
import json
from pathlib import Path
import numpy as np
from .pilot import decode, trace


def cross(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def crossing(a,b):
    """Proper crossings between densely sampled polylines; tangencies not certified."""
    p=a[:-1,None,:]; r=np.diff(a,axis=0)[:,None,:]
    q=b[None,:-1,:]; s=np.diff(b,axis=0)[None,:,:]
    den=cross(r,s)
    with np.errstate(divide='ignore',invalid='ignore'):
        t=cross(q-p,s)/den; u=cross(q-p,r)/den
    return bool(np.any((abs(den)>1e-12)&(t>=0)&(t<=1)&(u>=0)&(u<=1)))


def audit(out):
    summary=json.loads((out/'summary.json').read_text()); records=[]
    for idx,row in enumerate(summary['selected']):
        m=decode(row['genes']); shapes=[]
        for v,t,n,c,w in zip(m['vertices'],m['tangents'],m['normals'],m['curves'],m['widths']):
            u=np.linspace(-w,w,513)
            shapes.append(v+u[:,None]*t+(c*u*u)[:,None]*n)
        pairs=[[i+1,j+1] for i in range(3) for j in range(i) if crossing(shapes[i],shapes[j])]
        fine=trace(row['genes'],fields=21,pupils=201)
        record=dict(gallery_index=idx+1,mirror_crossings_sampled=pairs,
            finer_audit=fine,physical={k:v.tolist() for k,v in m.items()})
        records.append(record)
    (out/'geometry_audit.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    for r in records:
        f=r['finer_audit']
        spots=[x['rms_spot_mm'] for x in f['fields'] if x['rms_spot_mm'] is not None]
        print(json.dumps(dict(index=r['gallery_index'],support=f['worst_support'],
            worst_spot_mm=max(spots) if spots else None,loss=f['loss'],crossings=r['mirror_crossings_sampled'])))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('out',type=Path); audit(p.parse_args().out)
