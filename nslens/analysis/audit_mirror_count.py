"""Post-hoc independent density and sampled geometry audit; no search feedback."""
import sys,json
from pathlib import Path
import numpy as np
from nslens.analysis.variable_mirror import decode,trace
from nslens.three_mirror.audit import crossing

def main():
 out=Path(sys.argv[1]);rows=[]
 for p in out.glob('*.json'):
  r=json.loads(p.read_text())
  if isinstance(r,dict) and 'arm' in r:rows.append(r)
 audits=[]
 for r in rows:
  for k,old in r['coarse'].items():
   n=int(k);m=decode(old['genes'],n);shapes=[]
   for v,t,norm,c,w in zip(m['vertices'],m['tangents'],m['normals'],m['curves'],m['widths']):
    u=np.linspace(-w,w,513);shapes.append(v+u[:,None]*t+(c*u*u)[:,None]*norm)
   pairs=[[i,j] for i in range(n) for j in range(i) if crossing(shapes[i],shapes[j])]
   fine=trace(old['genes'],n,fields=21,pupils=201)
   audits.append(dict(seed=r['seed'],arm=r['arm'],n=n,crossings=pairs,fine=fine,screen_pass=not pairs and fine['worst_support']>=.99))
 (out/'geometry_audit.json').write_text(json.dumps(audits,indent=2))
 summary={arm:{str(n):dict(screen_pass=sum(a['screen_pass'] for a in audits if a['arm']==arm and a['n']==n),crossing_runs=sum(bool(a['crossings']) for a in audits if a['arm']==arm and a['n']==n),runs=sum(a['arm']==arm and a['n']==n for a in audits),median_fine_loss=float(np.median([a['fine']['loss'] for a in audits if a['arm']==arm and a['n']==n]))) for n in range(3,7)} for arm in ['restart','same_count','cross_count']}
 (out/'geometry_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
 with (out/'RESULTS.txt').open('a') as f:f.write('\nADDITIONAL POST-HOC AUDIT\n21 fields x201 rays per count incumbent, plus513 samples per surface for proper crossings. These extra audits are not included in the initial dense-call ledger and never affect selection.\nAdditional trace calls: '+str(len(audits))+'\nA screen pass is >=99% worst-field support and no sampled mirror crossings; not an imaging specification or certified mechanical clearance.\n'+json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
