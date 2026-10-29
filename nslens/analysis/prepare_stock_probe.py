"""Freeze a five-design, two-triangle/three-quadrilateral correction pilot."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
from nslens.workbench.stock_probe import dump,read,digest,measure,INSTRUCTIONS,SCHEMA
from nslens.discovery.polygon import PolygonCodec,exact_extent


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
 p.add_argument('--stock',type=Path,default=Path('runs/diverse_stock_dense_search_20260911_v1'))
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 manifest=read(a.stock/'stock_manifest.json')['designs']
 selected=[('good_triangle', 'good', manifest[0]),('good_quad','good',manifest[9]),('bad_quad_near','bad',manifest[14])]
 for name,env,k,target in [('bad_triangle','bottom_mid',3,.35),('bad_quad_low','bottom_right',4,.2)]:
  candidates=[]
  for path in sorted(a.stock.glob(f'{env}_k{k}_*/candidates.jsonl')):
   summary=read(path.parent/'summary.json')
   for row in [__import__('json').loads(line) for line in path.read_text().splitlines()]:
    if row['status']=='measured' and .05<row['support']<.55 and not row['feasible']:
     candidates.append((abs(row['support']-target),str(path),row,summary))
  candidates.sort(key=lambda x:(x[0],x[1],x[2]['evaluation']))
  for _,path,row,summary in candidates:
   d=dict(environment=env,corners=k,genes=row['genes'],codec=summary['config']['codec'],brief=summary['config']['brief'],
          source_archive=path,coarse_evaluation=row['evaluation'],route=row['route'],unit=summary['unit'])
   lo,hi=exact_extent(PolygonCodec(**d['codec']).decode(d['genes']))
   t=dict(design=d,initial_dimensions=(hi-lo).tolist())
   r=measure(t,d['genes'],'repair',audit=True)
   if not r['routing_feasible']:
    selected.append((name,'bad',d));break
  else:raise RuntimeError('No physically failing candidate found for '+name)
 for i,(name,label,d) in enumerate(selected):
  lo,hi=exact_extent(PolygonCodec(**d['codec']).decode(d['genes']))
  task=dict(schema_version=1,task_id=name,label=label,seed=2026091100+i*100,design=d,initial_dimensions=(hi-lo).tolist(),
            compaction='5% reduction of longest exact body bounding-box side; neither side may grow',
            full_fov_deg=dict(repair=1,compact=1,field=2),batches_per_probe=2,candidates_per_batch=40,
            scope='2D circular-arc routing adaptation; fixed environment and original codec bounds; no freeform imaging optimization')
  folder=a.out/name;folder.mkdir();dump(folder/'task.json',task)
  start=measure(task,d['genes'],'repair',audit=True)
  high=measure(task,d['genes'],'repair',audit=True,cap=400000)
  if start!=high or start['routing_feasible']!=(label=='good'):raise RuntimeError('label or capacity mismatch '+name)
  dump(folder/'selection_audit.json',dict(at_200k=start,at_400k=high,capacity_agrees=start==high))
  print(name,d['corners'],start['quality'],start['support'],flush=True)
 sources=[*Path('nslens/discovery').glob('*.py'),*Path('nslens/freeform_body').glob('*.py'),Path('nslens/search/cma_me.py'),Path(__file__),Path('nslens/workbench/stock_probe.py')]
 dump(a.out/'protocol.json',dict(schema_version=1,selection='Two stress-verified stock starts, one near-failing retained stock start; two search-log failures selected near support .35/.2; all labels rechecked at 200k/400k branches.',
  task_ids=[x[0] for x in selected],source_manifest_sha256=digest(a.stock/'stock_manifest.json'),sources={str(p):digest(p) for p in sources},
  search_grid='9 evenly spaced fields x 25 endpoint-inclusive pupil rays',audit_grid='9 fields x 101 pupil rays',branch_cap=200000,capacity_check=400000,
  proposal_limit=1200,search_ray_limit=270000,initial_ray_limit=3375,final_audit_ray_limit=54540,
  model='qwen3.8-27b',temperature=.2,max_tokens_per_response=4096,max_model_responses_per_design=12,
  limitations='Exploratory five-design pilot, one trajectory per probe, no model superiority claim; same-engine nested-grid audits; codec-bound failures do not establish physical impossibility.'))
 dump(a.out/'action_schema.json',SCHEMA);(a.out/'AGENT_INSTRUCTIONS.txt').write_text(INSTRUCTIONS+'\n')

if __name__=='__main__':main()
