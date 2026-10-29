"""Select six archived families and freeze their next angular repair targets."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from nslens.workbench.stock_probe import read, dump, measure, digest
from nslens.discovery.polygon import PolygonCodec, exact_extent

ROOT = Path('runs/angular_qwen_20260912_v1')
SOURCE = Path('runs/inception_long_20260912_6h_v1')
SEEDS = [910136, 910245, 910307, 910242, 910002, 910078]
INSTRUCTIONS = """You control six bounded CMA batches to widen INPUT angular acceptance of this 2D circular-face optical body. Only probe='field' is allowed. Each batch costs 40 candidates (4 generations of 10). Choose group (curvature, shape, placement_size, all), sigma in [.005,.15], and parent (start or best). Return exactly the five action-schema keys as JSON. Use per-field violations to choose edits. Minimize largest normalized violation before throughput. The task full_fov_deg.field is the FULL target angular width; preserve it throughout. Source, detector, beam width, material, codec bounds and clearance are fixed. Neither exact body bounding-box dimension may grow beyond initial_dimensions. Every field requires >=25% folded throughput, >=60% root support and >=2 TIR events with margin >=0.5 degrees. Search uses 9 fields x 25 aperture rays; final audit uses 9 x 101 and separate denser verification. This is angular routing, not imaging FOV. Do not request repair or compact. Use all six batches, adapting to measured feedback. Small changes may preserve fragile routes. No shell or code. Explain your choice briefly in reason."""

def prepare(seed):
    folder = next(SOURCE.glob(f'*seed{seed}'))
    summary = read(folder/'summary.json')
    audits = [x for x in read(folder/'archive.json')['audits'] if x['result']['feasible']]
    selected = max(audits, key=lambda x:x['result']['quality'])
    row = selected['result']
    design = dict(genes=row['genes'], corners=summary['corners'], codec=summary['config']['codec'],
                  brief=summary['config']['brief'], source_archive=str(folder/'archive.json'),
                  coarse_evaluation=selected['coarse_evaluation'], unit=folder.name)
    lo,hi = exact_extent(PolygonCodec(**design['codec']).decode(design['genes']))
    original = max(abs(x) for x in design['brief']['fields_deg']) * 2
    task = dict(task_id=f'seed{seed}', label='angular_expansion', seed=seed+10000, design=design,
                initial_dimensions=(hi-lo).tolist(), batch_limits={'field':6}, instructions=INSTRUCTIONS,
                original_full_fov_deg=original, full_fov_deg={'field':original})
    screen=[]
    for full in [original] + [x for x in [2.,4.,6.,8.,10.,12.] if x>original]:
        task['full_fov_deg']['field']=full
        r=measure(task,design['genes'],'field',audit=True)
        screen.append(dict(full_fov_deg=full,result=r))
        if full>original and not r['feasible']:break
    target=screen[-1]['full_fov_deg']
    task['full_fov_deg']['field']=target
    for arm in ['qwen','numerical']:
        dest=ROOT/arm/task['task_id'];dest.mkdir(parents=True,exist_ok=False)
        dump(dest/'task.json',task)
    dump(ROOT/f'selection_{seed}.json',dict(source_sha256=digest(folder/'archive.json'),screen=screen))
    print(task['task_id'], 'original',original,'target',target,'support',screen[-1]['result']['support'],flush=True)
    return dict(task_id=task['task_id'],unit=folder.name,original=original,target=target,
                selection_source_rays=sum(x['result']['source_rays'] for x in screen))

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as pool: rows=list(pool.map(prepare,SEEDS))
    dump(ROOT/'protocol.json',dict(selection='One best archived dense pass per six deliberately diverse jobs: two widest-angle, five-corner, right receiver, bottom receiver, thin right receiver.',
         tasks=rows,model='qwen3.8-27b',search_proposals_per_design_per_arm=240,
         numerical_plan='Six all-variable CMA batches, sigma .03, parent best; same seeds and budgets as Qwen.',
         sources={str(p):digest(p) for p in [Path(__file__),Path('nslens/workbench/stock_probe.py')]},
         limitations='Development selection, same-engine verification, no imaging certification. Targets are next failing tested wider range up to 12 degrees full width.'))
