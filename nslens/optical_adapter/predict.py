"""Load an adapter and propose one simulator-checked continuous correction."""
import argparse
import json
from pathlib import Path
import torch
from .network import NumericPolicy, tiny_qwen, pretrained_qwen, load_adapter
from .state import observe, apply_action
from .data import write
from .experiment import scalar_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',required=True)
    p.add_argument('--source',required=True,help='JSON with a 36-element genes array; fixed 18mm apertures only')
    p.add_argument('--out',required=True);p.add_argument('--radius',type=float,default=.05)
    p.add_argument('--device',default='cuda:1');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.threads)
    original=json.loads(Path(a.source).read_text())
    apertures=original.get('configuration',{}).get('aperture_radii',original.get('aperture_radii'))
    if apertures is not None and list(apertures)!=[18,18,18]:
        raise ValueError('v1 requires 18mm physical apertures; do not reinterpret a corrected-aperture artifact')
    genes=original['genes'];cells,state=observe(genes,a.radius)
    metadata=torch.load(a.adapter,map_location='cpu',weights_only=True)['metadata']
    if metadata['kind']=='numeric':policy=NumericPolicy(metadata['width']);device='cpu'
    elif metadata['kind']=='tiny_qwen':policy=tiny_qwen(metadata['seed'],metadata['width']);device='cpu'
    elif metadata['kind']=='pretrained_qwen':
        policy=pretrained_qwen(metadata['weights'],a.device,metadata['width']);device=a.device
    else:raise ValueError('Unknown adapter architecture')
    load_adapter(policy,a.adapter);policy.eval()
    with torch.no_grad():
        action=policy(torch.from_numpy(cells)[None].to(device),torch.from_numpy(state)[None].to(device))[0].cpu().numpy()
    candidate=apply_action(genes,action,a.radius)
    before=scalar_metrics(genes);after=scalar_metrics(candidate)
    accepted=after['loss']<before['loss']
    result=dict(genes=candidate.tolist() if accepted else genes,proposal_genes=candidate.tolist(),
        action=action.tolist(),accepted=accepted,radius=a.radius,aperture_radii=[18,18,18],
        adapter=str(Path(a.adapter).resolve()),source=str(Path(a.source).resolve()),
        before=before,raw_after=after,before_audit=scalar_metrics(genes,True),
        proposal_audit=scalar_metrics(candidate,True),forward_traces=5,
        interpretation='One continuous correction; coarse loss guard only, not physical certification or paper reproduction')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);write(out/'result.json',result)
    print(json.dumps(dict(accepted=accepted,before_loss=before['loss'],proposal_loss=after['loss'])))


if __name__=='__main__':main()
