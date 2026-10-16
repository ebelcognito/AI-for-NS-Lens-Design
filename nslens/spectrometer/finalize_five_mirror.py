"""Automatic fixed-prescription audit, figures, report and 40-second video."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .search import serial, write_json


def run(out):
    from . import powered_five_mirror as powered
    from .five_mirror_audit import audit
    from .five_mirror_render import render, build_report
    from .five_mirror_video import render_video
    root=Path(out);source=root/'result.json';data=json.loads(source.read_text());config=data['configuration']
    status_path=root/'finalization.json';started=time.monotonic()
    status=dict(status='auditing',source=str(source),automatic=True,requested_video_seconds=40)
    write_json(status_path,status)
    powered.configure(config['segment'],config['aperture'])
    attempts=[];rows=[dict(seed=data['best_seed'],row=data['best'],kind='selected')]
    # Archived candidate alternatives retain the same fixed masks and topology.
    for p in sorted(root.glob('seed_*/result.json')):
        r=json.loads(p.read_text())
        rows.append(dict(seed=r['seed'],row=r['best'],kind='other_seed'))
    seed_file=root/f"seed_{data['best_seed']}"/'inherited_seed.json'
    if seed_file.exists():rows.append(dict(seed=data['best_seed'],row=json.loads(seed_file.read_text()),kind='initial_scaffold'))
    code_files=['five_mirror_audit.py','slit_hardware.py','powered_five_mirror.py','powered_four_mirror.py','four_mirror.py','four_mirror_audit.py','physics_audit.py','entrance_access.py','model.py']
    validation_code={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in code_files}
    audit_file=root/'independent_audit.json'
    cached=json.loads(audit_file.read_text()) if audit_file.exists() else {}
    cache_valid=bool(cached.get('passed') and cached.get('source_sha256')==hashlib.sha256(source.read_bytes()).hexdigest() and cached.get('validation_code_sha256')==validation_code)
    if cache_valid:
        check=cached;selected=rows[0];attempts=cached.get('selection_attempts',[])
    else:
        seen=set();selected=None
        for candidate in rows:
            g=np.asarray(candidate['row']['genes']);key=g.tobytes()
            if key in seen:continue
            seen.add(key)
            check=audit(g,powered,hardware_constraints=config.get('hardware_constraints',False))
            attempts.append(dict(seed=candidate['seed'],kind=candidate['kind'],passed=check['passed']))
            if check['passed']:
                selected=candidate;break
    if selected is None:
        write_json(root/'independent_audit.json',check)
        status.update(status='audit_failed',attempts=attempts,elapsed_seconds=time.monotonic()-started)
        write_json(status_path,status)
        raise RuntimeError('No audited candidate passed. Search records retained; no accepted-design video generated.')
    if selected['kind']!='selected':
        shutil.copyfile(source,root/'pre_audit_selected_result.json')
        row=selected['row'];data.update(genes=row['genes'],best=row,best_seed=selected['seed'],layout=powered.decode(row['genes']),geometry=powered.geometry_metrics(row['genes']),selection='Independent-audit fallback: '+selected['kind'])
        write_json(source,data)
    check.update(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),selection_attempts=attempts,validation_code_sha256=validation_code)
    write_json(root/'independent_audit.json',check)
    layout=powered.decode(data['genes'])
    bounds_template=Path('experiments/spectrometer_powered_five_mirror_cma_v1/bounds.json')
    bounds=json.loads(bounds_template.read_text()) if bounds_template.exists() else dict(normalized_bounds=[-1,1],coarse_dimensions=33,fine_dimensions=75)
    labels=[];mirror=0
    for index in range(len(layout['surfaces'])):
        if index==layout['grating_index']:labels.append('grating')
        else:mirror+=1;labels.append(f'M{mirror}')
    bounds.update(physical_apertures_frozen_mm=[s['aperture'] for s in layout['surfaces']],insertion_segment=config['segment'],
        architecture=' -> '.join(['slit']+labels+['detector']),hardware_settings=config.get('hardware_settings'),
        limitations=['Other optical mounts and substrates excluded.','Finite slit width not propagated optically.','Hardware gates active in geometry search, CMA retention and held-out audit.','Inherited optical prescription; insertion-slot and parent-fold search, not unrestricted topology discovery.','Three independent CMA random seeds share the selected scaffold.'])
    write_json(root/'bounds.json',bounds)
    write_json(root/'prescription.json',dict(source_result=str(source),source_sha256=check['source_sha256'],genes=data['genes'],layout=layout,hardware_settings=config.get('hardware_settings'),units='Lengths in mm, surface sag polynomials in local mm coordinates.'))
    status.update(status='rendering',audit_passed=True,attempts=attempts);write_json(status_path,status)
    render(source,title='Five mirrors + grating')
    build_report(source)
    status['status']='encoding_video';write_json(status_path,status)
    video=render_video(source,duration=40,fps=24,width=1920,height=1080)
    write_json(root/'video_artifacts.json',serial(video))
    for name in ['finalize_five_mirror.py','five_mirror_audit.py','five_mirror_render.py','five_mirror_video.py','slit_hardware.py','powered_four_mirror.py','four_mirror.py','four_mirror_audit.py','four_mirror_search.py','physics_audit.py','entrance_access.py','model.py']:
        shutil.copyfile(Path(__file__).parent/name,root/(Path(name).stem+'_snapshot.py'))
    status.update(status='complete',video=serial(video),elapsed_seconds=time.monotonic()-started,source_sha256=check['source_sha256'])
    write_json(status_path,status)
    return status

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out');args=p.parse_args();print(json.dumps(serial(run(args.out)),indent=2))
