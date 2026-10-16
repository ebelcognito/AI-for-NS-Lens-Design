"""Sample pre-slit clearance of the prescribed input cone; optical masks only."""
import numpy as np
from . import model, four_mirror as fold

def upstream_access(layout, input_length=None, distance_mm=20., fields=9, pupil_rings=3, pupil_azimuths=16, exact=False):
    """Reverse ORIGINAL launch directions, independent of any added entrance fold.

    layout must contain surfaces and detector frame/bounds. Input_length is the
    unfolded original entrance leg used by the source illumination definition,
    not distance to the first added fold.
    """
    L=layout['input_length'] if input_length is None else input_length
    angles=np.arange(pupil_azimuths)*2*np.pi/pupil_azimuths
    pupil=np.vstack([np.zeros((1,2)),*[r*np.column_stack([np.cos(angles),np.sin(angles)]) for r in np.linspace(1/pupil_rings,1,pupil_rings)]])
    y=np.repeat(np.linspace(-5,5,fields),len(pupil));xy=np.tile(pupil,(fields,1));o=np.column_stack([0*y,y,0*y])
    d=-model.unit(np.array([0,0,L])-o+np.column_stack([xy,np.zeros(len(xy))])*L/7.6)
    fn=model.intersect
    if exact:
        from .physics_audit import exact_intersect
        fn=exact_intersect
    nearest=np.full(len(o),np.inf); blockers=[]
    for i,s in enumerate(layout['surfaces']):
        _,_,t,v=fn(o,d,s);ok=v&(t>1e-6)&(t<=distance_mm);nearest=np.minimum(nearest,np.where(ok,t,np.inf));blockers.append(int(ok.sum()))
    t,v=fold.detector_intersection(o,d,layout);ok=v&(t<=distance_mm);nearest=np.minimum(nearest,np.where(ok,t,np.inf))
    return dict(passed=bool(np.isinf(nearest).all()),distance_mm=float(distance_mm),rays=len(o),blocked_rays=int(np.isfinite(nearest).sum()),blocked_fraction=float(np.isfinite(nearest).mean()),surface_blocked_rays=blockers,detector_blocked_rays=int(ok.sum()),nearest_obstruction_mm=float(nearest.min()) if np.isfinite(nearest).any() else None,exact_polynomial_roots=bool(exact))
