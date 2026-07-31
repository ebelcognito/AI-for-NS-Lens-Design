"""Forward-match gate for the freeform trace.

With no polynomial terms the freeform faces ARE the circular arcs, so trace_ff(coeffs=None)
must reproduce forward.trace (the oracle-validated twin) bit-for-bit, and therefore the JS
oracle too. This is the end-to-end analog of validate_freeform.py's per-surface reduce-to-arc.

Checks per cell:
  (1) trace_ff vs forward.trace : sensor-y and energy, over all tracked rays  (~1e-8)
  (2) trace_ff vs JS oracle     : same tolerance the existing twin meets       (y<5e-3, e<5e-4)
  (3) twin dominant signature   : equals the frozen topology's event list

Run:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.validate_freeform_forward --cells 3,5 4,11 0,1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner  # noqa: E402

from . import topology as TOPO                                                        # noqa: E402
from .element_torch import PARAM_NAMES                                                # noqa: E402
from .forward import trace                                                            # noqa: E402
from .forward_freeform import trace_ff, dominant_signature_twin                       # noqa: E402

DTYPE = torch.float64
ARCHIVE = REPO / "ns_probe" / "results_fold_qd" / "archive.json"


def p_tensor(p0):
    return torch.tensor([p0[n] for n in PARAM_NAMES], dtype=DTYPE)


def check_cell(runner, cell, archive):
    topo = TOPO.extract(runner, archive, cell, n_rays=121, beam_w=8.0)
    p = p_tensor(topo.p0)
    with torch.no_grad():
        tr0 = trace(p, topo)                 # circular-arc twin (oracle-validated)
        trf = trace_ff(p, topo, coeffs=None) # freeform twin, pure conic == arc

    # (1) freeform vs arc twin
    dy = (trf.y - tr0.y).abs().max().item()
    de = (trf.energy - tr0.energy).abs().max().item()
    # (2) freeform vs oracle
    oy = np.abs(trf.y.numpy() - topo.oracle_y).max()
    oe = np.abs(trf.energy.numpy() - topo.oracle_b).max()
    # (3) twin signature vs frozen topology
    dom, _ = dominant_signature_twin(trf, topo.p0["cx"])
    topo_events = tuple((t, f) for (t, f) in topo.events)
    sig_ok = (dom == topo_events)
    reached = int(trf.reached.sum())

    print(f"  cell {str(cell):7s} | {topo.signature_str()}")
    print(f"      ff vs arc-twin : max|dy|={dy:.2e} mm   max|de|={de:.2e}")
    print(f"      ff vs oracle   : max|dy|={oy:.2e} mm   max|de|={oe:.2e}   (reached {reached}/{len(topo.dys)})")
    print(f"      twin signature : {'MATCH' if sig_ok else 'MISMATCH'}  twin_dom={dom}")
    ok = (dy < 1e-7) and (de < 1e-8) and (oy < 5e-3) and (oe < 5e-4) and sig_ok and reached == len(topo.dys)
    print(f"      --> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["3,5", "4,11", "0,1"])
    ap.add_argument("--archive", default=str(ARCHIVE))
    args = ap.parse_args()
    cells = [tuple(int(v) for v in c.split(",")) for c in args.cells]
    print("=" * 74)
    print("FREEFORM FORWARD-MATCH GATE  (trace_ff coeffs=None == forward.trace == oracle)")
    print("=" * 74)
    allok = True
    with PersistentRayOpticsRunner(timeout_s=30) as runner:
        for cell in cells:
            try:
                allok &= check_cell(runner, cell, Path(args.archive))
            except SystemExit as e:
                print(f"  cell {cell}: SKIP ({e})"); allok = False
    print("=" * 74)
    print(f"OVERALL: {'PASS' if allok else 'FAIL'}")


if __name__ == "__main__":
    main()
