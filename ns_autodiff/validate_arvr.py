"""Validate the AR/VR twin against the oracle.

(1) FORWARD MATCH on the bright bundle: for every export tracked ray whose recorded path is
    image-forming (only refract/tir, no partial-reflect ghost), the twin's exit angle and
    energy must equal the oracle's. (Ghost rays are stray light and are not modelled.)
(2) METRIC MATCH: reconstruct the 3 display fans, trace, and check the per-source
    collimation (theta, sigma, throughput) against the design's reference_metrics.

Usage:
    PY=/home/hisham26/Desktop/simu1/deeplens_env/bin/python
    $PY -m ns_autodiff.validate_arvr
"""
from __future__ import annotations

import numpy as np
import torch

from .arvr_data import (collimation_metrics, design_rays, load_export, p_tensor,
                        trace_fans, trace_rays_grouped)


def forward_match_bright(design):
    """Match only the bright tracked rays (events all in {refract,tir}), each traced for its
    own recorded bright-path length."""
    tr = design["tracked_rays"]
    keep = [i for i, r in enumerate(tr)
            if all(ev[0] in ("refract", "tir") for ev in r["events"])]
    if not keep:
        return None
    O = np.array([tr[i]["origin"] for i in keep], float)
    D = np.array([tr[i]["input_dir"] for i in keep], float)
    ne = np.array([int(tr[i]["n_events"]) for i in keep])
    p = p_tensor(design)
    with torch.no_grad():
        ang, en, py, rc, _m = trace_rays_grouped(p, O, D, ne)
    a_or = np.array([tr[i]["oracle_exit_angle_deg"] for i in keep])
    e_or = np.array([tr[i]["oracle_energy"] for i in keep])
    a_err = np.abs(ang.numpy() - a_or)
    e_err = np.abs(en.numpy() - e_or)
    return len(keep), len(tr), float(a_err.max()), float(e_err.max()), int(rc.sum())


def metric_match(design, common):
    p = p_tensor(design)
    with torch.no_grad():
        ang, en, py, rc, sid, _m = trace_fans(p, design, common)
    fields, fov, ordering = collimation_metrics(
        ang.numpy(), en.numpy(), py.numpy(), sid, rc.numpy(),
        n_fan=common["sources"]["n_fan"])
    ref = design["reference_metrics"]
    serr, terr = [], []
    for ft, fr in zip(fields, ref["fields"]):
        if ft["present"] and fr["present"]:
            serr.append(abs(ft["sigma"] - fr["sigma_deg"]))
            terr.append(abs(ft["theta"] - fr["theta_deg"]))
    return fields, fov, (max(serr) if serr else float("nan")), (max(terr) if terr else float("nan"))


def main():
    ex = load_export()
    common = ex["common"]
    print("=" * 80)
    print(f"AR/VR TWIN VALIDATION  ({ex['n_designs']} designs) -- {common['element_type']}")
    print("=" * 80)
    print(f"{'cell':>9} {'bright/all':>10} {'angle_err':>11} {'energy_err':>11} "
          f"{'theta_err':>10} {'sigma_err':>10} {'fov(twin/ref)':>16}")
    amax = emax = sgmax = thmax = 0.0
    for d in ex["designs"]:
        fm = forward_match_bright(d)
        fields, fov, serr, terr = metric_match(d, common)
        ref = d["reference_metrics"]
        a_e, e_e = (fm[2], fm[3]) if fm else (float("nan"), float("nan"))
        bright = f"{fm[0]}/{fm[1]}" if fm else "0/?"
        print(f"{str(tuple(d['cell'])):>9} {bright:>10} {a_e:>11.2e} {e_e:>11.2e} "
              f"{terr:>10.2e} {serr:>10.2e} {fov:>7.2f}/{ref['fov_deg']:>6.2f}")
        amax = max(amax, a_e if np.isfinite(a_e) else 0)
        emax = max(emax, e_e if np.isfinite(e_e) else 0)
        sgmax = max(sgmax, serr if np.isfinite(serr) else 0)
        thmax = max(thmax, terr if np.isfinite(terr) else 0)
    print("-" * 80)
    print(f"WORST: bright-ray angle {amax:.2e} deg, energy {emax:.2e}; "
          f"metric theta {thmax:.2e} deg, sigma {sgmax:.2e} deg")
    ok = amax < 1e-3 and emax < 1e-3
    print(f"BRIGHT-BUNDLE FORWARD-MATCH: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
