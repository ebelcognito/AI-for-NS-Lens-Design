"""Load the AR/VR autodiff export into tensors and trace whole designs.

Reads ns_probe/results_arvr_qd/autodiff_export/ (produced read-only by the QD side). Each
design ships its genotype, the per-ray explicit (origin, input_dir), the frozen per-source
topology length, and oracle validation targets. This module turns one design into batched
tensors and traces all its tracked rays (grouping by event count so each batch shares one
unrolled length).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .element_arvr import PARAM_NAMES
from .forward_arvr import trace_arvr

DTYPE = torch.float64
EXPORT_DIR = (Path(__file__).resolve().parents[1]
              / "ns_probe" / "results_arvr_qd" / "autodiff_export")


def load_export(path=None):
    path = Path(path) if path else (EXPORT_DIR / "arvr_autodiff_export.json")
    return json.loads(Path(path).read_text())


def design_rays(design):
    """Return per-ray numpy arrays + oracle targets for one design."""
    tr = design["tracked_rays"]
    return dict(
        origins=np.array([r["origin"] for r in tr], dtype=float),
        dirs=np.array([r["input_dir"] for r in tr], dtype=float),
        n_events=np.array([int(r["n_events"]) for r in tr]),
        source=np.array([int(r["source"]) for r in tr]),
        oracle_angle=np.array([float(r["oracle_exit_angle_deg"]) for r in tr]),
        oracle_energy=np.array([float(r["oracle_energy"]) for r in tr]),
        oracle_pupil_y=np.array([float(r["oracle_pupil_y"]) for r in tr]),
    )


def p_tensor(design, requires_grad=False):
    return torch.tensor(design["genotype_vector"], dtype=DTYPE, requires_grad=requires_grad)


def make_fans(common):
    """Reconstruct the 3 display point-source fans exactly as arvr_prism.scene_arvr emits
    them: sources at (display_x, display_y), each n_fan rays over source_aim ± fan_halfangle.
    Returns origins[m,2], dirs[m,2], source_id[m]."""
    s = common["sources"]
    xs = s["display_x"]; y = s["display_y"]
    aim = s["source_aim_deg"]; nf = s["n_fan"]; half = s["fan_halfangle_deg"]
    O, D, sid = [], [], []
    for si, sx in enumerate(xs):
        for a in np.radians(aim + np.linspace(-half, half, nf)):
            O.append([sx, y]); D.append([np.cos(a), np.sin(a)]); sid.append(si)
    return np.array(O, float), np.array(D, float), np.array(sid)


def trace_rays_grouped(p, origins, dirs, n_events):
    """Trace rays that may have different bright-path lengths by grouping on n_events.
    Returns exit_angle[m], energy[m], pupil_y[m], reached[m] in original order + min margin."""
    m = len(n_events)
    ang = torch.zeros(m, dtype=p.dtype); en = torch.zeros(m, dtype=p.dtype)
    py = torch.zeros(m, dtype=p.dtype); rc = torch.zeros(m, dtype=torch.bool)
    margins = []
    for ne in sorted(set(int(x) for x in n_events)):
        idx = np.where(np.asarray(n_events) == ne)[0]
        tr = trace_arvr(p, torch.tensor(origins[idx], dtype=p.dtype),
                        torch.tensor(dirs[idx], dtype=p.dtype), ne)
        ii = torch.tensor(idx)
        ang = ang.index_copy(0, ii, tr.exit_angle); en = en.index_copy(0, ii, tr.energy)
        py = py.index_copy(0, ii, tr.pupil_y); rc = rc.index_copy(0, ii, tr.reached)
        mm = tr.min_tir_margin_deg()
        if torch.is_tensor(mm) and torch.isfinite(mm):
            margins.append(float(mm))
    return ang, en, py, rc, (float(min(margins)) if margins else float("nan"))


TYPE_CODE = {"refract": 0, "tir": 1, "partial_reflect": 2}


def trace_typed(p, design):
    """Trace EVERY export tracked ray driven by its recorded event TYPES (faithful frozen
    topology, reproduces ghosts too). Returns exit_angle[m], energy[m], pupil_y[m] in order
    + oracle target arrays for comparison. Groups rays by their type tuple to batch."""
    tr = design["tracked_rays"]
    m = len(tr)
    ang = torch.zeros(m, dtype=p.dtype); en = torch.zeros(m, dtype=p.dtype); py = torch.zeros(m, dtype=p.dtype)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, r in enumerate(tr):
        groups[tuple(ev[0] for ev in r["events"])].append(i)
    for typ, idx in groups.items():
        O = torch.tensor([tr[i]["origin"] for i in idx], dtype=p.dtype)
        D = torch.tensor([tr[i]["input_dir"] for i in idx], dtype=p.dtype)
        codes = torch.tensor([[TYPE_CODE[t] for t in typ]] * len(idx), dtype=torch.long).T  # [n_ev, g]
        out = trace_arvr(p, O, D, len(typ), types=codes)
        ii = torch.tensor(idx)
        ang = ang.index_copy(0, ii, out.exit_angle); en = en.index_copy(0, ii, out.energy)
        py = py.index_copy(0, ii, out.pupil_y)
    return (ang.numpy(), en.numpy(), py.numpy(),
            np.array([r["oracle_exit_angle_deg"] for r in tr]),
            np.array([r["oracle_energy"] for r in tr]),
            np.array([r["oracle_pupil_y"] for r in tr]))


def source_n_events(design, n_src=3, default=3):
    pst = design["per_source_topology"]
    return [int(pst.get(str(s), {}).get("n_events", default)) for s in range(n_src)]


def trace_fans(p, design, common):
    """Trace the reconstructed 3 display fans; each source uses its bright-path event count
    (per_source_topology). Returns exit_angle, energy, pupil_y, reached, source_id, min_margin."""
    O, D, sid = make_fans(common)
    ne_per_src = source_n_events(design, n_src=len(common["sources"]["display_x"]))
    n_events_per_ray = np.array([ne_per_src[s] for s in sid])
    ang, en, py, rc, margin = trace_rays_grouped(p, O, D, n_events_per_ray)
    return ang, en, py, rc, sid, margin


def collimation_metrics(angle, energy, pupil_y, source, reached, *, n_fan,
                        eyebox_y=0.0, eyebox_half=9.0, n_src=3):
    """Per-source angular metrics of the bright bundle at the eyebox, matching
    arvr_prism.arvr_metrics (energy-weighted mean angle theta, std sigma, throughput T).
    Inputs are numpy. Returns (fields, fov, ordering_ok)."""
    fields = []
    for si in range(n_src):
        mask = (source == si) & reached & (energy > 0) & (np.abs(pupil_y - eyebox_y) <= eyebox_half)
        a, w = angle[mask], energy[mask]
        if w.sum() <= 0:
            fields.append(dict(present=False, T=0.0, theta=None, sigma=None, n=0)); continue
        theta = float(np.average(a, weights=w))
        sigma = float(np.sqrt(np.average((a - theta) ** 2, weights=w)))
        fields.append(dict(present=True, T=float(w.sum() / n_fan), theta=theta,
                           sigma=sigma, n=int(mask.sum())))
    thetas = [f["theta"] for f in fields]
    present = [f for f in fields if f["present"]]
    ordering_ok = (len(present) == 3 and
                   (thetas[0] < thetas[1] < thetas[2] or thetas[0] > thetas[1] > thetas[2]))
    fov = (abs(fields[2]["theta"] - fields[0]["theta"])
           if fields[0]["present"] and fields[2]["present"] else 0.0)
    return fields, float(fov), bool(ordering_ok)
