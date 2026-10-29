"""Price an exit-convergence gate on the designs we already have.

`clean_pack` cannot see the exit face -> detector leg (see Observation.exit_convergence
for why: leg widths are measured at the vertex that STARTS the leg, and a final-leg ray
crossing is deliberately excluded). So a bundle that leaves the glass narrow and fans out
on the way to the sensor passes every gate in the stack, and is only priced -- never
rejected -- by focused_throughput.

This does NOT pick a threshold and apply it. It measures several candidate verdicts on
two real populations and prints what each keeps and what each throws away, because the
one threshold this repo already chose by plausibility (etendue growth <= 2.0) turned out
to reject the two best designs in the run. See driver.py's gate_all_fields comment.

Every verdict is evaluated at ALL nine canonical fields, not the central one: 40/40
fold24_clean elites are clean_pack on axis and only 16/40 are clean everywhere, so a
field-0 measurement of a new gate would repeat exactly the mistake that finding names.

    python ns_probe/check_exit_convergence.py
    python ns_probe/check_exit_convergence.py --json runs/exit_convergence.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nslens.engine import Runner                            # noqa: E402
from nslens.qd.observation import measure                   # noqa: E402

sys.path.insert(0, str(_REPO / "ns_probe"))
from build_catalogue import (CANON_FIELDS, RELATIVE_SAG_RUNS,  # noqa: E402
                             hardened_problem)


#: `focused_throughput` scores energy within +-focus_w of the centroid, so the scoring
#: window is 2*focus_w wide -- 16 mm on FoldProblem. Every absolute threshold below is
#: expressed in units of that window rather than in millimetres, so the gate travels to
#: a problem with a different detector.
FOCUS_W = 8.0

#: candidate verdicts, each a function of every field's exit_convergence dict.
#:
#: The RATIO tests that look natural here are wrong, and the first run of this script is
#: the evidence: `sensor_ratio <= 0.8` keeps 1 of 53 elites and throws away every design
#: above 0.79, because the good designs focus EARLY (waist at 20-45 mm of an ~85 mm gap)
#: and re-expand to a spot that still lands inside the window. Ratio to the exit width
#: says nothing about whether the light arrives usefully -- the same error the repo
#: already made once with etendue growth (driver.py, gate_all_fields).
#:
#: What separates them is absolute size at the detector, plus one threshold-free fact:
#: whether the waist is ahead of the exit face at all. A negative focus_dist means the
#: bundle passed its focus inside the glass and is monotonically fanning out by the time
#: it leaves -- that is the "literally diverging" failure, and it needs no calibration.
def _fwd(fields):
    """waist ahead of the exit face at every field"""
    return all(e and e["measured"] and e["focus_dist"] is not None
               and e["focus_dist"] > 0.0 for e in fields)


def _lands(mult):
    def f(fields):
        return all(e and e["measured"] and e["width_sensor"] is not None
                   and e["width_sensor"] <= mult * 2 * FOCUS_W for e in fields)
    f.__name__ = f"width_sensor <= {mult} window"
    return f


GATES = [
    ("A: focus ahead of exit face", lambda fs: _fwd(fs)),
    ("B: lands inside the window", _lands(1.0)),
    ("C: A and B", lambda fs: _fwd(fs) and _lands(1.0)(fs)),
    ("D: A and within half a window", lambda fs: _fwd(fs) and _lands(0.5)(fs)),
    ("E: old weak (converges everywhere)", lambda fs: all(
        e and e["measured"] and e["converges"] for e in fs)),
]


def measure_design(runner, problem, x) -> dict:
    """Exit convergence at every canonical field. Worst field = the one that puts the
    WIDEST bundle on the detector -- absolute, not relative to the exit width, for the
    reason in the GATES comment."""
    p = problem.vec_to_params(x)
    face_of = problem.face_of(p)
    per_field, flags = [], []
    for f in CANON_FIELDS:
        try:
            raw, _ = runner.simulate(problem.layout(p, field_deg=f).to_scene_dict())
            obs = measure(raw, face_of, x_sensor=problem.x_sensor, allow_empty=True)
        except Exception:
            obs = None
        if obs is None:
            per_field.append(None)
            flags.append(False)          # a field with no bundle cannot be declared clean
            continue
        ec = obs.exit_convergence()
        per_field.append(ec)
        flags.append(bool(ec["measured"] and ec["converges"]))
    live = [e for e in per_field if e and e["measured"]]
    if not live:
        return dict(measured=False, flags=flags, per_field=per_field, worst={})
    worst = max(live, key=lambda e: (e.get("width_sensor") if e.get("width_sensor")
                                     is not None else float("inf")))
    return dict(measured=True, flags=flags, per_field=per_field, worst=worst,
                n_live=len(live))


def load_catalogue() -> list:
    path = _REPO / "catalogue" / "catalogue.json"
    if not path.exists():
        return []
    out = []
    for e in json.load(open(path))["entries"]:
        out.append(dict(src="catalogue", label=e["signature"],
                        genotype=e["genotype"], relative_sag=e["relative_sag"],
                        score=e["hardened_fitness"], verdict=e["verdict"]))
    return out


def load_elites(pattern: str) -> list:
    out = []
    for f in sorted(glob.glob(str(_REPO / pattern / "seed*" / "archive.json"))):
        arch = json.load(open(f))
        run = Path(f).parents[1].name
        for c in arch.get("cells", arch.get("elites", [])):
            g = c.get("genotype") or c.get("x")
            if g is None:
                continue
            out.append(dict(src=run, label=(c.get("info", {}) or {}).get(
                "dom_signature", ""), genotype=g,
                # every ab40k arm ran fold24_clean's genome, which is relative_sag
                relative_sag=True, score=c.get("fitness"), verdict="ELITE"))
    return out


def report(name, rows, results) -> dict:
    live = [(r, m) for r, m in zip(rows, results) if m["measured"]]
    print(f"\n=== {name}: {len(rows)} designs, {len(live)} with a measurable exit bundle ===")
    scored = [(r, m) for r, m in live if r["score"]]
    scored.sort(key=lambda rm: -rm[0]["score"])
    summary = {}
    for gname, fn in GATES:
        keep = [(r, m) for r, m in live if fn(m["per_field"])]
        kept_scored = [r["score"] for r, _ in keep if r["score"]]
        top = [r["score"] for r, m in scored[:10] if fn(m["per_field"])]
        print(f"  {gname:<32} keeps {len(keep):>4}/{len(live):<4} "
              f"({100.0*len(keep)/max(len(live),1):5.1f}%)  "
              f"best kept {max(kept_scored) if kept_scored else 0.0:.3f}  "
              f"top-10 survivors {len(top)}/10")
        summary[gname] = dict(kept=len(keep), of=len(live),
                              best=max(kept_scored) if kept_scored else None,
                              top10=len(top))
    if scored:
        print("\n  the top designs (worst field), and the two gate facts:")
        print("  %-6s %-9s %-9s %-7s %-7s  %s" % ("score", "w_sensor", "fwd_fields",
                                                  "A", "C", "signature"))
        for r, m in scored[:12]:
            w = m["worst"]
            nfwd = sum(1 for e in m["per_field"] if e and e["measured"]
                       and e["focus_dist"] is not None and e["focus_dist"] > 0)
            print("  %-6.3f %-9s %-9s %-7s %-7s  %s" % (
                r["score"],
                ("%.2f" % w["width_sensor"]) if w.get("width_sensor") is not None else "-",
                "%d/%d" % (nfwd, len(m["per_field"])),
                _fwd(m["per_field"]), GATES[2][1](m["per_field"]),
                (r["label"] or "")[:40]))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--elites", default="runs/ab40k_topo")
    a = ap.parse_args()

    pops = [("catalogue (all topologies)", load_catalogue()),
            (f"elites ({a.elites})", load_elites(a.elites)),
            ("elites (runs/ab40k_clean)", load_elites("runs/ab40k_clean"))]

    out, problems = {}, {rel: hardened_problem(rel) for rel in (False, True)}
    with Runner(timeout_s=20) as runner:
        for name, rows in pops:
            if not rows:
                print(f"\n=== {name}: nothing found, skipped ===")
                continue
            res = [measure_design(runner, problems[r["relative_sag"]],
                                  np.array(r["genotype"], float)) for r in rows]
            out[name] = dict(summary=report(name, rows, res),
                             rows=[dict(label=r["label"], score=r["score"],
                                        verdict=r["verdict"], src=r["src"],
                                        converges=all(m["flags"]),
                                        n_fields_converging=sum(m["flags"]),
                                        worst=m["worst"]) for r, m in zip(rows, res)])
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1, default=float)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
