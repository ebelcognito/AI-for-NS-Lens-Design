"""Read the pinned-index field ladder: what field width survives at a fixed nd?

Each arm is one half-FOV; the question is where the frontier is, not which arm "won".
Prints the per-arm best and the acceptance spectrum, then re-scores the surviving
designs under the clean_pack gate -- the ladder runs the cheaper fold24_wide gate, so
nothing in it is known to be caustic-free until this second pass.

    python ns_probe/analyze_pinned_index.py runs/fold24_nd151*
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(arm_dir: str):
    rows = []
    for p in sorted(glob.glob(f"{arm_dir}/seed*/archive.json")):
        d = json.loads(Path(p).read_text())
        m = d["meta"]
        ni = m["param_names"].index("n")
        for c in d["cells"]:
            rows.append(dict(fit=c["fitness"], seed=Path(p).parent.name,
                             folds=int(c["tir_count"]), n=c["genotype"][ni],
                             accept=c.get("accept_deg"), T=c.get("thinness_T"),
                             per_field=c.get("per_field"), genotype=c["genotype"],
                             half_fov=m.get("half_fov_deg"), fields=m.get("fields"),
                             invalid=m["invalid"], evals=m["evals"]))
    rows.sort(key=lambda r: -r["fit"])
    return rows


def rescore_clean(rows, half_fov, top=5):
    """Re-run the survivors under convergence_gate=clean_pack at the same fields."""
    from nslens.engine import Runner
    from nslens.qd import experiments
    from nslens.qd.driver import _evaluate

    cfg = experiments.get("fold24_wide")
    cfg.fields = tuple(round(float(f), 4)
                       for f in np.linspace(-half_fov, half_fov, 9))
    cfg.require_monotone_convergence = True
    cfg.convergence_gate = "clean_pack"
    cfg.gate_all_fields = True
    out = []
    with Runner(timeout_s=20) as r:
        for row in rows[:top]:
            ev = _evaluate(r, cfg.problem, cfg, np.asarray(row["genotype"], float))
            out.append((row, ev))
    return out


def main() -> None:
    arms = sys.argv[1:] or sorted(glob.glob("runs/fold24_nd151*"))
    for arm in arms:
        rows = load(arm)
        if not rows:
            print(f"\n### {arm}: no archives yet")
            continue
        hf = rows[0]["half_fov"]
        inv = 100 * rows[0]["invalid"] / max(rows[0]["evals"], 1)
        print(f"\n### {arm}   half-FOV +-{hf}deg ({2*hf:g}deg full)  "
              f"{len(rows)} elites  invalid {inv:.0f}%")
        print(f"{'fit':>6s} {'seed':6s} {'fld':>3s} {'n':>6s} {'acc':>5s} {'T':>6s}  per-field")
        for r in rows[:5]:
            pf = " ".join(f"{v:.2f}" for v in (r["per_field"] or []))
            print(f"{r['fit']:6.3f} {r['seed']:6s} {r['folds']:3d} {r['n']:6.4f} "
                  f"{(r['accept'] or 0):5.1f} {(r['T'] or 0):6.1f}  {pf}")
        # the frontier: best fitness at each acceptance level
        by = {}
        for r in rows:
            a = r["accept"]
            if a is not None and r["fit"] > by.get(a, (0,))[0]:
                by[a] = (r["fit"], r["folds"], r["T"])
        print("   acceptance spectrum: " + "  ".join(
            f"+-{a:g}deg:{f:.3f}({k}f,T{t:.0f})" for a, (f, k, t) in sorted(by.items())))

        print("   -- re-scored under clean_pack (all fields) --")
        for row, ev in rescore_clean(rows, hf):
            if ev is None:
                print(f"   {row['fit']:.3f} -> REJECT (caustic / not clean at all fields)")
            else:
                print(f"   {row['fit']:.3f} -> {ev['fitness']:.3f} SURVIVES  "
                      f"clean_all={ev['info']['beam_clean_all']} "
                      f"T={ev['info']['thinness_T']:.1f}")


if __name__ == "__main__":
    main()
