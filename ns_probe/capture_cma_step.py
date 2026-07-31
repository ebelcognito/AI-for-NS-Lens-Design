"""
INSTRUMENTATION (visualization only — changes no experiment result).

Replays the CMA-ME arm of discovery_control.py deterministically for one seed and records,
for every emitter generation, the triple the researcher asked to *see*:
  (1) the elite the emitter was SEEDED from (persistence: archive.random_elite)  -> the
      "solution sampled to start the CMA step",
  (2) the POPULATION em.ask() draws around it (popsize offspring)                 -> the
      "generated population",
  (3) which offspring the archive accepted, with per-member fitness/TIR/status    -> the
      "good solution that emerged among those".

The emitter loop is byte-identical to discovery_control.run_cmame (same rng draw order, same
evaluate/archive.add), so the trajectory reproduced here is the one the recorded run took.
We only READ em.mean / em._last_X and the eval dicts we already compute; we never draw extra
randomness. Populations are tiny (popsize*9 floats) so we log every generation and post-select.

    PYTHONPATH=.:ns_probe python ns_probe/capture_cma_step.py --seed 2 --evals 14000 \
        --out ns_probe/results_disc/cma_step_capture
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_fold_qd import evaluate, BC_BINS, BC_BOUNDS
import fold_demo as FD
from cma_me import CMAES, GridArchive
from optical_rl.utils.ray_optics_bridge_persistent import PersistentRayOpticsRunner

REPO = Path(__file__).resolve().parents[1]
GOOD, HIGH, NS_TIR = 0.5, 0.9, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--evals", type=int, default=14000)
    ap.add_argument("--emitters", type=int, default=5)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--sigma0", type=float, default=0.12)
    ap.add_argument("--out", default=str(REPO / "ns_probe" / "results_disc" / "cma_step_capture"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    archive = GridArchive(BC_BINS, BC_BOUNDS)
    rng = np.random.default_rng(args.seed)
    scale = FD.HI - FD.LO
    sig = args.sigma0 * float(np.mean(scale))

    # per-emitter provenance so we always know the ancestral seed and its age
    origin = {}       # ei -> dict(genotype, is_elite)
    age = {}          # ei -> generations since (re)seed

    def new_emitter(s, ei):
        x0 = archive.random_elite(rng)
        is_elite = x0 is not None
        if x0 is None:
            x0 = rng.uniform(FD.LO, FD.HI)
        origin[ei] = dict(genotype=[float(v) for v in x0], is_elite=bool(is_elite))
        age[ei] = 0
        return CMAES(np.asarray(x0, float), sig, popsize=args.popsize, seed=s,
                     bounds=(FD.LO, FD.HI))

    steps = []        # one record per emitter generation
    n = 0
    with PersistentRayOpticsRunner(timeout_s=20) as runner:
        emitters = [new_emitter(args.seed + i, i) for i in range(args.emitters)]
        while n < args.evals:
            for ei in range(args.emitters):
                if n >= args.evals:
                    break
                em = emitters[ei]
                mean_now = [float(v) for v in em.mean]      # center BEFORE this ask/tell
                sigma_now = float(em.sigma)
                X = em.ask()
                keys = np.empty(len(X)); added = False
                members = []
                for k, xx in enumerate(X):
                    ev = evaluate(runner, xx)
                    n += 1
                    if ev is None:
                        keys[k] = -1e9
                        members.append(dict(genotype=[float(v) for v in xx], valid=False,
                                            fitness=-1.0, tir=0, spot=None, status="invalid",
                                            dom=None))
                        continue
                    st, old = archive.add(xx, ev["fitness"], ev["bc"], info=ev["info"])
                    if st == "new":
                        keys[k] = 2e12 + ev["fitness"]; added = True
                    elif st == "improved":
                        keys[k] = 1e12 + (ev["fitness"] - (old or 0.0)); added = True
                    else:
                        keys[k] = ev["fitness"]
                    members.append(dict(
                        genotype=[float(v) for v in xx], valid=True,
                        fitness=float(ev["fitness"]), tir=int(ev["info"]["tir_count"]),
                        spot=float(ev["info"]["spot"]), status=st,
                        dom=ev["info"]["dom_signature"]))
                em.tell(np.argsort(-keys))

                # winner = accepted (new/improved) member with best fitness this generation
                acc = [(i, m) for i, m in enumerate(members)
                       if m["valid"] and m["status"] in ("new", "improved")]
                winner_i, winner_foc, winner_tir = -1, -1.0, 0
                if acc:
                    winner_i, wm = max(acc, key=lambda t: t[1]["fitness"])
                    winner_foc, winner_tir = wm["fitness"], wm["tir"]

                steps.append(dict(
                    n_evals=n, emitter=ei, emitter_age=age[ei], sigma=sigma_now,
                    origin=origin[ei], mean=mean_now, members=members,
                    winner_idx=winner_i, winner_foc=winner_foc, winner_tir=winner_tir))
                age[ei] += 1

                if (not added) or em.restart_needed:
                    emitters[ei] = new_emitter(int(rng.integers(0, 1 << 30)), ei)
            if n % 2000 < args.emitters * args.popsize:
                best = max((s["winner_foc"] for s in steps), default=0.0)
                print(f"[capture] evals={n:6d} steps={len(steps):5d} filled={len(archive.cells):3d} "
                      f"best_winner_foc={best:.3f}", flush=True)

    # ---- post-select highlight candidates: accepted folded winners, best foc first ----
    cands = [s for s in steps if s["winner_idx"] >= 0 and s["winner_tir"] >= NS_TIR
             and s["winner_foc"] >= GOOD]
    cands.sort(key=lambda s: (s["winner_foc"], -s["emitter_age"]), reverse=True)
    top = cands[:12]

    (out / "steps_summary.json").write_text(json.dumps(dict(
        seed=args.seed, evals=n, n_steps=len(steps), n_folded_accepted=len(cands),
        highlights=top), indent=2))
    # also keep the full log compactly in case another generation is wanted later
    np.savez_compressed(out / "all_steps.npz",
                        obj=np.array(json.dumps(steps), dtype=object))
    print(f"[capture] wrote {out/'steps_summary.json'}  "
          f"({len(cands)} folded-accepted winners; top foc={top[0]['winner_foc']:.3f} "
          f"tir={top[0]['winner_tir']} at emitter_age={top[0]['emitter_age']}, "
          f"seed_is_elite={top[0]['origin']['is_elite']})" if top else
          f"[capture] wrote {out/'steps_summary.json'} (no folded accepted winner found)",
          flush=True)


if __name__ == "__main__":
    main()
