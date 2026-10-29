"""The RL reward must agree with the QD driver about what a design is worth.

`nslens/rl/reward.py` re-implements the measurement path so that a gated design comes back
graded instead of as None. That freedom is only safe if the two paths still produce the
same numbers wherever both produce numbers at all -- otherwise the RL result cannot be
compared against the CMA-ME baseline, which is the whole point of doc 18 §6 item 3.

So: for every archive elite that the driver accepts under the hardened config, the fitness
and the behaviour coordinates from `reward.evaluate` must match to floating-point equality.

    python tests/test_rl_reward.py            (needs the engine)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.engine import Runner                                    # noqa: E402
from nslens.qd.driver import _evaluate                              # noqa: E402
from nslens.rl import env_config, parse, reward                     # noqa: E402

#: how many archive elites to cross-check. All 95 takes ~15 s; this is the gate, so keep
#: it quick but wide enough to hit several topologies.
N_ELITES = 40


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'} {label}" + (f": {detail}" if detail else ""))
    return ok


def _elites(n: int):
    cells = [c for f in sorted(glob.glob(str(REPO / "runs/ab40k_*/seed*/archive.json")))
             for c in json.loads(Path(f).read_text())["cells"]]
    cells.sort(key=lambda c: -c["fitness"])
    return cells[:n]


def check_agrees_with_driver() -> bool:
    env = env_config.get("fold")
    cells = _elites(N_ELITES)
    if not cells:
        return _check("driver agreement", False, "no ab40k archives on disk")

    n_cmp, bad = 0, []
    with Runner(timeout_s=20) as r:
        for c in cells:
            x = np.asarray(c["genotype"], float)
            ref = _evaluate(r, env.problem, env.cfg, x)
            got = reward.evaluate(r, env, x)
            if ref is None:
                # the driver rejected it; reward.py must not be claiming a clean score
                if got.stage == "scored" and got.gates.get("clean_all") \
                        and got.gates.get("exit_ok_all"):
                    bad.append((c["fitness"], "driver rejected a design reward.py "
                                              "reports as passing every gate"))
                continue
            n_cmp += 1
            if not got.scored:
                bad.append((c["fitness"], f"driver accepted, reward.py says {got.stage}"))
            elif ref["fitness"] != got.fitness:
                bad.append((c["fitness"],
                            f"fitness {ref['fitness']!r} != {got.fitness!r}"))
            elif tuple(ref["bc"]) != tuple(got.bc):
                bad.append((c["fitness"], f"bc {ref['bc']} != {got.bc}"))
    return _check(f"fitness and bc match the driver on {n_cmp} accepted elites",
                  not bad, "; ".join(f"[{f:.3f}] {m}" for f, m in bad[:4]))


def check_reward_ordering() -> bool:
    """A better design must not score worse. Checked against the driver's own ranking on
    the elites, holding the spec fixed so only the design varies."""
    env = env_config.get("fold")
    cells = _elites(N_ELITES)
    rows = []
    with Runner(timeout_s=20) as r:
        for c in cells:
            v = reward.evaluate(r, env, c["genotype"])
            if v.scored:
                rows.append((v.fitness, v))
    if len(rows) < 5:
        return _check("reward ordering", False, f"only {len(rows)} scored elites")

    spec = reward.Spec("fold", rows[0][1].topology, rows[0][1].thinness_bin,
                       achievable=max(f for f, _ in rows))
    same = [(f, reward.reward_of(v, spec, env).total) for f, v in rows
            if v.topology == spec.topology and v.thinness_bin == spec.thinness_bin]
    if len(same) < 3:
        return _check("reward ordering", True, "too few same-cell elites to rank")
    corr = float(np.corrcoef([f for f, _ in same], [t for _, t in same])[0, 1])
    return _check(f"reward tracks fitness within a cell (rho={corr:.3f}, n={len(same)})",
                  corr > 0.8)


def check_stages_are_ordered() -> bool:
    """Every stage is worth at least as much as the ones below it."""
    env = env_config.get("fold")
    spec = reward.Spec("fold", 21, 3, achievable=0.878)
    totals = []
    for stage in ("degenerate", "no_light", "off_class"):
        totals.append(reward.reward_of(reward.Verdict(stage), spec, env).total)
    best = reward.Verdict("scored", fitness=0.878, topology=21, thinness_bin=3,
                          gates=dict(clean_frac=1.0, exit_ok_frac=1.0, crit_margin=9.0))
    totals.append(reward.reward_of(best, spec, env).total)
    return _check(f"stages are monotone {[round(t, 3) for t in totals]}",
                  all(a <= b for a, b in zip(totals, totals[1:]))
                  and abs(totals[-1] - reward.R_MAX) < 1e-9)


def check_parse() -> bool:
    env = env_config.get("fold")
    good = parse.format_design(env, [0.0, 0.0, 60.0, 20.0, 0.1, 0.1, 5.0, -5.0, 1.7])
    cases = [
        ("valid", good, True, ""),
        ("prose only", "I would use a prism.", False, "no_design_block"),
        ("bad json", "<design>{cx: 60}</design>", False, "bad_json"),
        ("array", "<design>[1,2,3]</design>", False, "not_an_object"),
        ("missing key", '<design>{"cx": 60.0}</design>', False, "missing_keys"),
        ("out of bounds", parse.format_design(
            env, [0.0, 0.0, 60.0, 20.0, 0.1, 0.1, 5.0, -5.0, 9.9]), False,
         "out_of_bounds"),
        ("last block wins", "<design>{\"bad\": 1}</design>" + good, True, ""),
    ]
    ok = True
    for label, text, want_ok, want_failure in cases:
        r = parse.parse(text, env)
        ok &= _check(f"parse {label}", r.ok == want_ok and r.failure == want_failure,
                     f"ok={r.ok} failure={r.failure!r} detail={r.detail[:60]}")
    r = parse.parse(cases[5][1], env)
    ok &= _check("out-of-bounds scores above malformed",
                 r.reward_floor > parse.parse("nothing", env).reward_floor)
    return ok


def check_topology_distance() -> bool:
    """Vocabulary index distance is meaningless; multiset distance is not."""
    from nslens.qd.descriptors import TIR_FACE_INDEX
    a = TIR_FACE_INDEX[(("A", 1), ("B", 1), ("T", 1))]
    b = TIR_FACE_INDEX[(("A", 1), ("B", 1), ("D", 1))]
    same = TIR_FACE_INDEX[(("A", 1), ("B", 1), ("T", 1))]
    ok = _check("identical multisets are distance 0",
                reward.topology_distance(a, same) == 0)
    ok &= _check("one swapped bounce is distance 2",
                 reward.topology_distance(a, b) == 2,
                 f"got {reward.topology_distance(a, b)}")
    return ok


def main() -> int:
    print("RL reward")
    ok = all([
        check_parse(),
        check_topology_distance(),
        check_stages_are_ordered(),
        check_agrees_with_driver(),
        check_reward_ordering(),
    ])
    print("ALL RL REWARD CHECKS PASS" if ok else "RL REWARD CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
