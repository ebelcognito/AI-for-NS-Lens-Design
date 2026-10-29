"""Contracts for the proxy -> full-beam experiment scaffold.

    python3 tests/test_rl_multifidelity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.rl import env_config, reward                         # noqa: E402
from nslens.rl.data.build_multifidelity import (                  # noqa: E402
    _label, make_samples,
)
from nslens.rl.multifidelity import (                             # noqa: E402
    FIDELITIES, get_fidelity, with_fidelity,
)
from nslens.learning.multifidelity_baseline import run as run_baseline  # noqa: E402
from nslens.rl.tasks.certify import Certified                     # noqa: E402


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'} {label}" + (f": {detail}" if detail else ""))
    return ok


def _cert(topology: int, suffix: int) -> Certified:
    return Certified(
        env="fold", fov=float(4 + 2 * (suffix % 5)), focus_w=8.0,
        topology=topology, thinness_bin=2, achievable=0.5, difficulty=0.5,
        seeds_hit=1, seeds_run=2, from_seed=False,
        genotype=[0.0, 45.0, 60.0, 22.0, 0.0, 0.0, 4.0, -4.0, 1.7],
        signature="", crit_margin=1.0, spec_id=f"cert_{topology}_{suffix}")


def check_ladder() -> bool:
    base = env_config.get("fold", fov=12.0, focus_w=6.0)
    observed = []
    ok = True
    for fidelity in FIDELITIES:
        view = with_fidelity(base, fidelity)
        observed.append((fidelity.name, view.problem.n_rays, len(view.fields)))
        ok &= view.problem.n_rays == fidelity.n_rays
        ok &= len(view.fields) == fidelity.n_fields
        ok &= 0.0 in view.fields
        ok &= view.fov == base.fov and view.problem.focus_w == base.problem.focus_w
    ok &= base.problem.n_rays == 17 and len(base.fields) == 9
    return _check("fidelity views preserve the cached base env", ok, repr(observed))


def check_authority() -> bool:
    verdict = reward.Verdict(
        "scored", fitness=0.4, topology=20, thinness_bin=2,
        gates={"clean_all": True, "exit_ok_all": True, "margin_ok": True})
    cheap = _label(verdict, get_fidelity("pencil_onaxis"))
    full = _label(verdict, get_fidelity("full_multifov"))
    return _check("only the full label can claim hard acceptance",
                  cheap["hard_pass"] is None and full["hard_pass"] is True)


def check_sampling() -> bool:
    certs = [_cert(topology, i) for i, topology in enumerate(range(20, 40))]
    a = make_samples(certs, 100, seed=19, uniform_fraction=0.0)
    b = make_samples(certs, 100, seed=19, uniform_fraction=0.0)
    train_topologies = {
        row["anchor"].topology for row in a if row["split"] == "train"
    }
    held_topologies = {
        row["anchor"].topology for row in a if row["split"] == "test_cell"
    }
    deterministic = all(
        x["sample_id"] == y["sample_id"]
        and np.array_equal(x["genotype"], y["genotype"])
        for x, y in zip(a, b))
    bounded = all(np.all((row["normalized_genotype"] >= 0.0)
                        & (row["normalized_genotype"] <= 1.0)) for row in a)
    return _check("sampling is deterministic, bounded, and topology-disjoint",
                  len(a) == 100 and deterministic and bounded
                  and not (train_topologies & held_topologies),
                  f"train={len(train_topologies)} held={len(held_topologies)}")


def check_baseline_contracts() -> bool:
    """Tiny argument checks should fail clearly instead of singular-fitting."""
    ok = True
    for kwargs in (
        dict(proxy_sizes=[], hf_sizes=[2]),
        dict(proxy_sizes=[1], hf_sizes=[2]),
        dict(proxy_sizes=[2], hf_sizes=[]),
    ):
        try:
            run_baseline([], **kwargs)
        except ValueError:
            pass
        except Exception:
            ok = False
        else:
            ok = False
    return _check("baseline rejects underspecified scaling sweeps", ok)


def main() -> int:
    print("RL multifidelity")
    ok = all((check_ladder(), check_authority(), check_sampling(),
              check_baseline_contracts()))
    print("ALL RL MULTIFIDELITY CHECKS PASS" if ok else
          "RL MULTIFIDELITY CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
