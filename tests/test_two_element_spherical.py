from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.qd.multi_element import TwoElementSphericalProblem


def test_two_element_scene_is_uncoated_and_separated() -> None:
    problem = TwoElementSphericalProblem()
    genotype = (np.asarray(problem.lo) + np.asarray(problem.hi)) / 2
    params = problem.vec_to_params(genotype)
    scene = problem.layout(params).to_scene_dict()
    glasses = [obj for obj in scene["objs"] if obj.get("type") == "Glass"]
    assert len(glasses) == 2
    assert all("reflectivity" not in glass for glass in glasses)
    assert problem.is_valid(params)
    first, second = problem.elements(params)
    assert first.x_extent()[1] + problem.inter_element_clearance < second.x_extent()[0]


if __name__ == "__main__":
    test_two_element_scene_is_uncoated_and_separated()
    print("ALL TWO-ELEMENT SPHERICAL CHECKS PASS")
