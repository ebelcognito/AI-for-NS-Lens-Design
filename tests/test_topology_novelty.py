"""Contract checks for conservative topology novelty."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.qd.novelty import (TopologyExclusionSet, canonical_topology,
                               edit_distance)


def test_surface_renaming_is_not_novel() -> None:
    old = "T@A -> TIR@D -> TIR@A -> T@B"
    renamed = "T@left -> TIR@top -> TIR@left -> T@right"
    assert canonical_topology(old) == canonical_topology(renamed)
    assert not TopologyExclusionSet([old]).report(renamed).passes


def test_repeat_pattern_is_preserved() -> None:
    repeated = canonical_topology("T@A -> TIR@B -> TIR@A -> T@B")
    distinct = canonical_topology("T@A -> TIR@B -> TIR@D -> T@B")
    assert repeated != distinct
    assert edit_distance(repeated, distinct) == 1


def test_two_edit_and_fraction_thresholds_are_both_required() -> None:
    old = "T@A -> TIR@B -> TIR@A -> T@B"
    one_edit = "T@A -> TIR@B -> PR@A -> T@B"
    two_edits = "T@A -> TIR@D -> TIR@B -> TIR@A -> T@B"
    exclusion = TopologyExclusionSet([old], min_edits=2,
                                     min_normalized_distance=0.25)
    assert not exclusion.report(one_edit).passes
    assert exclusion.report(two_edits).passes


if __name__ == "__main__":
    test_surface_renaming_is_not_novel()
    test_repeat_pattern_is_preserved()
    test_two_edit_and_fraction_thresholds_are_both_required()
    print("ALL TOPOLOGY NOVELTY CHECKS PASS")
