"""Contract tests for canonical interaction fingerprint v1."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.engine.fingerprints import (FINGERPRINT_INVARIANCES, FINGERPRINT_SCHEMA,
                                        fingerprint_record,
                                        make_single_element_contact_coordinate)


def rotate(x, y, angle_deg):
    angle = math.radians(angle_deg)
    return (math.cos(angle) * x - math.sin(angle) * y,
            math.sin(angle) * x + math.cos(angle) * y)


def transform_record(record, *, angle_deg=0.0, dx=0.0, dy=0.0,
                     reflect=False, relabel=None, energy=None):
    relabel = relabel or {}
    transformed = {**record, "seq": []}
    if energy is not None:
        transformed["b"] = energy
    for source in record["seq"]:
        vertex = dict(source)
        x, y = float(vertex.get("x", 0)), float(vertex.get("y", 0))
        if reflect:
            x = -x
        x, y = rotate(x, y, angle_deg)
        vertex["x"], vertex["y"] = x + dx, y + dy
        if "surface" in vertex:
            vertex["surface"] = relabel.get(vertex["surface"], vertex["surface"])
        if vertex.get("u") is not None and reflect:
            vertex["u"] = -vertex["u"]
        for prefix in ("dirIn", "dirOut"):
            vx, vy = vertex.get(prefix + "X"), vertex.get(prefix + "Y")
            if vx is None or vy is None:
                continue
            if reflect:
                vx = -vx
            vertex[prefix + "X"], vertex[prefix + "Y"] = rotate(vx, vy, angle_deg)
        transformed["seq"].append(vertex)
    return transformed


def main() -> None:
    failures = []

    def check(name, condition):
        print(f"  [{'ok ' if condition else 'FAIL'}] {name}")
        if not condition:
            failures.append(name)

    base = {
        "b": 0.8,
        "outcome": "delivered",
        "historyComplete": True,
        "aggregate": False,
        "seq": [
            {"t": "source", "x": 0, "y": 0},
            {"t": "refract", "surface": "left", "u": 0.2, "x": 2, "y": 1,
             "dirInX": 1, "dirInY": 0, "dirOutX": 0.8, "dirOutY": 0.6},
            {"t": "tir", "surface": "right", "u": -0.3, "x": 5, "y": 2,
             "dirInX": 0.8, "dirInY": 0.6, "dirOutX": -0.8, "dirOutY": 0.6},
            {"t": "tir", "surface": "left", "u": -0.4, "x": 2, "y": 3,
             "dirInX": -0.8, "dirInY": 0.6, "dirOutX": 0.8, "dirOutY": 0.6},
            {"t": "sensor", "x": 10, "y": 4},
        ],
    }
    surface_of = lambda vertex: vertex["surface"]
    coordinate_of = lambda vertex, surface: vertex["u"]
    fp = fingerprint_record(base, surface_of, coordinate_of)

    moved = transform_record(base, angle_deg=71, dx=123, dy=-88,
                             reflect=True, relabel={"left": "zeta", "right": "alpha"})
    moved_fp = fingerprint_record(moved, surface_of, coordinate_of)
    check("declares canonical v1 schema", fp["schema_version"] == FINGERPRINT_SCHEMA)
    check("declares every promised invariance",
          set(FINGERPRINT_INVARIANCES) == set(fp["invariances"]))
    check("rigid motion, reflection, and relabeling preserve path hash",
          fp["path_hash"] == moved_fp["path_hash"])
    check("rigid motion, reflection, and relabeling preserve terminal hash",
          fp["fingerprint_hash"] == moved_fp["fingerprint_hash"])
    check("first/repeat surface indices are explicit",
          [(e["surface"], e["repeat_index"]) for e in fp["events"]]
          == [("s0", 0), ("s1", 0), ("s0", 1)])

    energy_changed = transform_record(base, energy=0.01)
    check("energy weights do not alter identity",
          fingerprint_record(energy_changed, surface_of, coordinate_of)["fingerprint_hash"]
          == fp["fingerprint_hash"])

    footprint_changed = transform_record(base)
    footprint_changed["seq"][3]["u"] = 0.7
    check("materially different repeated footprint changes path hash",
          fingerprint_record(footprint_changed, surface_of, coordinate_of)["path_hash"]
          != fp["path_hash"])

    fate_changed = {**base, "outcome": "escaped"}
    fate_fp = fingerprint_record(fate_changed, surface_of, coordinate_of)
    check("same optical path retains path hash across terminal fate",
          fate_fp["path_hash"] == fp["path_hash"])
    check("terminal fate changes full fingerprint hash",
          fate_fp["fingerprint_hash"] != fp["fingerprint_hash"])

    mech_changed = transform_record(base)
    mech_changed["seq"][1]["t"] = "partial_reflect"
    mech_changed["seq"][1]["mech"] = "forced"
    check("interaction mechanism changes path identity",
          fingerprint_record(mech_changed, surface_of, coordinate_of)["path_hash"]
          != fp["path_hash"])

    contact_of = make_single_element_contact_coordinate(60, 40, 10, -5, 30)
    check("single-element side coordinate is height-normalized",
          abs(contact_of({"x": 40, "y": 15}, "A") - 0.5) < 1e-12)
    check("single-element edge coordinates span -1 to +1",
          abs(contact_of({"x": 40 + math.tan(math.radians(10)) * 30, "y": -30}, "T") + 1)
          < 1e-12
          and abs(contact_of({"x": 80 + math.tan(math.radians(-5)) * 30, "y": -30}, "T") - 1)
          < 1e-12)

    if failures:
        raise SystemExit(f"FAILED: {len(failures)} check(s): {failures}")
    print("ALL FINGERPRINT CHECKS PASS")


if __name__ == "__main__":
    main()
