"""Deterministic checks for the spherical-face element with sampled corners.

Arc geometry is exactly computable, so these assert closed-form values rather than
tolerances: a 44 x 48 rectangle has area 2112, and a 14 mm sag on a 48 mm chord is an
arc of radius (24^2 + 14^2) / 28 = 27.571 mm.
"""
from __future__ import annotations

import sys
from math import isclose, pi
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.freeform_body import (                                   # noqa: E402
    ArcPolygonBody,
    ArcPolygonCodec,
    DisplayTarget,
    FreeformRelayProblem,
    InputRay,
    trace_bundle,
)
from nslens.freeform_body.arc_codec import NOMINAL_CORNERS            # noqa: E402

RECTANGLE = ((38.0, -24.0), (82.0, -24.0), (82.0, 24.0), (38.0, 24.0))
FLAT = (0.0, 0.0, 0.0, 0.0)


def test_flat_faced_body_is_exactly_the_polygon_it_was_given():
    body = ArcPolygonBody(RECTANGLE, FLAT, 1.60)
    report = body.geometry_report()
    assert report.valid
    # Area comes from the sampled polyline, which cuts each corner very slightly;
    # the perimeter is summed analytically from the edges and is exact.
    assert isclose(report.signed_area, 44.0 * 48.0, rel_tol=2e-3)
    assert isclose(report.perimeter, 2.0 * (44.0 + 48.0), rel_tol=1e-12)
    assert report.max_abs_curvature == 0.0
    assert np.allclose(body.curvature(np.linspace(0.0, 1.0, 17, endpoint=False)), 0.0)


def test_a_sag_makes_a_sphere_of_the_radius_the_chord_and_bulge_imply():
    # right-hand face: chord 48 mm, sag 14 mm -> R = (24^2 + 14^2) / (2*14)
    body = ArcPolygonBody(RECTANGLE, (0.0, 14.0, 0.0, 0.0), 1.60)
    expected_radius = (24.0 ** 2 + 14.0 ** 2) / (2.0 * 14.0)
    hit = body.nearest_intersection((120.0, 0.0), (-1.0, 0.0))
    assert isclose(hit.point[0], 82.0 + 14.0, abs_tol=1e-9), "apex sits a full sag out"
    assert isclose(hit.curvature, 1.0 / expected_radius, rel_tol=1e-12)
    assert np.allclose(hit.outward_normal, (1.0, 0.0), atol=1e-12)


def test_a_negative_sag_is_concave_and_reports_negative_curvature():
    convex = ArcPolygonBody(RECTANGLE, (0.0, 10.0, 0.0, 0.0), 1.60)
    concave = ArcPolygonBody(RECTANGLE, (0.0, -10.0, 0.0, 0.0), 1.60)
    hit_out = convex.nearest_intersection((120.0, 0.0), (-1.0, 0.0))
    hit_in = concave.nearest_intersection((120.0, 0.0), (-1.0, 0.0))
    assert hit_out.point[0] > 82.0 > hit_in.point[0]
    assert hit_out.curvature > 0.0 > hit_in.curvature
    assert isclose(hit_out.curvature, -hit_in.curvature, rel_tol=1e-12)
    # Either way the reported normal still faces out of the glass.
    assert hit_out.outward_normal[0] > 0 and hit_in.outward_normal[0] > 0


def test_a_sag_larger_than_the_half_chord_stays_a_single_major_arc():
    """|sag| > half-chord means the arc passes beyond the circle's widest point."""
    body = ArcPolygonBody(((38.0, -6.0), (82.0, -6.0), (82.0, 6.0), (38.0, 6.0)),
                          (0.0, 11.0, 0.0, 0.0), 1.60)
    assert body.geometry_report().valid
    hit = body.nearest_intersection((120.0, 0.0), (-1.0, 0.0))
    assert isclose(hit.point[0], 82.0 + 11.0, abs_tol=1e-9)


def test_reversing_the_cage_carries_each_sag_to_the_same_physical_face():
    """A clockwise cage must decode to the identical body, sags included.

    Reversing the vertex list also reverses which edge each sag belongs to, so the
    sag list has to be permuted, not merely flipped -- get that wrong and a face's
    curvature silently lands on its neighbour.
    """
    ccw = ArcPolygonBody(RECTANGLE, (1.0, 14.0, 2.0, 3.0), 1.60)
    # Q = [P3,P2,P1,P0], so Q0->Q1 is edge 2 reversed, Q1->Q2 is edge 1, and so on.
    cw = ArcPolygonBody(tuple(reversed(RECTANGLE)), (2.0, 14.0, 1.0, 3.0), 1.60)
    assert np.allclose(ccw.corners, cw.corners)
    assert np.allclose(ccw.sags, cw.sags)
    u = np.linspace(0.0, 1.0, 128, endpoint=False)
    assert np.allclose(ccw.evaluate(u), cw.evaluate(u))
    assert np.allclose(ccw.curvature(u), cw.curvature(u))
    assert np.allclose(ccw.outward_normal(u), cw.outward_normal(u))

    # The 14 mm sag is on the eyebox-facing face in both, not rotated onto another.
    for body in (ccw, cw):
        hit = body.nearest_intersection((120.0, 0.0), (-1.0, 0.0))
        assert isclose(hit.point[0], 82.0 + 14.0, abs_tol=1e-9)


def test_containment_is_exact_rather_than_polyline_approximated():
    body = ArcPolygonBody(RECTANGLE, FLAT, 1.60)
    assert body.contains((60.0, 0.0))
    assert body.contains((38.5, 23.5)), "a point just inside a corner is still inside"
    assert not body.contains((82.001, 0.0))
    assert not body.contains((0.0, 0.0))
    assert not body.contains((60.0, 40.0))


def test_a_ray_through_the_body_crosses_exactly_twice_nearest_first():
    body = ArcPolygonBody(RECTANGLE, (0.0, 9.0, 0.0, -5.0), 1.60)
    hits = body.all_intersections((120.0, 0.0), (-1.0, 0.0))
    assert len(hits) == 2
    assert hits[0].t < hits[1].t
    assert hits[0].point[0] > hits[1].point[0]
    assert body.nearest_intersection((120.0, 0.0), (-1.0, 0.0)).t == hits[0].t
    assert all(0.0 <= hit.arc_s < 1.0 for hit in hits)


def test_crossing_faces_are_rejected_not_repaired():
    bowtie = ((38.0, -24.0), (82.0, 20.0), (82.0, -24.0), (38.0, 28.0))
    report = ArcPolygonBody(bowtie, FLAT, 1.60).geometry_report()
    assert not report.valid
    assert any("cross" in reason for reason in report.reasons)
    # A sag deep enough to sweep one face through another is caught the same way.
    swallowed = ArcPolygonBody(RECTANGLE, (0.0, -70.0, 0.0, 0.0), 1.60).geometry_report()
    assert not swallowed.valid
    # A degenerate cage is refused outright rather than reported.
    try:
        ArcPolygonBody(((38.0, -24.0), (82.0, 24.0), (82.0, -24.0), (38.0, 24.0)),
                       FLAT, 1.60)
    except ValueError as exc:
        assert "zero signed area" in str(exc)
    else:
        raise AssertionError("a zero-area cage was accepted")


def test_a_face_that_turns_the_body_inside_out_is_invalid():
    """A deep concave face can invert the region without any two faces crossing.

    Winding is canonicalized on the corner cage, so an inverted body would keep
    reporting normals that point into the glass.
    """
    report = ArcPolygonBody(RECTANGLE, (0.0, -60.0, 0.0, 0.0), 1.60).geometry_report()
    assert report.simple, "this one genuinely does not cross itself"
    assert report.signed_area < 0.0
    assert not report.valid
    assert any("invert" in reason for reason in report.reasons)


def test_each_corner_moves_only_inside_its_own_region():
    codec = ArcPolygonCodec()
    assert codec.gene_size == 3 * codec.corner_count == 12
    nominal = np.asarray(NOMINAL_CORNERS, dtype=float)
    assert np.allclose(codec.decode(np.zeros(codec.gene_size)).corners, nominal)

    pushed = codec.decode(np.concatenate([np.ones(8), np.zeros(4)])).corners
    assert np.allclose(pushed - nominal, codec.corner_region)
    for corner, moved in zip(nominal, pushed):
        assert float(np.max(np.abs(moved - corner))) <= codec.corner_region + 1e-12


def test_corner_regions_may_not_overlap():
    """Overlapping regions would let two corners swap and invert the polygon."""
    try:
        ArcPolygonCodec(corner_region=30.0)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("codec accepted overlapping corner regions")


def test_the_index_is_problem_configuration_and_never_a_gene():
    codec = ArcPolygonCodec(n=1.55)
    assert codec.gene_size == 12, "12 = 8 corner offsets + 4 sags, no index"
    for genes in (np.zeros(12), np.ones(12), -np.ones(12)):
        assert codec.decode(genes).n == 1.55
    for bad in (1.40, 1.80):
        try:
            ArcPolygonCodec(n=bad)
        except ValueError:
            continue
        raise AssertionError(f"codec accepted n={bad}")


def test_genes_must_stay_inside_the_normalized_box():
    codec = ArcPolygonCodec()
    for bad in (np.full(12, 1.5), np.full(11, 0.0), np.full(12, np.nan)):
        try:
            codec.decode(bad)
        except ValueError:
            continue
        raise AssertionError("codec accepted out-of-range genes")


def test_the_body_is_parameterized_by_arclength_so_u_is_arc_s():
    body = ArcPolygonBody(RECTANGLE, (0.0, 12.0, 0.0, 0.0), 1.60)
    u = np.linspace(0.0, 1.0, 512, endpoint=False)
    points = body.evaluate(u)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # Equal steps in u are equal steps in distance, to the sampling error of an arc.
    assert steps.std() / steps.mean() < 0.02
    assert np.allclose(body.arclength_coordinates(u), u)


def test_emergent_tir_still_happens_on_a_spherical_face():
    body = ArcPolygonBody(RECTANGLE, (0.0, 6.0, 0.0, 0.0), 1.60)
    # A steep internal ray exceeds the critical angle without anyone asking for it.
    rays = (InputRay(origin=(60.0, 0.0), direction=(0.30, 1.0)),)
    trace = trace_bundle(body, rays, max_interactions=8, energy_floor=1e-4)
    kinds = {event.kind for ray in trace.exits for event in ray.events}
    kinds |= {event.kind for ray in trace.terminations for event in ray.events}
    assert "tir" in kinds
    for ray in trace.exits:
        for event in ray.events:
            if event.kind == "tir":
                assert event.critical_margin_deg is not None
                assert event.critical_margin_deg > 0.0


def test_the_relay_problem_accepts_this_geometry_unchanged():
    codec = ArcPolygonCodec()
    problem = FreeformRelayProblem(codec=codec, field_degs=(0.0,),
                                   targets=(DisplayTarget(0.0),),
                                   minimum_field_throughput=1e-3)
    rng = np.random.default_rng(0)
    evaluations = [problem.evaluate(rng.uniform(-1.0, 1.0, codec.gene_size))
                   for _ in range(24)]
    assert any(e.feasible for e in evaluations), "spec must be satisfiable"
    for evaluation in evaluations:
        if evaluation.trace_result is None:
            # rejected before tracing: no fields to report, and ranked below band 0
            assert evaluation.fields == () and evaluation.search_score < 0.0
            continue
        assert len(evaluation.fields) == len(problem.targets)
        assert evaluation.shape is not None and evaluation.interaction is not None
        if evaluation.feasible:
            assert evaluation.route_keys


def main() -> None:
    checks = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for check in checks:
        check()
        print(f"  PASS  {check.__name__}")
    print("ALL ARC-POLYGON CHECKS PASS")


if __name__ == "__main__":
    main()
