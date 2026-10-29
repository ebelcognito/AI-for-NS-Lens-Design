"""Deterministic Stage-1 checks for the continuous freeform optical element."""
from __future__ import annotations

import sys
from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.freeform_body import (                                   # noqa: E402
    BoundaryEvent,
    DiverseDiscoveryArchive,
    ExitRay,
    FreeformBody2D,
    FreeformBodyCodec,
    FreeformDiscoveryProblem,
    FreeformDiscoverySearch,
    InputRay,
    SensorTarget,
    TraceResult,
    interaction_descriptor,
    ranked_route_signatures,
    route_signature,
    shape_niche_key,
    shape_descriptor,
    trace_bundle,
)


def ellipse_points(rx=10.0, ry=8.0, center=(0.0, 0.0), count=24):
    theta = np.linspace(0.0, 2.0 * pi, count, endpoint=False)
    return np.stack((center[0] + rx * np.cos(theta),
                     center[1] + ry * np.sin(theta)), axis=1)


def test_refractive_index_is_a_hard_problem_constraint():
    points = ellipse_points()
    FreeformBody2D(points, 1.50)
    FreeformBody2D(points, 1.70)
    for bad in (1.499, 1.701, 2.0):
        try:
            FreeformBody2D(points, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"out-of-domain index {bad} was accepted")


def test_nonfinite_configuration_and_ray_energy_are_rejected():
    for build in (
        lambda: FreeformBodyCodec(n=float("nan")),
        lambda: FreeformBodyCodec(center=(0.0, float("inf"))),
        lambda: InputRay(origin=(0.0, 0.0), direction=(1.0, 0.0),
                         energy_s=float("nan")),
        lambda: SensorTarget(source_id=0, y=float("nan"), half_height=1.0),
    ):
        try:
            build()
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite Stage-1 configuration was accepted")


def test_periodic_boundary_has_no_faces_and_intersects_as_one_surface():
    body = FreeformBody2D(ellipse_points(), 1.60)
    report = body.geometry_report()
    assert report.valid, report.reasons
    assert body.control_count == 24
    assert not hasattr(body, "faces")
    assert np.allclose(body.evaluate(0.0), body.evaluate(1.0), atol=1e-12)
    assert np.allclose(body.evaluate(0.0, 1), body.evaluate(1.0, 1), atol=1e-12)

    hits = body.all_intersections((-20.0, 0.0), (1.0, 0.0))
    assert len(hits) == 2
    assert abs(hits[0].point[0] + 10.0) < 2e-2
    assert abs(hits[1].point[0] - 10.0) < 2e-2
    assert all(0.0 <= hit.arc_s < 1.0 for hit in hits)


def test_default_geometry_audit_resolves_high_control_count_boundaries():
    body = FreeformBody2D(ellipse_points(count=160), 1.60)
    report = body.geometry_report()
    assert report.valid, report.reasons
    assert report.perimeter > 0.0


def test_multiresolution_codec_fixes_index_and_changes_the_whole_boundary():
    coarse = FreeformBodyCodec(center=(0.0, 0.0), base_radii=(10.0, 8.0),
                               n=1.57, control_count=16, harmonic_count=2)
    base = coarse.decode(np.zeros(coarse.gene_size))
    genes = np.zeros(coarse.gene_size)
    genes[5] = 0.7
    genes[8] = -0.45
    deformed = coarse.decode(genes)
    assert base.n == deformed.n == 1.57
    assert base.geometry_report().valid
    assert deformed.geometry_report().valid
    distance = np.linalg.norm(base.sample()[1] - deformed.sample()[1], axis=1)
    assert np.count_nonzero(distance > 0.1) > len(distance) // 2


def test_boundary_represents_a_smooth_concave_object_not_only_rounded_quads():
    theta = np.linspace(0.0, 2.0 * pi, 16, endpoint=False)
    radius = np.full(16, 10.0)
    radius[0] = 4.0
    points = np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=1)
    body = FreeformBody2D(points, 1.60)
    report = body.geometry_report()
    descriptor = shape_descriptor(body).as_dict()
    assert report.valid, report.reasons
    assert descriptor["concave_fraction"] > 0.1
    assert descriptor["curvature_sign_changes"] >= 2


def test_uncoated_branching_trace_records_emergent_boundary_events():
    body = FreeformBody2D(ellipse_points(), 1.60)
    result = trace_bundle(
        body,
        [InputRay(origin=(-20.0, 0.0), direction=(1.0, 0.0))],
        max_interactions=8,
        energy_floor=1e-9,
    )
    dominant = result.dominant_exit()
    assert dominant is not None
    assert [event.kind for event in dominant.events] == ["refract", "refract"]
    assert dominant.energy > 0.85
    assert all(not hasattr(event, "face") for event in dominant.events)
    assert abs(result.closure_residual) < 1e-10


def test_internal_high_incidence_ray_generates_tir_without_requesting_it():
    body = FreeformBody2D(ellipse_points(rx=10.0, ry=10.0, count=32), 1.60)
    result = trace_bundle(
        body,
        [InputRay(origin=(0.0, 8.0), direction=(1.0, 0.0))],
        max_interactions=4,
        energy_floor=0.0,
    )
    limited = [r for r in result.terminations if r.reason == "interaction_limit"]
    assert len(limited) == 1
    assert len(limited[0].events) == 4
    assert all(event.kind == "tir" for event in limited[0].events)
    assert all(event.critical_margin_deg > 10.0 for event in limited[0].events)
    assert abs(result.closure_residual) < 1e-12


def test_zero_power_ray_terminates_even_when_energy_floor_is_zero():
    body = FreeformBody2D(ellipse_points(), 1.60)
    result = trace_bundle(
        body,
        [InputRay(origin=(-20.0, 0.0), direction=(1.0, 0.0),
                  energy_s=0.0, energy_p=0.0)],
        energy_floor=0.0,
        max_branches=4,
    )
    assert len(result.terminations) == 1
    assert result.terminations[0].reason == "energy_floor"
    assert result.branch_limited_energy == 0.0
    assert result.closure_residual == 0.0


def test_empty_exit_descriptor_preserves_termination_energy_fractions():
    body = FreeformBody2D(ellipse_points(rx=10.0, ry=10.0, count=32), 1.60)
    result = trace_bundle(
        body,
        [InputRay(origin=(0.0, 8.0), direction=(1.0, 0.0))],
        max_interactions=1,
        energy_floor=0.0,
    )
    assert not result.exits
    descriptor = interaction_descriptor(result).as_dict()
    assert abs(descriptor["depth_limited_energy_fraction"] - 1.0) < 1e-12


def test_shape_descriptor_is_pose_scale_and_control_start_invariant():
    points = ellipse_points(rx=10.0, ry=7.0, count=24)
    a = FreeformBody2D(points, 1.60)
    theta = 0.73
    rotation = np.array(((np.cos(theta), -np.sin(theta)),
                         (np.sin(theta), np.cos(theta))))
    transformed = 2.4 * (np.roll(points, 7, axis=0) @ rotation.T) + np.array((31.0, -12.0))
    b = FreeformBody2D(transformed, 1.60)
    da, db = shape_descriptor(a), shape_descriptor(b)
    assert da.names == db.names
    assert np.allclose(da.values, db.values, rtol=2e-8, atol=2e-8)


def test_route_signature_uses_relative_boundary_footprints_not_face_labels():
    def event(kind, arc_s):
        return BoundaryEvent(kind=kind, point=(0.0, 0.0), u=arc_s, arc_s=arc_s,
                             curvature=0.0, incidence_deg=30.0,
                             critical_margin_deg=None, energy_in=1.0, energy_out=1.0)

    route_a = (event("refract", 0.10), event("tir", 0.42), event("refract", 0.83))
    route_b = tuple(replace(e, u=(e.u + 0.27) % 1.0, arc_s=(e.arc_s + 0.27) % 1.0)
                    for e in route_a)
    assert route_signature(route_a) == route_signature(route_b)


def test_dominant_route_aggregates_energy_across_many_branches():
    def event(kind, arc_s):
        return BoundaryEvent(kind=kind, point=(0.0, 0.0), u=arc_s, arc_s=arc_s,
                             curvature=0.0, incidence_deg=30.0,
                             critical_margin_deg=None, energy_in=1.0, energy_out=1.0)

    def exit_ray(events, energy, source_id):
        return ExitRay(source_id=source_id, origin=(0.0, 0.0), direction=(1.0, 0.0),
                       energy_s=energy, energy_p=0.0, events=events)

    single_bright = (event("refract", 0.10), event("tir", 0.40))
    shared_route = (event("refract", 0.20), event("refract", 0.80))
    result = TraceResult(
        exits=(
            exit_ray(single_bright, 0.70, 0),
            exit_ray(shared_route, 0.40, 0),
            exit_ray(shared_route, 0.40, 1),
            # A brighter ray that never touched the body is not an optical route.
            exit_ray((), 1.00, 2),
        ),
        terminations=(), emitted_energy=2.5, pruned_energy=0.0,
        depth_limited_energy=0.0, branch_limited_energy=0.0,
    )

    ranked = ranked_route_signatures(result)
    assert ranked[0] == route_signature(shared_route)
    assert ranked[1] == route_signature(single_bright)
    assert ranked_route_signatures(result, minimum_tir_bounces=1) == (
        route_signature(single_bright),)


def test_interaction_descriptor_reports_direct_and_ghost_route_mixture():
    body = FreeformBody2D(ellipse_points(), 1.60)
    result = trace_bundle(
        body,
        [InputRay(origin=(-20.0, y), direction=(1.0, 0.0), source_id=i)
         for i, y in enumerate((-2.0, 0.0, 2.0))],
        max_interactions=8,
        energy_floor=1e-7,
    )
    descriptor = interaction_descriptor(result).as_dict()
    assert descriptor["mean_interactions"] > 0.0
    assert descriptor["refract_fraction"] > 0.0
    assert descriptor["fresnel_reflect_fraction"] > 0.0
    assert descriptor["footprint_count"] >= 2.0
    assert 0.0 < descriptor["dominant_route_share"] < 1.0


def test_zero_event_exits_do_not_dilute_surface_event_type_fractions():
    body = FreeformBody2D(ellipse_points(), 1.60)
    result = trace_bundle(
        body,
        [InputRay(origin=(-20.0, 0.0), direction=(1.0, 0.0), source_id=0),
         InputRay(origin=(-20.0, 30.0), direction=(1.0, 0.0), source_id=1)],
        max_interactions=8,
        energy_floor=1e-7,
    )
    descriptor = interaction_descriptor(result).as_dict()
    event_fraction = sum(descriptor[name] for name in (
        "tir_fraction", "fresnel_reflect_fraction", "refract_fraction"
    ))
    assert abs(event_fraction - 1.0) < 1e-12
    assert descriptor["zero_event_energy_fraction"] > 0.0


def test_stage1_problem_scores_outcomes_and_exports_diversity_separately():
    codec = FreeformBodyCodec(center=(0.0, 0.0), base_radii=(10.0, 8.0),
                              n=1.60, control_count=16, harmonic_count=2,
                              translation_limit=(2.0, 2.0), rotation_limit_deg=20.0)
    problem = FreeformDiscoveryProblem(
        codec=codec,
        rays=(InputRay(origin=(-20.0, 0.0), direction=(1.0, 0.0), source_id=0),),
        sensor_x=20.0,
        targets=(SensorTarget(source_id=0, y=0.0, half_height=5.0),),
        spot_scale=1.0,
        minimum_field_throughput=0.1,
        max_interactions=8,
        energy_floor=1e-8,
    )
    result = problem.evaluate(np.zeros(codec.gene_size))
    assert result.feasible, result.reason
    assert result.fitness > 0.8
    assert result.fields[0].rms_error < 1e-8
    assert result.shape is not None and result.interaction is not None
    assert result.route_keys
    # The fixed material is owned by the problem codec, not present as an optimizable gene.
    assert codec.gene_size == 5 + 4 * codec.harmonic_count


def test_joint_archive_preserves_route_and_shape_families():
    codec = FreeformBodyCodec(center=(0.0, 0.0), base_radii=(10.0, 8.0),
                              n=1.60, control_count=16, harmonic_count=2,
                              translation_limit=(2.0, 2.0), rotation_limit_deg=20.0)
    problem = FreeformDiscoveryProblem(
        codec=codec,
        rays=(InputRay(origin=(-20.0, 0.0), direction=(1.0, 0.0), source_id=0),),
        sensor_x=20.0,
        targets=(SensorTarget(source_id=0, y=0.0, half_height=5.0),),
        minimum_field_throughput=0.1,
        max_interactions=8,
        energy_floor=1e-8,
    )
    genes = np.zeros(codec.gene_size)
    evaluation = problem.evaluate(genes)
    archive = DiverseDiscoveryArchive()
    status, _ = archive.add(genes, evaluation)
    assert status == "new"
    status, old = archive.add(genes, replace(evaluation, fitness=evaluation.fitness * 0.9))
    assert status == "none" and old == evaluation.fitness

    distinct_route = ((('tir',), ()),)
    status, _ = archive.add(genes, replace(evaluation, route_keys=distinct_route))
    assert status == "new"
    assert archive.route_family_count == 2
    assert archive.joint_niche_count == 2
    assert archive.random_elite(np.random.default_rng(4)).shape == genes.shape

    # Construction is inert: Stage-1 search only runs when an explicit budget is given.
    search = FreeformDiscoverySearch(problem, archive=archive, emitters=2, popsize=4, seed=3)
    assert search.progress().evaluations == 0
    # An explicit budget is a hard cap even when smaller than one CMA population.
    assert search.run(max_evaluations=1).evaluations == 1


def test_shape_niches_separate_convex_and_concave_morphologies():
    convex = FreeformBody2D(ellipse_points(rx=10.0, ry=10.0, count=16), 1.60)
    theta = np.linspace(0.0, 2.0 * pi, 16, endpoint=False)
    radius = np.full(16, 10.0)
    radius[0] = 4.0
    concave = FreeformBody2D(
        np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=1), 1.60)
    assert shape_niche_key(shape_descriptor(convex)) != shape_niche_key(shape_descriptor(concave))


def main() -> None:
    checks = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for check in checks:
        check()
        print(f"  PASS  {check.__name__}")
    print("ALL FREEFORM-BODY CHECKS PASS")


if __name__ == "__main__":
    main()
