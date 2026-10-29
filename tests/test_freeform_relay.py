"""Deterministic checks for the reverse-traced three-field AR/VR relay objective."""
from __future__ import annotations

import sys
from dataclasses import replace
from math import cos, radians, sin
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nslens.freeform_body import (                                   # noqa: E402
    DisplayTarget,
    DiverseDiscoveryArchive,
    EyeboxSpec,
    FreeformBody2D,
    FreeformBodyCodec,
    FreeformDiscoverySearch,
    FreeformRelayProblem,
    InputRay,
    ObjectPlane,
    VerticalBlocker,
    collimated_eyebox_rays,
    project_exits_to_line,
    project_exits_to_plane,
    project_exits_to_x,
    relay_field_report,
    trace_bundle,
)
from nslens.freeform_body.relay import (                              # noqa: E402
    EYEBOX_TILT_DEG, EYEBOX_X, EYEBOX_Y, FIELD_DEGS, OBJECT_CENTER,
    OBJECT_TILT_DEG, TARGET_OFFSETS)

CODEC = FreeformBodyCodec()


def random_genes(count: int, seed: int = 0) -> list[np.ndarray]:
    """The fixture protocol: successive draws from one seeded generator."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1.0, 1.0, CODEC.gene_size) for _ in range(count)]


def permissive_problem(**kw) -> FreeformRelayProblem:
    """A problem pinned to the old 1e-3 floor.

    Tests that need a specific random gene to be *feasible* must not ride the default
    floor, or raising it turns a mechanism test into a fixture failure.  Tests about
    the default itself state it explicitly instead.
    """
    return FreeformRelayProblem(codec=CODEC, minimum_field_throughput=1e-3, **kw)


def fake_records(coords, energies, *, events=(("refract",),)):
    return [{"coord": float(c), "energy": float(e), "events": events}
            for c, e in zip(coords, energies)]


# --- the fixed eyebox ---------------------------------------------------------------

def test_every_field_fills_the_whole_fixed_eyebox_with_a_collimated_bundle():
    eyebox = EyeboxSpec()
    rays = collimated_eyebox_rays(eyebox, FIELD_DEGS)
    assert len(rays) == eyebox.ray_count * len(FIELD_DEGS)
    for source_id in range(len(FIELD_DEGS)):
        bundle = [r for r in rays if r.source_id == source_id]
        assert len(bundle) == eyebox.ray_count
        center = np.array((eyebox.x, eyebox.y_center))
        positions = np.array([r.origin for r in bundle]) - center
        offsets = sorted(positions @ eyebox.along)
        # the box is spanned exactly, so no field can score off a thin slice of it
        assert np.isclose(offsets[0], -eyebox.half_height)
        assert np.isclose(offsets[-1], eyebox.half_height)
        assert np.allclose(positions @ eyebox.inward_normal, 0.0)
        # collimated: one shared direction across the whole aperture
        assert len({r.direction for r in bundle}) == 1


def test_single_on_axis_field_uses_the_full_pupil_bundle():
    problem = FreeformRelayProblem(
        codec=CODEC,
        field_degs=(0.0,),
        targets=(DisplayTarget(0.0),),
    )
    assert len(problem.rays) == problem.eyebox.ray_count
    assert {ray.source_id for ray in problem.rays} == {0}
    assert all(np.allclose(ray.direction, problem.eyebox.inward_normal)
               for ray in problem.rays)
    evaluation = problem.evaluate(random_genes(1)[0])
    assert len(evaluation.fields) == 1
    assert evaluation.parity == 1


def test_one_pupil_sample_is_the_centred_chief_ray():
    eyebox = EyeboxSpec(ray_count=1)
    rays = collimated_eyebox_rays(eyebox, (0.0,))
    assert len(rays) == 1
    assert np.allclose(rays[0].origin, (eyebox.x, eyebox.y_center))
    assert np.allclose(rays[0].direction, eyebox.inward_normal)


def test_vertical_blocker_absorbs_only_segments_that_cross_its_finite_extent():
    blocker = VerticalBlocker(x=20.0, y_center=5.0, half_height=10.0)
    assert blocker.intersection((50.0, 5.0), (0.0, 5.0)) == (20.0, 5.0)
    assert blocker.intersection((50.0, 30.0), (0.0, 30.0)) is None
    assert blocker.intersection((10.0, 5.0), (0.0, 5.0)) is None


def test_blocked_direct_problem_only_reports_tir_routes():
    problem = FreeformRelayProblem(
        codec=CODEC,
        field_degs=(0.0,),
        targets=(DisplayTarget(0.0),),
        minimum_tir_bounces=1,
    )
    for genes in random_genes(12, seed=8):
        evaluation = problem.evaluate(genes)
        assert all("tir" in signature[0] for signature in evaluation.route_keys)
        if evaluation.feasible:
            assert evaluation.route_keys


def test_reverse_bundles_enter_at_the_eyebox_travelling_back_toward_the_element():
    rays = collimated_eyebox_rays(EyeboxSpec(ray_count=2), (-6.0, 0.0, 6.0))
    eyebox = EyeboxSpec(ray_count=2)
    for degrees_, source_id in ((-6.0, 0), (0.0, 1), (6.0, 2)):
        theta = radians(eyebox.tilt_deg + degrees_)
        expected = (-cos(theta), -sin(theta))
        got = next(r.direction for r in rays if r.source_id == source_id)
        assert np.allclose(got, expected)
        assert got[0] < 0.0                      # travelling -x, away from the eye
    chief = next(r.direction for r in rays if r.source_id == 1)
    assert np.allclose(chief, eyebox.inward_normal)


def test_the_eyebox_is_not_searchable_and_must_be_sampled():
    for bad in (dict(half_height=0.0), dict(half_height=float("nan")),
                dict(tilt_deg=float("nan")), dict(ray_count=0)):
        try:
            EyeboxSpec(**bad)
        except ValueError:
            continue
        raise AssertionError(f"EyeboxSpec accepted {bad}")


# --- the feasibility ladder ---------------------------------------------------------

def test_ladder_bands_run_from_untouched_to_imaging():
    target = DisplayTarget(0.0, half_width=2.0)
    common = dict(source_id=0, target=target, minimum_throughput=1e-3,
                  spot_scale=1.0, distance_scale=10.0)

    untouched = relay_field_report(records=[], emitted=1.0, entered=0.0, escaped=0.0,
                                   **common)
    trapped = relay_field_report(records=[], emitted=1.0, entered=0.8, escaped=0.0,
                                 **common)
    escaped = relay_field_report(records=[], emitted=1.0, entered=0.8, escaped=0.6,
                                 **common)
    missed = relay_field_report(records=fake_records([25.0], [0.5]), emitted=1.0,
                                entered=0.8, escaped=0.6, **common)
    imaging = relay_field_report(records=fake_records([0.1, -0.1], [0.4, 0.4]),
                                 emitted=1.0, entered=0.9, escaped=0.9, **common)

    assert [r.band for r in (untouched, trapped, escaped, missed, imaging)] == [0, 1, 2, 3, 4]
    scores = [r.score for r in (untouched, trapped, escaped, missed, imaging)]
    assert scores == sorted(scores), scores
    assert imaging.fitness > 0.0
    assert all(r.fitness == 0.0 for r in (untouched, trapped, escaped, missed))


def test_progress_can_never_lift_a_field_into_the_band_above():
    target = DisplayTarget(0.0, half_width=2.0)
    common = dict(source_id=0, target=target, minimum_throughput=1e-3,
                  spot_scale=1.0, distance_scale=10.0)
    # entered == emitted would give progress 1.0 uncapped, which would tie band 1 with
    # band 2 at zero progress and make the ladder non-monotonic.
    saturated = relay_field_report(records=[], emitted=1.0, entered=1.0, escaped=0.0,
                                   **common)
    next_band = relay_field_report(records=[], emitted=1.0, entered=1e-9, escaped=1e-12,
                                   **common)
    assert saturated.band == 1 and next_band.band == 2
    assert saturated.score < next_band.score
    assert saturated.progress <= 1.0 and saturated.score < 2.0


def test_a_nearer_miss_outranks_a_further_miss_within_the_same_band():
    target = DisplayTarget(0.0, half_width=2.0)
    common = dict(source_id=0, target=target, minimum_throughput=1e-3,
                  spot_scale=1.0, distance_scale=10.0, emitted=1.0,
                  entered=0.9, escaped=0.9)
    near = relay_field_report(records=fake_records([3.0], [0.5]), **common)
    far = relay_field_report(records=fake_records([40.0], [0.5]), **common)
    assert near.band == far.band == 3
    assert near.score > far.score


def test_the_ladder_grades_designs_the_optical_fitness_cannot_separate():
    problem = FreeformRelayProblem(codec=CODEC)
    evaluations = [problem.evaluate(g) for g in random_genes(24)]
    dead = [e for e in evaluations if e.fitness == 0.0]
    assert len(dead) > 10, "expected most random bodies to deliver nothing"
    # Flat where it matters: every one of these scores zero on the real objective.
    assert len({e.fitness for e in dead}) == 1
    # Yet the emitter still has somewhere to go.
    assert len({round(e.search_score, 9) for e in dead}) > 0.8 * len(dead)


def test_scoring_reads_delivered_energy_only_and_never_the_route():
    target = DisplayTarget(0.0, half_width=2.0)
    common = dict(source_id=0, target=target, minimum_throughput=1e-3,
                  spot_scale=1.0, distance_scale=10.0, emitted=1.0,
                  entered=0.9, escaped=0.9)
    direct = relay_field_report(
        records=fake_records([0.4], [0.7], events=(("refract",),)), **common)
    folded = relay_field_report(
        records=fake_records([0.4], [0.7],
                             events=(("tir",), ("tir",), ("tir",), ("refract",))),
        **common)
    assert direct.score == folded.score
    assert direct.fitness == folded.fitness


# --- reflection parity --------------------------------------------------------------

def test_reflection_parity_is_discovered_rather_than_prescribed():
    problem = permissive_problem(
        eyebox=EyeboxSpec(x=120.0, y_center=0.0, tilt_deg=0.0))
    genes = random_genes(3)[2]
    evaluation = problem.evaluate(genes)
    assert evaluation.feasible
    # This body images with field angle mapping to *decreasing* display x.  Pinning one
    # pairing would exclude every design of the opposite reflection parity outright.
    assert evaluation.parity == -1
    reports = evaluation.field_reports
    assert [r.target_offset for r in reports] == list(reversed(list(TARGET_OFFSETS)))
    assert all(r.band == 4 for r in reports)


def test_both_parities_are_scored_and_the_better_one_is_kept():
    problem = FreeformRelayProblem(codec=CODEC)
    for genes in random_genes(8):
        evaluation = problem.evaluate(genes)
        if not evaluation.field_reports:
            continue
        assigned = [r.target_offset for r in evaluation.field_reports]
        forward = list(TARGET_OFFSETS)
        reverse = list(reversed(forward))
        assert assigned in (forward, reverse)
        assert evaluation.parity == (1 if assigned == forward else -1)


# --- the fixed display plane --------------------------------------------------------

def test_display_targets_must_be_ordered_and_their_windows_disjoint():
    overlapping = (DisplayTarget(-6.0, half_width=4.0),
                   DisplayTarget(0.0, half_width=4.0),
                   DisplayTarget(6.0, half_width=4.0))
    for bad in (dict(targets=overlapping),
                dict(targets=(DisplayTarget(6.0), DisplayTarget(-6.0),
                              DisplayTarget(0.0))),
                dict(field_degs=(0.0, -6.0, 6.0)),
                dict(field_degs=(-6.0, 6.0))):
        try:
            FreeformRelayProblem(codec=CODEC, **bad)
        except ValueError:
            continue
        raise AssertionError(f"FreeformRelayProblem accepted {bad}")


def test_disjoint_windows_make_a_scrambled_image_unsatisfiable():
    """Ordering needs no separate gate: the windows cannot both be hit by one spot."""
    problem = FreeformRelayProblem(codec=CODEC)
    offsets = [t.offset for t in problem.targets]
    widths = [t.half_width for t in problem.targets]
    for i in range(len(offsets) - 1):
        assert widths[i] + widths[i + 1] < offsets[i + 1] - offsets[i]


def test_a_body_reaching_a_fixed_plane_is_rejected():
    problem = permissive_problem()
    evaluation = problem.evaluate(random_genes(5)[4])
    assert not evaluation.feasible
    assert "object plane" in (evaluation.reason or "")
    assert evaluation.trace_result is None, "a rejected body must not be traced"


def test_rejected_geometry_ranks_below_every_traced_candidate():
    problem = FreeformRelayProblem(codec=CODEC)
    evaluations = [problem.evaluate(g) for g in random_genes(24)]
    rejected = [e for e in evaluations if e.trace_result is None]
    traced = [e for e in evaluations if e.trace_result is not None]
    assert rejected and traced
    assert max(e.search_score for e in rejected) < 0.0 <= min(
        e.search_score for e in traced)


def disc(radius: float, center=(60.0, 0.0)) -> FreeformBody2D:
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    return FreeformBody2D(np.stack((center[0] + radius * np.cos(theta),
                                    center[1] + radius * np.sin(theta)), axis=1), 1.60)


def test_clearance_is_measured_against_the_real_body_not_the_control_cage():
    problem = FreeformRelayProblem(codec=CODEC, clearance=2.0)
    # The element centre stands 42.4 mm off the tilted plane, so a disc larger than
    # that crosses it -- a test the old axis-aligned height check could not express.
    reasons, overshoot = problem.clearance_report(disc(48.0))
    assert reasons and overshoot > 0.0
    assert problem.clearance_report(disc(20.0)) == ((), 0.0)


# --- plane projection ---------------------------------------------------------------

def test_a_body_far_past_a_fixed_plane_ranks_below_one_just_past_it():
    """The reject score must point at the thing the emitter needs to change."""
    problem = FreeformRelayProblem(codec=CODEC)
    _, small = problem.clearance_report(disc(48.0))
    _, large = problem.clearance_report(disc(70.0))
    assert 0.0 < small < large
    # Grading these by self-intersection distance instead would score both identically,
    # since neither body self-intersects at all.
    assert problem.clearance_report(disc(20.0)) == ((), 0.0)


def test_projection_handles_horizontal_planes_and_reverse_travel():
    body = FreeformBody2D(
        np.stack((10.0 * np.cos(np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)),
                  10.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False))),
                 axis=1), 1.60)
    rays = (InputRay(origin=(40.0, 0.0), direction=(-1.0, 0.0)),)
    trace = trace_bundle(body, rays, max_interactions=6, energy_floor=1e-3)

    forward = project_exits_to_plane(trace, axis=0, value=-40.0, travel_sign=-1)
    assert forward and all(r["direction"][0] < 0 for r in forward)
    assert all(r["x"] == -40.0 and r["coord"] == r["y"] for r in forward)
    # nothing travels +x far enough to cross a plane behind the source
    assert project_exits_to_plane(trace, axis=0, value=-40.0, travel_sign=1) == []

    horizontal = project_exits_to_plane(trace, axis=1, value=50.0, travel_sign=1)
    assert all(r["y"] == 50.0 and r["coord"] == r["x"] for r in horizontal)
    assert all(r["direction"][1] > 0 for r in horizontal)

    for bad in (dict(axis=2, value=0.0), dict(axis=0, value=float("inf")),
                dict(axis=0, value=0.0, travel_sign=2)):
        try:
            project_exits_to_plane(trace, **bad)
        except ValueError:
            continue
        raise AssertionError(f"project_exits_to_plane accepted {bad}")


def test_the_vertical_sensor_wrapper_is_unchanged_by_the_generalization():
    body = FreeformBody2D(
        np.stack((10.0 * np.cos(np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)),
                  8.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False))),
                 axis=1), 1.60)
    rays = (InputRay(origin=(-40.0, 1.0), direction=(1.0, 0.0)),)
    trace = trace_bundle(body, rays, max_interactions=6, energy_floor=1e-3)
    wrapped = project_exits_to_x(trace, 60.0)
    direct = project_exits_to_plane(trace, axis=0, value=60.0, travel_sign=1)
    assert [r["y"] for r in wrapped] == [r["y"] for r in direct]
    assert all(r["x"] == 60.0 for r in wrapped)
    assert len(project_exits_to_x(trace, 60.0, forward_only=False)) >= len(wrapped)


# --- archive and emitter contracts --------------------------------------------------

def test_the_archive_admits_only_top_band_designs():
    # Pin the historical fixture pose; this test is about archive admission.
    problem = permissive_problem(eyebox=EyeboxSpec(tilt_deg=0.0))
    archive = DiverseDiscoveryArchive()
    admitted = 0
    for genes in random_genes(24):
        evaluation = problem.evaluate(genes)
        status, _ = archive.add(genes, evaluation)
        if status == "none":
            continue
        admitted += 1
        assert evaluation.feasible
        assert all(r.band == 4 for r in evaluation.field_reports)
    assert admitted >= 1
    assert archive.joint_niche_count == admitted
    assert all(record.evaluation.feasible for record in archive.records)


def test_the_ladder_orders_infeasible_candidates_without_outranking_feasible_ones():
    problem = FreeformRelayProblem(codec=CODEC)
    search = FreeformDiscoverySearch(problem, popsize=4, emitters=1, seed=3)
    progress = search.run(max_evaluations=8)
    assert progress.evaluations == 8, "the budget is a hard cap"
    assert all(record.evaluation.feasible for record in search.archive.records)

    # The emitter ranks an infeasible candidate at -1e9 + search_score.  That stays
    # below every feasible rank only while the ladder is bounded, so pin the bound:
    # an infeasible design never reaches the top band, hence never scores 4 or more.
    evaluations = [problem.evaluate(g) for g in random_genes(16, seed=5)]
    scores = [e.search_score for e in evaluations if not e.feasible]
    assert scores, "expected some infeasible candidates"
    assert max(scores) < 4.0
    # 0.0 is the floor of a feasible candidate's rank, so the penalised reject stays
    # below it with a margin no ladder value can close.
    assert -1e9 + max(scores) < 0.0


def test_relay_problem_satisfies_the_discovery_problem_contract():
    problem = FreeformRelayProblem(codec=CODEC)
    evaluation = problem.evaluate(random_genes(1)[0])
    for attribute in ("feasible", "fitness", "fields", "shape", "interaction",
                      "route_keys", "search_score"):
        assert hasattr(evaluation, attribute), attribute
    assert len(evaluation.fields) == len(problem.targets)
    assert problem.codec is CODEC
    assert len(problem.rays) == EyeboxSpec().ray_count * len(FIELD_DEGS)


def test_a_dead_field_cannot_hide_behind_two_good_ones():
    # Pin the historical fixture pose: project-level layout defaults are intentionally
    # edited during experiments, while this test is about field aggregation only.
    problem = permissive_problem(
        eyebox=EyeboxSpec(x=120.0, y_center=0.0, tilt_deg=0.0))
    genes = random_genes(3)[2]
    baseline = problem.evaluate(genes)
    assert baseline.feasible and baseline.fitness > 0.0
    # Move one target somewhere the light does not go; the geometric mean must collapse.
    moved = replace(problem, targets=(DisplayTarget(-6.0), DisplayTarget(0.0),
                                      DisplayTarget(200.0)))
    starved = moved.evaluate(genes)
    assert not starved.feasible
    assert starved.fitness == 0.0
    assert starved.search_score < baseline.search_score


def test_the_throughput_floor_excludes_designs_that_leak_almost_all_the_light():
    """The floor is archive admission, not a search penalty."""
    fixture_eyebox = EyeboxSpec(x=120.0, y_center=0.0, tilt_deg=0.0)
    default = FreeformRelayProblem(codec=CODEC, eyebox=fixture_eyebox)
    assert default.minimum_field_throughput == 1e-2

    genes = random_genes(3)[2]
    loose = permissive_problem(eyebox=fixture_eyebox).evaluate(genes)
    strict = default.evaluate(genes)
    # Same optics, same delivered energy -- only admission changes.
    assert loose.feasible and not strict.feasible
    assert min(r.throughput for r in loose.field_reports) < 1e-2
    assert [r.throughput for r in loose.field_reports] == [
        r.throughput for r in strict.field_reports]
    # Rejected, but not thrown away: it still sits above a design that reached nothing.
    assert strict.search_score > 3.0
    assert max(r.band for r in strict.field_reports) <= 4


def test_the_specification_uses_the_configured_object_and_eyebox_pose():
    problem = FreeformRelayProblem(codec=CODEC)
    plane = problem.object_plane
    assert problem.eyebox.x == EYEBOX_X
    assert problem.eyebox.y_center == EYEBOX_Y
    assert problem.eyebox.tilt_deg == EYEBOX_TILT_DEG == -10.0
    assert tuple(t.offset for t in problem.targets) == TARGET_OFFSETS
    assert problem.field_degs == FIELD_DEGS
    assert abs(problem.field_degs[-1] - problem.field_degs[0]) == 12.0

    assert plane.center == OBJECT_CENTER
    assert plane.tilt_deg == OBJECT_TILT_DEG == 135.0
    assert np.allclose(plane.along, (-np.sqrt(0.5), np.sqrt(0.5)))
    # The normal faces the element, so light arrives against it.
    assert plane.normal[0] > 0 and np.dot(plane.along, plane.normal) < 1e-12
    assert float(plane.signed_distance(
        [(EYEBOX_X, EYEBOX_Y)])[0]) > 0.0
    # Targets lie ON the tilted plane, not on a horizontal line.
    for target in problem.targets:
        assert abs(float(plane.signed_distance([plane.point_at(target.offset)])[0])) < 1e-9


def test_general_plane_projection_measures_position_along_the_tilted_plane():
    plane = ObjectPlane()
    body = disc(10.0, center=(60.0, 0.0))
    rays = (InputRay(origin=(120.0, 0.0), direction=(-1.0, 0.0)),)
    trace = trace_bundle(body, rays, max_interactions=6, energy_floor=1e-3)
    records = project_exits_to_line(trace, point=plane.center, normal=plane.normal)
    assert records
    for record in records:
        crossing = np.array((record["x"], record["y"]))
        # every crossing lies on the plane, and coord is its signed position along it
        assert abs(float(plane.signed_distance([crossing])[0])) < 1e-9
        assert np.allclose(crossing, plane.point_at(record["coord"]), atol=1e-9)
        assert float(np.dot(record["direction"], plane.normal)) < 0.0

    for bad in (dict(point=(0.0, 0.0), normal=(0.0, 0.0)),
                dict(point=(0.0, float("nan")), normal=(1.0, 0.0)),
                dict(point=(0.0,), normal=(1.0, 0.0))):
        try:
            project_exits_to_line(trace, **bad)
        except ValueError:
            continue
        raise AssertionError(f"project_exits_to_line accepted {bad}")


def main() -> None:
    checks = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for check in checks:
        check()
        print(f"  PASS  {check.__name__}")
    print("ALL FREEFORM-RELAY CHECKS PASS")


if __name__ == "__main__":
    main()
