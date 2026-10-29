"""Geometry, finite-aperture acceptance, replay, and hard-budget contracts."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nslens.discovery.polygon import PolygonCodec, exact_extent
from nslens.discovery.folded import RoutingBrief, evaluate, ranking
from nslens.discovery.run import run_one
from nslens.freeform_body.trace import BoundaryEvent, ExitRay, TraceResult
from nslens.search.cma_me import CMAES


class PolygonTests(unittest.TestCase):
    def test_corner_counts_and_scale_relative_curvature(self):
        rng = np.random.default_rng(27)
        for k in (3, 4, 5):
            codec = PolygonCodec(k)
            for _ in range(20):
                body = codec.decode(rng.uniform(-1, 1, codec.gene_size))
                self.assertEqual(len(body.corners), k)
                half = np.linalg.norm(np.roll(body.corners, -1, axis=0)-body.corners, axis=1)/2
                self.assertTrue(np.all(abs(body.sags)/half <= codec.max_sag_ratio + 1e-12))
                for edge in body._edges:
                    if edge["kind"] == "arc":
                        self.assertGreaterEqual(edge["radius"] + 1e-10, codec.min_radius_mm)
                lo, hi = exact_extent(body)
                points = body.sample(4096)[1]
                self.assertTrue(np.all(points >= lo-1e-10))
                self.assertTrue(np.all(points <= hi+1e-10))

    def test_minimum_radius_binds_small_edges(self):
        codec = PolygonCodec(5, min_radius_mm=100.)
        genes = np.zeros(codec.gene_size)
        genes[-5:] = 1
        body = codec.decode(genes)
        self.assertTrue(all(abs(edge["radius"]-100.) < 1e-8 for edge in body._edges))

    def test_finite_aperture_is_repeated_for_each_field(self):
        brief = RoutingBrief()
        rays = brief.rays()
        self.assertEqual(len({r.source_id for r in rays}), 27)
        for i in range(3):
            x = [r.origin[0] for r in rays[i*9:(i+1)*9]]
            self.assertAlmostEqual(max(x)-min(x), 4.)
        with self.assertRaises(ValueError):
            RoutingBrief(beam_width_mm=.001)
        with self.assertRaises(ValueError):
            RoutingBrief(rays_per_field=1)

    def test_fresnel_descendants_cannot_inflate_root_support(self):
        codec, brief = PolygonCodec(4), RoutingBrief(min_folded_throughput=.05)
        events = tuple(BoundaryEvent("tir", (60., 0.), u, u, 0., 60., 10., 1., 1.)
                       for u in (.1, .6))
        exits = []
        for i in range(27):
            if i % 9 == 0:
                for _ in range(5):
                    exits.append(ExitRay(i, (80., 0.), (1., 0.), .1, .1, events))
            else:
                exits.append(ExitRay(i, (80., 0.), (-1., 0.), .5, .5, ()))
        trace = TraceResult(tuple(exits), (), 27., 0., 0., 0.)
        with patch("nslens.discovery.folded.trace_bundle", return_value=trace):
            row = evaluate(codec, np.zeros(codec.gene_size), brief)
        self.assertEqual(row["status"], "measured")
        self.assertGreater(row["quality"], brief.min_folded_throughput)
        self.assertAlmostEqual(row["support"], 1/9)
        self.assertFalse(row["feasible"])

    def test_horizontal_detector_accepts_only_downward_crossings(self):
        codec = PolygonCodec(3, center_x=(51., 59.), center_y=(-4., 4.),
                             half_width=(10., 16.), half_height=(6., 10.),
                             radial_fraction=(.75, 1.), angular_jitter_fraction=.15)
        brief = RoutingBrief(beam_width_mm=4.8, source_center=(82., 0.), aim_deg=180.,
                             detector_axis=1, detector_travel_sign=-1,
                             detector_position=-35., detector_center=55.,
                             detector_half_length=6.)
        events = tuple(BoundaryEvent("tir", (55., -5.), u, u, 0., 60., 10., 1., 1.)
                       for u in (.1, .6))
        exits = tuple(ExitRay(i, (55., -15.), (0., -1.), .5, .5, events)
                      for i in range(len(brief.rays())))
        trace = TraceResult(exits, (), float(len(exits)), 0., 0., 0.)
        with patch("nslens.discovery.folded.trace_bundle", return_value=trace):
            row = evaluate(codec, np.zeros(codec.gene_size), brief)
        self.assertTrue(row["feasible"])
        self.assertGreaterEqual(row["detector_clearance"], brief.clearance_mm)
        self.assertEqual(brief.detector(), (1, -35., 55., 6., -1))

    def test_spot_objective_prefers_tighter_bundle_at_equal_throughput(self):
        codec = PolygonCodec(4)
        brief = RoutingBrief(min_folded_throughput=.05, spot_weight=1.)
        events = tuple(BoundaryEvent("tir", (60., 0.), u, u, 0., 60., 10., 1., 1.)
                       for u in (.1, .6))
        def result(spread):
            exits = tuple(ExitRay(i, (80., spread*((i % 9)-4)/4), (1., 0.), .5, .5, events)
                          for i in range(len(brief.rays())))
            return TraceResult(exits, (), float(len(exits)), 0., 0., 0.)
        with patch("nslens.discovery.folded.trace_bundle", return_value=result(0.)):
            sharp = evaluate(codec, np.zeros(codec.gene_size), brief)
        with patch("nslens.discovery.folded.trace_bundle", return_value=result(4.)):
            broad = evaluate(codec, np.zeros(codec.gene_size), brief)
        self.assertAlmostEqual(sharp["worst_folded_throughput"], broad["worst_folded_throughput"])
        self.assertEqual(sharp["worst_spot_rms"], 0.)
        self.assertGreater(broad["worst_spot_rms"], 0.)
        self.assertGreater(sharp["quality"], broad["quality"])

    def test_declared_clearance_ranges_are_enforced(self):
        with self.assertRaises(ValueError):
            RoutingBrief(source_clearance_bounds=(1., 10.))
        brief = RoutingBrief(source_clearance_bounds=(3., 10.),
                             detector_clearance_bounds=(16., 22.))
        row = evaluate(PolygonCodec(3), np.zeros(PolygonCodec(3).gene_size), brief)
        self.assertEqual(row["status"], "geometry")
        self.assertIn("clearance bounds", row["reason"])

    def test_actual_tracer_closure_and_unfolded_rejection(self):
        codec = PolygonCodec(4)
        row = evaluate(codec, np.zeros(codec.gene_size), RoutingBrief())
        self.assertEqual(row["status"], "measured")
        self.assertLess(row["closure"], 1e-9)
        self.assertFalse(row["feasible"])

    def test_reflection_replay_and_legacy_default(self):
        bounds = (-np.ones(3), np.ones(3))
        em = CMAES(np.zeros(3), 5., popsize=12, seed=3, bounds=bounds, boundary_handling="reflect")
        snapshot = em.state_snapshot()
        restored = CMAES.from_state_snapshot(snapshot, bounds=bounds)
        x = em.ask()
        np.testing.assert_array_equal(x, restored.ask())
        self.assertTrue(np.all(abs(x) < 1))
        em.tell(np.argsort(np.sum(x*x, axis=1)))
        self.assertTrue(np.isfinite(em.C).all())
        self.assertTrue(np.all(np.linalg.eigvalsh(em.C) > 0))
        legacy = CMAES(np.zeros(3), 5., popsize=12, seed=3, bounds=bounds)
        self.assertTrue(np.any(abs(legacy.ask()) == 1))

    def test_partial_generation_and_ray_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = dict(out=str(Path(tmp)/"run"), codec=asdict(PolygonCodec(3)),
                       brief=asdict(RoutingBrief()), seed=0, arm="cma", evals=3,
                       ray_budget=54, popsize=10, sigma=.15, boundary="reflect",
                       feedback=True, patience=5, audit_rays=25)
            result = run_one(job)
            self.assertLessEqual(result["evaluations"], 3)
            self.assertLessEqual(result["source_rays"], 54)
            trace = [json.loads(s) for s in (Path(job["out"])/"candidates.jsonl").read_text().splitlines()]
            self.assertEqual(result["source_rays"], sum(r["source_rays"] for r in trace))
            with self.assertRaises(FileExistsError):
                run_one(job)


if __name__ == "__main__":
    unittest.main()
