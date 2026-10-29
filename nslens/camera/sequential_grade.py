"""Search-only, surface-to-surface progression for the three-body camera.

The ordered body contacts are a construction prior. They never change the final
imaging acceptance test, which scores every optical path reaching the sensor.
"""
import numpy as np


def _ray_distance(origin, direction, points):
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    delta = points - origin
    along = np.maximum(0., delta @ direction)
    residual = delta - along[:, None] * direction
    return float(np.min(np.linalg.norm(residual, axis=1)))


def grade(scene, trace, brief, parity):
    """Return a worst-field-first graded score and per-field stage diagnostics.

    Stages are contact with body 0, then body 1, then body 2, then image-plane
    targeting. Within a stage, forward ray distance to the next body's boundary
    provides a continuous score. Each branch contributes only its surviving energy.
    """
    samples = [body.sample(64)[1] for body in scene.bodies]
    fields = np.linspace(-brief.half_fov_deg, brief.half_fov_deg, brief.field_count)
    reports = []
    for field_index, field in enumerate(fields):
        total = 0.
        reached = np.zeros(len(scene.bodies) + 1)
        target_y = parity * brief.focal_length_mm * np.tan(np.deg2rad(field))
        for ray in trace.exits:
            if ray.source_id // brief.rays_per_field != field_index or ray.energy <= 0:
                continue
            stage = 0
            for event in ray.events:
                if stage < len(scene.bodies) and event.body_id == stage:
                    stage += 1
            reached[stage] += ray.energy
            if stage < len(scene.bodies):
                distance = _ray_distance(ray.origin, ray.direction, samples[stage])
            else:
                distance = _ray_distance(ray.origin, ray.direction,
                                         np.asarray([[brief.max_length_mm, target_y]]))
            progress = min(.99, 1. / (1. + distance / brief.sensor_half_mm))
            total += ray.energy * (stage + progress)
        for ray in trace.terminations:
            if ray.source_id // brief.rays_per_field != field_index or ray.energy <= 0:
                continue
            stage = 0
            for event in ray.events:
                if stage < len(scene.bodies) and event.body_id == stage:
                    stage += 1
            reached[stage] += ray.energy
            total += ray.energy * stage
        score = total / brief.rays_per_field
        reports.append(dict(field_deg=float(field), score=float(score),
                            energy_by_stage=(reached / brief.rays_per_field).tolist()))
    scores = [r['score'] for r in reports]
    return dict(fields=reports, worst_score=float(min(scores)),
                mean_score=float(np.mean(scores)))
