import math
import sys
from pathlib import Path

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.utils.scene_builder import scene_randomized_ray_with_sag_slab_at_y


def _glass_path(scene):
    glass = next(obj for obj in scene["objs"] if obj["type"] == "Glass")
    return glass["path"]


def test_sag_path_structure_and_arc_point():
    scene = scene_randomized_ray_with_sag_slab_at_y(
        source_x=0,
        source_y=50,
        slope=0.05,
        detector_x=300,
        detector_center_y=65,
        detector_half_height=2.5,
        quad_center_x=150,
        quad_center_y=40,
        quad_size=40,
        sag_depth_frac=0.25,
        ref_index=1.5,
        ray_dx=30,
    )
    path = _glass_path(scene)
    assert len(path) == 5
    assert [p["arc"] for p in path] == [False, False, False, False, True]


def test_signed_sag_maps_to_control_point_x():
    quad_center_x = 150
    quad_center_y = 40
    quad_size = 40
    half = quad_size / 2

    scene_pos = scene_randomized_ray_with_sag_slab_at_y(
        source_x=0,
        source_y=50,
        slope=0.05,
        detector_x=300,
        detector_center_y=65,
        detector_half_height=2.5,
        quad_center_x=quad_center_x,
        quad_center_y=quad_center_y,
        quad_size=quad_size,
        sag_depth_frac=0.5,
        ref_index=1.5,
        ray_dx=30,
    )
    scene_neg = scene_randomized_ray_with_sag_slab_at_y(
        source_x=0,
        source_y=50,
        slope=0.05,
        detector_x=300,
        detector_center_y=65,
        detector_half_height=2.5,
        quad_center_x=quad_center_x,
        quad_center_y=quad_center_y,
        quad_size=quad_size,
        sag_depth_frac=-0.5,
        ref_index=1.5,
        ray_dx=30,
    )

    pos_mid = _glass_path(scene_pos)[4]
    neg_mid = _glass_path(scene_neg)[4]

    expected_base_x = quad_center_x - half
    assert math.isclose(pos_mid["x"], expected_base_x + 0.5 * quad_size)
    assert math.isclose(neg_mid["x"], expected_base_x - 0.5 * quad_size)
    assert math.isclose(pos_mid["y"], quad_center_y)
    assert math.isclose(neg_mid["y"], quad_center_y)


def test_scene_values_are_finite():
    scene = scene_randomized_ray_with_sag_slab_at_y(
        source_x=0,
        source_y=50,
        slope=0.05,
        detector_x=300,
        detector_center_y=65,
        detector_half_height=2.5,
        quad_center_x=150,
        quad_center_y=40,
        quad_size=40,
        sag_depth_frac=0.1,
        ref_index=1.5,
        ray_dx=30,
    )
    for obj in scene["objs"]:
        if obj["type"] == "Glass":
            for pt in obj["path"]:
                assert math.isfinite(float(pt["x"]))
                assert math.isfinite(float(pt["y"]))
