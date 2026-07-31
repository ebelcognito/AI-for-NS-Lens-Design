import sys
from pathlib import Path

# Add project root for optical_rl imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optical_rl.utils.scene_builder import scene_single_ray_with_dual_curved_slab


def test_scene_geometry_matches_spec_and_arc_layout():
    scene = scene_single_ray_with_dual_curved_slab(
        ray_origin_x=10,
        ray_origin_y=106.5,
        ray_target_x=290,
        ray_target_y=150,
        slab_right_x=260,
        slab_thickness=3,
        slab_center_y=150,
        slab_height=15,
        left_sag_mm=0.25,
        right_sag_mm=-0.40,
        detector_x=290,
        detector_center_y=150,
        detector_half_height=1.5,
        ref_index=1.5,
    )

    glass = next(obj for obj in scene["objs"] if obj["type"] == "Glass")
    detector = next(obj for obj in scene["objs"] if obj["type"] == "Detector")

    path = glass["path"]
    assert len(path) == 6

    left_x = 257.0
    right_x = 260.0
    top_y = 142.5
    bottom_y = 157.5

    assert path[0]["x"] == left_x and path[0]["y"] == top_y and path[0]["arc"] is False
    assert path[1]["x"] == right_x and path[1]["y"] == top_y and path[1]["arc"] is False
    assert path[2]["x"] == right_x - 0.40 and path[2]["y"] == 150.0 and path[2]["arc"] is True
    assert path[3]["x"] == right_x and path[3]["y"] == bottom_y and path[3]["arc"] is False
    assert path[4]["x"] == left_x and path[4]["y"] == bottom_y and path[4]["arc"] is False
    assert path[5]["x"] == left_x + 0.25 and path[5]["y"] == 150.0 and path[5]["arc"] is True

    assert sum(1 for p in path if p["arc"]) == 2

    assert detector["p1"]["x"] == 290.0
    assert detector["p2"]["x"] == 290.0
    assert detector["p1"]["y"] == 148.5
    assert detector["p2"]["y"] == 151.5
