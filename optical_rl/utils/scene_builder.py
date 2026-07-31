def _base_scene():
    return {
        "version": 5,
        "objs": [],
    }


def _source():
    return {
        "type": "SingleRay",
        "p1": {"x": 0, "y": 150},
        "p2": {"x": 50, "y": 150},
    }


def _detector():
    return {
        "type": "Detector",
        "p1": {"x": 300, "y": 140},
        "p2": {"x": 300, "y": 160},
    }


def _glass_quad(quad_x, quad_size=50, ref_index=1.5):
    half = quad_size / 2
    return {
        "type": "Glass",
        "path": [
            {"x": quad_x - half, "y": 125, "arc": False},
            {"x": quad_x + half, "y": 125, "arc": False},
            {"x": quad_x + half, "y": 175, "arc": False},
            {"x": quad_x - half, "y": 175, "arc": False},
        ],
        "refIndex": ref_index,
        "partialReflect": True,
    }


def scene_direct():
    scene = _base_scene()
    scene["objs"].extend([_source(), _detector()])
    return scene


def scene_with_glass():
    scene = _base_scene()
    scene["objs"].extend([_source(), _glass_quad(quad_x=150), _detector()])
    return scene


def scene_with_quad_at_x(quad_x):
    scene = _base_scene()
    scene["objs"].extend([_source(), _glass_quad(quad_x=quad_x), _detector()])
    return scene


def scene_angled_ray_with_quad_at_y(
    quad_center_y,
    quad_center_x=50,
    quad_size=40,
    detector_x=100,
    detector_center_y=70,
    detector_half_height=30,
    ref_index=1.5,
):
    half = quad_size / 2
    scene = _base_scene()
    scene["objs"].extend(
        [
            {
                "type": "SingleRay",
                "p1": {"x": 0, "y": 0},
                "p2": {"x": 30, "y": 30},
            },
            {
                "type": "Glass",
                "path": [
                    {
                        "x": quad_center_x - half,
                        "y": quad_center_y - half,
                        "arc": False,
                    },
                    {
                        "x": quad_center_x + half,
                        "y": quad_center_y - half,
                        "arc": False,
                    },
                    {
                        "x": quad_center_x + half,
                        "y": quad_center_y + half,
                        "arc": False,
                    },
                    {
                        "x": quad_center_x - half,
                        "y": quad_center_y + half,
                        "arc": False,
                    },
                ],
                "refIndex": ref_index,
                "partialReflect": True,
            },
            {
                "type": "Detector",
                "p1": {
                    "x": detector_x,
                    "y": detector_center_y - detector_half_height,
                },
                "p2": {
                    "x": detector_x,
                    "y": detector_center_y + detector_half_height,
                },
            },
            {
                "type": "CropBox",
                "p1": {"x": -20, "y": -20},
                "p4": {"x": 200, "y": 200},
                "width": 800,
            },
        ]
    )
    return scene


def scene_randomized_ray_with_slab_at_y(
    source_x,
    source_y,
    slope,
    detector_x,
    detector_center_y,
    detector_half_height,
    quad_center_x,
    quad_center_y,
    quad_size=40,
    ref_index=1.5,
    ray_dx=30,
):
    half = quad_size / 2
    ray_end_x = source_x + ray_dx
    ray_end_y = source_y + slope * ray_dx
    scene = _base_scene()
    scene["objs"].extend(
        [
            {
                "type": "SingleRay",
                "p1": {"x": source_x, "y": source_y},
                "p2": {"x": ray_end_x, "y": ray_end_y},
            },
            {
                "type": "Glass",
                "path": [
                    {
                        "x": quad_center_x - half,
                        "y": quad_center_y - half,
                        "arc": False,
                    },
                    {
                        "x": quad_center_x + half,
                        "y": quad_center_y - half,
                        "arc": False,
                    },
                    {
                        "x": quad_center_x + half,
                        "y": quad_center_y + half,
                        "arc": False,
                    },
                    {
                        "x": quad_center_x - half,
                        "y": quad_center_y + half,
                        "arc": False,
                    },
                ],
                "refIndex": ref_index,
                "partialReflect": True,
            },
            {
                "type": "Detector",
                "p1": {
                    "x": detector_x,
                    "y": detector_center_y - detector_half_height,
                },
                "p2": {
                    "x": detector_x,
                    "y": detector_center_y + detector_half_height,
                },
            },
        ]
    )
    return scene


def scene_randomized_ray_with_slab_at_xy(
    source_x,
    source_y,
    slope,
    detector_x,
    detector_center_y,
    detector_half_height,
    quad_center_x,
    quad_center_y,
    quad_size=40,
    ref_index=1.5,
    ray_dx=30,
):
    half = quad_size / 2
    ray_end_x = source_x + ray_dx
    ray_end_y = source_y + slope * ray_dx
    scene = _base_scene()
    scene["objs"].extend(
        [
            {
                "type": "SingleRay",
                "p1": {"x": source_x, "y": source_y},
                "p2": {"x": ray_end_x, "y": ray_end_y},
            },
            {
                "type": "Glass",
                "path": [
                    {
                        "x": quad_center_x - half,
                        "y": quad_center_y - half,
                        "arc": False,
                    },
                    {
                        "x": quad_center_x + half,
                        "y": quad_center_y - half,
                        "arc": False,
                    },
                    {
                        "x": quad_center_x + half,
                        "y": quad_center_y + half,
                        "arc": False,
                    },
                    {
                        "x": quad_center_x - half,
                        "y": quad_center_y + half,
                        "arc": False,
                    },
                ],
                "refIndex": ref_index,
                "partialReflect": True,
            },
            {
                "type": "Detector",
                "p1": {
                    "x": detector_x,
                    "y": detector_center_y - detector_half_height,
                },
                "p2": {
                    "x": detector_x,
                    "y": detector_center_y + detector_half_height,
                },
            },
        ]
    )
    return scene


def scene_randomized_ray_with_sag_slab_at_y(
    source_x,
    source_y,
    slope,
    detector_x,
    detector_center_y,
    detector_half_height,
    quad_center_x,
    quad_center_y,
    quad_size=40,
    sag_depth_frac=0.0,
    ref_index=1.5,
    ray_dx=30,
):
    half = quad_size / 2
    sag_offset = sag_depth_frac * quad_size
    ray_end_x = source_x + ray_dx
    ray_end_y = source_y + slope * ray_dx

    top_left = {"x": quad_center_x - half, "y": quad_center_y - half, "arc": False}
    top_right = {"x": quad_center_x + half, "y": quad_center_y - half, "arc": False}
    bottom_right = {"x": quad_center_x + half, "y": quad_center_y + half, "arc": False}
    bottom_left = {"x": quad_center_x - half, "y": quad_center_y + half, "arc": False}
    left_arc_mid = {
        "x": quad_center_x - half + sag_offset,
        "y": quad_center_y,
        "arc": True,
    }

    scene = _base_scene()
    scene["objs"].extend(
        [
            {
                "type": "SingleRay",
                "p1": {"x": source_x, "y": source_y},
                "p2": {"x": ray_end_x, "y": ray_end_y},
            },
            {
                "type": "Glass",
                "path": [top_left, top_right, bottom_right, bottom_left, left_arc_mid],
                "refIndex": ref_index,
                "partialReflect": True,
            },
            {
                "type": "Detector",
                "p1": {
                    "x": detector_x,
                    "y": detector_center_y - detector_half_height,
                },
                "p2": {
                    "x": detector_x,
                    "y": detector_center_y + detector_half_height,
                },
            },
        ]
    )
    return scene


def scene_two_rays_with_custom_glass(
    source_x,
    source_y,
    ray1_target_x,
    ray1_target_y,
    ray2_target_x,
    ray2_target_y,
    detector_x,
    detector_center_y,
    detector_half_height,
    points,
    ref_index=1.5,
    ray3_target_x=None,
    ray3_target_y=None,
    ray4_target_x=None,
    ray4_target_y=None,
):
    """
    Build a scene with two SingleRay sources, a custom 8-point Glass path
    (4 corners + 4 arc control points), and a single detector.

    Path indexing is expected to be:
      0 TL (arc False)
      1 TopArc (arc True)
      2 TR (arc False)
      3 RightArc (arc True)
      4 BR (arc False)
      5 BottomArc (arc True)
      6 BL (arc False)
      7 LeftArc (arc True)
    """
    if len(points) != 8:
        raise ValueError(f"Expected 8 points, got {len(points)}")

    ray_targets = [
        (ray1_target_x, ray1_target_y),
        (ray2_target_x, ray2_target_y),
    ]
    if ray3_target_x is not None and ray3_target_y is not None:
        ray_targets.append((ray3_target_x, ray3_target_y))
    if ray4_target_x is not None and ray4_target_y is not None:
        ray_targets.append((ray4_target_x, ray4_target_y))

    return scene_ray_bundle_with_custom_glass(
        source_x=source_x,
        source_y=source_y,
        ray_targets=ray_targets,
        detector_x=detector_x,
        detector_center_y=detector_center_y,
        detector_half_height=detector_half_height,
        points=points,
        ref_index=ref_index,
    )


def scene_ray_bundle_with_custom_glass(
    source_x,
    source_y,
    ray_targets,
    detector_x,
    detector_center_y,
    detector_half_height,
    points,
    ref_index=1.5,
):
    """
    Build a scene with N SingleRay sources, a custom 8-point Glass path
    (4 corners + 4 arc control points), and a single detector.

    `ray_targets` is an iterable of (target_x, target_y) pairs.

    Path indexing is expected to be:
      0 TL (arc False)
      1 TopArc (arc True)
      2 TR (arc False)
      3 RightArc (arc True)
      4 BR (arc False)
      5 BottomArc (arc True)
      6 BL (arc False)
      7 LeftArc (arc True)
    """
    if len(points) != 8:
        raise ValueError(f"Expected 8 points, got {len(points)}")

    arc_flags = [False, True, False, True, False, True, False, True]
    path = [
        {"x": float(x), "y": float(y), "arc": bool(arc_flags[i])}
        for i, (x, y) in enumerate(points)
    ]

    scene = _base_scene()

    for target_x, target_y in ray_targets:
        scene["objs"].append(
            {
                "type": "SingleRay",
                "p1": {"x": float(source_x), "y": float(source_y)},
                "p2": {"x": float(target_x), "y": float(target_y)},
            }
        )

    scene["objs"].extend(
        [
            {
                "type": "Glass",
                "path": path,
                "refIndex": float(ref_index),
                "partialReflect": True,
            },
            {
                "type": "Detector",
                "p1": {
                    "x": float(detector_x),
                    "y": float(detector_center_y - detector_half_height),
                },
                "p2": {
                    "x": float(detector_x),
                    "y": float(detector_center_y + detector_half_height),
                },
            },
        ]
    )
    return scene


def scene_single_ray_with_left_curved_slab(
    ray_origin_x,
    ray_origin_y,
    ray_target_x,
    ray_target_y,
    slab_right_x,
    slab_thickness,
    slab_center_y,
    slab_height,
    left_sag_mm,
    detector_x,
    detector_center_y,
    detector_half_height,
    ref_index=1.5,
):
    """
    Build a scene with:
      - one SingleRay defined by (origin -> target),
      - one rectangular slab where only the left face is curved,
      - one detector.

    Glass path ordering:
      0 top-left corner (arc False)
      1 top-right corner (arc False)
      2 bottom-right corner (arc False)
      3 bottom-left corner (arc False)
      4 left-face arc control point (arc True)
    """
    left_x = float(slab_right_x) - float(slab_thickness)
    half_h = float(slab_height) / 2.0
    top_y = float(slab_center_y) - half_h
    bottom_y = float(slab_center_y) + half_h
    arc_x = left_x + float(left_sag_mm)
    arc_y = float(slab_center_y)

    scene = _base_scene()
    scene["objs"].extend(
        [
            {
                "type": "SingleRay",
                "p1": {"x": float(ray_origin_x), "y": float(ray_origin_y)},
                "p2": {"x": float(ray_target_x), "y": float(ray_target_y)},
            },
            {
                "type": "Glass",
                "path": [
                    {"x": left_x, "y": top_y, "arc": False},
                    {"x": float(slab_right_x), "y": top_y, "arc": False},
                    {"x": float(slab_right_x), "y": bottom_y, "arc": False},
                    {"x": left_x, "y": bottom_y, "arc": False},
                    {"x": arc_x, "y": arc_y, "arc": True},
                ],
                "refIndex": float(ref_index),
                "partialReflect": True,
            },
            {
                "type": "Detector",
                "p1": {
                    "x": float(detector_x),
                    "y": float(detector_center_y - detector_half_height),
                },
                "p2": {
                    "x": float(detector_x),
                    "y": float(detector_center_y + detector_half_height),
                },
            },
        ]
    )
    return scene


def scene_single_ray_with_dual_curved_slab(
    ray_origin_x,
    ray_origin_y,
    ray_target_x,
    ray_target_y,
    slab_right_x,
    slab_thickness,
    slab_center_y,
    slab_height,
    left_sag_mm,
    right_sag_mm,
    detector_x,
    detector_center_y,
    detector_half_height,
    ref_index=1.5,
):
    """
    Build a scene with:
      - one SingleRay defined by (origin -> target),
      - one rectangular slab where both left and right vertical faces are curved,
      - one detector.

    Glass path ordering:
      0 top-left corner (arc False)
      1 top-right corner (arc False)
      2 right-face arc control point (arc True)
      3 bottom-right corner (arc False)
      4 bottom-left corner (arc False)
      5 left-face arc control point (arc True)
    """
    left_x = float(slab_right_x) - float(slab_thickness)
    right_x = float(slab_right_x)
    half_h = float(slab_height) / 2.0
    top_y = float(slab_center_y) - half_h
    bottom_y = float(slab_center_y) + half_h

    left_arc_x = left_x + float(left_sag_mm)
    right_arc_x = right_x + float(right_sag_mm)
    arc_y = float(slab_center_y)

    scene = _base_scene()
    scene["objs"].extend(
        [
            {
                "type": "SingleRay",
                "p1": {"x": float(ray_origin_x), "y": float(ray_origin_y)},
                "p2": {"x": float(ray_target_x), "y": float(ray_target_y)},
            },
            {
                "type": "Glass",
                "path": [
                    {"x": left_x, "y": top_y, "arc": False},
                    {"x": right_x, "y": top_y, "arc": False},
                    {"x": right_arc_x, "y": arc_y, "arc": True},
                    {"x": right_x, "y": bottom_y, "arc": False},
                    {"x": left_x, "y": bottom_y, "arc": False},
                    {"x": left_arc_x, "y": arc_y, "arc": True},
                ],
                "refIndex": float(ref_index),
                "partialReflect": True,
            },
            {
                "type": "Detector",
                "p1": {
                    "x": float(detector_x),
                    "y": float(detector_center_y - detector_half_height),
                },
                "p2": {
                    "x": float(detector_x),
                    "y": float(detector_center_y + detector_half_height),
                },
            },
        ]
    )
    return scene
