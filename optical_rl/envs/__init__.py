"""Environment exports.

These imports are optional to allow parts of the repo (e.g. QD utilities, docs,
basic unit tests) to be used without installing `gymnasium`.
"""

__all__ = []

try:
    from .simple_steering_env import SimpleSteeringEnv, SimpleSteeringConfig

    __all__ += ["SimpleSteeringEnv", "SimpleSteeringConfig"]
except ImportError:
    pass

try:
    from .angled_ray_env import AngledRayEnv, AngledRayConfig

    __all__ += ["AngledRayEnv", "AngledRayConfig"]
except ImportError:
    pass

try:
    from .randomized_slab_env import RandomizedSlabEnv, RandomizedSlabConfig

    __all__ += ["RandomizedSlabEnv", "RandomizedSlabConfig"]
except ImportError:
    pass

try:
    from .randomized_slab_env_v2 import RandomizedSlab2DEnv, RandomizedSlab2DConfig

    __all__ += ["RandomizedSlab2DEnv", "RandomizedSlab2DConfig"]
except ImportError:
    pass

try:
    from .randomized_sag_slab_env_v3 import (
        RandomizedSagSlabEnvV3,
        RandomizedSagSlabConfig,
    )

    __all__ += ["RandomizedSagSlabEnvV3", "RandomizedSagSlabConfig"]
except ImportError:
    pass

try:
    from .quad_curve_glass_env import QuadCurveGlassEnv, QuadCurveGlassConfig

    __all__ += ["QuadCurveGlassEnv", "QuadCurveGlassConfig"]
except ImportError:
    pass

try:
    from .quad_curve_glass_env_v2 import QuadCurveGlassEnvV2, QuadCurveGlassConfigV2

    __all__ += ["QuadCurveGlassEnvV2", "QuadCurveGlassConfigV2"]
except ImportError:
    pass

try:
    from .quad_curve_glass_env_v3 import QuadCurveGlassEnvV3, QuadCurveGlassConfigV3

    __all__ += ["QuadCurveGlassEnvV3", "QuadCurveGlassConfigV3"]
except ImportError:
    pass

try:
    from .single_surface_curvature_env_v1 import (
        SingleSurfaceCurvatureConfigV1,
        SingleSurfaceCurvatureEnvV1,
    )

    __all__ += ["SingleSurfaceCurvatureEnvV1", "SingleSurfaceCurvatureConfigV1"]
except ImportError:
    pass

try:
    from .dual_surface_curvature_env_v2 import (
        DualSurfaceCurvatureConfigV2,
        DualSurfaceCurvatureEnvV2,
    )

    __all__ += ["DualSurfaceCurvatureEnvV2", "DualSurfaceCurvatureConfigV2"]
except ImportError:
    pass
