from .config import load_config

# Optional imports that require additional dependencies
try:
    from .metrics import MetricsLogger
    _has_torch = True
except ImportError:
    MetricsLogger = None
    _has_torch = False

try:
    from .visualization import plot_training_progress
    _has_viz = True
except ImportError:
    plot_training_progress = None
    _has_viz = False

__all__ = ["load_config"]
if _has_torch:
    __all__.append("MetricsLogger")
if _has_viz:
    __all__.append("plot_training_progress")
