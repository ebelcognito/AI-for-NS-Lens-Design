"""Continuous-boundary freeform optics for topology-discovering NS design.

This package intentionally does not use the legacy four-face ``Element`` model.  A
``FreeformBody2D`` has one periodic C2 boundary, and interactions are identified by
normalized boundary arclength rather than by A/B/T/D face labels.
"""

from .boundary import BoundaryHit, FreeformBody2D, GeometryReport
from .arc_codec import ArcPolygonCodec
from .arc_polygon import ArcPolygonBody
from .codec import FreeformBodyCodec
from .descriptors import (
    Descriptor,
    interaction_descriptor,
    ranked_route_signatures,
    route_signature,
    shape_descriptor,
)
from .trace import (
    BoundaryEvent,
    ExitRay,
    InputRay,
    TerminatedRay,
    TraceResult,
    project_exits_to_line,
    project_exits_to_plane,
    project_exits_to_x,
    trace_bundle,
)
from .problem import (
    DiscoveryEvaluation,
    DiscoveryProblem,
    FieldScore,
    FreeformDiscoveryProblem,
    SensorTarget,
)
from .relay import (
    DisplayTarget,
    EyeboxSpec,
    ObjectPlane,
    FreeformRelayProblem,
    RelayEvaluation,
    RelayFieldReport,
    VerticalBlocker,
    collimated_eyebox_rays,
    relay_field_report,
)
from .archive import (
    ArchiveRecord,
    DiverseDiscoveryArchive,
    ShapeNicheResolution,
    interaction_niche_key,
    shape_niche_key,
)
from .search import FreeformDiscoverySearch, SearchProgress

__all__ = [
    "BoundaryEvent",
    "BoundaryHit",
    "ArchiveRecord",
    "ArcPolygonBody",
    "ArcPolygonCodec",
    "Descriptor",
    "DiscoveryEvaluation",
    "DiscoveryProblem",
    "DisplayTarget",
    "DiverseDiscoveryArchive",
    "ExitRay",
    "EyeboxSpec",
    "FreeformBody2D",
    "FreeformBodyCodec",
    "FreeformDiscoveryProblem",
    "FreeformDiscoverySearch",
    "FreeformRelayProblem",
    "GeometryReport",
    "ObjectPlane",
    "InputRay",
    "FieldScore",
    "RelayEvaluation",
    "RelayFieldReport",
    "SensorTarget",
    "SearchProgress",
    "ShapeNicheResolution",
    "TerminatedRay",
    "TraceResult",
    "VerticalBlocker",
    "interaction_descriptor",
    "interaction_niche_key",
    "collimated_eyebox_rays",
    "project_exits_to_line",
    "project_exits_to_plane",
    "project_exits_to_x",
    "relay_field_report",
    "ranked_route_signatures",
    "route_signature",
    "shape_descriptor",
    "shape_niche_key",
    "trace_bundle",
]
