"""Conservative novelty checks for interaction-sequence discovery.

The historical signature labels (A/B/T/D) are geometry-specific names.  Calling a
path new merely because those names changed is the exact false-positive that the v1
interaction fingerprint was designed to prevent.  This module therefore canonicalises
surfaces by first use and records repeat indices before comparing paths.

Novelty is deliberately defined against a frozen reference set.  A generation run must
write that set (and its source files) into its manifest so that "new" remains auditable
after more archives are added to the workspace.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


CanonicalEvent = tuple[str, str, int]
CanonicalTopology = tuple[CanonicalEvent, ...]


def canonical_topology(signature: str) -> CanonicalTopology:
    """Parse a legacy signature and remove arbitrary physical-surface naming.

    ``T@A -> TIR@D -> TIR@A`` becomes
    ``(('T', 's0', 0), ('TIR', 's1', 0), ('TIR', 's0', 1))``.
    Mechanism annotations such as ``PR[fresnel]`` remain part of the interaction token.
    """
    if not isinstance(signature, str) or not signature.strip():
        raise ValueError("interaction signature must be a non-empty string")
    surface_names: dict[str, str] = {}
    repeats: Counter[str] = Counter()
    events: list[CanonicalEvent] = []
    for raw in signature.split("->"):
        token = raw.strip()
        if "@" not in token:
            raise ValueError(f"signature token {token!r} must contain '@'")
        interaction, surface = (part.strip() for part in token.split("@", 1))
        # Audit renderings append #repeat.  Recompute it rather than trusting input.
        surface = surface.split("#", 1)[0]
        if not interaction or not surface:
            raise ValueError(f"invalid signature token {token!r}")
        canonical = surface_names.setdefault(surface, f"s{len(surface_names)}")
        events.append((interaction, canonical, repeats[surface]))
        repeats[surface] += 1
    return tuple(events)


def topology_text(topology: Sequence[CanonicalEvent]) -> str:
    return " -> ".join(f"{interaction}@{surface}#{repeat}"
                       for interaction, surface, repeat in topology)


def edit_distance(left: Sequence[CanonicalEvent],
                  right: Sequence[CanonicalEvent]) -> int:
    """Unit-cost Levenshtein distance between canonical interaction events."""
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for i, event_r in enumerate(right, 1):
        current = [i]
        for j, event_l in enumerate(left, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (event_l != event_r),
            ))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class NoveltyReport:
    signature: str
    canonical: CanonicalTopology
    nearest_signature: str | None
    nearest_canonical: CanonicalTopology | None
    edit_distance: int
    normalized_distance: float
    exact_signature_seen: bool
    exact_canonical_seen: bool
    passes: bool

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "canonical_topology": topology_text(self.canonical),
            "nearest_signature": self.nearest_signature,
            "nearest_canonical_topology": (
                topology_text(self.nearest_canonical)
                if self.nearest_canonical is not None else None),
            "edit_distance": self.edit_distance,
            "normalized_distance": self.normalized_distance,
            "exact_signature_seen": self.exact_signature_seen,
            "exact_canonical_seen": self.exact_canonical_seen,
            "passes": self.passes,
        }


class TopologyExclusionSet:
    """Frozen reference paths plus an explicit dissimilarity threshold."""

    def __init__(self, signatures: Iterable[str], *, min_edits: int = 2,
                 min_normalized_distance: float = 0.25):
        if min_edits < 1:
            raise ValueError("min_edits must be positive")
        if not 0.0 < min_normalized_distance <= 1.0:
            raise ValueError("min_normalized_distance must lie in (0, 1]")
        self.min_edits = int(min_edits)
        self.min_normalized_distance = float(min_normalized_distance)
        self.signatures = tuple(sorted({s.strip() for s in signatures
                                        if isinstance(s, str) and "@" in s}))
        canonical_to_signature: dict[CanonicalTopology, str] = {}
        for signature in self.signatures:
            try:
                topology = canonical_topology(signature)
            except ValueError:
                continue
            canonical_to_signature.setdefault(topology, signature)
        self.canonical_to_signature = canonical_to_signature
        self.canonical = tuple(canonical_to_signature)

    def report(self, signature: str) -> NoveltyReport:
        candidate = canonical_topology(signature)
        if not self.canonical:
            distance, normalized = len(candidate), 1.0
            nearest_topology = nearest_signature = None
        else:
            nearest_topology = min(
                self.canonical,
                key=lambda known: (edit_distance(candidate, known),
                                   abs(len(candidate) - len(known)), known),
            )
            distance = edit_distance(candidate, nearest_topology)
            normalized = distance / max(len(candidate), len(nearest_topology), 1)
            nearest_signature = self.canonical_to_signature[nearest_topology]
        exact_signature = signature.strip() in self.signatures
        exact_canonical = candidate in self.canonical_to_signature
        passes = bool(
            not exact_signature
            and not exact_canonical
            and distance >= self.min_edits
            and normalized >= self.min_normalized_distance
        )
        return NoveltyReport(
            signature=signature,
            canonical=candidate,
            nearest_signature=nearest_signature,
            nearest_canonical=nearest_topology,
            edit_distance=distance,
            normalized_distance=float(normalized),
            exact_signature_seen=exact_signature,
            exact_canonical_seen=exact_canonical,
            passes=passes,
        )


__all__ = [
    "CanonicalEvent", "CanonicalTopology", "NoveltyReport",
    "TopologyExclusionSet", "canonical_topology", "edit_distance",
    "topology_text",
]
