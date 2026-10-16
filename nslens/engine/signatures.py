"""
Typed interaction-sequence tools for the instrumented NS engine.

The engine (with scene flag `recordInteractions: true`) now returns:

  result["raySequences"]: one record per detector crossing:
      {x, y, dirx, diry, b, seq: [vertex, ...]}
    where each vertex is
      {x, y, i,            # position + scene-object index (-1 = source point)
       t,                  # interaction type: source | refract | tir | partial_reflect |
                           #                   mirror | beam_split | sensor | absorbed | other
       bIn, bOut,          # energy arriving at / continuing past this event on THIS branch
       mech}               # for partial_reflect: 'fresnel' (physical) or 'forced' (coating knob)

  result["energyLedger"]: {emitted, delivered, escaped, absorbed, truncated, unfinished}
    delivered = terminal energy of rays that crossed a detector; escaped = left the scene
    without one; truncated = sub-threshold ghosts + depth-budget kills; unfinished =
    rays alive when the simulator stopped early. Closure: emitted == sum of the rest.

  result["rayTerminations"]: one record for every terminal branch, including rays that
    never reach a detector. Exact records retain the full ``seq``; sub-threshold energy
    that was never instantiated as a ray is explicitly marked ``aggregate: true`` and
    ``historyComplete: false``. ``rayTerminationOverflow`` reports any record-cap loss.

This module turns those into checkable statements: classify each sensor-arriving ray's
sequence into a signature like (('refract','A'),('tir','B'),('tir','A'),('refract','B')),
aggregate energy per signature, and verify whether a REQUESTED target sequence carries
the dominant energy.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Vertex = dict
Signature = Tuple[Tuple[str, str], ...]

# Spec-token aliases for writing target sequences like "T@A TIR@B TIR@A T@B"
TYPE_ALIASES = {
    "t": "refract", "transmit": "refract", "refract": "refract",
    "tir": "tir",
    "pr": "partial_reflect", "partial": "partial_reflect", "partial_reflect": "partial_reflect",
    "m": "mirror", "mirror": "mirror",
    "bs": "beam_split", "beam_split": "beam_split",
}


def parse_target(spec: str) -> Signature:
    """Parse a target-sequence spec, e.g. "T@A,TIR@B,TIR@A,T@B" or "T@A TIR@B ...".

    Each token is TYPE@FACE; FACE may be '*' to accept any face.
    """
    tokens = [tok for tok in spec.replace(",", " ").split() if tok]
    out = []
    for tok in tokens:
        if "@" not in tok:
            raise ValueError(f"target token {tok!r} must look like TYPE@FACE (e.g. TIR@B)")
        t, face = tok.split("@", 1)
        t_lower = t.strip().lower()
        if t_lower not in TYPE_ALIASES:
            raise ValueError(f"unknown interaction type {t!r} in target spec")
        out.append((TYPE_ALIASES[t_lower], face.strip()))
    return tuple(out)


def make_single_element_face_classifier(cx: float, elem_h: float, eps: float = 1e-3
                                        ) -> Callable[[Vertex], str]:
    """Face labels for the single hexagonal element used by the demos:
    'A' = left optical face, 'B' = right optical face, 'T'/'D' = flat top/bottom edges."""
    def face_of(v: Vertex) -> str:
        if v["y"] <= -elem_h + eps:
            return "T"
        if v["y"] >= elem_h - eps:
            return "D"
        return "A" if v["x"] < cx else "B"
    return face_of


def signature_of(record: dict, face_of: Callable[[Vertex], str]) -> Optional[Signature]:
    """The ordered (type, face) tuple of optic events between source and the FIRST sensor
    crossing. Returns None for re-crossing records (a sensor event before the last vertex),
    so each ray's energy is aggregated exactly once."""
    seq = record["seq"]
    events = []
    for k, v in enumerate(seq):
        t = v.get("t")
        if t == "source":
            continue
        if t == "sensor":
            return tuple(events) if k == len(seq) - 1 else None
        events.append((t or "other", face_of(v)))
    # No sensor vertex at all should not happen for raySequences records; treat as invalid.
    return None


def matches(sig: Signature, target: Signature) -> bool:
    if len(sig) != len(target):
        return False
    return all(ts == ss and (tf == "*" or tf == sf)
               for (ss, sf), (ts, tf) in zip(sig, target))


def aggregate_by_signature(records: Iterable[dict], face_of: Callable[[Vertex], str],
                           forward_only: bool = True) -> Dict[Signature, float]:
    """Energy (at sensor crossing) per interaction-sequence signature."""
    out: Dict[Signature, float] = {}
    for rec in records:
        if forward_only and rec.get("dirx", 0) <= 0:
            continue
        sig = signature_of(rec, face_of)
        if sig is None:
            continue
        out[sig] = out.get(sig, 0.0) + rec["b"]
    return out


def format_signature(sig: Sequence[Tuple[str, str]]) -> str:
    pretty = {"refract": "T", "tir": "TIR", "partial_reflect": "PR",
              "mirror": "M", "beam_split": "BS"}
    return " -> ".join(f"{pretty.get(t, t)}@{f}" for t, f in sig)


def ledger_closure(ledger: dict) -> float:
    """Residual of: emitted - (delivered + escaped + absorbed + truncated + unfinished)."""
    return ledger["emitted"] - (ledger["delivered"] + ledger["escaped"]
                                + ledger["absorbed"] + ledger["truncated"]
                                + ledger.get("unfinished", 0.0))


def audit_ray_energy(record: dict, tol: float = 1e-9) -> List[str]:
    """Per-ray bookkeeping audit: along one branch, the energy continuing past event k
    must equal the energy arriving at event k+1 (free-space propagation is loss-free),
    and TIR events must be exactly loss-free."""
    problems = []
    seq = [v for v in record["seq"] if v.get("t") != "source"]
    for k, v in enumerate(seq):
        if v.get("t") == "tir" and abs(v["bIn"] - v["bOut"]) > tol:
            problems.append(f"event {k}: TIR not loss-free (bIn={v['bIn']}, bOut={v['bOut']})")
        if k + 1 < len(seq) and abs(v["bOut"] - seq[k + 1]["bIn"]) > tol:
            problems.append(f"event {k}->{k+1}: energy jump in free space "
                            f"({v['bOut']} -> {seq[k+1]['bIn']})")
    return problems


def dominance_report(result: dict, face_of: Callable[[Vertex], str], target: Signature,
                     dominance_threshold: float = 0.5) -> dict:
    """The verification artifact: per-signature energy table, target share of the INPUT
    energy, ledger closure, and the dominance verdict.

    Dominant := energy delivered via exactly the target sequence is (a) more than
    `dominance_threshold` of all emitted energy and (b) larger than every other
    sensor-arriving signature class.
    """
    ledger = result.get("energyLedger") or {}
    by_sig = aggregate_by_signature(result.get("raySequences", []), face_of)
    emitted = ledger.get("emitted", 0.0)
    target_energy = sum(e for s, e in by_sig.items() if matches(s, target))
    others = {s: e for s, e in by_sig.items() if not matches(s, target)}
    biggest_other = max(others.values(), default=0.0)
    share = target_energy / emitted if emitted > 0 else 0.0
    return {
        "by_signature": dict(sorted(by_sig.items(), key=lambda kv: -kv[1])),
        "target": tuple(target),
        "target_energy": target_energy,
        "target_share_of_input": share,
        "biggest_other_class": biggest_other,
        "ledger": ledger,
        "ledger_residual": ledger_closure(ledger) if ledger else None,
        "dominant": share > dominance_threshold and target_energy > biggest_other,
    }


def print_report(report: dict, max_rows: int = 12) -> None:
    led = report["ledger"]
    print(f"  input (emitted)      : {led['emitted']:.6f}")
    print(f"  delivered to sensor  : {led['delivered']:.6f}")
    print(f"  reflected-away/missed: {led['escaped']:.6f}")
    print(f"  absorbed             : {led['absorbed']:.6f}")
    print(f"  truncated            : {led['truncated']:.6f}")
    if led.get("unfinished"):
        print(f"  unfinished           : {led['unfinished']:.6f}")
    print(f"  ledger closure resid : {report['ledger_residual']:.2e}")
    print(f"  -- sensor-arriving energy by interaction sequence --")
    for k, (sig, e) in enumerate(report["by_signature"].items()):
        if k >= max_rows:
            rest = sum(list(report["by_signature"].values())[max_rows:])
            print(f"    ... {len(report['by_signature']) - max_rows} more classes, total {rest:.6f}")
            break
        mark = " <== TARGET" if matches(sig, report["target"]) else ""
        print(f"    {e:8.6f}  {format_signature(sig)}{mark}")
    print(f"  target share of input: {report['target_share_of_input']:.1%} "
          f"(largest competing class: {report['biggest_other_class']:.6f})")
    print(f"  DOMINANT-ENERGY SEQUENCE VERDICT: {'PASS' if report['dominant'] else 'FAIL'}")
