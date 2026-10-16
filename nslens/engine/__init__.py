"""Engine layer: the ray-optics process, and the typed-interaction vocabulary.

`runner` is the persistent Node.js ray-optics process. `signatures` turns its typed
interaction records into comparable sequences like `T@A -> TIR@D -> T@B`, which is the
vocabulary every fitness and descriptor in the project is written against.
"""
from .fingerprints import (FINGERPRINT_INVARIANCES, FINGERPRINT_SCHEMA,
                           fingerprint_record, make_single_element_contact_coordinate,
                           terminal_energy_by_outcome, terminal_ledger_residuals)
from .runner import Runner
from .signatures import (aggregate_by_signature, audit_ray_energy, dominance_report,
                         format_signature, ledger_closure,
                         make_single_element_face_classifier, matches, parse_target,
                         print_report, signature_of)

__all__ = ["Runner", "signature_of", "aggregate_by_signature", "format_signature",
           "make_single_element_face_classifier", "matches", "parse_target",
           "ledger_closure", "audit_ray_energy", "dominance_report", "print_report",
           "FINGERPRINT_SCHEMA", "FINGERPRINT_INVARIANCES", "fingerprint_record",
           "make_single_element_contact_coordinate", "terminal_energy_by_outcome",
           "terminal_ledger_residuals"]
