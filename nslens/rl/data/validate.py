"""Validate JSON or JSONL against a versioned training-data contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import (validate_codev_transition, validate_hardened_action_sft,
                     validate_hardened_design,
                     validate_proposer, validate_search_group,
                     validate_search_preference)


def _rows(path: Path):
    if path.suffix == ".jsonl":
        for lineno, line in enumerate(path.open(), 1):
            if line.strip():
                yield lineno, json.loads(line)
    else:
        yield 1, json.loads(path.read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=("proposer", "codev-transition", "search-group",
                                     "search-preference", "hardened-design",
                                     "hardened-action-sft"))
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    check = {
        "proposer": validate_proposer,
        "codev-transition": validate_codev_transition,
        "search-group": validate_search_group,
        "search-preference": validate_search_preference,
        "hardened-design": validate_hardened_design,
        "hardened-action-sft": validate_hardened_action_sft,
    }[args.kind]
    n, bad = 0, 0
    for raw in args.paths:
        path = Path(raw)
        for lineno, row in _rows(path):
            n += 1
            errors = check(row)
            if errors:
                bad += 1
                print(f"{path}:{lineno}: " + "; ".join(errors))
    print(f"validated {n} records: {n - bad} pass, {bad} fail")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
