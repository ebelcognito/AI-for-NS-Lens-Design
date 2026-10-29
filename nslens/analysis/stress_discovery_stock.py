"""Stress every stored passing audit in a discovery run at 101 rays x 9 fields."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from nslens.discovery.folded import RoutingBrief, evaluate
from nslens.discovery.polygon import PolygonCodec
from nslens.discovery.run import write


def _check(item):
    summary, stored = item
    brief = RoutingBrief(**summary["config"]["brief"])
    fields = sorted(brief.fields_deg)
    stress = replace(brief, rays_per_field=101,
                     fields_deg=tuple(np.linspace(fields[0], fields[-1], 9)))
    result = evaluate(PolygonCodec(**summary["config"]["codec"]), stored["genes"], stress,
                      geometry_samples=4096)
    return {"unit": summary["unit"], "feasible": result["feasible"], "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = []
    for summary in json.loads((args.run / "summary.json").read_text()):
        archive = json.loads((Path(summary["config"]["out"]) / "archive.json").read_text())
        jobs.extend((summary, audit["result"]) for audit in archive["audits"]
                    if audit["result"]["feasible"])
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_check, jobs)
    output = {"source_run": str(args.run.resolve()), "tested": len(results),
              "passed": sum(row["feasible"] for row in results),
              "source_rays": sum(row["result"]["source_rays"] for row in results),
              "results": results}
    write(args.run / "stock_stress_all.json", output)
    print(json.dumps({key: output[key] for key in ("tested", "passed", "source_rays")}, indent=2))


if __name__ == "__main__":
    main()
