"""Replay every selected audit in every arm, checking frozen sources and budgets."""
import argparse
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from nslens.discovery.audit_benchmark import check
from nslens.discovery.run import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    protocol = json.loads((args.run/"protocol.json").read_text())
    for path, digest in protocol["sources"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed: {path}")
    summaries = json.loads((args.run/"summary.json").read_text())
    jobs = []
    for summary in summaries:
        root = Path(summary["config"]["out"])
        rows = [json.loads(line) for line in (root/"candidates.jsonl").read_text().splitlines()]
        audits = json.loads((root/"archive.json").read_text())["audits"]
        if not (len(rows) == summary["proposals"] <= 4*summary["config"]["search_calls"]
                and sum(r["source_rays"] for r in rows) == summary["search_source_rays"]
                and sum(r["source_rays"] > 0 for r in rows) == summary["search_calls"] <= summary["config"]["search_calls"]
                and len(audits) <= summary["config"]["audit_limit"]
                and sum(a["result"]["source_rays"] for a in audits) == summary["audit_source_rays"]
                and summary["search_source_rays"]+summary["audit_source_rays"] <= summary["reserved_source_rays"]):
            raise RuntimeError(f"Budget mismatch: {root}")
        jobs.extend((summary, audit["result"], False) for audit in audits)
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(check, jobs)
    details = [dict(arm=job[0]["arm"], **result) for job, result in zip(jobs, results)]
    result = dict(runs_checked=len(summaries), audits_replayed=len(details),
                  exact_replays=sum(r["exact_replay"] for r in details),
                  all_budgets_match=True, all_frozen_source_hashes_match=True,
                  additional_source_rays=sum(r["source_rays"] for r in details), audits=details)
    write(args.run/"all_arm_replay.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "audits"}, indent=2))
    if result["exact_replays"] != result["audits_replayed"]:
        raise RuntimeError("Audit replay mismatch")


if __name__ == "__main__":
    main()
