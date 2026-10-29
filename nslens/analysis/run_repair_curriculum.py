"""Run a reproducible, budgeted freeform data-generation / learning experiment.

python -m nslens.analysis.run_repair_curriculum --out runs/repair_loop_v1

All strategies share an initial corpus, representation, model, teacher, task family
pool, validation tasks, and untouched test tasks. Only subsequent sampling differs.
No test outcome is read until all training and checkpoint selection have finished.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import joblib
import numpy as np

from nslens.learning.freeform_repair import (
    BOUND, SCHEMA, SEVERITIES, ImagingSpec, Recorder, ResponsePolicy,
    curriculum_weights, features, solve, teacher_step,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_worker(spec_dict):
    spec = ImagingSpec(**spec_dict)
    rec = Recorder(spec, spec.family_id + "/witness", "certification")
    x = np.zeros(4)
    v = rec(x, source="witness_initial")
    for _ in range(4):
        x, v = teacher_step(x, v, rec)
    dense = rec(x, "dense", source="witness_admission")
    return dict(spec=spec_dict, witness=x.tolist(), witness_verdict=asdict(dense),
                certified=dense.success, records=rec.records)


def make_specs(seed, counts):
    rng = np.random.default_rng(seed)
    specs = []
    for split, count in zip(("train", "validation", "test"), counts):
        for i in range(count):
            n = float(rng.uniform(1.52, 1.68))
            f = float(rng.uniform(36., 54.))
            ry = float(rng.uniform(18., 23.))
            rx = ry ** 2 / (2 * (n - 1) * f) * float(rng.uniform(.93, 1.07))
            specs.append(asdict(ImagingSpec(
                family_id=f"seed{seed}/{split}/{i:03d}", split=split,
                radius_x=rx, radius_y=ry, index=n, focal_distance=f,
                half_field_deg=float(rng.uniform(1., 3.)),
                chief_deg=float(rng.uniform(-3., 3.)), pupil_half=float(rng.uniform(2., 4.)),
            )))
    return specs


def make_task(family, severity, rng, task_id):
    witness = np.asarray(family["witness"])
    start = np.clip(witness + rng.normal(0., SEVERITIES[severity], 4), -BOUND, BOUND)
    # The witness is deliberately not included in the task sent to the solver.
    return dict(spec=family["spec"], start=start.tolist(), severity=int(severity),
                task_id=task_id, seed=int(rng.integers(2**31)))


def train_worker(job):
    task, policy, stage = job
    spec = ImagingSpec(**task["spec"])
    x = np.asarray(task["start"])
    rec = Recorder(spec, task["task_id"], stage)
    initial = rec(x, source="initial")
    proposed = np.clip(x + policy.action(spec, x, initial), -BOUND, BOUND)
    probe = rec(proposed, source="learner_probe", parent=x.tolist())
    end, result = teacher_step(x, initial, rec)
    example = None
    if result.physical_valid and result.rms < initial.rms - 1e-8:
        example = dict(schema_version=SCHEMA, task_id=task["task_id"],
                       family_id=spec.family_id, split=spec.split,
                       features=features(spec, x, initial).tolist(),
                       action=(end - x).tolist(), before_rms=initial.rms,
                       after_rms=result.rms)
    probes = [row for row in rec.records if row["source"] == "jacobian"]
    if initial.physical_valid and all(row["verdict"]["physical_valid"] for row in probes):
        jac = np.array([(np.asarray(row["verdict"]["residual"]) - initial.residual) /
                        (np.asarray(row["x"]) - x)[j] for j, row in enumerate(probes)]).T
        if example is None:
            example = dict(schema_version=SCHEMA, task_id=task["task_id"],
                           family_id=spec.family_id, split=spec.split,
                           features=features(spec, x, initial).tolist(), action=[0.] * 4,
                           before_rms=initial.rms, after_rms=result.rms)
        example.update(response_features=np.r_[spec.context(), x].tolist(), jacobian=jac.ravel().tolist())
    return dict(task=task, example=example, probe_success=probe.success,
                initial_success=initial.success, teacher_success=result.success,
                records=rec.records)


def eval_worker(job):
    task, policy, algorithm, stage, budget, audit = job
    spec = ImagingSpec(**task["spec"])
    rec = Recorder(spec, task["task_id"], stage)
    end, before, after = solve(spec, task["start"], rec, policy=policy,
                               algorithm=algorithm, budget=budget, seed=task["seed"])
    dense = rec(end, "dense", source="final_dense")
    # Reconstruct the best state after exactly two candidate calls. This is an
    # evaluation-only snapshot, never feedback to the search. It measures useful
    # sample efficiency before eight iterations saturate this small domain.
    candidates = [row for row in rec.records if row["fidelity"] == "coarse"][:3]
    prefix = min(candidates, key=lambda row: (not row["verdict"]["physical_valid"], row["verdict"]["rms"]))
    early = rec(prefix["x"], "dense", source="two_call_dense")
    initial_dense = rec(task["start"], "dense", source="initial_dense")
    row = dict(task_id=task["task_id"], family_id=spec.family_id,
               severity=task["severity"], stage=stage, algorithm=algorithm,
               initial_success=initial_dense.success, initial_rms=initial_dense.rms,
               coarse=asdict(after), dense=asdict(dense), early=asdict(early), final_x=end.tolist())
    if audit:
        row["audit"] = asdict(rec(end, "audit", source="independent_grid_audit"))
        row["early_audit"] = asdict(rec(prefix["x"], "audit", source="two_call_grid_audit"))
    return row, rec.records


def metrics(rows, fidelity="dense"):
    return dict(tasks=len(rows), success_rate=float(np.mean([r[fidelity]["success"] for r in rows])),
                physical_valid_rate=float(np.mean([r[fidelity]["physical_valid"] for r in rows])),
                median_rms=float(np.median([r[fidelity]["rms"] for r in rows])),
                mean_rms=float(np.mean([r[fidelity]["rms"] for r in rows])),
                improvement_rate=float(np.mean([r[fidelity]["rms"] < r["initial_rms"] for r in rows])),
                initially_solved_rate=float(np.mean([r["initial_success"] for r in rows])))


class Ledger:
    def __init__(self, root):
        self.file = (root / "evaluations.jsonl").open("w")
        self.counts = {}

    def add(self, records):
        for record in records:
            self.file.write(json.dumps(record, allow_nan=False) + "\n")
            key = record["stage"] + "/" + record["fidelity"]
            item = self.counts.setdefault(key, dict(evaluations=0, launched_rays=0))
            item["evaluations"] += 1
            item["launched_rays"] += record["verdict"]["rays"]
        self.file.flush()

    def close(self):
        self.file.close()


def paired_summary(left, right, seed=42, fidelity="audit"):
    """Family-clustered bootstrap: task siblings are not independent replicates."""
    l = {r["task_id"]: r for r in left}
    r = {r["task_id"]: r for r in right}
    if l.keys() != r.keys():
        raise ValueError("paired evaluation must have identical task IDs")
    groups = {}
    for task_id, row in l.items():
        other = r[task_id]
        groups.setdefault(row["family_id"], []).append([
            float(row[fidelity]["success"]) - float(other[fidelity]["success"]),
            other[fidelity]["rms"] - row[fidelity]["rms"],
        ])
    values = np.array([np.mean(v, axis=0) for v in groups.values()])
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return dict(families=len(values), success_rate_delta=float(values[:, 0].mean()),
                success_delta_95ci=np.quantile(draws[:, 0], [.025, .975]).tolist(),
                rms_reduction=float(values[:, 1].mean()),
                rms_reduction_95ci=np.quantile(draws[:, 1], [.025, .975]).tolist())


def run_seed(args, seed, pool):
    root = args.out / f"seed{seed}"
    root.mkdir()
    ledger = Ledger(root)
    started = time.monotonic()
    specs = make_specs(seed, (args.train_families, args.validation_families, args.test_families))
    write_json(root / "specs.json", specs)
    families = []
    for result in pool.map(seed_worker, specs):
        ledger.add(result.pop("records"))
        families.append(result)
    write_json(root / "families.json", families)
    usable = {split: [f for f in families if f["certified"] and f["spec"]["split"] == split]
              for split in ("train", "validation", "test")}
    print(f"seed {seed}: certified families " + str({k: len(v) for k, v in usable.items()}), flush=True)
    if any(len(v) < 2 for v in usable.values()):
        ledger.close()
        write_json(root / "accounting.json", ledger.counts)
        raise RuntimeError("insufficient certified families; inspect witness failures")

    # Freeze tasks before training. Withheld families and their witnesses are never
    # supplied to the learner, sampler, model selection, or training worker.
    eval_tasks = {}
    for offset, split in enumerate(("validation", "test")):
        rng = np.random.default_rng(seed + 10000 + offset)
        eval_tasks[split] = [make_task(f, level, rng, f["spec"]["family_id"] + f"/s{level}")
                             for f in usable[split] for level in range(4)]
    write_json(root / "heldout_tasks.json", eval_tasks)
    replay = []
    train_log = (root / "training_examples.jsonl").open("w")
    tasks_log = (root / "training_tasks.jsonl").open("w")

    def collect(tasks, policy, stage):
        examples, counts, successes = [], np.zeros(4), np.zeros(4)
        for result in pool.map(train_worker, [(t, policy, stage) for t in tasks]):
            ledger.add(result.pop("records"))
            task = result["task"]
            counts[task["severity"]] += 1
            successes[task["severity"]] += result["probe_success"]
            if result["example"] is not None:
                examples.append(result["example"])
                train_log.write(json.dumps(dict(stage=stage, **result["example"])) + "\n")
            tasks_log.write(json.dumps(dict(stage=stage, **{k: v for k, v in result.items()
                                                           if k != "example"})) + "\n")
        train_log.flush()
        tasks_log.flush()
        return examples, counts, successes

    def sample_tasks(weights, stage, round_index, count=None):
        # Identical random stream across arms; differing task draws arise solely
        # from their sampling weights. New episode perturbations every round.
        rng = np.random.default_rng(seed * 1000 + round_index + 500)
        return [make_task(usable["train"][int(rng.integers(len(usable["train"])))],
                          int(rng.choice(4, p=weights)), rng, f"{stage}/task{i:04d}")
                for i in range(args.batch if count is None else count)]

    replay, attempts0, success0 = collect(sample_tasks(np.full(4, .25), "bootstrap", 0, args.bootstrap),
                                         ResponsePolicy(), "bootstrap")
    initial_policy = ResponsePolicy().fit(replay)
    joblib.dump(initial_policy, root / "bootstrap.joblib")
    history, selection, policies = [], {}, {}

    def evaluate_tasks(tasks, policy, algorithm, stage, audit=False):
        rows = []
        for row, records in pool.map(eval_worker, [(t, policy, algorithm, stage,
                                                    args.budget, audit) for t in tasks]):
            ledger.add(records)
            rows.append(row)
        return rows

    baseline_val = evaluate_tasks(eval_tasks["validation"], initial_policy, "learned", "bootstrap_validation")
    write_json(root / "bootstrap_validation.json", baseline_val)
    for strategy in ("uniform", "fixed", "adaptive"):
        examples = list(replay)
        policy = initial_policy
        attempts, successes = attempts0.copy(), success0.copy()
        checkpoints = [(0, root / "bootstrap.joblib", metrics(baseline_val, "early"))]
        incumbent_round = 0
        incumbent_score = checkpoints[0][2]
        for round_index in range(args.rounds):
            weights = curriculum_weights(strategy, successes, attempts, round_index, args.rounds)
            stage = f"{strategy}/round{round_index + 1}"
            extra, count, passed = collect(sample_tasks(weights, stage, round_index + 1), policy, stage)
            # Recency-weighted observations follow the changing learner.
            attempts, successes = .5 * attempts + count, .5 * successes + passed
            examples.extend(extra)
            candidate_policy = ResponsePolicy().fit(examples)
            path = root / f"{strategy}_round{round_index + 1}.joblib"
            joblib.dump(candidate_policy, path)
            rows = evaluate_tasks(eval_tasks["validation"], candidate_policy, "learned", stage + "/validation")
            write_json(root / f"{strategy}_round{round_index + 1}_validation.json", rows)
            score = metrics(rows, "early")
            checkpoints.append((round_index + 1, path, score))
            promoted = ((score["success_rate"], -score["mean_rms"]) >
                        (incumbent_score["success_rate"], -incumbent_score["mean_rms"]))
            if promoted:
                policy, incumbent_score, incumbent_round = candidate_policy, score, round_index + 1
            history.append(dict(strategy=strategy, round=round_index + 1,
                                sampling_weights=weights.tolist(), examples=len(examples),
                                promoted=promoted, incumbent_round=incumbent_round,
                                probe_attempts=count.tolist(), probe_successes=passed.tolist(), **score))
            write_json(root / "history.json", history)
            print(f"seed {seed} {stage}: {len(examples)} examples, validation pass "
                  f"{score['success_rate']:.3f}, median RMS {score['median_rms']:.5f}", flush=True)
        # Fixed validation rule. Test stays unopened throughout this selection.
        chosen = max(checkpoints, key=lambda c: (c[2]["success_rate"], -c[2]["mean_rms"]))
        selection[strategy] = dict(round=chosen[0], checkpoint=chosen[1].name,
                                   sha256=digest(chosen[1]), validation=chosen[2])
        policies[strategy] = joblib.load(chosen[1])
    write_json(root / "selection.json", selection)
    train_log.close()
    tasks_log.close()

    # Locked test: evaluate frozen selections and baselines only after every arm is trained.
    test_rows = {}
    arms = [("bootstrap", initial_policy, "learned")]
    arms += [(strategy, policies[strategy], "learned") for strategy in policies]
    arms += [(name, None, name) for name in ("random", "reset", "teacher")]
    for name, policy, algorithm in arms:
        rows = evaluate_tasks(eval_tasks["test"], policy, algorithm, f"test/{name}", audit=True)
        write_json(root / f"test_{name}.json", rows)
        test_rows[name] = rows
        print(f"seed {seed} test {name}: {metrics(rows, 'audit')}", flush=True)
    summary = dict(seed=seed, certified={k: len(v) for k, v in usable.items()},
                   rejected_witnesses=sum(not f["certified"] for f in families),
                   selection=selection,
                   test={name: metrics(rows, "early_audit") for name, rows in test_rows.items()},
                   test_eight_calls={name: metrics(rows, "audit") for name, rows in test_rows.items()},
                   adaptive_vs={name: paired_summary(test_rows["adaptive"], rows, fidelity="early_audit")
                                for name, rows in test_rows.items() if name != "adaptive"},
                   seconds=time.monotonic() - started)
    ledger.close()
    write_json(root / "accounting.json", ledger.counts)
    write_json(root / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--train-families", type=int, default=24)
    parser.add_argument("--validation-families", type=int, default=6)
    parser.add_argument("--test-families", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=16)
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args(argv)
    if min(args.rounds, args.batch, args.workers, args.bootstrap) < 1 or args.budget < 8 or args.budget % 8:
        parser.error("positive rounds/batch/workers and a candidate budget divisible by eight required")
    if args.out.exists():
        parser.error("output directory already exists; use a new run to preserve provenance")
    args.out.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), repo / "nslens/learning/freeform_repair.py"]
    sources += sorted((repo / "nslens/freeform_body").glob("*.py"))
    # Preserve executable source as well as hashes: this workspace already contains
    # uncommitted work, and git HEAD alone cannot reproduce the run.
    for source in sources:
        destination = args.out / "source_snapshot" / source.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = dict(schema_version=SCHEMA, config={**vars(args), "out": str(args.out)},
                    sources={str(p.relative_to(repo)): digest(p) for p in sources},
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                    versions={"numpy": np.__version__, "sklearn": __import__("sklearn").__version__},
                    scope="2-D monochromatic transmitting spline; four radial Fourier genes; fixed material per task",
                    limitations=["No folded-route or 3-D transfer claim", "Dense audit shares the same tracer",
                                 "Adaptive weights estimate learnability, not causal learning gain",
                                 "Training strategies have equal candidate allowances, not equal wall time",
                                 "Witness certification conditions this study on solvable families"],
                    selection="best validation two-candidate dense success rate, then mean RMS; bootstrap eligible",
                    budget="10 coarse evaluations per generated training task; primary metric is two-candidate prefix, secondary is eight candidates; dense/audit accounted separately")
    write_json(args.out / "manifest.json", manifest)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        summaries = [run_seed(args, seed, pool) for seed in args.seeds]
    aggregate = {}
    for name in ("bootstrap", "uniform", "fixed", "adaptive", "random", "reset", "teacher"):
        rows = []
        for seed in args.seeds:
            rows.extend(json.loads((args.out / f"seed{seed}/test_{name}.json").read_text()))
        aggregate[name] = rows
    report = dict(seeds=summaries, test={k: metrics(v, "early_audit") for k, v in aggregate.items()},
                  test_eight_calls={k: metrics(v, "audit") for k, v in aggregate.items()},
                  adaptive_vs={k: paired_summary(aggregate["adaptive"], v, fidelity="early_audit")
                               for k, v in aggregate.items() if k != "adaptive"})
    write_json(args.out / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "seeds"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
