from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .cloud import initialize_root, job_status, submit_job
from .protocol import ALLOWED_CAPTURES
from .worker import Worker, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codev-bridge",
        description="Declarative SFTP job bridge for a local CODE V worker",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create an unprivileged queue tree")
    init.add_argument("--root", type=Path, required=True)

    submit = commands.add_parser("submit", help="atomically publish a staged-case job")
    submit.add_argument("--root", type=Path, required=True)
    submit.add_argument("--case-id", required=True)
    submit.add_argument("--case-file", type=Path, required=True)
    submit.add_argument("--codev-release", default="2026")
    submit.add_argument(
        "--capture",
        action="append",
        choices=sorted(ALLOWED_CAPTURES),
        dest="captures",
    )
    submit.add_argument("--timeout-seconds", type=int, default=300)
    submit.add_argument("--max-result-bytes", type=int, default=256 * 1024)
    submit.add_argument("--max-artifact-bytes", type=int, default=16 * 1024 * 1024)

    status = commands.add_parser("status", help="inspect one job without consuming it")
    status.add_argument("--root", type=Path, required=True)
    status.add_argument("job_id")

    worker = commands.add_parser("worker", help="run the outbound Windows poller")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--verbose", action="store_true")
    worker.add_argument("--log-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        initialize_root(args.root)
        print(args.root.resolve())
        return 0
    if args.command == "submit":
        job, path = submit_job(
            args.root,
            case_id=args.case_id,
            case_file=args.case_file,
            codev_release=args.codev_release,
            captures=args.captures,
            timeout_seconds=args.timeout_seconds,
            max_result_bytes=args.max_result_bytes,
            max_artifact_bytes=args.max_artifact_bytes,
        )
        print(json.dumps({"job_id": job["job_id"], "path": str(path)}, indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(job_status(args.root, args.job_id), indent=2, sort_keys=True))
        return 0
    if args.command == "worker":
        level = logging.DEBUG if args.verbose else logging.INFO
        log_format = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if args.log_file:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                args.log_file, maxBytes=2 * 1024 * 1024, backupCount=3
            )
            file_handler.setFormatter(log_format)
            handlers.append(file_handler)
        logging.basicConfig(level=level, format=log_format._fmt, handlers=handlers)
        worker = Worker(load_config(args.config))
        if args.once:
            print(worker.run_once())
        else:
            worker.run_forever()
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
