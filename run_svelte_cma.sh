#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Preserve the caller's cwd, so relative --config and --out paths stay intuitive.
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${repo_dir}/.venv/bin/python" -m experiments.gelsight_svelte.cma_control "$@"
