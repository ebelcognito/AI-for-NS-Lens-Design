#!/usr/bin/env bash
# The clean topology-axis comparison: same seed, same budget, 3 seeds per arm.
#
# Fixes both confounds in the 6k pilot (doc 19 section 5.3):
#   common seed   both arms resume from runs/fold24_common_seed, which carries BOTH
#                 tir_count and tir_faces on every cell, so each arm reads the
#                 coordinate its own cell_bc_keys names. The pilot gave the topology
#                 arm a seed built from catalogue survivors -- stronger AND more
#                 diverse -- which makes its fitness column unquotable.
#   full budget   40k evals, what every archive on disk was built at. The pilot's 6k
#                 is ~900 valid designs under a ~15% gate, and neither arm moved off
#                 its seed.
#
#   tmux new-session -d -s topo_ab_40k 'bash ns_probe/run_topo_ab_40k.sh'
#   tmux attach -t topo_ab_40k
set -u
cd "$(dirname "$0")/.."
PY=/playpen/hisham26/optifusion_env/bin/python3
EVALS=${EVALS:-40000}
SEED_DIR=runs/fold24_common_seed
LOGS=runs/ab40k_logs
mkdir -p "$LOGS"

echo "=== topology-axis A/B: $EVALS evals, 3 seeds/arm, common seed $SEED_DIR ==="
date

for s in 0 1 2; do
  # -u so the logs are live rather than block-buffered
  $PY -u -m nslens.qd fold24_clean --evals "$EVALS" --seed "$s" --trace \
      --resume-from "$SEED_DIR" --out "runs/ab40k_clean/seed$s" \
      > "$LOGS/clean_$s.log" 2>&1 &
  $PY -u -m nslens.qd fold24_topo  --evals "$EVALS" --seed "$s" --trace \
      --resume-from "$SEED_DIR" --out "runs/ab40k_topo/seed$s" \
      > "$LOGS/topo_$s.log" 2>&1 &
done

echo "launched 6 runs; pids: $(jobs -p | tr '\n' ' ')"
wait
echo "=== all runs finished ==="
date
$PY ns_probe/compare_topo_axis.py --a runs/ab40k_clean --b runs/ab40k_topo \
    2>&1 | tee "$LOGS/comparison.txt"
