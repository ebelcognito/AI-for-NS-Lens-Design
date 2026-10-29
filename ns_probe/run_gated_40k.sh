#!/usr/bin/env bash
# The gated runs: doc 19's comparison re-run with the exit-convergence leak closed,
# plus a third curved face.
#
# Three arms, 3 seeds each, 40k evals -- the budget every archive on disk was built at.
#
#   fold24_clean_gated   fold-count axis   2 curved faces   exit gate + det_clearance
#   fold24_topo_gated    topology axis     2 curved faces   exit gate + det_clearance
#   fold24_topo3_gated   topology axis     3 curved faces   exit gate + det_clearance
#
# The first two answer "does doc 19 section 5.4's 2.1x retention survive gating", and they
# share ONE seed (runs/fold24_gated_seed) carrying both tir_count and tir_faces so each
# arm reads its own coordinate -- the confound doc 19 section 5.3 records.
#
# The third answers "what does a third curved surface buy". Its seed is the same
# genotypes with dsag=0 (a FLAT bottom face -- the identical optic), so it starts from
# exactly the 2-curved designs and can only win by using the new surface.
#
#   tmux new-session -d -s gated40k 'bash ns_probe/run_gated_40k.sh'
#   tmux attach -t gated40k
set -u
cd "$(dirname "$0")/.."
PY=/playpen/hisham26/optifusion_env/bin/python3
EVALS=${EVALS:-40000}
# The configured sigma0 (0.12) is mismatched to this gate stack and it dominates
# everything else. Measured by perturbing seed elites, acceptance is:
#     sigma 0.05   9.0% (2-curved)   3.1% (3-curved)
#     sigma 0.12   0.7% (2-curved)   0.2% (3-curved)
# and the 10k runs at 0.12 landed on that row exactly -- 0.63/0.71/0.30%, i.e. 30,000
# evals bought 190/213/89 valid designs. At 0.05 the same budget is worth ~13x more
# valid designs, which is the difference between a populated archive and three arms
# still sitting on their seed. See exit-convergence notes in driver.py.
SIGMA=${SIGMA:-0.05}
LOGS=runs/gated40k_logs
mkdir -p "$LOGS"

echo "=== gated: $EVALS evals, sigma0=$SIGMA, 3 arms x 3 seeds ==="
date

for s in 0 1 2; do
  for arm in clean topo topo3; do
    $PY -u -m nslens.qd "fold24_${arm}_gated" --evals "$EVALS" --seed "$s" --trace \
        --sigma0 "$SIGMA" --out "runs/gated40k_${arm}/seed$s" \
        > "$LOGS/${arm}_$s.log" 2>&1 &
  done
done

echo "launched 9 runs; pids: $(jobs -p | tr '\n' ' ')"
wait
echo "=== all runs finished ==="
date

# retention + fitness, same table as the ungated comparison
$PY ns_probe/compare_topo_axis.py --a runs/gated40k_clean --b runs/gated40k_topo \
    2>&1 | tee "$LOGS/comparison_axis.txt"
# what the third curved face bought, against the same axis at the same budget
$PY ns_probe/compare_topo_axis.py --a runs/gated40k_topo --b runs/gated40k_topo3 \
    2>&1 | tee "$LOGS/comparison_3curved.txt"
