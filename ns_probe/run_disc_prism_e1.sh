#!/usr/bin/env bash
# E1 head-to-head on the PERPENDICULAR prism-fold (parallel bundle from below, detector on the
# right). 3 modes x 5 seeds x 14000 evals (matched to the collinear fold experiment), seeds in
# parallel within each mode. Plus a 50k random pool (hit-rate + novelty). All arms share engine/
# fitness/bounds/archive.
set -u
cd /home/hisham26/Desktop/NS_lensdesign
PY=/playpen/hisham26/deeplens_env/bin/python
export PYTHONPATH=.:ns_probe
OUT=ns_probe/results_disc_prism

for mode in cmame random restart_ls; do
  echo "=== mode=$mode : seeds 0..4 in parallel ($(date +%H:%M:%S)) ==="
  pids=()
  for seed in 0 1 2 3 4; do
    $PY ns_probe/discovery_control_prism.py --mode $mode --evals 14000 --seed $seed \
        --out $OUT/$mode/seed$seed > logs_disc_prism/${mode}_seed${seed}.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "=== mode=$mode done ($(date +%H:%M:%S)) ==="
done

echo "=== E2a: 50k random pool ($(date +%H:%M:%S)) ==="
$PY ns_probe/discovery_control_prism.py --mode random --evals 50000 --seed 100 --dump-valid \
    --out $OUT/random50k > logs_disc_prism/random50k.log 2>&1

echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
