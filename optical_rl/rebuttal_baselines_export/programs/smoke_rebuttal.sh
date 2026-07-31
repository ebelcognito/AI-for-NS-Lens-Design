#!/bin/bash
set -uo pipefail
cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
SIX=optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml
D=/tmp/smoke_rebuttal; mkdir -p $D
echo "### [$(date +%H:%M:%S)] import timing"
python - <<'PY'
import time,sys; t=time.time(); sys.path.insert(0,"DeepLens")
from optical_rl.qd.lensmega.run import _build_descriptor
print(f"  import OK in {time.time()-t:.1f}s")
PY
echo "### [$(date +%H:%M:%S)] m5 cmaes budget=200"
$HPY m5_cmaes_restarts.py --config $SIX --budget 200 --seed 42 --out $D/m5.json 2>&1 | tail -4
echo "### [$(date +%H:%M:%S)] m4 multistart budget=200"
$HPY m4_multistart_lbfgs.py --config $SIX --budget 200 --n-starts 50 --max-steps 30 --seed 42 --out $D/m4.json 2>&1 | tail -4
echo "### [$(date +%H:%M:%S)] refine m5 pool via json mode"
$HPY tools/refine_all_parallel.py --pool-from json --json-pool $D/m5.json \
   --problem-config $SIX --threshold -5.0 --top-n 8 --workers 4 --out $D/m5_refined.json 2>&1 | tail -4
echo "### [$(date +%H:%M:%S)] paramspace QD iters=300 (lens_param descriptor + archive)"
$HPY -m optical_rl.qd.lensmega.run --config configs/rebuttal_sixlet/smoke_p5_paramspace.yaml --iterations 300 2>&1 | tail -6
echo "### [$(date +%H:%M:%S)] SMOKE DONE"
