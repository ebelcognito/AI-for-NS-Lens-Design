#!/bin/bash
# ============================================================================
# Rebuttal sixlet baseline suite
#
# Adds the reviewer-requested baselines to the sixlet head-to-head, all at
# MATCHED 100k forward evals, all pushed through the IDENTICAL L-BFGS refine
# pipeline, then compared on eps-net diversity + refined-RMS.
#
#   P1 multistart_lbfgs  (m4)  random-restart + L-BFGS-B          [R1.3a, R2.4]
#   P2 cmaes_restarts    (m5)  derivative-free CMA-ES + restarts  [R1.3b]
#   P3 randbc            (QD)  CMA-MEGA, random-projection BC      [R1.3d, R2.3]
#   P4 nograd            (QD)  MAP-Elites, Seidel BC, no gradients [R1.3c]
#   P5 paramspace        (QD)  CMA-MEGA, raw-curvature BC (Antonov)[R1.3e,R4.1]
#
# Reference incumbents (reused, already refined): Seidel[S2,S3] + SA+GA.
#
# Usage:
#   bash run_sixlet_baselines.sh <ALLOC_JOBID> [budget=100000] [seeds="42 43"]
#
# Node-portable: sizes srun steps to the allocation's NumCPUs (32 or 64).
# ============================================================================
set -uo pipefail

ALLOC=${1:?usage: $0 <ALLOC_JOBID> [budget=100000] [seeds="42 43"]}
BUDGET=${2:-100000}
SEEDS_STR=${3:-"42 43"}
read -ra SEEDS <<< "$SEEDS_STR"

cd /nas/longleaf/home/hisham26/Desktop/hyperlens
HPY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python
SIXLET_CFG=optical_rl/qd/lensmega/config/examples/sixlet_apsc_discovery_v2.yaml

# --- size srun steps to the allocation ---
NCPU=$(scontrol show job "$ALLOC" -o 2>/dev/null | grep -oP 'NumCPUs=\K[0-9]+' | head -1)
NCPU=${NCPU:-32}
SERIAL_CPUS=2                              # per serial baseline (m4/m5, single-proc)
N_SERIAL=$(( ${#SEEDS[@]} * 2 ))          # m4 + m5 per seed
SERIAL_TOTAL=$(( N_SERIAL * SERIAL_CPUS ))
QD_CPUS=$(( NCPU - SERIAL_TOTAL - 2 ))     # QD overlaps the bg serial baselines
(( QD_CPUS < 4 )) && QD_CPUS=4
QD_WORKERS=$QD_CPUS
REFINE_CPUS=$(( NCPU - 2 )); (( REFINE_CPUS < 4 )) && REFINE_CPUS=4
REFINE_WORKERS=$(( NCPU - 4 )); (( REFINE_WORKERS < 2 )) && REFINE_WORKERS=2

TS=$(date +%Y%m%d_%H%M%S)
ROOT=results/rebuttal_sixlet_${TS}
LOG=$ROOT/logs; POOL=$ROOT/pools; REF=$ROOT/refined; CFGD=$ROOT/configs
mkdir -p "$LOG" "$POOL" "$REF" "$CFGD"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/master.log"; }
node(){ local c=$1 m=$2; shift 2
  srun --jobid=$ALLOC --overlap --exact --ntasks=1 --cpus-per-task=$c --mem=$m \
       --chdir=/nas/longleaf/home/hisham26/Desktop/hyperlens "$@"; }

log "ROOT=$ROOT  ALLOC=$ALLOC (NCPU=$NCPU)  BUDGET=$BUDGET  SEEDS=${SEEDS[*]}"
log "QD_CPUS=$QD_CPUS/QD_WORKERS=$QD_WORKERS  SERIAL=$SERIAL_TOTAL  REFINE=$REFINE_CPUS/$REFINE_WORKERS"

# --- reference pools (already refined; reuse) ---
declare -A REF_SEIDEL REF_SAGA
REF_SEIDEL[42]=results/lensmega/sixlet_apsc_discovery_v2_seed42_20260506_034601/refined_top500_seed42_hl.json
REF_SEIDEL[43]=results/lensmega/sixlet_apsc_discovery_v2_seed43_20260506_042806/refined_top500_seed43_hl.json
REF_SAGA[42]=results/diversity_run_20260506_033447/sixlet/seed42/sa_ga/refined_above_-1.0_seed42_sa_ga.json
REF_SAGA[43]=results/diversity_run_20260506_033447/sixlet/seed43/sa_ga/refined_above_-1.0_seed43_sa_ga.json

declare -A QD_BASE
QD_BASE[randbc]=configs/rebuttal_sixlet/p3_randbc.yaml
QD_BASE[nograd]=configs/rebuttal_sixlet/p4_nograd.yaml
QD_BASE[paramspace]=configs/rebuttal_sixlet/p5_paramspace.yaml

clone_qd(){ local arm=$1 seed=$2 out="$CFGD/${arm}_seed${2}.yaml"
  $HPY -c "
import yaml
c=yaml.safe_load(open('${QD_BASE[$arm]}'))
c['algorithm']['cma_seed']=$seed
c['io']['run_name']='sixlet_rebuttal_${arm}_seed${seed}'
c['io']['parallel_workers']=$QD_WORKERS
yaml.safe_dump(c, open('$out','w'), sort_keys=False)"
  echo "$out"; }

# =====================================================================
# PHASE 1a: serial baselines (m4, m5) — all seeds concurrent, background
# =====================================================================
log "===== PHASE 1a: serial baselines (m4 multistart, m5 cmaes) bg ====="
declare -a BG_PIDS
for seed in "${SEEDS[@]}"; do
  node $SERIAL_CPUS 6G $HPY m4_multistart_lbfgs.py --config $SIXLET_CFG \
     --budget $BUDGET --n-starts 20000 --max-steps 100 --seed $seed \
     --out "$POOL/multistart_seed${seed}.json" --method-name multistart_lbfgs \
     > "$LOG/m4_seed${seed}.log" 2>&1 &  BG_PIDS+=($!)
  node $SERIAL_CPUS 6G $HPY m5_cmaes_restarts.py --config $SIXLET_CFG \
     --budget $BUDGET --seed $seed \
     --out "$POOL/cmaes_seed${seed}.json" --method-name cmaes_restarts \
     > "$LOG/m5_seed${seed}.log" 2>&1 &  BG_PIDS+=($!)
done
log "  launched ${#BG_PIDS[@]} serial steps"

# =====================================================================
# PHASE 1b: QD searches (sequential, seed-major so seed42 lands complete first)
# =====================================================================
log "===== PHASE 1b: QD searches (randbc, nograd, paramspace) ====="
declare -A QD_DIR
for seed in "${SEEDS[@]}"; do
  for arm in randbc nograd paramspace; do
    cfg=$(clone_qd $arm $seed)
    log "----- QD ${arm} seed=${seed} -----"
    node $QD_CPUS 40G $HPY -m optical_rl.qd.lensmega.run \
        --config "$cfg" --iterations $BUDGET \
        > "$LOG/qd_${arm}_seed${seed}.log" 2>&1 \
      && log "  ${arm} seed=${seed} done" || log "  ${arm} seed=${seed} FAILED ($?)"
    QD_DIR["${arm}_${seed}"]=$(ls -td results/lensmega/sixlet_rebuttal_${arm}_seed${seed}_* 2>/dev/null | head -1)
    log "  dir: ${QD_DIR[${arm}_${seed}]}"
  done
done

log "  waiting on serial baselines..."
for pid in "${BG_PIDS[@]}"; do wait $pid || log "  serial baseline exited nonzero"; done
log "===== PHASE 1 complete ====="

# =====================================================================
# PHASE 2: identical L-BFGS refinement of every NEW pool (top-500 >= -1.0)
# =====================================================================
log "===== PHASE 2: refinement (identical L-BFGS, top-500 >= -1.0) ====="
for seed in "${SEEDS[@]}"; do
  for arm in randbc nograd paramspace; do
    dir="${QD_DIR[${arm}_${seed}]}"
    log "  refine ${arm} seed=${seed} (archive.pkl)"
    if [[ -n "$dir" && -f "$dir/archive.pkl" ]]; then
      # Pin the problem config to the seidel sixlet config so EVERY arm is
      # refined with byte-identical problem+refine settings (keeps "identical
      # refinement" airtight and avoids building non-seidel descriptors here).
      node $REFINE_CPUS 40G $HPY tools/refine_all_parallel.py \
        --pool-from hl --hl-run "$dir" --problem-config $SIXLET_CFG \
        --threshold -1.0 --top-n 500 \
        --workers $REFINE_WORKERS --out "$REF/${arm}_seed${seed}.json" \
        > "$LOG/refine_${arm}_seed${seed}.log" 2>&1 && log "   ok" || log "   FAILED"
    else
      log "   SKIP (no archive.pkl in $dir)"
    fi
  done
  for arm in multistart cmaes; do
    pf="$POOL/${arm}_seed${seed}.json"
    log "  refine ${arm} seed=${seed} (json pool)"
    if [[ -f "$pf" ]]; then
      node $REFINE_CPUS 40G $HPY tools/refine_all_parallel.py \
        --pool-from json --json-pool "$pf" --problem-config $SIXLET_CFG \
        --threshold -1.0 --top-n 500 \
        --workers $REFINE_WORKERS --out "$REF/${arm}_seed${seed}.json" \
        > "$LOG/refine_${arm}_seed${seed}.log" 2>&1 && log "   ok" || log "   FAILED"
    else
      log "   SKIP (no pool file $pf)"
    fi
  done
done
log "===== PHASE 2 complete ====="

# =====================================================================
# PHASE 3: build manifest + eps-net analysis
# =====================================================================
log "===== PHASE 3: analysis ====="
MANIFEST="$ROOT/manifest.json"
$HPY - <<PY
import json
seeds = "${SEEDS[*]}".split()
ref_seidel = {"42": "${REF_SEIDEL[42]}", "43": "${REF_SEIDEL[43]}"}
ref_saga   = {"42": "${REF_SAGA[42]}",   "43": "${REF_SAGA[43]}"}
REF = "$REF"
def m(arm): return {s: f"{REF}/{arm}_seed{s}.json" for s in seeds}
man = {"sixlet_config": "$SIXLET_CFG", "methods": {
    "seidel_s2s3":      {s: ref_seidel.get(s, "") for s in seeds},
    "sa_ga":            {s: ref_saga.get(s, "")   for s in seeds},
    "multistart_lbfgs": m("multistart"),
    "cmaes_restarts":   m("cmaes"),
    "randbc":           m("randbc"),
    "nograd":           m("nograd"),
    "paramspace":       m("paramspace"),
}}
json.dump(man, open("$MANIFEST","w"), indent=2)
print("wrote $MANIFEST")
PY
node 4 8G $HPY analyze_rebuttal_sixlet.py --manifest "$MANIFEST" \
   --eps 0.20 --ref seidel_s2s3 --out "$ROOT/table.json" 2>&1 | tee "$LOG/analysis.log"

log "===== ALL DONE ====="
log "Results root: $ROOT"
log "Table:        $ROOT/table.json"
log "Analysis log: $LOG/analysis.log"
