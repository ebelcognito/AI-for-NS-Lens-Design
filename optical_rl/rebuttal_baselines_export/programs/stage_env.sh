#!/bin/bash
# Stage the hyperlens conda env onto the compute node's LOCAL xfs (/tmp) so
# torch imports read from local disk instead of the slow NFS home.
# Idempotent: skips if a working local env is already present.
# Usage: bash stage_env.sh <JOBID>   -> prints local python path on success.
set -uo pipefail
JOBID=${1:?usage: stage_env.sh <JOBID>}
SRC=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens
DSTPARENT=/tmp/hlenv_stage
LOCPY=$DSTPARENT/hyperlens/bin/python
REPO=/nas/longleaf/home/hisham26/Desktop/hyperlens

srun --jobid=$JOBID --overlap --exact --ntasks=1 --cpus-per-task=8 --mem=16G \
     --chdir="$REPO" bash -c '
set -uo pipefail
SRC="'"$SRC"'"; DSTPARENT="'"$DSTPARENT"'"; LOCPY="'"$LOCPY"'"
node=$(hostname)
if [ -x "$LOCPY" ] && "$LOCPY" -c "import torch" >/dev/null 2>&1; then
  echo "[stage@$node] already staged and importable: $LOCPY"
else
  echo "[stage@$node] copying env to node-local $DSTPARENT ... (one-time)"
  rm -rf "$DSTPARENT"; mkdir -p "$DSTPARENT"
  t0=$SECONDS
  # tar-pipe: sequential read beats per-file rsync for the many small files
  tar -C "$(dirname "$SRC")" -cf - "$(basename "$SRC")" | tar -C "$DSTPARENT" -xf -
  echo "[stage@$node] copy took $((SECONDS-t0))s"
fi
echo "[stage@$node] verifying LOCAL import (timed)..."
cd "'"$REPO"'"
/usr/bin/time -f "  local import wall=%es" "$LOCPY" -c "import sys; sys.path.insert(0,\"DeepLens\"); from optical_rl.qd.lensmega.run import _build_descriptor; import cma; print(\"[stage@$(hostname)] LOCAL import OK (torch+lensmega+cma)\")" 2>&1
'
