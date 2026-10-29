#!/usr/bin/env bash
# fold24_clean: 2-4 folds, wide field, and the beam must pack WITHOUT an internal focus.
#
# This is fold24_pack with convergence_gate=clean_pack. The width-only gate that
# fold24_pack uses does not do what it claims: its own completed 40k archive came back
# 11 of 15 niches caustic, with both top designs among them (best 0.766, best
# caustic-free 0.613). The ray-order-inversion test in the clean_pack verdict is what
# closes that.
#
#   ./ns_probe/run_fold24_clean.sh [evals] [session] [outdir]
#
# outdir defaults to runs/fold24_clean. Pass it explicitly to keep an earlier archive:
# the driver overwrites archive.json/trace.jsonl in place, so re-running with the
# default silently destroys the run you are comparing against.
#
# Attach with:  tmux attach -t fold24clean      (detach: ctrl-b then d)
# Watch:        tmux capture-pane -pt fold24clean:seed0 | tail -20
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/playpen/hisham26/deeplens_env/bin/python"
EVALS="${1:-40000}"
SESSION="${2:-fold24clean}"
OUT="${3:-$REPO/runs/fold24_clean}"

cd "$REPO"
mkdir -p "$OUT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists -- attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

for s in 0 1 2; do
  mkdir -p "$OUT/seed$s"
  # each seed gets its own window so one can be watched without the others scrolling past
  CMD="cd '$REPO' && PYTHONPATH=. '$PY' -u -m nslens.qd fold24_clean \
        --evals $EVALS --seed $s --trace --out '$OUT/seed$s' \
        2>&1 | tee '$OUT/seed$s/run.log'; echo; echo '[seed$s EXITED rc='\$?']'; sleep infinity"
  if [ "$s" = "0" ]; then
    tmux new-session -d -s "$SESSION" -n "seed$s" "bash -lc \"$CMD\""
  else
    tmux new-window -t "$SESSION" -n "seed$s" "bash -lc \"$CMD\""
  fi
done

echo "launched 3 x $EVALS evals in tmux session '$SESSION'"
echo "  attach : tmux attach -t $SESSION"
echo "  peek   : tmux capture-pane -pt $SESSION:seed0 | tail -20"
echo "  logs   : $OUT/seed{0,1,2}/run.log"
