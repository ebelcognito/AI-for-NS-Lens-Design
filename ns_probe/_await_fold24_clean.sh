#!/usr/bin/env bash
# Block until all three fold24_clean seeds have written archive.json, then render.
set -uo pipefail
REPO="/home/hisham26/Desktop/NS_lensdesign"
OUT="$REPO/runs/fold24_clean"
PY="/playpen/hisham26/deeplens_env/bin/python"
cd "$REPO"
for i in $(seq 1 240); do
  n=$(ls "$OUT"/seed*/archive.json 2>/dev/null | wc -l)
  [ "$n" -ge 3 ] && break
  sleep 30
done
echo "archives present: $(ls "$OUT"/seed*/archive.json 2>/dev/null | wc -l)/3"
PYTHONPATH=.:ns_probe "$PY" -u ns_probe/render_best_designs.py --top 10 2>&1 | tail -20
PYTHONPATH=.:ns_probe "$PY" -u ns_probe/render_best_designs.py --per-niche \
  --out "$OUT/gallery_niches" 2>&1 | tail -5
echo "RENDER_DONE"
