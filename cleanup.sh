#!/bin/bash
# cleanup.sh — delete all generated output files from the project directory.
# Run this on the cluster BEFORE submitting fresh runs to avoid stale data.
#
# Keeps: all *.py, *.sh source files and hidden dirs (.git, .claude)
# Deletes: results/, logs/, *.pt, *.txt, *.csv, __pycache__
#
# Usage:
#   bash cleanup.sh           # dry run — shows what would be deleted
#   bash cleanup.sh --confirm # actually deletes

DRY=1
[ "$1" = "--confirm" ] && DRY=0

_rm() {
    if [ "$DRY" = "1" ]; then
        echo "[dry-run] would remove: $*"
    else
        rm -rf "$@" && echo "[deleted] $*"
    fi
}

echo "============================================================"
echo "  PQC cluster cleanup"
echo "  Mode: $([ "$DRY" = "1" ] && echo "DRY RUN (pass --confirm to delete)" || echo "CONFIRMED DELETE")"
echo "  Dir : $(pwd)"
echo "============================================================"

# Output directories
_rm results/
_rm logs/

# Stray output files in project root (from runs not using OUT_DIR)
for f in *.pt; do [ -f "$f" ] && _rm "$f"; done
for f in *.txt; do [ -f "$f" ] && _rm "$f"; done
for f in sweep_results.csv summary*.csv gap_*.csv topo_sweep.csv barren_plateau.csv; do
    [ -f "$f" ] && _rm "$f"
done

# Python cache
_rm __pycache__/

# Recreate empty output dirs for new runs
if [ "$DRY" = "0" ]; then
    mkdir -p logs results
    echo "Created empty logs/ and results/"
fi

echo "============================================================"
echo "  Source files preserved:"
ls *.py *.sh 2>/dev/null | sort
echo "============================================================"
