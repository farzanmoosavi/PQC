#!/bin/bash
# ============================================================
# submit_rungJ_topup.sh
#
# Fills in the 59 missing tw1p0/n4 seeds from the timed-out
# rungJ_20260614_2130 job, then re-runs gap analysis and
# aggregation so the full 12-seed dataset is complete.
#
# Usage:
#   sbatch submit_rungJ_topup.sh
#
# Wall time: 12 h is conservative; expect ~4-6 h on 32 CPUs.
# ============================================================

#SBATCH --job-name=CE-PDPTW-J-topup
#SBATCH --account=def-bfarooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-12:00
#SBATCH --mail-user=farzanmoosavi368@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/qrl-%x-%j.out
#SBATCH --error=logs/qrl-%x-%j.err

# ---- environment ----
module purge
module load python/3.10 scipy-stack

_VENV=""
for _V in \
    "$HOME/py310_nibi/bin/activate" \
    "$HOME/py310_fir/bin/activate"  \
    "$HOME/py310_env/bin/activate"  \
    "$HOME/py310/bin/activate"; do
    [ -f "$_V" ] && { _VENV="$_V"; break; }
done
[ -z "$_VENV" ] && { echo "ERROR: no virtualenv found."; exit 1; }
source "$_VENV"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- project directory ----
_PROJ=""
for _C in \
    "$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW/PQC" \
    "$HOME/links/projects/def-bfarooq/farzan97/PQC" \
    "$HOME/projects/def-bfarooq/farzan97/CE-PDPTW/PQC" \
    "$HOME/projects/def-bfarooq/farzan97/PQC" \
    "$HOME/scratch/CE-PDPTW/PQC" \
    "$HOME/scratch/PQC"; do
    [ -f "$_C/quantum_qnet.py" ] && { _PROJ="$_C"; break; }
done
[ -z "$_PROJ" ] && _PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$_PROJ" || { echo "ERROR: cannot cd to $_PROJ"; exit 1; }
mkdir -p logs

# Write into the SAME directory as the original job so gap analysis
# can see all 12 seeds together.
OUT_DIR="results/rungJ_20260614_2130"
[ -d "$OUT_DIR" ] || { echo "ERROR: $OUT_DIR does not exist — wrong machine?"; exit 1; }

echo "============================================================"
echo "  Job    : $SLURM_JOB_ID"
echo "  TopUp  : tw1p0 / n4 (59 missing seeds)"
echo "  OutDir : $OUT_DIR"
echo "  CPUs   : $SLURM_CPUS_PER_TASK   Start: $(date)"
echo "============================================================"

# ---- shared parameters (must match original job) ----
CAPACITY=5
EPISODES=1000
N_QUBITS=9          # default; flat models use this
N_LAYERS=4
ENCODING="ry"
TW=1.0
TW_LABEL="1p0"
N_SIZE=4
NQ=9                # 2*4+1

# ---- run_bg helper ----
run_bg() {
    local max_jobs="${SLURM_CPUS_PER_TASK:-32}"
    while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
        sleep 2
    done
    local label="$1"; shift
    echo "[start] $label"
    python3 -u "$@" > "$OUT_DIR/${label}.log" 2>&1 &
    echo $! >> "$OUT_DIR/.pids_topup"
}

wait_all() {
    local failed=0
    if [ -f "$OUT_DIR/.pids_topup" ]; then
        while read -r pid; do
            wait "$pid" || { echo "FAILED pid $pid"; failed=1; }
        done < "$OUT_DIR/.pids_topup"
        rm -f "$OUT_DIR/.pids_topup"
    fi
    [ $failed -ne 0 ] && echo "WARNING: one or more jobs failed"
}

# ---- missing seeds per model ----
# (determined from: for s in $(seq 0 11); do [ -f <prefix>_s${s}_rewards.txt ] || echo missing; done)

MISSING_quantum="1 3 4 5 6 7 8 9 10 11"
MISSING_qaoa="11"
MISSING_node_quantum="0 1 2 3 4 5 6 7 8 9 10 11"
MISSING_node_qaoa="0 1 2 3 4 5 6 7 8 9 10 11"
MISSING_classical="0 1 2 3 4 5 6 7 8 9 10 11"
MISSING_classical_large="0 1 2 3 4 5 6 7 8 9 10 11"

launch_model() {
    local MODEL="$1"
    local SEEDS="$2"
    local MODEL_NQ
    case "$MODEL" in
        node-*) MODEL_NQ=$NQ ;;
        *)       MODEL_NQ=$N_QUBITS ;;
    esac
    for SEED in $SEEDS; do
        run_bg "j_tw${TW_LABEL}_n${N_SIZE}_${MODEL}_s${SEED}" train_qrl.py \
            --model "$MODEL" --node "$N_SIZE" --capacity "$CAPACITY" \
            --episodes "$EPISODES" \
            --n-qubits "$MODEL_NQ" --n-layers "$N_LAYERS" \
            --seed "$SEED" --fixed-instance --encoding "$ENCODING" \
            --tw-tightness "$TW" \
            --out-prefix "$OUT_DIR/j_tw${TW_LABEL}_n${N_SIZE}"
    done
}

echo "--- launching missing tw${TW_LABEL} n${N_SIZE} seeds ---"
launch_model "quantum"        "$MISSING_quantum"
launch_model "qaoa"           "$MISSING_qaoa"
launch_model "node-quantum"   "$MISSING_node_quantum"
launch_model "node-qaoa"      "$MISSING_node_qaoa"
launch_model "classical"      "$MISSING_classical"
launch_model "classical-large" "$MISSING_classical_large"

wait_all
echo "--- training done: $(date) ---"

# ---- re-run gap analysis for tw1p0/n4 with the full 12-seed dataset ----
ALL_SEEDS="0 1 2 3 4 5 6 7 8 9 10 11"
J_MODELS="quantum qaoa node-quantum node-qaoa classical classical-large"

echo "--- gap analysis: tw${TW_LABEL} n${N_SIZE} ---"
python3 -u gap_analysis.py \
    --prefix       "$OUT_DIR/j_tw${TW_LABEL}_n${N_SIZE}" \
    --models       $J_MODELS \
    --seeds        $ALL_SEEDS \
    --node         "$N_SIZE" --n-qubits "$NQ" --n-layers "$N_LAYERS" \
    --encoding     "$ENCODING" --mode fixed \
    --tw-tightness "$TW" \
    --out-csv      "$OUT_DIR/gap_tw${TW_LABEL}_n${N_SIZE}.csv" \
    >> "$OUT_DIR/gap_j.log" 2>&1
echo "Gap CSV -> $OUT_DIR/gap_tw${TW_LABEL}_n${N_SIZE}.csv"

# ---- re-run aggregate for tw1p0/n4 ----
echo "--- aggregate: tw${TW_LABEL} n${N_SIZE} ---"
python3 -u aggregate_results.py \
    --prefix  "$OUT_DIR/j_tw${TW_LABEL}_n${N_SIZE}" \
    --models  "$J_MODELS" \
    --seeds   "$ALL_SEEDS" \
    --out-csv "$OUT_DIR/summary_tw${TW_LABEL}_n${N_SIZE}.csv" \
    >> "$OUT_DIR/aggregate.log" 2>&1
echo "Summary CSV -> $OUT_DIR/summary_tw${TW_LABEL}_n${N_SIZE}.csv"

echo ""
echo "============================================================"
echo "  TopUp complete: $(date)"
echo "  Check: ls $OUT_DIR/j_tw1p0_n4_*_rewards.txt | wc -l  (expect 72)"
echo "  Gap  : $OUT_DIR/gap_tw1p0_n4.csv"
echo "============================================================"
