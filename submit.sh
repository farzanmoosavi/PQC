#!/bin/bash
# ============================================================
# Alliance Canada — CPDPTW Quantum-RL experiments (Chapter 6)
# Clusters: nibi (H100), rorqual (L40S), fir (H100 NVL)
#
# All models run on CPU.  Quantum circuits use PennyLane default.qubit;
# classical MLP also runs on CPU (fast enough for n=5 problem size).
#
# One-time setup (run once on ANY login node — $HOME is shared):
#   source ~/py310_nibi/bin/activate
#   pip install pennylane "numpy>=2.0"
#
# Usage:
#   sbatch submit.sh                              # Rung A (DQN, default)
#   sbatch --export=RUNG=B submit.sh              # Rung B: REINFORCE fixed-instance
#   sbatch --export=RUNG=C submit.sh              # Rung C: REINFORCE policy-learning
#   sbatch --export=RUNG=D submit.sh              # Rung D: hyperparameter sweep
#   sbatch --export=RUNG=T submit.sh              # Rung T: smoke-test only
#   sbatch --export=RUNG=A,EPISODES=500 submit.sh
#   sbatch --export=RUNG=B,SEEDS="0 1 2 3 4" submit.sh
# ============================================================

#SBATCH --job-name=CE-PDPTW-qrl
#SBATCH --account=def-bfarooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=farzanmoosavi368@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/qrl-%x-%j.out
#SBATCH --error=logs/qrl-%x-%j.err

# ============================================================
# Environment
# ============================================================
module purge
module load python/3.10 scipy-stack

source "$HOME/py310_nibi/bin/activate" || {
    echo "ERROR: ~/py310_nibi not found."
    echo "Run the one-time setup commands shown at the top of this file."
    exit 1
}

# ============================================================
# Locate project directory
# Handles rorqual (~/links/projects/...) and nibi/fir (~/projects/...)
# ============================================================
_CLUSTER="${SLURM_CLUSTER_NAME:-unknown}"
echo "[cluster] $_CLUSTER"

_PROJ=""
for _C in \
    "$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW/PQC" \
    "$HOME/links/projects/def-bfarooq/farzan97/PQC" \
    "$HOME/projects/def-bfarooq/farzan97/CE-PDPTW/PQC" \
    "$HOME/projects/def-bfarooq/farzan97/PQC" \
    "$HOME/scratch/CE-PDPTW/PQC" \
    "$HOME/scratch/PQC" ; do
    [ -f "$_C/quantum_qnet.py" ] && { _PROJ="$_C"; break; }
done
[ -z "$_PROJ" ] && _PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[project] $_PROJ"
cd "$_PROJ" || { echo "ERROR: cannot cd to $_PROJ"; exit 1; }
mkdir -p logs results

echo "============================================================"
echo "  Job   : $SLURM_JOB_ID"
echo "  Rung  : ${RUNG:-A}   Cluster: $_CLUSTER"
echo "  Dir   : $(pwd)"
echo "  CPUs  : $SLURM_CPUS_PER_TASK   Start: $(date)"
echo "============================================================"

# ============================================================
# Experiment defaults (override with --export=VAR=value)
# ============================================================
RUNG="${RUNG:-A}"
NODE="${NODE:-5}"
CAPACITY="${CAPACITY:-5}"
EPISODES="${EPISODES:-1000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
N_QUBITS="${N_QUBITS:-11}"
N_LAYERS="${N_LAYERS:-4}"
LR="${LR:-5e-4}"
ENTROPY="${ENTROPY:-0.05}"
FRESH="${FRESH:-0}"

DQN_MODELS="quantum qaoa classical"
PG_MODELS="quantum qaoa node-quantum node-qaoa classical"

OUT_DIR="results/rung${RUNG}_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT_DIR"

# ============================================================
# Helpers
# ============================================================

run_bg() {
    local label="$1"; shift
    echo "[start] $label"
    python3 -u "$@" > "$OUT_DIR/${label}.log" 2>&1 &
    echo $! >> "$OUT_DIR/.pids"
}

wait_all() {
    local failed=0
    if [ -f "$OUT_DIR/.pids" ]; then
        while read -r pid; do
            wait "$pid" || { echo "FAILED pid $pid"; failed=1; }
        done < "$OUT_DIR/.pids"
        rm -f "$OUT_DIR/.pids"
    fi
    [ $failed -ne 0 ] && echo "WARNING: one or more jobs failed — check $OUT_DIR/*.log"
}

aggregate() {
    local prefix="$1" models="$2"
    python3 -u aggregate_results.py \
        --prefix "$prefix" \
        --models "$models" \
        --seeds  "$SEEDS" \
        --delete-seeds \
        >> "$OUT_DIR/aggregate.log" 2>&1
}

# ============================================================
# Rung T — smoke test only
# ============================================================
if [ "$RUNG" = "T" ]; then
    echo "=== Rung T: smoke test ==="
    python3 -u test_cluster.py --quick
    echo "Exit: $?"
    exit 0
fi

# ============================================================
# Pre-flight smoke test
# ============================================================
echo "--- pre-flight smoke test ---"
python3 -u test_cluster.py --quick || { echo "ABORT: smoke test failed."; exit 1; }
echo "--- smoke test passed ---"

# ============================================================
# Rung A — DQN, fixed instance
# ============================================================
if [ "$RUNG" = "A" ]; then
    echo "=== Rung A: DQN fixed-instance ==="
    [ "$FRESH" = "1" ] && rm -f "$OUT_DIR"/../rungA_*/*.pt "$OUT_DIR"/../rungA_*/*.txt 2>/dev/null

    for MODEL in $DQN_MODELS; do
        for SEED in $SEEDS; do
            run_bg "${MODEL}_s${SEED}" train_qrl.py \
                --model "$MODEL" --node "$NODE" --capacity "$CAPACITY" \
                --episodes "$EPISODES" --n-qubits "$N_QUBITS" --n-layers "$N_LAYERS" \
                --seed "$SEED" --fixed-instance --out-prefix "$OUT_DIR/dqn"
        done
    done

    wait_all
    aggregate "$OUT_DIR/dqn" "$DQN_MODELS"
    echo "=== Rung A done: $(date) ==="
fi

# ============================================================
# Rung B — REINFORCE, fixed instance
# ============================================================
if [ "$RUNG" = "B" ]; then
    echo "=== Rung B: REINFORCE fixed-instance ==="
    [ "$FRESH" = "1" ] && rm -f "$OUT_DIR"/../rungB_*/*.pt "$OUT_DIR"/../rungB_*/*.txt 2>/dev/null

    for MODEL in $PG_MODELS; do
        for SEED in $SEEDS; do
            run_bg "${MODEL}_s${SEED}" reinforce_qrl.py \
                --model "$MODEL" --node "$NODE" --capacity "$CAPACITY" \
                --episodes "$EPISODES" --n-qubits "$N_QUBITS" --n-layers "$N_LAYERS" \
                --lr "$LR" --entropy "$ENTROPY" \
                --seed "$SEED" --fixed-instance --out-prefix "$OUT_DIR/pg"
        done
    done

    wait_all
    aggregate "$OUT_DIR/pg" "$PG_MODELS"
    echo "=== Rung B done: $(date) ==="
fi

# ============================================================
# Rung C — REINFORCE, policy learning
# ============================================================
if [ "$RUNG" = "C" ]; then
    echo "=== Rung C: REINFORCE policy-learning ==="

    for MODEL in $PG_MODELS; do
        for SEED in $SEEDS; do
            run_bg "${MODEL}_s${SEED}" reinforce_qrl.py \
                --model "$MODEL" --node "$NODE" --capacity "$CAPACITY" \
                --episodes "$EPISODES" --n-qubits "$N_QUBITS" --n-layers "$N_LAYERS" \
                --lr "$LR" --entropy "$ENTROPY" \
                --seed "$SEED" --out-prefix "$OUT_DIR/policy"
        done
    done

    wait_all
    aggregate "$OUT_DIR/policy" "$PG_MODELS"

    echo "--- evaluating generalisation on held-out seeds ---"
    for MODEL in $PG_MODELS; do
        python3 -u policy_eval.py \
            --model "$MODEL" --node "$NODE" --capacity "$CAPACITY" \
            --train-episodes "$EPISODES" --eval-seeds 20 \
            --n-qubits "$N_QUBITS" --n-layers "$N_LAYERS" \
            >> "$OUT_DIR/policy_eval_${MODEL}.log" 2>&1 &
    done
    wait

    echo "=== Rung C done: $(date) ==="
fi

# ============================================================
# Rung D — Hyperparameter sweep
# ============================================================
if [ "$RUNG" = "D" ]; then
    echo "=== Rung D: hyperparameter sweep ==="
    python3 -u sweep_experiment.py \
        --node "$NODE" --episodes "$EPISODES" \
        --out "$OUT_DIR/sweep.csv" \
        2>&1 | tee "$OUT_DIR/sweep.log"
    echo "=== Rung D done: $(date) ==="
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  Results : $OUT_DIR"
ls -lh "$OUT_DIR"/*.log 2>/dev/null || echo "  (no logs)"
ls -lh "$OUT_DIR"/*.pt  2>/dev/null || echo "  (no checkpoints)"
echo "  End: $(date)"
echo "============================================================"
