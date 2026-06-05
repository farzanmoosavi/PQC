#!/bin/bash
# ============================================================
# Alliance Canada — CPDPTW Quantum-RL experiments (Chapter 6)
# Tested clusters: nibi (H100), rorqual (L40S), fir (H100 NVL)
#
# Usage:
#   sbatch submit.sh                              # Rung A: DQN fixed-instance
#   sbatch --export=RUNG=B submit.sh              # Rung B: REINFORCE fixed-instance
#   sbatch --export=RUNG=C submit.sh              # Rung C: REINFORCE policy-learning
#   sbatch --export=RUNG=D submit.sh              # Rung D: hyperparameter sweep
#   sbatch --export=RUNG=T submit.sh              # Rung T: smoke-test only
#   sbatch --export=RUNG=A,EPISODES=200 submit.sh
#   sbatch --export=RUNG=B,SEEDS="0 1" submit.sh
#   sbatch --export=RUNG=A,FRESH=1 submit.sh      # wipe previous results, restart
#
# One-time setup on login node before first sbatch:
#   module purge
#   module load python/3.10 scipy-stack cuda/12.2
#   virtualenv ~/py310_env
#   source ~/py310_env/bin/activate
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
#   pip install pennylane pennylane-lightning
#   pip install numpy --upgrade      # PennyLane requires numpy >= 2.0
#
# GPU note:
#   Classical networks run fully on the allocated GPU.
#   Quantum PQC circuits simulate on CPU via lightning.qubit (C++ accelerated).
#   GPU memory is used by the PyTorch compressor and output head layers only.
#   Requesting 1 GPU is still correct for the classical baseline comparison
#   and for any future lightning.gpu (cuQuantum) upgrade.
# ============================================================

#SBATCH --job-name=CE-PDPTW-qrl
#SBATCH --account=def-bfarooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00
#SBATCH --mail-user=farzanmoosavi368@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/qrl-%x-%j.out
#SBATCH --error=logs/qrl-%x-%j.err

# ============================================================
# Environment setup
# ============================================================
module purge
module load python/3.10 scipy-stack

source ~/py310_env/bin/activate

# Load CUDA matching the installed torch wheel.
_TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")
_CUDA_LOADED=0
for _CV in "$_TORCH_CUDA" 13.2 13.1 12.6 12.2; do
    [ -z "$_CV" ] && continue
    if module load cuda/$_CV 2>/dev/null; then
        echo "[cuda] Loaded cuda/$_CV"
        _CUDA_LOADED=1
        break
    fi
done
[ $_CUDA_LOADED -eq 0 ] && echo "INFO: no CUDA module loaded — classical model uses CPU."

# ============================================================
# Locate project directory
#
# Probe common Alliance layouts in order; use the first that contains
# quantum_qnet.py.  This handles rorqual (~/links/), nibi/fir (~/projects/),
# and direct lustre paths (e.g. /lustre09/project/<id>/farzan97/PQC).
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

# Last resort: the directory this script lives in is the project root.
[ -z "$_PROJ" ] && _PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[project] $_PROJ"
cd "$_PROJ" || { echo "ERROR: cannot cd to project dir: $_PROJ"; exit 1; }
mkdir -p logs results

echo "============================================================"
echo "  Job: $SLURM_JOB_ID   Rung: ${RUNG:-A}   Cluster: $_CLUSTER"
echo "  Dir: $(pwd)"
echo "  CPUs: $SLURM_CPUS_PER_TASK   Start: $(date)"
echo "============================================================"

# ============================================================
# Defaults (can be overridden via --export)
# ============================================================
RUNG="${RUNG:-A}"
NODE="${NODE:-5}"
CAPACITY="${CAPACITY:-5}"
EPISODES="${EPISODES:-1000}"
SEEDS="${SEEDS:-0 1 2}"
N_QUBITS="${N_QUBITS:-7}"
N_LAYERS="${N_LAYERS:-3}"
LR="${LR:-5e-4}"
ENTROPY="${ENTROPY:-0.01}"
FRESH="${FRESH:-0}"

# Models for each rung (node models are REINFORCE-only due to DQN replay-buffer
# coord issue when fixed_instance=False; safe with DQN+fixed_instance=True).
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
    python3 "$@" > "$OUT_DIR/${label}.log" 2>&1 &
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
    [ $failed -ne 0 ] && echo "WARNING: one or more background jobs failed — check $OUT_DIR/*.log"
}

# ============================================================
# Rung T — smoke test only
# ============================================================
if [ "$RUNG" = "T" ]; then
    echo "=== Rung T: smoke test ==="
    python3 test_cluster.py --quick
    echo "Test exit code: $?"
    exit 0
fi

# ============================================================
# Always run smoke test before any real experiment
# ============================================================
echo "--- pre-flight smoke test ---"
python3 test_cluster.py --quick || { echo "ABORT: smoke test failed."; exit 1; }
echo "--- smoke test passed ---"

# ============================================================
# Rung A — DQN, fixed instance, all flat+classical models
# ============================================================
if [ "$RUNG" = "A" ]; then
    echo "=== Rung A: DQN fixed-instance, node=$NODE, eps=$EPISODES, seeds=($SEEDS) ==="
    [ "$FRESH" = "1" ] && rm -f "$OUT_DIR"/../rungA_*/*.pt "$OUT_DIR"/../rungA_*/*.txt 2>/dev/null

    for MODEL in $DQN_MODELS; do
        for SEED in $SEEDS; do
            LABEL="${MODEL}_s${SEED}"
            run_bg "$LABEL" train_qrl.py \
                --model "$MODEL" \
                --node "$NODE" \
                --capacity "$CAPACITY" \
                --episodes "$EPISODES" \
                --n-qubits "$N_QUBITS" \
                --n-layers "$N_LAYERS" \
                --seed "$SEED" \
                --fixed-instance \
                --out-prefix "$OUT_DIR/dqn"
        done
    done

    wait_all
    echo "=== Rung A done: $(date) ==="
fi

# ============================================================
# Rung B — REINFORCE, fixed instance, all models (incl. node)
# ============================================================
if [ "$RUNG" = "B" ]; then
    echo "=== Rung B: REINFORCE fixed-instance, node=$NODE, eps=$EPISODES, seeds=($SEEDS) ==="
    [ "$FRESH" = "1" ] && rm -f "$OUT_DIR"/../rungB_*/*.pt "$OUT_DIR"/../rungB_*/*.txt 2>/dev/null

    for MODEL in $PG_MODELS; do
        for SEED in $SEEDS; do
            LABEL="${MODEL}_s${SEED}"
            run_bg "$LABEL" reinforce_qrl.py \
                --model "$MODEL" \
                --node "$NODE" \
                --capacity "$CAPACITY" \
                --episodes "$EPISODES" \
                --n-qubits "$N_QUBITS" \
                --n-layers "$N_LAYERS" \
                --lr "$LR" \
                --entropy "$ENTROPY" \
                --seed "$SEED" \
                --fixed-instance \
                --out-prefix "$OUT_DIR/pg"
        done
    done

    wait_all
    echo "=== Rung B done: $(date) ==="
fi

# ============================================================
# Rung C — REINFORCE, policy learning (fixed_instance=False)
# ============================================================
if [ "$RUNG" = "C" ]; then
    echo "=== Rung C: REINFORCE policy-learning, node=$NODE, eps=$EPISODES, seeds=($SEEDS) ==="

    for MODEL in $PG_MODELS; do
        for SEED in $SEEDS; do
            LABEL="${MODEL}_s${SEED}"
            run_bg "$LABEL" reinforce_qrl.py \
                --model "$MODEL" \
                --node "$NODE" \
                --capacity "$CAPACITY" \
                --episodes "$EPISODES" \
                --n-qubits "$N_QUBITS" \
                --n-layers "$N_LAYERS" \
                --lr "$LR" \
                --entropy "$ENTROPY" \
                --seed "$SEED" \
                --out-prefix "$OUT_DIR/policy"
        done
    done

    wait_all

    echo "--- evaluating generalisation on held-out seeds ---"
    for MODEL in $PG_MODELS; do
        python3 policy_eval.py \
            --model "$MODEL" \
            --node "$NODE" \
            --capacity "$CAPACITY" \
            --train-episodes "$EPISODES" \
            --eval-seeds 20 \
            --n-qubits "$N_QUBITS" \
            --n-layers "$N_LAYERS" \
            >> "$OUT_DIR/policy_eval_${MODEL}.log" 2>&1 &
    done
    wait

    echo "=== Rung C done: $(date) ==="
fi

# ============================================================
# Rung D — Hyperparameter sweep (sweep_experiment.py)
# ============================================================
if [ "$RUNG" = "D" ]; then
    echo "=== Rung D: hyperparameter sweep, node=$NODE ==="
    python3 sweep_experiment.py \
        --node "$NODE" \
        --episodes "$EPISODES" \
        --out "$OUT_DIR/sweep.csv" \
        2>&1 | tee "$OUT_DIR/sweep.log"
    echo "=== Rung D done: $(date) ==="
fi

# ============================================================
# Summarise outputs
# ============================================================
echo ""
echo "============================================================"
echo "  Results saved to: $OUT_DIR"
echo "  Log files:"
ls -lh "$OUT_DIR"/*.log 2>/dev/null || echo "  (none)"
echo "  Checkpoints:"
ls -lh "$OUT_DIR"/*.pt  2>/dev/null || echo "  (none)"
echo "  End: $(date)"
echo "============================================================"
