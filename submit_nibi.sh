#!/bin/bash
# ============================================================
# Nibi (H100) — CPDPTW Quantum-RL experiments (Chapter 6).
#
# Usage:
#   sbatch submit_nibi.sh                         # Rung A: DQN fixed-instance
#   sbatch --export=RUNG=B submit_nibi.sh         # Rung B: REINFORCE fixed-instance
#   sbatch --export=RUNG=C submit_nibi.sh         # Rung C: REINFORCE policy-learning
#   sbatch --export=RUNG=D submit_nibi.sh         # Rung D: hyperparameter sweep
#   sbatch --export=RUNG=T submit_nibi.sh         # Rung T: smoke-test only
#   sbatch --export=RUNG=A,EPISODES=200 submit_nibi.sh
#   sbatch --export=RUNG=B,SEEDS="0 1" submit_nibi.sh
#   sbatch --export=RUNG=A,FRESH=1 submit_nibi.sh   # wipe previous results, restart
#
# One-time setup on login node before first sbatch:
#   module purge
#   module load python/3.10 scipy-stack cuda/12.2
#   virtualenv ~/py310_nibi
#   source ~/py310_nibi/bin/activate
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
#   pip install pennylane pennylane-lightning
#   pip install numpy --upgrade      # PennyLane requires numpy >= 2.0
# ============================================================

#SBATCH --job-name=CE-PDPTW-qrl
#SBATCH --account=def-bfarooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
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

source ~/py310_nibi/bin/activate

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
[ $_CUDA_LOADED -eq 0 ] && echo "WARNING: no CUDA module loaded — quantum models run on CPU (expected)."

# ============================================================
# Locate project directory
# ============================================================
_PROJ="$HOME/projects/def-bfarooq/farzan97/CE-PDPTW/PQC"
[ -d "$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW/PQC" ] \
    && _PROJ="$HOME/links/projects/def-bfarooq/farzan97/CE-PDPTW/PQC"
cd "$_PROJ" || { echo "ERROR: project dir not found: $_PROJ"; exit 1; }
mkdir -p logs results

echo "============================================================"
echo "  Job: $SLURM_JOB_ID   Rung: ${RUNG:-A}"
echo "  Dir: $(pwd)"
echo "  CPUs: $SLURM_CPUS_PER_TASK   Start: $(date)"
echo "============================================================"

# ============================================================
# Defaults (can be overridden via --export)
# ============================================================
RUNG="${RUNG:-A}"
NODE="${NODE:-5}"             # number of pickup-delivery request pairs
CAPACITY="${CAPACITY:-5}"
EPISODES="${EPISODES:-500}"
SEEDS="${SEEDS:-0 1 2}"       # space-separated list of random seeds
N_QUBITS="${N_QUBITS:-7}"     # for flat models (quantum, qaoa); node models auto-use 2n+1
N_LAYERS="${N_LAYERS:-3}"
LR="${LR:-5e-4}"
ENTROPY="${ENTROPY:-0.01}"
FRESH="${FRESH:-0}"

# Models for each rung (node models are REINFORCE-only due to replay-buffer coord issue).
DQN_MODELS="quantum qaoa classical"
PG_MODELS="quantum qaoa node-quantum node-qaoa classical"
OUT_DIR="results/rung${RUNG}_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT_DIR"

# ============================================================
# Helpers
# ============================================================

run_bg() {
    # run_bg <label> <python-command...>
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
        # Greedy rollout evaluation uses the checkpoint from seed 0
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
        --capacity "$CAPACITY" \
        --episodes "$EPISODES" \
        --out-prefix "$OUT_DIR/sweep" \
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
