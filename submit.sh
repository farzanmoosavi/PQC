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
#   sbatch --export=RUNG=E submit.sh              # Rung E: scaling n=3,4 + gap analysis
#   sbatch --export=RUNG=F submit.sh              # Rung F: barren plateau gradient-variance
#   sbatch --export=RUNG=G submit.sh              # Rung G: topology × depth sensitivity
#   sbatch --export=RUNG=H submit.sh              # Rung H: critic ablation (REINFORCE vs AC)
#   sbatch --export=RUNG=T submit.sh              # Rung T: smoke-test only
#   sbatch --export=RUNG=A,EPISODES=500 submit.sh
#   sbatch --export=RUNG=B,SEEDS="0 1 2 3 4" submit.sh
#
# Recommended wall times (override default 24h with --time=HH:MM:SS):
#   T  0:20:00    A  20:00:00    B  16:00:00    C  16:00:00
#   D  8:00:00    E  22:00:00    F  1:00:00     G  8:00:00    H  20:00:00
# Example: sbatch --time=1:00:00 --export=RUNG=F submit.sh
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

_VENV=""
for _V in \
    "$HOME/py310_nibi/bin/activate" \
    "$HOME/py310_fir/bin/activate"  \
    "$HOME/py310_env/bin/activate"  \
    "$HOME/py310/bin/activate"; do
    [ -f "$_V" ] && { _VENV="$_V"; break; }
done
[ -z "$_VENV" ] && {
    echo "ERROR: no virtualenv found. Tried py310_nibi, py310_env, py310."
    echo "Run: python3 -m venv ~/py310_nibi && source ~/py310_nibi/bin/activate && pip install pennylane torch numpy"
    exit 1
}
echo "[venv] $_VENV"
source "$_VENV"

# Pin each Python process to exactly 1 CPU thread so N parallel processes
# cleanly occupy N cores without spawning competing PyTorch/NumPy thread pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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
SEEDS="${SEEDS:-0 1 2 3 4 5 6}"
N_QUBITS="${N_QUBITS:-9}"     # capped at 9 (≤10 qubits) to stay within 1-day wall time
N_LAYERS="${N_LAYERS:-4}"
LR="${LR:-5e-4}"
ENTROPY="${ENTROPY:-0.05}"
ENCODING="${ENCODING:-ry}"    # qubit encoding: ry | rz | ryrz
FRESH="${FRESH:-0}"

DQN_MODELS="quantum qaoa classical"
PG_MODELS="quantum qaoa node-quantum node-qaoa classical"

OUT_DIR="results/rung${RUNG}_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT_DIR"

# ============================================================
# Helpers
# ============================================================

run_bg() {
    # Throttle to SLURM_CPUS_PER_TASK concurrent processes so we never
    # over-subscribe the node (each process uses exactly 1 CPU thread).
    local max_jobs="${SLURM_CPUS_PER_TASK:-32}"
    while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
        sleep 2
    done
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
        --prefix  "$prefix" \
        --models  "$models" \
        --seeds   "$SEEDS" \
        --out-csv "$OUT_DIR/summary.csv" \
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
                --seed "$SEED" --fixed-instance --encoding "$ENCODING" \
                --out-prefix "$OUT_DIR/dqn"
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
                --lr "$LR" --entropy "$ENTROPY" --encoding "$ENCODING" \
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
                --lr "$LR" --entropy "$ENTROPY" --encoding "$ENCODING" \
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
        --n-jobs "$SLURM_CPUS_PER_TASK" \
        --out "$OUT_DIR/sweep.csv" \
        2>&1 | tee "$OUT_DIR/sweep.log"
    echo "=== Rung D done: $(date) ==="
fi

# ============================================================
# Rung E — Scaling + optimality gap (n=3, n=4)
#
# All node models use ≤9 qubits (2n+1: n=3→7q, n=4→9q) so the
# full job stays well within the 1-day wall limit.
#
# Three sub-experiments per node size:
#   fixed/ry    : fixed-instance REINFORCE with RY encoding
#   fixed/ryrz  : fixed-instance REINFORCE with RY+RZ encoding
#   policy/ry   : policy-learning (new instance each ep) with RY
#
# After training, gap_analysis.py produces the combined comparison
# table: model × encoding × mode → (gap%, total_params, pqc_params).
#
# Episode budgets (per seed) fit within ~20h on 32 CPUs:
#   n=3 (7q):  500 eps per sub-experiment
#   n=4 (9q):  300 eps per sub-experiment
# ============================================================
if [ "$RUNG" = "E" ]; then
    echo "=== Rung E: scaling + gap analysis (n=3, n=4) ==="

    declare -A E_EPS=( [3]=500 [4]=300 )

    for N_SIZE in 3 4; do
        NQ=$(( 2 * N_SIZE + 1 ))
        EPS_N=${E_EPS[$N_SIZE]}
        echo "--- n=$N_SIZE  qubits=$NQ  episodes=$EPS_N ---"

        for ENC in ry ryrz; do
            # Fixed-instance: train on one problem, tests route memorisation
            for MODEL in $PG_MODELS; do
                for SEED in $SEEDS; do
                    run_bg "e_n${N_SIZE}_${ENC}_fixed_${MODEL}_s${SEED}" reinforce_qrl.py \
                        --model "$MODEL" --node "$N_SIZE" --capacity "$CAPACITY" \
                        --episodes "$EPS_N" \
                        --n-qubits "$NQ" --n-layers "$N_LAYERS" \
                        --lr "$LR" --entropy "$ENTROPY" --encoding "$ENC" \
                        --seed "$SEED" --fixed-instance \
                        --out-prefix "$OUT_DIR/e_n${N_SIZE}_${ENC}_fixed"
                done
            done
        done

        # Policy learning: new instance each episode, tests generalisation
        for MODEL in $PG_MODELS; do
            for SEED in $SEEDS; do
                run_bg "e_n${N_SIZE}_ry_policy_${MODEL}_s${SEED}" reinforce_qrl.py \
                    --model "$MODEL" --node "$N_SIZE" --capacity "$CAPACITY" \
                    --episodes "$EPS_N" \
                    --n-qubits "$NQ" --n-layers "$N_LAYERS" \
                    --lr "$LR" --entropy "$ENTROPY" --encoding "ry" \
                    --seed "$SEED" \
                    --out-prefix "$OUT_DIR/e_n${N_SIZE}_ry_policy"
            done
        done
    done

    wait_all

    # Aggregate + gap analysis for every sub-experiment
    for N_SIZE in 3 4; do
        NQ=$(( 2 * N_SIZE + 1 ))
        for ENC in ry ryrz; do
            aggregate "$OUT_DIR/e_n${N_SIZE}_${ENC}_fixed" "$PG_MODELS"
            python3 -u gap_analysis.py \
                --prefix   "$OUT_DIR/e_n${N_SIZE}_${ENC}_fixed" \
                --models   $PG_MODELS \
                --seeds    $SEEDS \
                --node     "$N_SIZE" --n-qubits "$NQ" --n-layers "$N_LAYERS" \
                --encoding "$ENC" --mode fixed \
                --out-csv  "$OUT_DIR/gap_n${N_SIZE}_${ENC}_fixed.csv" \
                >> "$OUT_DIR/gap_n${N_SIZE}.log" 2>&1
        done

        aggregate "$OUT_DIR/e_n${N_SIZE}_ry_policy" "$PG_MODELS"
        python3 -u gap_analysis.py \
            --prefix   "$OUT_DIR/e_n${N_SIZE}_ry_policy" \
            --models   $PG_MODELS \
            --seeds    $SEEDS \
            --node     "$N_SIZE" --n-qubits "$NQ" --n-layers "$N_LAYERS" \
            --encoding "ry" --mode policy \
            --out-csv  "$OUT_DIR/gap_n${N_SIZE}_ry_policy.csv" \
            >> "$OUT_DIR/gap_n${N_SIZE}.log" 2>&1

        echo "Gap CSVs for n=$N_SIZE -> $OUT_DIR/gap_n${N_SIZE}_*.csv"
    done

    echo "=== Rung E done: $(date) ==="
fi

# ============================================================
# Rung F — Barren plateau gradient-variance analysis
#
# Scans: qubit width, layer depth, entanglement topology, encoding,
# and initialisation strategy.  All five scans use n=3 (7q) so the
# 2^7 statevector fits in memory with fast backprop.
#
# Runtime: ~30-60 min on 32 CPUs with 50 random initialisations.
# ============================================================
if [ "$RUNG" = "F" ]; then
    echo "=== Rung F: barren plateau gradient-variance analysis ==="
    python3 -u barren_plateau.py \
        --scan qubits layers topology encoding hinit \
        --trials 50 --node 3 \
        --n-jobs "$SLURM_CPUS_PER_TASK" \
        --out "$OUT_DIR/barren_plateau.csv" \
        2>&1 | tee "$OUT_DIR/barren_plateau.log"
    echo "=== Rung F done: $(date) ==="
fi

# ============================================================
# Rung G — Circuit sensitivity: topology × layer-depth sweep
#
# Sweeps ring / brick / star entanglement at n=3 and n=4, layers=1-4,
# seeds=0-2, for quantum and qaoa models only (classical is topology-free).
# Produces topo_sweep.csv for Rung G figure: converge_ep vs n_layers,
# coloured by topology, showing which topology reaches lowest gap fastest.
#
# Runtime: ~8h on 32 CPUs (3 topologies × 3 qubit sizes × 4 layers ×
#          3 seeds × 2 models × n=[3,4]).
# ============================================================
if [ "$RUNG" = "G" ]; then
    echo "=== Rung G: topology × layer-depth sensitivity ==="
    python3 -u sweep_experiment.py \
        --node 3 4 \
        --layers 1 2 3 4 \
        --topologies ring brick star \
        --models quantum qaoa \
        --seeds 0 1 2 \
        --episodes 200 \
        --n-jobs "$SLURM_CPUS_PER_TASK" \
        --out "$OUT_DIR/topo_sweep.csv" \
        2>&1 | tee "$OUT_DIR/topo_sweep.log"
    echo "=== Rung G done: $(date) ==="
fi

# ============================================================
# Rung H — Critic ablation: pure REINFORCE vs actor-critic
#
# Compares value_coef=0 (no baseline) vs value_coef=0.5 (actor-critic)
# for all five PG models at n=4 (9 qubits).  500 episodes per run.
# Shows whether the critic baseline is essential for quantum PG stability.
#
# Runtime: ~10h on 32 CPUs (5 models × 7 seeds × 2 conditions × 500 eps).
# ============================================================
if [ "$RUNG" = "H" ]; then
    echo "=== Rung H: critic ablation (pure REINFORCE vs actor-critic) ==="
    H_MODELS="quantum qaoa node-quantum node-qaoa classical"

    for CONDITION in nocritic critic; do
        VC="0.0"
        [ "$CONDITION" = "critic" ] && VC="0.5"
        for MODEL in $H_MODELS; do
            for SEED in $SEEDS; do
                run_bg "h_${CONDITION}_${MODEL}_s${SEED}" reinforce_qrl.py \
                    --model "$MODEL" --node 4 --capacity "$CAPACITY" \
                    --episodes 500 --n-qubits 9 --n-layers "$N_LAYERS" \
                    --lr "$LR" --entropy "$ENTROPY" --encoding ry \
                    --value-coef "$VC" \
                    --seed "$SEED" --fixed-instance \
                    --out-prefix "$OUT_DIR/h_${CONDITION}"
            done
        done
    done

    wait_all
    aggregate "$OUT_DIR/h_nocritic" "$H_MODELS"
    aggregate "$OUT_DIR/h_critic"   "$H_MODELS"
    echo "=== Rung H done: $(date) ==="
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
