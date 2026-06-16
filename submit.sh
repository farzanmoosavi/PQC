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
#   sbatch --export=RUNG=I submit.sh              # Rung I: PPO fixed-instance + policy-learning
#   sbatch --export=RUNG=J submit.sh              # Rung J: tw_tightness sweep [0, 0.5, 1.0]
#   sbatch --export=RUNG=T submit.sh              # Rung T: smoke-test only
#   sbatch --export=RUNG=A,EPISODES=500 submit.sh
#   sbatch --export=RUNG=B,SEEDS="0 1 2 3 4" submit.sh
#
# Recommended wall times (override default 24h with --time=HH:MM:SS):
#   T  0:20:00    A  24:00:00    B  22:00:00    C  24:00:00
#   D  8:00:00    E  24:00:00    F  1:00:00     G  14:00:00   H  24:00:00
#   I  24:00:00   J  24:00:00
# Example: sbatch --time=1:00:00 --export=RUNG=F submit.sh
# ============================================================

#SBATCH --job-name=CE-PDPTW-qrl
#SBATCH --account=def-bfarooq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --partition=cpubase_bycore_b3
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
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11}"   # 12 seeds — tighter confidence intervals
N_QUBITS="${N_QUBITS:-9}"     # capped at 9 (≤10 qubits) to stay within 1-day wall time
N_LAYERS="${N_LAYERS:-4}"
LR="${LR:-5e-4}"
ENTROPY="${ENTROPY:-0.05}"
ENCODING="${ENCODING:-ry}"    # qubit encoding: ry | rz | ryrz
FRESH="${FRESH:-0}"

DQN_MODELS="quantum qaoa node-quantum node-qaoa classical classical-large"
PG_MODELS="quantum qaoa node-quantum node-qaoa classical classical-large"
PPO_MODELS="quantum qaoa node-quantum node-qaoa classical classical-large"

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
    # aggregate <prefix> <models> [out_csv]
    # Default out_csv = $OUT_DIR/summary.csv
    local prefix="$1" models="$2" out="${3:-$OUT_DIR/summary.csv}"
    python3 -u aggregate_results.py \
        --prefix  "$prefix" \
        --models  "$models" \
        --seeds   "$SEEDS" \
        --out-csv "$out" \
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
    echo "=== Rung A: DQN fixed-instance (n=4 and n=5, all models) ==="

    for N_SIZE in 4 5; do
        NQ_NAT=$(( 2 * N_SIZE + 1 ))   # natural qubit count (node models always use this)
        echo "--- n=$N_SIZE  natural_qubits=$NQ_NAT ---"
        for MODEL in $DQN_MODELS; do
            # flat models: cap at N_QUBITS to avoid 11-qubit slowdown at n=5
            case "$MODEL" in
                node-*) MODEL_NQ=$NQ_NAT ;;
                *)       MODEL_NQ=$N_QUBITS ;;
            esac
            for SEED in $SEEDS; do
                run_bg "a_n${N_SIZE}_${MODEL}_s${SEED}" train_qrl.py \
                    --model "$MODEL" --node "$N_SIZE" --capacity "$CAPACITY" \
                    --episodes "$EPISODES" --n-qubits "$MODEL_NQ" --n-layers "$N_LAYERS" \
                    --seed "$SEED" --fixed-instance --encoding "$ENCODING" \
                    --out-prefix "$OUT_DIR/dqn_n${N_SIZE}"
            done
        done
    done

    wait_all
    for N_SIZE in 4 5; do
        aggregate "$OUT_DIR/dqn_n${N_SIZE}" "$DQN_MODELS" \
                  "$OUT_DIR/summary_dqn_n${N_SIZE}.csv"
    done
    echo "=== Rung A done: $(date) ==="
fi

# ============================================================
# Rung B — REINFORCE, fixed instance
# ============================================================
if [ "$RUNG" = "B" ]; then
    echo "=== Rung B: REINFORCE fixed-instance (ry + ryrz encodings) ==="

    for ENC in ry ryrz; do
        for MODEL in $PG_MODELS; do
            for SEED in $SEEDS; do
                run_bg "b_${ENC}_${MODEL}_s${SEED}" reinforce_qrl.py \
                    --model "$MODEL" --node "$NODE" --capacity "$CAPACITY" \
                    --episodes "$EPISODES" --n-qubits "$N_QUBITS" --n-layers "$N_LAYERS" \
                    --lr "$LR" --entropy "$ENTROPY" --encoding "$ENC" \
                    --seed "$SEED" --fixed-instance \
                    --out-prefix "$OUT_DIR/pg_${ENC}"
            done
        done
    done

    wait_all
    for ENC in ry ryrz; do
        aggregate "$OUT_DIR/pg_${ENC}" "$PG_MODELS" \
                  "$OUT_DIR/summary_${ENC}.csv"
    done
    echo "=== Rung B done: $(date) ==="
fi

# ============================================================
# Rung C — REINFORCE, policy learning
# ============================================================
if [ "$RUNG" = "C" ]; then
    echo "=== Rung C: REINFORCE policy-learning (2000 episodes) ==="
    # Policy learning needs 2× the episode budget of fixed-instance to differentiate models.
    C_EPISODES="${EPISODES:-2000}"

    for MODEL in $PG_MODELS; do
        for SEED in $SEEDS; do
            run_bg "c_${MODEL}_s${SEED}" reinforce_qrl.py \
                --model "$MODEL" --node "$NODE" --capacity "$CAPACITY" \
                --episodes "$C_EPISODES" --n-qubits "$N_QUBITS" --n-layers "$N_LAYERS" \
                --lr "$LR" --entropy "$ENTROPY" --encoding "$ENCODING" \
                --seed "$SEED" --out-prefix "$OUT_DIR/policy"
        done
    done

    wait_all
    aggregate "$OUT_DIR/policy" "$PG_MODELS"

    echo "--- evaluating generalisation on held-out seeds ---"
    for MODEL in $PG_MODELS; do
        run_bg "c_eval_${MODEL}" policy_eval.py \
            --model "$MODEL" --node "$NODE" --capacity "$CAPACITY" \
            --train-episodes "$EPISODES" --eval-seeds 20 \
            --n-qubits "$N_QUBITS" --n-layers "$N_LAYERS" \
            --out-prefix "$OUT_DIR/policy_eval"
    done
    wait_all

    echo "=== Rung C done: $(date) ==="
fi

# ============================================================
# Rung D — Hyperparameter sweep
# ============================================================
if [ "$RUNG" = "D" ]; then
    echo "=== Rung D: hyperparameter sweep ==="
    python3 -u sweep_experiment.py \
        --node 3 4 --episodes "${EPISODES:-150}" \
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
    echo "=== Rung E: scaling + gap analysis (n=3, n=4, n=5) ==="

    # Fixed-instance episode budgets (shorter at n=5 due to 11-qubit simulation cost)
    declare -A E_EPS=( [3]=500 [4]=300 [5]=150 )
    # Policy-learning needs more episodes to differentiate models
    declare -A E_POL_EPS=( [3]=1000 [4]=600 [5]=250 )

    for N_SIZE in 3 4 5; do
        NQ=$(( 2 * N_SIZE + 1 ))
        EPS_N=${E_EPS[$N_SIZE]}
        EPS_POL=${E_POL_EPS[$N_SIZE]}
        echo "--- n=$N_SIZE  qubits=$NQ  fixed_eps=$EPS_N  policy_eps=$EPS_POL ---"

        for ENC in ry ryrz; do
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

        # Policy learning: new instance each episode — more eps than fixed
        for MODEL in $PG_MODELS; do
            for SEED in $SEEDS; do
                run_bg "e_n${N_SIZE}_ry_policy_${MODEL}_s${SEED}" reinforce_qrl.py \
                    --model "$MODEL" --node "$N_SIZE" --capacity "$CAPACITY" \
                    --episodes "$EPS_POL" \
                    --n-qubits "$NQ" --n-layers "$N_LAYERS" \
                    --lr "$LR" --entropy "$ENTROPY" --encoding "ry" \
                    --seed "$SEED" \
                    --out-prefix "$OUT_DIR/e_n${N_SIZE}_ry_policy"
            done
        done
    done

    wait_all

    # Gap analysis FIRST — needs .pt checkpoints before aggregate deletes them
    for N_SIZE in 3 4 5; do
        NQ=$(( 2 * N_SIZE + 1 ))
        for ENC in ry ryrz; do
            python3 -u gap_analysis.py \
                --prefix   "$OUT_DIR/e_n${N_SIZE}_${ENC}_fixed" \
                --models   $PG_MODELS \
                --seeds    $SEEDS \
                --node     "$N_SIZE" --n-qubits "$NQ" --n-layers "$N_LAYERS" \
                --encoding "$ENC" --mode fixed \
                --out-csv  "$OUT_DIR/gap_n${N_SIZE}_${ENC}_fixed.csv" \
                >> "$OUT_DIR/gap_n${N_SIZE}.log" 2>&1
        done

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

    # Aggregate AFTER gap analysis
    for N_SIZE in 3 4 5; do
        for ENC in ry ryrz; do
            aggregate "$OUT_DIR/e_n${N_SIZE}_${ENC}_fixed" "$PG_MODELS" \
                      "$OUT_DIR/summary_n${N_SIZE}_${ENC}_fixed.csv"
        done
        aggregate "$OUT_DIR/e_n${N_SIZE}_ry_policy" "$PG_MODELS" \
                  "$OUT_DIR/summary_n${N_SIZE}_ry_policy.csv"
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
    # 5 topologies (ring/brick/star/all/none), 500 eps for reliable convergence signal,
    # 5 seeds for tighter variance, n=3 and n=4.
    python3 -u sweep_experiment.py \
        --node 3 4 \
        --layers 1 2 3 4 \
        --topologies ring brick star all none \
        --models quantum qaoa \
        --seeds 0 1 2 3 4 \
        --episodes 500 \
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
    echo "=== Rung H: value_coef sweep (0.0, 0.1, 0.25, 0.5) ==="
    H_MODELS="quantum qaoa node-quantum node-qaoa classical"

    for VC in 0.0 0.1 0.25 0.5; do
        VC_LABEL=$(echo "$VC" | tr '.' 'p')   # 0.25 → 0p25
        for MODEL in $H_MODELS; do
            for SEED in $SEEDS; do
                run_bg "h_vc${VC_LABEL}_${MODEL}_s${SEED}" reinforce_qrl.py \
                    --model "$MODEL" --node 4 --capacity "$CAPACITY" \
                    --episodes 500 --n-qubits 9 --n-layers "$N_LAYERS" \
                    --lr "$LR" --entropy "$ENTROPY" --encoding ry \
                    --value-coef "$VC" \
                    --seed "$SEED" --fixed-instance \
                    --out-prefix "$OUT_DIR/h_vc${VC_LABEL}"
            done
        done
    done

    wait_all
    for VC in 0.0 0.1 0.25 0.5; do
        VC_LABEL=$(echo "$VC" | tr '.' 'p')
        aggregate "$OUT_DIR/h_vc${VC_LABEL}" "$H_MODELS" \
                  "$OUT_DIR/summary_vc${VC_LABEL}.csv"
    done

    # Convergence-speed analysis: episode at which each (model, vc) first reaches
    # 80% of its best reward — shows critic benefit independently of final performance.
    python3 - <<'PYEOF' >> "$OUT_DIR/convergence_speed.log" 2>&1
import glob, numpy as np, csv, os

OUT_DIR = os.environ.get("OUT_DIR", ".")
threshold = 0.80
rows = []
for vc_label, vc in [("0p0","0.0"),("0p1","0.1"),("0p25","0.25"),("0p5","0.5")]:
    prefix = f"{OUT_DIR}/h_vc{vc_label}"
    for model in ["quantum","qaoa","node-quantum","node-qaoa","classical"]:
        files = glob.glob(f"{prefix}_{model}_s*_rewards.txt")
        if not files:
            continue
        curves = [np.loadtxt(f) for f in files]
        mean = np.stack(curves).mean(axis=0)
        best, worst = mean.max(), mean.min()
        span = best - worst
        if span < 1e-6:
            conv_ep = len(mean)
        else:
            thr = worst + threshold * span
            conv_ep = next((i for i,v in enumerate(mean) if v >= thr), len(mean))
        rows.append({"value_coef": vc, "model": model,
                     "conv_ep_80pct": conv_ep, "n_seeds": len(files),
                     "final_reward": float(mean[-20:].mean())})

rows.sort(key=lambda r: (r["model"], float(r["value_coef"])))
print(f"\n{'model':14s} {'vc':5s}  {'conv_ep':>8s}  {'final_reward':>12s}")
print("-" * 45)
for r in rows:
    print(f"{r['model']:14s} {r['value_coef']:5s}  {r['conv_ep_80pct']:8d}  "
          f"{r['final_reward']:12.3f}")

if rows:
    keys = list(rows[0].keys())
    with open(f"{OUT_DIR}/convergence_speed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {OUT_DIR}/convergence_speed.csv")
PYEOF

    echo "=== Rung H done: $(date) ==="
fi

# ============================================================
# Rung I — PPO: fixed-instance + policy-learning
#
# PPO replaces per-episode REINFORCE with multi-epoch updates on a
# buffer of 10 complete episodes, using GAE(lambda=0.95) advantages and
# a clipped surrogate objective (eps=0.2).  This prevents the large
# single-step gradient updates that destabilise PQC training.
#
# Includes classical-large (hidden=max(32,F)) as the properly-sized
# classical baseline to test quantum parameter efficiency:
#   node-quantum (~294 params) vs classical-large (~2900 params).
#
# Two sub-experiments:
#   fixed : fixed instance — convergence speed vs DQN / REINFORCE
#   policy: new instance each episode — generalisation with less data
#
# Runtime: ~18h on 32 CPUs (6 models × 7 seeds × 2 modes × 300 eps).
# ============================================================
if [ "$RUNG" = "I" ]; then
    echo "=== Rung I: PPO fixed-instance + policy-learning (n=3,4,5) ==="

    declare -A I_EPS=(     [3]=300  [4]=200  [5]=100  )
    declare -A I_POL_EPS=( [3]=500  [4]=300  [5]=150  )

    for N_SIZE in 3 4 5; do
        NQ=$(( 2 * N_SIZE + 1 ))
        EPS_I=${I_EPS[$N_SIZE]}
        EPS_P=${I_POL_EPS[$N_SIZE]}
        echo "--- n=$N_SIZE  qubits=$NQ  fixed_eps=$EPS_I  policy_eps=$EPS_P ---"

        for MODEL in $PPO_MODELS; do
            for SEED in $SEEDS; do
                run_bg "i_n${N_SIZE}_fixed_${MODEL}_s${SEED}" ppo_qrl.py \
                    --model "$MODEL" --node "$N_SIZE" --capacity "$CAPACITY" \
                    --episodes "$EPS_I" \
                    --n-qubits "$NQ" --n-layers "$N_LAYERS" \
                    --lr "$LR" --entropy "$ENTROPY" --encoding "$ENCODING" \
                    --seed "$SEED" --fixed-instance \
                    --out-prefix "$OUT_DIR/i_n${N_SIZE}_fixed"
            done
        done

        for MODEL in $PPO_MODELS; do
            for SEED in $SEEDS; do
                run_bg "i_n${N_SIZE}_policy_${MODEL}_s${SEED}" ppo_qrl.py \
                    --model "$MODEL" --node "$N_SIZE" --capacity "$CAPACITY" \
                    --episodes "$EPS_P" \
                    --n-qubits "$NQ" --n-layers "$N_LAYERS" \
                    --lr "$LR" --entropy "$ENTROPY" --encoding "$ENCODING" \
                    --seed "$SEED" \
                    --out-prefix "$OUT_DIR/i_n${N_SIZE}_policy"
            done
        done
    done

    wait_all

    for N_SIZE in 3 4 5; do
        NQ=$(( 2 * N_SIZE + 1 ))
        python3 -u gap_analysis.py \
            --prefix  "$OUT_DIR/i_n${N_SIZE}_fixed" \
            --models  $PPO_MODELS \
            --seeds   $SEEDS \
            --node    "$N_SIZE" --n-qubits "$NQ" --n-layers "$N_LAYERS" \
            --encoding "$ENCODING" --mode fixed \
            --out-csv "$OUT_DIR/gap_ppo_n${N_SIZE}_fixed.csv" \
            >> "$OUT_DIR/gap_ppo_n${N_SIZE}.log" 2>&1

        python3 -u gap_analysis.py \
            --prefix  "$OUT_DIR/i_n${N_SIZE}_policy" \
            --models  $PPO_MODELS \
            --seeds   $SEEDS \
            --node    "$N_SIZE" --n-qubits "$NQ" --n-layers "$N_LAYERS" \
            --encoding "$ENCODING" --mode policy \
            --out-csv "$OUT_DIR/gap_ppo_n${N_SIZE}_policy.csv" \
            >> "$OUT_DIR/gap_ppo_n${N_SIZE}.log" 2>&1

        echo "PPO gap CSVs for n=$N_SIZE -> $OUT_DIR/gap_ppo_n${N_SIZE}_*.csv"
    done

    for N_SIZE in 3 4 5; do
        aggregate "$OUT_DIR/i_n${N_SIZE}_fixed"  "$PPO_MODELS" \
                  "$OUT_DIR/summary_ppo_n${N_SIZE}_fixed.csv"
        aggregate "$OUT_DIR/i_n${N_SIZE}_policy" "$PPO_MODELS" \
                  "$OUT_DIR/summary_ppo_n${N_SIZE}_policy.csv"
    done

    echo "=== Rung I done: $(date) ==="
fi

# ============================================================
# Rung J — Time-window tightness sweep
#
# Trains DQN with tw_tightness ∈ {0.0, 0.5, 1.0} for all models
# at n=3 and n=4.  The key hypothesis: tighter windows make the RL
# objective harder to satisfy with a greedy-distance heuristic, so
# models with richer temporal representations (quantum or node-based)
# should pull ahead of flat classical MLPs.
#
# Gap analysis compares each (model, tightness) pair against the
# exact reference solver (also run with the same tightness so the
# reference cost reflects the true optimum under that window width).
#
# Runtime: ~20h on 32 CPUs (6 models × 12 seeds × 3 tightness ×
#          2 node sizes × 500 eps).
# ============================================================
if [ "$RUNG" = "J" ]; then
    echo "=== Rung J: tw_tightness sweep [0.0, 0.5, 1.0] ==="
    J_MODELS="quantum qaoa node-quantum node-qaoa classical classical-large"
    J_SEEDS="${SEEDS:-0 1 2 3 4 5 6}"
    J_EPISODES="${EPISODES:-500}"

    for TW in 0.0 0.5 1.0; do
        TW_LABEL=$(echo "$TW" | tr '.' 'p')   # 0.5 → 0p5
        for N_SIZE in 3 4; do
            NQ=$(( 2 * N_SIZE + 1 ))
            echo "--- tw=$TW  n=$N_SIZE  qubits=$NQ ---"
            for MODEL in $J_MODELS; do
                case "$MODEL" in
                    node-*) MODEL_NQ=$NQ ;;
                    *)       MODEL_NQ=$N_QUBITS ;;
                esac
                for SEED in $J_SEEDS; do
                    run_bg "j_tw${TW_LABEL}_n${N_SIZE}_${MODEL}_s${SEED}" train_qrl.py \
                        --model "$MODEL" --node "$N_SIZE" --capacity "$CAPACITY" \
                        --episodes "$J_EPISODES" \
                        --n-qubits "$MODEL_NQ" --n-layers "$N_LAYERS" \
                        --seed "$SEED" --fixed-instance --encoding "$ENCODING" \
                        --tw-tightness "$TW" \
                        --out-prefix "$OUT_DIR/j_tw${TW_LABEL}_n${N_SIZE}"
                done
            done
        done
    done

    wait_all

    # Gap analysis for each (tightness, node_size) pair
    for TW in 0.0 0.5 1.0; do
        TW_LABEL=$(echo "$TW" | tr '.' 'p')
        for N_SIZE in 3 4; do
            NQ=$(( 2 * N_SIZE + 1 ))
            python3 -u gap_analysis.py \
                --prefix       "$OUT_DIR/j_tw${TW_LABEL}_n${N_SIZE}" \
                --models       $J_MODELS \
                --seeds        $J_SEEDS \
                --node         "$N_SIZE" --n-qubits "$NQ" --n-layers "$N_LAYERS" \
                --encoding     "$ENCODING" --mode fixed \
                --tw-tightness "$TW" \
                --out-csv      "$OUT_DIR/gap_tw${TW_LABEL}_n${N_SIZE}.csv" \
                >> "$OUT_DIR/gap_j.log" 2>&1
        done
    done

    # Aggregate reward/dist/complete curves per (tightness, node_size)
    for TW in 0.0 0.5 1.0; do
        TW_LABEL=$(echo "$TW" | tr '.' 'p')
        for N_SIZE in 3 4; do
            aggregate "$OUT_DIR/j_tw${TW_LABEL}_n${N_SIZE}" "$J_MODELS" \
                      "$OUT_DIR/summary_tw${TW_LABEL}_n${N_SIZE}.csv"
        done
    done

    echo "Gap CSVs: $OUT_DIR/gap_tw*.csv"
    echo "=== Rung J done: $(date) ==="
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
