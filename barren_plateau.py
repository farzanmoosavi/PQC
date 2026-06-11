"""
barren_plateau.py

Gradient variance diagnostic for Chapter 6 sensitivity analysis.

Measures Var_theta(dL/dtheta_k) -- the variance of each PQC parameter's
gradient across random initialisations -- as a function of circuit width and
depth, entanglement topology, encoding strategy, and initialisation.

Theory background
-----------------
McClean et al. (2018): for a random circuit of depth L on n qubits, the
gradient of a GLOBAL observable (one involving all qubits) concentrates
exponentially: Var ~ exp(-n).  This is the barren plateau problem.

Cerezo et al. (2021): for LOCAL observables (acting on O(1) qubits), the
gradient variance decays only polynomially for shallow circuits:
    Var ~ O(1 / poly(n))    (local cost, depth L = O(log n))
    Var ~ O(exp(-n))        (local cost, depth L = O(poly(n)))

Our measurements are 2-local (Z_i and Z_i*Z_{i+1}), so we expect polynomial
decay -- the circuit should remain trainable at the qubit counts we use.
QAOA's structured cost+mixer inductive bias further resists barren plateaus
(Larocca et al. 2022).

Four scans
----------
1. qubit_scan  : Var vs n_qubits, fixed depth=3.  Main barren-plateau plot.
2. layer_scan  : Var vs n_layers, fixed width=6.  Shows depth sensitivity.
3. topology_scan: Var for ring/brick/all/star at fixed (n_qubits=6, n_layers=3).
4. encoding_scan: Var for ry/rz/ryrz at fixed (n_qubits=6, n_layers=3).
5. hinit_scan  : Var for H-init vs |0> init at same config.

Usage
-----
    python barren_plateau.py              # all five scans, save CSV + plots
    python barren_plateau.py --quick      # 20 trials, faster sanity check
    python barren_plateau.py --scan qubits layers   # only named scans
    python barren_plateau.py --node 3     # use n=3 env (faster)
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from typing import Any

import numpy as np
import torch

from cpdptw_env import CPDPTWEnv

device = torch.device("cpu")   # gradient variance must run on CPU (TorchLayer)

# --------------------------------------------------------------------------- #
# Default scan grids
# --------------------------------------------------------------------------- #

QUBIT_SCAN  = [3, 4, 5, 6, 7, 9, 11]   # fixed depth=3
LAYER_SCAN  = [1, 2, 3, 4, 5]           # fixed width=6
FIXED_DEPTH = 3
FIXED_WIDTH = 6
DEFAULT_TRIALS  = 50
QUICK_TRIALS    = 20
DEFAULT_NODE    = 3    # smallest env — network architecture is what matters here


# --------------------------------------------------------------------------- #
# Core measurement
# --------------------------------------------------------------------------- #

def measure_grad_var(
    model_kind: str,
    n_qubits: int,
    n_layers: int,
    n_trials: int = DEFAULT_TRIALS,
    node: int = DEFAULT_NODE,
    entanglement: str = "ring",
    encoding: str = "ry",
    h_init: bool = True,
) -> dict[str, Any]:
    """
    Measure gradient variance of PQC parameters across random initialisations.

    Algorithm:
      1. Build the network once (expensive -- PennyLane circuit compilation).
      2. For each trial:
           a. Re-initialise PQC weights ~ U(0, 2pi).
           b. Forward pass on a fixed random state.
           c. Backward pass.
           d. Collect all PQC parameter gradients.
      3. Compute Var_theta(dL/dtheta_k) for each k, then report mean/min/max.

    The loss is the mean of all Q-value outputs on a random unit-norm state --
    a simple scalar that exercises the full circuit output.  Using a fixed
    state removes input randomness from the variance estimate.
    """
    from quantum_qnet import QuantumQNetwork, QAOAQNetwork

    env = CPDPTWEnv(node=node, vehicle_capacity=5, rng_seed=0)
    env.reset(regenerate=True)
    state, _ = env.reset(regenerate=False)
    state = state.to(device)

    t0 = time.perf_counter()

    if model_kind == "quantum":
        net = QuantumQNetwork(env, n_qubits=n_qubits, n_layers=n_layers,
                              entanglement=entanglement, encoding=encoding,
                              h_init=h_init, torch_device=device)
        def _pqc_params():
            return [net.qlayer.weights]
    elif model_kind == "qaoa":
        net = QAOAQNetwork(env, n_qubits=n_qubits, n_layers=n_layers,
                           encoding=encoding, h_init=h_init,
                           torch_device=device)
        def _pqc_params():
            return [net.qlayer.gamma, net.qlayer.beta]
    else:
        raise ValueError(f"model_kind must be 'quantum' or 'qaoa', got '{model_kind}'")

    grads_all: list[np.ndarray] = []

    for _ in range(n_trials):
        # Re-initialise PQC weights to U(0, 2pi) -- samples a random point on
        # the parameter manifold, matching the standard barren-plateau protocol.
        with torch.no_grad():
            for p in _pqc_params():
                p.uniform_(0.0, 2.0 * math.pi)

        net.zero_grad()
        loss = net(state).mean()
        loss.backward()

        flat_grads = torch.cat([
            p.grad.flatten() for p in _pqc_params() if p.grad is not None
        ]).detach().cpu().numpy()
        grads_all.append(flat_grads)

    grads = np.array(grads_all)                     # (n_trials, n_pqc_params)
    var_per_param = np.var(grads, axis=0)           # variance across trials
    mean_grad     = np.abs(grads).mean()

    elapsed = time.perf_counter() - t0

    return {
        "model":        model_kind,
        "n_qubits":     n_qubits,
        "n_layers":     n_layers,
        "entanglement": entanglement,
        "encoding":     encoding,
        "h_init":       h_init,
        "n_trials":     n_trials,
        "n_pqc_params": grads.shape[1],
        "mean_var":     float(np.mean(var_per_param)),
        "min_var":      float(np.min(var_per_param)),
        "max_var":      float(np.max(var_per_param)),
        "mean_abs_grad": float(mean_grad),
        "wall_sec":     elapsed,
    }


# --------------------------------------------------------------------------- #
# Process-pool helpers (module-level so workers are picklable)
# --------------------------------------------------------------------------- #

def _measure_task(args_kwargs: tuple) -> dict:
    """ProcessPoolExecutor entry point — wraps measure_grad_var."""
    args, kwargs = args_kwargs
    return measure_grad_var(*args, **kwargs)


def _run_parallel(configs: list[tuple], n_jobs: int) -> list[dict]:
    """Run a list of (args, kwargs) measure_grad_var configs, optionally in parallel."""
    if n_jobs == 1:
        rows = []
        for args, kwargs in configs:
            r = measure_grad_var(*args, **kwargs)
            print(f"  {r['model']:8s}  q={r['n_qubits']:2d}  l={r['n_layers']}  "
                  f"mean_var={r['mean_var']:.2e}  ({r['wall_sec']:.1f}s)")
            rows.append(r)
        return rows
    from concurrent.futures import ProcessPoolExecutor
    print(f"  launching {len(configs)} configs across {n_jobs} workers ...")
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(configs))) as ex:
        rows = list(ex.map(_measure_task, configs))
    for r in rows:
        print(f"  {r['model']:8s}  q={r['n_qubits']:2d}  l={r['n_layers']}  "
              f"mean_var={r['mean_var']:.2e}  ({r['wall_sec']:.1f}s)")
    return rows


# --------------------------------------------------------------------------- #
# Scan drivers
# --------------------------------------------------------------------------- #

def scan_qubits(n_trials: int, node: int, n_jobs: int = 1) -> list[dict]:
    """Var vs n_qubits at fixed depth=FIXED_DEPTH for quantum and qaoa."""
    print(f"\n[scan_qubits]  n_layers={FIXED_DEPTH}  trials={n_trials}  n_jobs={n_jobs}")
    configs = [
        ((mk, nq, FIXED_DEPTH, n_trials, node), {})
        for nq in QUBIT_SCAN for mk in ("quantum", "qaoa")
    ]
    return _run_parallel(configs, n_jobs)


def scan_layers(n_trials: int, node: int, n_jobs: int = 1) -> list[dict]:
    """Var vs n_layers at fixed width=FIXED_WIDTH for quantum and qaoa."""
    print(f"\n[scan_layers]  n_qubits={FIXED_WIDTH}  trials={n_trials}  n_jobs={n_jobs}")
    configs = [
        ((mk, FIXED_WIDTH, nl, n_trials, node), {})
        for nl in LAYER_SCAN for mk in ("quantum", "qaoa")
    ]
    return _run_parallel(configs, n_jobs)


def scan_topology(n_trials: int, node: int, n_jobs: int = 1) -> list[dict]:
    """Var for ring/brick/all/star at fixed (FIXED_WIDTH, FIXED_DEPTH).
    Only applicable to QuantumQNetwork (HEA); QAOA topology is always ring."""
    print(f"\n[scan_topology]  n_qubits={FIXED_WIDTH}  n_layers={FIXED_DEPTH}"
          f"  trials={n_trials}  n_jobs={n_jobs}")
    configs = [
        (("quantum", FIXED_WIDTH, FIXED_DEPTH, n_trials, node), {"entanglement": topo})
        for topo in ("ring", "brick", "all", "star")
    ]
    return _run_parallel(configs, n_jobs)


def scan_encoding(n_trials: int, node: int, n_jobs: int = 1) -> list[dict]:
    """Var for ry/rz/ryrz encoding at fixed (FIXED_WIDTH, FIXED_DEPTH)."""
    print(f"\n[scan_encoding]  n_qubits={FIXED_WIDTH}  n_layers={FIXED_DEPTH}"
          f"  trials={n_trials}  n_jobs={n_jobs}")
    configs = [
        ((mk, FIXED_WIDTH, FIXED_DEPTH, n_trials, node), {"encoding": enc})
        for enc in ("ry", "rz", "ryrz") for mk in ("quantum", "qaoa")
    ]
    return _run_parallel(configs, n_jobs)


def scan_hinit(n_trials: int, node: int, n_jobs: int = 1) -> list[dict]:
    """Var for H-init vs |0>-init at fixed (FIXED_WIDTH, FIXED_DEPTH)."""
    print(f"\n[scan_hinit]  n_qubits={FIXED_WIDTH}  n_layers={FIXED_DEPTH}"
          f"  trials={n_trials}  n_jobs={n_jobs}")
    configs = [
        ((mk, FIXED_WIDTH, FIXED_DEPTH, n_trials, node), {"h_init": hi})
        for hi in (True, False) for mk in ("quantum", "qaoa")
    ]
    return _run_parallel(configs, n_jobs)


# --------------------------------------------------------------------------- #
# Print helpers
# --------------------------------------------------------------------------- #

def _print_qubit_table(rows: list[dict]) -> None:
    print("\n" + "="*68)
    print("Gradient variance vs n_qubits  (n_layers={})".format(FIXED_DEPTH))
    print(f"{'model':8s} {'n_q':4s} {'n_l':4s} {'n_pqc':6s} "
          f"{'mean_var':12s} {'min_var':12s} {'max_var':12s}")
    print("-"*68)
    for r in rows:
        print(f"{r['model']:8s} {r['n_qubits']:4d} {r['n_layers']:4d} "
              f"{r['n_pqc_params']:6d} {r['mean_var']:12.4e} "
              f"{r['min_var']:12.4e} {r['max_var']:12.4e}")
    print("="*68)


def _print_layer_table(rows: list[dict]) -> None:
    print("\n" + "="*68)
    print("Gradient variance vs n_layers  (n_qubits={})".format(FIXED_WIDTH))
    print(f"{'model':8s} {'n_q':4s} {'n_l':4s} {'n_pqc':6s} "
          f"{'mean_var':12s}")
    print("-"*68)
    for r in rows:
        print(f"{r['model']:8s} {r['n_qubits']:4d} {r['n_layers']:4d} "
              f"{r['n_pqc_params']:6d} {r['mean_var']:12.4e}")
    print("="*68)


def _print_sensitivity_table(rows: list[dict], title: str, key: str) -> None:
    print("\n" + "="*60)
    print(title)
    print(f"{'model':8s} {key:12s} {'n_pqc':6s} {'mean_var':12s} "
          f"{'mean_abs_grad':14s}")
    print("-"*60)
    for r in rows:
        print(f"{r['model']:8s} {str(r[key]):12s} {r['n_pqc_params']:6d} "
              f"{r['mean_var']:12.4e} {r['mean_abs_grad']:14.4e}")
    print("="*60)


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #

def run_all(
    scans: list[str],
    n_trials: int,
    node: int,
    out_csv: str,
    n_jobs: int = 1,
) -> list[dict]:
    all_rows: list[dict] = []
    fieldnames: list[str] = []

    scan_map = {
        "qubits":   lambda: scan_qubits(n_trials, node, n_jobs),
        "layers":   lambda: scan_layers(n_trials, node, n_jobs),
        "topology": lambda: scan_topology(n_trials, node, n_jobs),
        "encoding": lambda: scan_encoding(n_trials, node, n_jobs),
        "hinit":    lambda: scan_hinit(n_trials, node, n_jobs),
    }
    print_map = {
        "qubits":   lambda r: _print_qubit_table(r),
        "layers":   lambda r: _print_layer_table(r),
        "topology": lambda r: _print_sensitivity_table(r, "Topology sensitivity", "entanglement"),
        "encoding": lambda r: _print_sensitivity_table(r, "Encoding sensitivity", "encoding"),
        "hinit":    lambda r: _print_sensitivity_table(r, "Initialisation sensitivity", "h_init"),
    }

    for scan_name in scans:
        rows = scan_map[scan_name]()
        print_map[scan_name](rows)
        all_rows.extend(rows)
        if not fieldnames and rows:
            fieldnames = list(rows[0].keys())
        # Incremental write.
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()),
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\nFull results -> {out_csv}")
    return all_rows


# --------------------------------------------------------------------------- #
# Optional: matplotlib plots
# --------------------------------------------------------------------------- #

def plot_results(csv_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib/pandas not available -- skipping plots.")
        return

    df = pd.read_csv(csv_path)

    # Plot 1: Var vs n_qubits
    qdf = df[(df["n_layers"] == FIXED_DEPTH) &
             (df["entanglement"] == "ring") &
             (df["encoding"] == "ry") &
             (df["h_init"] == True)]  # noqa: E712
    if not qdf.empty:
        fig, ax = plt.subplots()
        for mk, grp in qdf.groupby("model"):
            ax.semilogy(grp["n_qubits"], grp["mean_var"], marker="o", label=mk)
        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Grad variance (log scale)")
        ax.set_title(f"Barren plateau: Var vs n_qubits  (L={FIXED_DEPTH})")
        ax.legend()
        fig.tight_layout()
        fig.savefig("bp_qubits.png", dpi=150)
        print("Saved bp_qubits.png")

    # Plot 2: Var vs n_layers
    ldf = df[(df["n_qubits"] == FIXED_WIDTH) &
             (df["entanglement"] == "ring") &
             (df["encoding"] == "ry") &
             (df["h_init"] == True)]  # noqa: E712
    if not ldf.empty:
        fig, ax = plt.subplots()
        for mk, grp in ldf.groupby("model"):
            ax.semilogy(grp["n_layers"], grp["mean_var"], marker="s", label=mk)
        ax.set_xlabel("Number of layers")
        ax.set_ylabel("Grad variance (log scale)")
        ax.set_title(f"Barren plateau: Var vs n_layers  (nq={FIXED_WIDTH})")
        ax.legend()
        fig.tight_layout()
        fig.savefig("bp_layers.png", dpi=150)
        print("Saved bp_layers.png")

    plt.close("all")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Barren plateau gradient variance analysis")
    ap.add_argument("--scan",    nargs="+",
                    choices=["qubits", "layers", "topology", "encoding", "hinit"],
                    default=["qubits", "layers", "topology", "encoding", "hinit"],
                    help="Which scans to run (default: all five)")
    ap.add_argument("--trials",  type=int, default=DEFAULT_TRIALS,
                    help=f"Random initialisations per config (default: {DEFAULT_TRIALS})")
    ap.add_argument("--node",    type=int, default=DEFAULT_NODE,
                    help=f"n_requests for env (default: {DEFAULT_NODE}, smallest)")
    ap.add_argument("--out",     default="barren_plateau.csv",
                    help="Output CSV path")
    ap.add_argument("--n-jobs",   type=int, default=1,
                    help="Parallel workers for config-level parallelism (default: 1). "
                         "Set to $SLURM_CPUS_PER_TASK on the cluster.")
    ap.add_argument("--quick",   action="store_true",
                    help=f"Use {QUICK_TRIALS} trials instead of {DEFAULT_TRIALS}")
    ap.add_argument("--plot",    action="store_true",
                    help="Save matplotlib plots (requires matplotlib+pandas)")
    args = ap.parse_args()

    n_trials = QUICK_TRIALS if args.quick else args.trials
    rows = run_all(args.scan, n_trials, args.node, args.out, n_jobs=args.n_jobs)

    if args.plot:
        plot_results(args.out)
