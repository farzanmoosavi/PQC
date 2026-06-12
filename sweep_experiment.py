"""
sweep_experiment.py

Matched-parameter quantum-vs-classical comparison harness for the CPDPTW PQC
chapter.  Sweeps seeds × qubit counts × layer depths, trains the PQC and a
parameter-matched MLP on identical instances, and writes per-run results to
CSV.

Qubit-count rationale
---------------------
A CPDPTW instance with n_request pairs has:
    1 depot  +  n_request pickups  +  n_request deliveries  =  2*n_request + 1 nodes.

The "natural" qubit count is therefore 2*n_request + 1 — one qubit per node
including the depot.  The "compact" count is ceil(log2(2*n_request + 1)) — the
minimum register needed to address all actions in binary.  The sweep covers
both ends so the chapter can report the expressivity-vs-cost trade-off.

Usage
-----
    python sweep_experiment.py               # full grid, saves sweep_results.csv
    python sweep_experiment.py --quick       # 1 seed, 80 episodes per run
    python sweep_experiment.py --node 3      # only the 3-request problems
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from typing import Any

# --------------------------------------------------------------------------- #
# Grid definition
# --------------------------------------------------------------------------- #

# n_request values to sweep.  Keep ≤ 5 for simulation feasibility:
# the "natural" qubit count 2n+1 means a 2^(2n+1) statevector simulation.
DEFAULT_NODES    = [3, 4]          # 2n+1 qubits: n=3→7q, n=4→9q — both ≤10 qubit cap
DEFAULT_LAYERS   = [1, 2, 3, 4]
DEFAULT_SEEDS    = [0, 1, 2]
DEFAULT_EPISODES = 150
DEFAULT_TOPOLOGIES = ["ring"]      # add "brick" "star" for topology sensitivity (Rung G)
# Models compared.  "classical" and "classical-qaoa" are parameter-matched
# to "quantum" and "qaoa" respectively via match_classical_width().
DEFAULT_MODELS   = ["quantum", "qaoa", "classical", "classical-qaoa",
                    "node-quantum", "node-qaoa"]


def natural_qubits(node: int) -> int:
    """One qubit per CPDPTW node including depot: 2*n_request + 1."""
    return 2 * node + 1


def compact_qubits(node: int) -> int:
    """Minimum binary-address register: ceil(log2(2*n_request + 1))."""
    return math.ceil(math.log2(2 * node + 1))


def qubit_sizes(node: int) -> list[int]:
    """Return [compact, mid, natural] deduplicated and sorted."""
    c = compact_qubits(node)
    n = natural_qubits(node)
    mid = (c + n) // 2
    return sorted(set([c, mid, n]))


# --------------------------------------------------------------------------- #
# Process-pool task wrapper (module-level so it is picklable)
# --------------------------------------------------------------------------- #

def _run_one_task(args: tuple) -> dict:
    """Called by ProcessPoolExecutor workers — must live at module level."""
    model, node, nq, nl, seed, episodes, topo = args
    try:
        return run_one(model, node, nq, nl, seed, episodes, entanglement=topo)
    except Exception as exc:
        return {"model": model, "node": node, "n_qubits": nq,
                "n_layers": nl, "seed": seed, "entanglement": topo,
                "error": str(exc)}


# --------------------------------------------------------------------------- #
# Single-configuration runner
# --------------------------------------------------------------------------- #

def run_one(
    model_kind: str,
    node: int,
    n_qubits: int,
    n_layers: int,
    seed: int,
    episodes: int,
    capacity: int = 5,
    entanglement: str = "ring",
) -> dict[str, Any]:
    """
    Train one (model, qubit_count, layers, seed, entanglement) configuration.
    Returns a flat dict of metrics for the results CSV.
    """
    from train_qrl import train

    out_prefix = f"sweep_n{node}_q{n_qubits}_l{n_layers}_{entanglement}_s{seed}"
    t0 = time.perf_counter()

    quantum_models = ("quantum", "qaoa", "node-quantum", "node-qaoa")
    result = train(
        model_kind=model_kind,
        node=node,
        capacity=capacity,
        episodes=episodes,
        seed=seed,
        fixed_instance=True,
        out_prefix=out_prefix,
        n_qubits=n_qubits,
        n_layers=n_layers,
        entanglement=entanglement,
    )
    net = result["net"]
    params = net.param_report()
    total_params = params["total"]

    if model_kind in quantum_models:
        pqc_params  = params.get("pqc_var", 0)   # quantum models report "pqc_var"
        head_params = params.get("head", 0)
        # compact models use "compressor"; node models use "node_encoder"
        comp_params = params.get("compressor", 0) or params.get("node_encoder", 0)
    else:  # classical
        pqc_params = head_params = comp_params = 0

    elapsed = time.perf_counter() - t0

    rewards  = result["rewards"]
    dists    = result["dists"]
    complete = result["complete"]
    tail     = max(1, episodes // 5)   # last 20% of episodes

    return {
        "model":        model_kind,
        "node":         node,
        "n_qubits":     n_qubits,
        "n_layers":     n_layers,
        "entanglement": entanglement,
        "seed":         seed,
        "episodes":     episodes,
        "total_params": total_params,
        "pqc_params":   pqc_params,
        "comp_params":  comp_params,
        "head_params":  head_params,
        # last-tail-episodes means (lower is better since reward = negative cost)
        "final_reward_mean":   sum(rewards[-tail:]) / tail,
        "final_dist_mean":     sum(dists[-tail:])   / tail,
        "final_complete_mean": sum(complete[-tail:]) / tail,
        "best_reward":         max(rewards),
        "best_dist":           min(dists),
        "wall_sec":            elapsed,
        # rough convergence: first episode where mean(last 10) crosses 50% of best
        "converge_ep":         _convergence_episode(rewards),
    }


def _convergence_episode(rewards: list[float], window: int = 10) -> int:
    """
    Episode index at which the smoothed reward first reaches 50% of the total
    improvement (from worst to best).  Works correctly for negative rewards.
    """
    worst = min(rewards)
    best  = max(rewards)
    span  = best - worst
    if span < 1e-9:
        return len(rewards)
    threshold = worst + 0.5 * span   # midpoint between worst and best
    for i in range(window, len(rewards)):
        if sum(rewards[i - window:i]) / window >= threshold:
            return i
    return len(rewards)


# --------------------------------------------------------------------------- #
# Sweep driver
# --------------------------------------------------------------------------- #

def sweep(
    nodes:      list[int] = DEFAULT_NODES,
    layers:     list[int] = DEFAULT_LAYERS,
    seeds:      list[int] = DEFAULT_SEEDS,
    episodes:   int       = DEFAULT_EPISODES,
    models:     list[str] = DEFAULT_MODELS,
    topologies: list[str] = DEFAULT_TOPOLOGIES,
    out_csv:    str       = "sweep_results.csv",
    n_jobs:     int       = 1,
) -> list[dict]:
    _node_models = {"node-quantum", "node-qaoa"}

    # Build full config list; filter node-model redundancies upfront.
    all_configs = [
        (model, node, nq, nl, s, topo)
        for node  in nodes
        for nq    in qubit_sizes(node)
        for nl    in layers
        for s     in seeds
        for model in models
        for topo  in topologies
    ]
    valid = [
        c for c in all_configs
        if not (c[0] in _node_models and c[2] != natural_qubits(c[1]))
    ]
    total   = len(valid)
    skipped = len(all_configs) - total

    print(f"Sweep: {total} runs  ({skipped} node-model redundancies skipped)")
    print(f"  {len(nodes)} nodes × "
          f"{len([nq for node in nodes for nq in qubit_sizes(node)])} qubit configs × "
          f"{len(layers)} layers × {len(seeds)} seeds × "
          f"{len(models)} models × {len(topologies)} topologies")
    print(f"Qubit counts by node: { {n: qubit_sizes(n) for n in nodes} }")
    print(f"Natural encoding: 2n+1 qubits  "
          f"({', '.join(f'n={n}->{natural_qubits(n)}q' for n in nodes)})")
    print(f"Topologies: {topologies}  n_jobs={n_jobs}")
    print()

    rows:       list[dict] = []
    fieldnames: list[str]  = []

    def _save(rows, fieldnames):
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[-1].keys()),
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    if n_jobs == 1:
        for i, (model, node, nq, nl, seed, topo) in enumerate(valid, 1):
            print(f"[{i:3d}/{total}]  {model:9s}  n={node}  q={nq}  l={nl}  seed={seed}  topo={topo}")
            try:
                row = run_one(model, node, nq, nl, seed, episodes, entanglement=topo)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                row = {"model": model, "node": node, "n_qubits": nq,
                       "n_layers": nl, "seed": seed, "entanglement": topo, "error": str(exc)}
            rows.append(row)
            if not fieldnames and "error" not in row:
                fieldnames = list(row.keys())
            _save(rows, fieldnames)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        tasks = [(model, node, nq, nl, seed, episodes, topo) for (model, node, nq, nl, seed, topo) in valid]
        print(f"Launching {total} configs across {n_jobs} workers ...")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            future_map = {ex.submit(_run_one_task, t): t for t in tasks}
            done = 0
            for fut in as_completed(future_map):
                done += 1
                model, node, nq, nl, seed, _, topo = future_map[fut]
                row = fut.result()
                rows.append(row)
                if not fieldnames and "error" not in row:
                    fieldnames = list(row.keys())
                status = f"ERROR: {row['error']}" if "error" in row \
                    else f"reward={row.get('final_reward_mean', 0):7.2f}"
                print(f"[{done:3d}/{total}]  {model:9s}  n={node}  q={nq}  "
                      f"l={nl}  seed={seed}  topo={topo}  {status}")
                _save(rows, fieldnames)

    _print_summary(rows, nodes)
    print(f"\nFull results -> {out_csv}")
    return rows


def _print_summary(rows: list[dict], nodes: list[int]) -> None:
    print("\n" + "=" * 72)
    print(f"{'model':10s} {'node':4s} {'n_q':4s} {'n_l':4s} "
          f"{'params':7s} {'reward':9s} {'dist':8s} {'complete':8s} {'sec':6s}")
    print("-" * 72)
    for r in rows:
        if "error" in r:
            continue
        print(f"{r['model']:10s} {r['node']:4d} {r['n_qubits']:4d} {r['n_layers']:4d} "
              f"{r['total_params']:7d} {r['final_reward_mean']:9.3f} "
              f"{r['final_dist_mean']:8.3f} {r['final_complete_mean']:8.3f} "
              f"{r['wall_sec']:6.1f}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PQC sweep experiment")
    ap.add_argument("--node",     type=int,  nargs="+", default=DEFAULT_NODES,
                    help="n_request values to sweep (default: 3 5)")
    ap.add_argument("--layers",   type=int,  nargs="+", default=DEFAULT_LAYERS,
                    help="n_layers values (default: 2 3)")
    ap.add_argument("--seeds",    type=int,  nargs="+", default=DEFAULT_SEEDS,
                    help="RNG seeds (default: 0 1 2)")
    ap.add_argument("--models",   type=str,  nargs="+", default=DEFAULT_MODELS,
                    choices=["quantum", "qaoa", "classical", "classical-qaoa",
                             "node-quantum", "node-qaoa"],
                    help="Model types to include (default: all six)")
    ap.add_argument("--episodes", type=int,  default=DEFAULT_EPISODES,
                    help="Training episodes per run (default: 150)")
    ap.add_argument("--out",      default="sweep_results.csv",
                    help="Output CSV path")
    ap.add_argument("--topologies", type=str, nargs="+", default=DEFAULT_TOPOLOGIES,
                    choices=["ring", "brick", "all", "star"],
                    help="Entanglement topologies to sweep (default: ring)")
    ap.add_argument("--n-jobs",   type=int, default=1,
                    help="Parallel workers (default: 1 = sequential). "
                         "Set to $SLURM_CPUS_PER_TASK on the cluster.")
    ap.add_argument("--quick",    action="store_true",
                    help="1 seed, 80 episodes — for a fast sanity check")
    args = ap.parse_args()

    if args.quick:
        args.seeds    = [0]
        args.episodes = 80

    sweep(
        nodes      = args.node,
        layers     = args.layers,
        seeds      = args.seeds,
        episodes   = args.episodes,
        models     = args.models,
        topologies = args.topologies,
        out_csv    = args.out,
        n_jobs     = args.n_jobs,
    )
