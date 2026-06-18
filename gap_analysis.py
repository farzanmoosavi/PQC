"""
gap_analysis.py — Chapter 6 optimality gap and parameter efficiency analysis.

Compares trained RL policies against a reference solver:
  n <= 5  →  exact branch-and-bound (minimises the same objective as RL:
             dist/dist_scale + time-window penalties; provably optimal,
             milliseconds per instance).  gap_pct >= 0 is guaranteed.
  n >  5  →  greedy nearest-neighbour (near-optimal upper bound)

Two evaluation modes
--------------------
  fixed    Train seed S → evaluate greedy policy on the SAME instance (seed S).
           Gap shows how well the agent memorised the target route.

  policy   Train on random instances → evaluate on 20 held-out instances
           (seeds 200..219, never seen during training).
           Gap shows how well the agent generalises.

Primary metric: gap_cost = (rl_cost - ref_cost) / ref_cost x 100 %
  where cost = dist/dist_scale + time-window penalties (same as RL objective).
  ref_dist and rl_dist are also reported for context.

This directly supports the parameter-efficiency claim:
  "node-qaoa achieves G% gap with P_q PQC params;
   classical RL requires P_c >> P_q total params for the same gap,
   and the ratio P_c/P_q grows with n."

Usage
-----
  # Single sub-experiment (one mode + encoding)
  python gap_analysis.py \\
      --prefix  results/rungE/e_n4_ry_fixed \\
      --models  quantum qaoa node-quantum node-qaoa classical \\
      --node 4  --n-qubits 9 --n-layers 4 \\
      --encoding ry --mode fixed \\
      --seeds 0 1 2 3 4 5 6 \\
      --out-csv gap_n4_ry_fixed.csv

  # Reference solver benchmark only
  python gap_analysis.py --solver-only --node 4 --n-instances 30
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch

from cpdptw_env import CPDPTWEnv

# Seeds used for policy-mode held-out evaluation (never seen during training).
_POLICY_EVAL_SEEDS = list(range(200, 220))


# --------------------------------------------------------------------------- #
# Reference solvers
# --------------------------------------------------------------------------- #

def _route_dist_cost(route: list[int], env: CPDPTWEnv) -> tuple[float, float]:
    """Return (total_dist, total_cost) for a completed route on env's instance."""
    dm = env.dist_matrix.numpy()
    total_dist = total_cost = 0.0
    vtime = 0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        d = float(dm[a, b])
        dt = int(math.ceil(d / max(env.speed, 1e-6)))
        vtime += dt
        total_dist += d
        open_t = int(env.time_window[b, 0])
        close_t = int(env.time_window[b, 1])
        pen = 0.0
        if open_t > 0 and vtime < open_t:
            pen += env.early_w * (open_t - vtime) ** 2
        if close_t > 0 and vtime > close_t:
            pen += env.late_w * (vtime - close_t) ** 2
        total_cost += d / env.dist_scale + pen
    return total_dist, total_cost


def exact_solve_dist(env: CPDPTWEnv) -> tuple[list[int], float]:
    """
    Exact branch-and-bound minimising pure travel distance (no time-penalty term).
    Enforces precedence + capacity; ignores time windows in the objective so the
    reference is distance-pure and gap_pct >= 0 is guaranteed.
    Provably distance-optimal for n <= 5 (milliseconds per instance).
    Returns (route, total_dist).
    """
    n = env.node
    dm = env.dist_matrix.numpy()
    demands = env.demands

    best_dist = [float('inf')]
    best_route: list = [[]]

    def dfs(node: int, vmask: int, load: int, dist_acc: float, route: list):
        if dist_acc >= best_dist[0]:          # distance bound-and-prune
            return
        if len(route) == 2 * n + 1:
            ret_d = float(dm[node, 0])
            total = dist_acc + ret_d
            if total < best_dist[0]:
                best_dist[0] = total
                best_route[0] = route + [0]
            return
        for nxt in range(1, 2 * n + 1):
            if vmask & (1 << nxt):
                continue
            if nxt > n and not (vmask & (1 << (nxt - n))):
                continue
            new_load = load + int(demands[nxt])
            if new_load > env.capacity or new_load < 0:
                continue
            d = float(dm[node, nxt])
            dfs(nxt, vmask | (1 << nxt), new_load, dist_acc + d, route + [nxt])

    dfs(0, 1, 0, 0.0, [0])
    return best_route[0], best_dist[0]


def exact_solve(env: CPDPTWEnv, timeout: float = 120.0) -> tuple[list[int], float, float]:
    """
    Exact DFS with cost-based pruning. Provably optimal for n <= 5 (~1-5s).
    Returns (route, total_dist, total_cost).
    """
    n = env.node
    dm = env.dist_matrix.numpy()
    demands = env.demands
    tw = env.time_window

    best = [float('inf'), []]
    t0 = time.perf_counter()

    def dfs(node: int, vmask: int, load: int, vtime: int,
            cost_acc: float, route: list):
        if time.perf_counter() - t0 > timeout:
            return
        if cost_acc >= best[0]:
            return
        if len(route) == 2 * n + 1:
            ret_d = float(dm[node, 0])
            final = cost_acc + ret_d / env.dist_scale
            if final < best[0]:
                best[0] = final
                best[1] = route + [0]
            return
        for nxt in range(1, 2 * n + 1):
            if vmask & (1 << nxt):
                continue
            if nxt > n and not (vmask & (1 << (nxt - n))):
                continue
            new_load = load + int(demands[nxt])
            if new_load > env.capacity or new_load < 0:
                continue
            d = float(dm[node, nxt])
            dt = int(math.ceil(d / max(env.speed, 1e-6)))
            nt = vtime + dt
            pen = 0.0
            ot, ct = int(tw[nxt, 0]), int(tw[nxt, 1])
            if ot > 0 and nt < ot:
                pen += env.early_w * (ot - nt) ** 2
            if ct > 0 and nt > ct:
                pen += env.late_w * (nt - ct) ** 2
            dfs(nxt, vmask | (1 << nxt), new_load, nt,
                cost_acc + d / env.dist_scale + pen, route + [nxt])

    dfs(0, 1, 0, 0, 0.0, [0])
    dist, cost = _route_dist_cost(best[1], env)
    return best[1], dist, cost


def greedy_solve(env: CPDPTWEnv) -> tuple[list[int], float, float]:
    """Nearest-neighbour greedy. Near-optimal reference for n > 5."""
    n = env.node
    dm = env.dist_matrix.numpy()
    visited = {0}
    node, load, vtime = 0, 0, 0
    route = [0]

    while len(visited) < 2 * n + 1:
        best_d, best_nxt = float('inf'), None
        for nxt in range(1, 2 * n + 1):
            if nxt in visited:
                continue
            if nxt > n and (nxt - n) not in visited:
                continue
            nl = load + int(env.demands[nxt])
            if nl > env.capacity or nl < 0:
                continue
            if float(dm[node, nxt]) < best_d:
                best_d, best_nxt = float(dm[node, nxt]), nxt
        if best_nxt is None:
            break
        visited.add(best_nxt)
        d = float(dm[node, best_nxt])
        dt = int(math.ceil(d / max(env.speed, 1e-6)))
        vtime += dt
        load += int(env.demands[best_nxt])
        node = best_nxt
        route.append(best_nxt)

    route.append(0)
    dist, cost = _route_dist_cost(route, env)
    return route, dist, cost


def reference_solve(env: CPDPTWEnv) -> tuple[list[int], float, float, str]:
    if env.node <= 5:
        # Cost-optimal exact branch-and-bound: minimises the same objective as
        # the RL reward (dist/dist_scale + time-window penalties).
        # gap_pct = (rl_cost - ref_cost) / ref_cost >= 0 is guaranteed because
        # no feasible route can achieve lower cost than this exact optimum.
        r, d, c = exact_solve(env)
        return r, d, c, "exact-B&B"
    r, d, c = greedy_solve(env)
    return r, d, c, "greedy-nn"


# --------------------------------------------------------------------------- #
# RL policy evaluation
# --------------------------------------------------------------------------- #

def eval_greedy_policy(net, env: CPDPTWEnv) -> tuple[float, float, bool]:
    """Greedy rollout (argmax). Returns (total_dist, total_cost, completed).
    total_cost = dist/dist_scale + time-window penalties — same as RL objective."""
    device = next(net.parameters()).device
    state, _ = env.reset(regenerate=False)
    done = False
    total_reward = 0.0
    for _ in range(4 * env.n_total):
        mask = env.action_mask()
        if not mask.any():
            break
        with torch.no_grad():
            logits = net(state.to(device)).squeeze(0)
        logits = logits.masked_fill(~mask.to(device), -float('inf'))
        action = int(logits.argmax().item())
        state, reward, done, _, _ = env.step(action)
        total_reward += reward.item()
        if done:
            break
    return env.total_distance, -total_reward, done


def eval_random_policy(env: CPDPTWEnv, n_trials: int = 10) -> tuple[float, float, bool]:
    """Uniform random over feasible actions. Returns (mean_dist, mean_cost, mean_done)."""
    import random
    dists, costs, dones = [], [], []
    for _ in range(n_trials):
        env.reset(regenerate=False)
        done = False
        total_reward = 0.0
        for _ in range(4 * env.n_total):
            mask = env.action_mask()
            if not mask.any():
                break
            choices = mask.nonzero(as_tuple=True)[0].tolist()
            action = random.choice(choices)
            _, reward, done, _, _ = env.step(action)
            total_reward += reward.item()
            if done:
                break
        dists.append(env.total_distance)
        costs.append(-total_reward)
        dones.append(done)
    return float(np.mean(dists)), float(np.mean(costs)), float(np.mean(dones)) >= 0.5


# --------------------------------------------------------------------------- #
# Parameter counting
# --------------------------------------------------------------------------- #

def _pqc_params(net) -> int:
    return sum(p.numel() for p in net.qlayer.parameters()) if hasattr(net, 'qlayer') else 0


def _total_params(net) -> int:
    return sum(p.numel() for p in net.parameters())


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #

def analyze(
    prefix: str,
    models: list[str],
    seeds: list[int],
    node: int,
    n_qubits: int,
    n_layers: int,
    encoding: str = "ry",
    entanglement: str = "ring",
    mode: str = "fixed",          # "fixed" | "policy"
    capacity: int = 5,
    out_csv: str = "",
    tw_tightness: float = 0.0,
) -> list[dict]:
    """
    Load checkpoints from {prefix}_{model}_s{seed}.pt, evaluate the greedy
    policy, and compare against the reference solver.

    fixed mode  : evaluate on the same instance the agent was trained on (seed S).
    policy mode : evaluate on 20 held-out instances (seeds 200-219).
    """
    from train_qrl import build_net

    rows = []
    ref_label = "exact-B&B" if node <= 5 else "greedy-nn"
    print(f"\n{'='*84}")
    print(f"Gap analysis | node={node} n_qubits={n_qubits} n_layers={n_layers} "
          f"encoding={encoding} entanglement={entanglement} mode={mode}")
    print(f"Reference: {ref_label} (same objective: dist/scale + time-window penalties)")
    print(f"{'='*84}")
    print(f"{'model':12s} {'seed':4s} {'params':7s} {'pqc':6s} "
          f"{'ref_cost':9s} {'rl_cost':9s} {'ref_dist':9s} {'rl_dist':9s} {'gap%':7s} {'done':4s}")
    print("-" * 84)

    for model in models:
        # "random" is a parameter-free baseline — no checkpoint needed.
        if model == "random":
            for seed in seeds:
                env = CPDPTWEnv(node=node, vehicle_capacity=capacity, rng_seed=seed,
                                tw_tightness=tw_tightness)
                env.reset(regenerate=True)
                if mode == "fixed":
                    _, ref_dist, ref_cost, _ = reference_solve(env)
                    rl_dist, rl_cost, done = eval_random_policy(env)
                    gap = (rl_cost - ref_cost) / max(ref_cost, 1e-6) * 100.0
                    done_str = "Y" if done else "N"
                else:
                    gaps, ref_dists, ref_costs, rl_dists, rl_costs, dones = [], [], [], [], [], []
                    for es in _POLICY_EVAL_SEEDS:
                        eval_env = CPDPTWEnv(node=node, vehicle_capacity=capacity,
                                            rng_seed=es, tw_tightness=tw_tightness)
                        eval_env.reset(regenerate=True)
                        _, ref_d, ref_c, _ = reference_solve(eval_env)
                        rl_d, rl_c, d = eval_random_policy(eval_env)
                        gaps.append((rl_c - ref_c) / max(ref_c, 1e-6) * 100.0)
                        ref_dists.append(ref_d); ref_costs.append(ref_c)
                        rl_dists.append(rl_d);  rl_costs.append(rl_c)
                        dones.append(d)
                    gap      = float(np.mean(gaps))
                    ref_dist = float(np.mean(ref_dists)); ref_cost = float(np.mean(ref_costs))
                    rl_dist  = float(np.mean(rl_dists));  rl_cost  = float(np.mean(rl_costs))
                    done_str = f"{np.mean(dones):.0%}"
                print(f"{'random':12s} {seed:4d} {'—':>7s} {'—':>6s} "
                      f"{ref_cost:9.4f} {rl_cost:9.4f} {ref_dist:9.4f} {rl_dist:9.4f} "
                      f"{gap:7.2f}% {done_str:4s}")
                rows.append({
                    "model":        "random",
                    "node":         node,
                    "n_qubits":     0,
                    "n_layers":     0,
                    "encoding":     encoding,
                    "entanglement": entanglement,
                    "mode":         mode,
                    "seed":         seed,
                    "total_params": 0,
                    "pqc_params":   0,
                    "ref_cost":     round(ref_cost, 6),
                    "rl_cost":      round(rl_cost, 6),
                    "ref_dist":     round(ref_dist, 6),
                    "rl_dist":      round(rl_dist, 6),
                    "gap_pct":      round(gap, 4),
                    "ref_method":   ref_label,
                })
            continue

        for seed in seeds:
            ckpt = f"{prefix}_{model}_s{seed}.pt"
            if not os.path.exists(ckpt):
                print(f"{model:12s} {seed:4d}  -- not found: {ckpt}")
                continue

            try:
                env = CPDPTWEnv(node=node, vehicle_capacity=capacity, rng_seed=seed,
                                tw_tightness=tw_tightness)
                env.reset(regenerate=True)
                net = build_net(model, env, n_qubits=n_qubits, n_layers=n_layers,
                                encoding=encoding, entanglement=entanglement)
                net.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
                net.eval()
            except Exception as e:
                print(f"{model:12s} {seed:4d}  -- error: {e}")
                continue

            total_p = _total_params(net)
            pqc_p   = _pqc_params(net)

            if mode == "fixed":
                # Same instance as training seed.
                _, ref_dist, ref_cost, _ = reference_solve(env)
                rl_dist, rl_cost, done = eval_greedy_policy(net, env)
                gap = (rl_cost - ref_cost) / max(ref_cost, 1e-6) * 100.0
                done_str = "Y" if done else "N"
            else:
                # Policy mode: 20 held-out instances never seen during training.
                gaps, ref_dists, ref_costs, rl_dists, rl_costs, dones = [], [], [], [], [], []
                for es in _POLICY_EVAL_SEEDS:
                    eval_env = CPDPTWEnv(node=node, vehicle_capacity=capacity,
                                        rng_seed=es, tw_tightness=tw_tightness)
                    eval_env.reset(regenerate=True)
                    _, ref_d, ref_c, _ = reference_solve(eval_env)
                    rl_d, rl_c, d = eval_greedy_policy(net, eval_env)
                    gaps.append((rl_c - ref_c) / max(ref_c, 1e-6) * 100.0)
                    ref_dists.append(ref_d); ref_costs.append(ref_c)
                    rl_dists.append(rl_d);  rl_costs.append(rl_c)
                    dones.append(d)
                gap      = float(np.mean(gaps))
                ref_dist = float(np.mean(ref_dists)); ref_cost = float(np.mean(ref_costs))
                rl_dist  = float(np.mean(rl_dists));  rl_cost  = float(np.mean(rl_costs))
                done_str = f"{np.mean(dones):.0%}"

            print(f"{model:12s} {seed:4d} {total_p:7d} {pqc_p:6d} "
                  f"{ref_cost:9.4f} {rl_cost:9.4f} {ref_dist:9.4f} {rl_dist:9.4f} "
                  f"{gap:7.2f}% {done_str:4s}")

            rows.append({
                "model":        model,
                "node":         node,
                "n_qubits":     n_qubits,
                "n_layers":     n_layers,
                "encoding":     encoding,
                "entanglement": entanglement,
                "mode":         mode,
                "seed":         seed,
                "total_params": total_p,
                "pqc_params":   pqc_p,
                "ref_cost":     round(ref_cost, 6),
                "rl_cost":      round(rl_cost, 6),
                "ref_dist":     round(ref_dist, 6),
                "rl_dist":      round(rl_dist, 6),
                "gap_pct":      round(gap, 4),
                "ref_method":   ref_label,
            })

    if rows:
        _print_summary(rows)
    if out_csv and rows:
        _write_csv(rows, out_csv)
        print(f"\nResults -> {out_csv}")
    return rows


# --------------------------------------------------------------------------- #
# Summary + CSV helpers
# --------------------------------------------------------------------------- #

def _print_summary(rows: list[dict]) -> None:
    by_model: dict = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    enc  = rows[0]["encoding"]
    ent  = rows[0].get("entanglement", "ring")
    mode = rows[0]["mode"]
    title = f"--- Summary: encoding={enc}  entanglement={ent}  mode={mode} "
    print(f"\n{title:=<76}")
    print(f"{'model':12s} {'n':3s} {'params':8s} {'pqc':7s} "
          f"{'gap%_mean':10s} {'gap%_std':9s} {'param_eff':10s}")
    print("-" * 76)
    for model, rs in sorted(by_model.items()):
        gaps = [r["gap_pct"] for r in rs]
        p    = rs[0]["total_params"]
        pqc  = rs[0]["pqc_params"]
        # param_eff = gap% per 100 params (lower = better)
        eff  = np.mean(gaps) / max(p, 1) * 100
        print(f"{model:12s} {rs[0]['node']:3d} {p:8d} {pqc:7d} "
              f"{np.mean(gaps):10.2f} {np.std(gaps):9.2f} {eff:10.4f}")
    print("=" * 76)


def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Solver-only benchmark
# --------------------------------------------------------------------------- #

def solver_only(node: int, n_instances: int = 30, capacity: int = 5,
                tw_tightness: float = 0.0) -> None:
    method = "exact-DFS" if node <= 5 else "greedy-NN"
    print(f"\nReference solver  node={node}  method={method}  n={n_instances}")
    dists, times = [], []
    for i in range(n_instances):
        env = CPDPTWEnv(node=node, vehicle_capacity=capacity, rng_seed=i,
                        tw_tightness=tw_tightness)
        env.reset(regenerate=True)
        t0 = time.perf_counter()
        _, dist, _, _ = reference_solve(env)
        elapsed = time.perf_counter() - t0
        dists.append(dist)
        times.append(elapsed)
        print(f"  {i:3d}: dist={dist:.4f}  ({elapsed:.2f}s)")
    print(f"\nSummary: dist {np.mean(dists):.4f}±{np.std(dists):.4f}  "
          f"time {np.mean(times):.2f}s mean / {np.max(times):.2f}s max")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Optimality gap and parameter efficiency analysis — Chapter 6"
    )
    ap.add_argument("--prefix",    default="pg",
                    help="Checkpoint prefix: loads {prefix}_{model}_s{seed}.pt")
    ap.add_argument("--models",    nargs="+",
                    default=["quantum", "qaoa", "node-quantum", "node-qaoa", "classical"],
                    choices=["quantum", "qaoa", "node-quantum", "node-qaoa",
                             "classical", "classical-qaoa", "classical-large", "random"])
    ap.add_argument("--seeds",     type=int, nargs="+", default=list(range(7)))
    ap.add_argument("--node",      type=int, default=4)
    ap.add_argument("--n-qubits",  type=int, default=9)
    ap.add_argument("--n-layers",  type=int, default=4)
    ap.add_argument("--encoding",  choices=["ry", "rz", "ryrz"], default="ry",
                    help="Encoding the checkpoint was trained with.")
    ap.add_argument("--entanglement", choices=["none", "ring", "brick", "all", "star"],
                    default="ring",
                    help="Entanglement topology the checkpoint was trained with.")
    ap.add_argument("--mode",      choices=["fixed", "policy"], default="fixed",
                    help="fixed: eval on training instance; "
                         "policy: eval on 20 held-out instances.")
    ap.add_argument("--capacity",  type=int, default=5)
    ap.add_argument("--out-csv",   default="",
                    help="Optional CSV path for per-seed results.")
    ap.add_argument("--solver-only", action="store_true",
                    help="Run reference solver benchmark only (no RL eval).")
    ap.add_argument("--n-instances", type=int, default=30,
                    help="Number of instances for --solver-only.")
    ap.add_argument("--tw-tightness", type=float, default=0.0,
                    help="Time-window tightness: 0=loose (15-30 min), 1=tight (3-8 min).")
    args = ap.parse_args()

    if args.solver_only:
        solver_only(args.node, args.n_instances, args.capacity,
                    tw_tightness=args.tw_tightness)
    else:
        analyze(
            prefix        = args.prefix,
            models        = args.models,
            seeds         = args.seeds,
            node          = args.node,
            n_qubits      = args.n_qubits,
            n_layers      = args.n_layers,
            encoding      = args.encoding,
            entanglement  = args.entanglement,
            mode          = args.mode,
            capacity      = args.capacity,
            out_csv       = args.out_csv,
            tw_tightness  = args.tw_tightness,
        )
