"""
route_validator.py

Three tools for Chapter 6's constraint and quality analysis:

  1. greedy_rollout   — load a trained checkpoint, produce a deterministic route
                        (no exploration) and report per-step constraint compliance.

  2. validate_route   — verify every CPDPTW constraint on any given route and
                        return a structured violation report.

  3. exact_solve      — find the true optimal route via depth-first search with
                        constraint pruning.  Exact for n <= 5 (10 nodes) in < 1 s.
                        For n > 6 substitute OR-Tools routing or Gurobi MIP.

  4. compare          — load checkpoints for all trained models, run each greedy,
                        run exact, and print a comparison table.

Constraint summary
------------------
Hard (structural — enforced by action mask, NOT by the network):
  * Capacity:   load in [0, Q] at every step
  * Precedence: delivery i+n only after pickup i

Soft (learned — enforced via reward penalties):
  * Pickup earliness : early_w * (open_t - t_arrive)^2  if t_arrive < open_t
  * Delivery lateness: late_w * (t_arrive - close_t)^2  if t_arrive > close_t

The validate_route function checks BOTH hard and soft constraints so we can
distinguish "the mask works" (hard, should always pass) from "the policy learned
good timing" (soft, improves with training).

Usage
-----
    # Compare all four models on the default n=5 instance
    python route_validator.py

    # Single checkpoint
    python route_validator.py --checkpoint qrl_quantum.pt --model quantum

    # Smaller instance (faster exact solve, good for sanity checks)
    python route_validator.py --node 3
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import numpy as np
import torch

from cpdptw_env import CPDPTWEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Checkpoint loader
# --------------------------------------------------------------------------- #

def load_checkpoint(
    path: str,
    model_kind: str,
    env: CPDPTWEnv,
    n_qubits: int = 6,
    n_layers: int = 3,
) -> torch.nn.Module:
    from train_qrl import build_net
    net = build_net(model_kind, env, n_qubits, n_layers).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net


# --------------------------------------------------------------------------- #
# Greedy rollout
# --------------------------------------------------------------------------- #

def greedy_rollout(
    net: torch.nn.Module,
    env: CPDPTWEnv,
    algo: str = "dqn",
) -> tuple[list[int], float, float]:
    """
    Run one deterministic episode using the trained policy.

    DQN:       argmax of masked Q-values (eps=0).
    REINFORCE: argmax of masked softmax logits (mode of the policy).

    Returns (route, total_reward, total_distance).
    """
    state, _ = env.reset(regenerate=False)
    state = state.to(device)
    total_r = 0.0

    for _ in range(4 * env.n_total):
        mask = env.action_mask().to(device)
        if not mask.any():
            break
        with torch.no_grad():
            logits = net(state).squeeze(0)               # (n_actions,)
            logits = logits.masked_fill(~mask, -float("inf"))
            action = logits.argmax().item()

        state, reward, done, _, _ = env.step(action)
        state = state.to(device)
        total_r += reward.item()
        if done:
            break

    return env.get_route(), total_r, env.total_distance


# --------------------------------------------------------------------------- #
# Route validator
# --------------------------------------------------------------------------- #

def validate_route(env: CPDPTWEnv, route: list[int]) -> dict:
    """
    Check every CPDPTW constraint for the given route on the given env instance.

    Returns a dict:
      hard_ok       — bool, True if all hard constraints pass
      soft_ok       — bool, True if no time-window violations
      complete      — bool, all 2n nodes visited
      capacity_viol — list of strings describing capacity violations
      precedence_viol — list of strings describing precedence violations
      early_viol    — list of (node, arrive_t, open_t) tuples
      late_viol     — list of (node, arrive_t, close_t) tuples
      time_penalty  — total quadratic time-window cost (0.0 = perfect)
      total_dist    — closed-tour distance (including return to depot)
    """
    n = env.node
    capacity = env.capacity
    speed = env.speed
    demands = env.demands
    tw = env.time_window
    dm = env.dist_matrix
    early_w = env.early_w
    late_w = env.late_w
    dist_scale = env.dist_scale

    cap_viol = []
    prec_viol = []
    early_viol = []
    late_viol = []
    time_pen = 0.0
    total_dist = 0.0

    load = 0
    time = 0
    visited_pickups: set[int] = set()
    visited_nodes: set[int] = set(route)

    complete = all(i in visited_nodes for i in range(1, 2 * n + 1))

    for idx in range(len(route) - 1):
        src = route[idx]
        dst = route[idx + 1]

        dist = float(dm[src, dst].item())
        total_dist += dist
        dt = int(np.ceil(dist / max(speed, 1e-6)))
        time += dt

        if dst == 0:
            continue                        # return-to-depot leg

        # Precedence
        if dst > n and (dst - n) not in visited_pickups:
            prec_viol.append(f"node {dst}: delivery before pickup {dst - n}")

        # Capacity
        load += int(demands[dst])
        if load > capacity:
            cap_viol.append(f"node {dst}: load={load} > Q={capacity}")
        if load < 0:
            cap_viol.append(f"node {dst}: load={load} < 0")

        # Time windows
        open_t, close_t = int(tw[dst, 0]), int(tw[dst, 1])
        if open_t > 0 and time < open_t:
            early_viol.append((dst, time, open_t))
            time_pen += early_w * (open_t - time) ** 2
        if close_t > 0 and time > close_t:
            late_viol.append((dst, time, close_t))
            time_pen += late_w * (time - close_t) ** 2

        if dst <= n:
            visited_pickups.add(dst)

    hard_ok = (not cap_viol) and (not prec_viol) and complete
    soft_ok = (not early_viol) and (not late_viol)

    return {
        "hard_ok":        hard_ok,
        "soft_ok":        soft_ok,
        "complete":       complete,
        "capacity_viol":  cap_viol,
        "precedence_viol": prec_viol,
        "early_viol":     early_viol,
        "late_viol":      late_viol,
        "time_penalty":   time_pen,
        "total_dist":     total_dist,
    }


# --------------------------------------------------------------------------- #
# Exact solver (depth-first search with constraint pruning)
# --------------------------------------------------------------------------- #

def exact_solve(env: CPDPTWEnv) -> tuple[Optional[list[int]], float, float]:
    """
    Find the globally optimal route for the given env instance by exhaustive
    depth-first search with constraint pruning.

    Pruning applied at each branch:
      * Already-visited nodes skipped.
      * Delivery blocked until its pickup is visited (precedence).
      * Capacity check before expanding.

    Exact for n <= 5 (10 delivery nodes) — typically runs in milliseconds.
    Returns (route, reward, total_distance) or (None, -inf, inf) if infeasible.
    """
    n = env.node
    capacity = env.capacity
    speed = env.speed
    demands = env.demands
    tw = env.time_window
    dm = env.dist_matrix
    dist_scale = env.dist_scale
    early_w = env.early_w
    late_w = env.late_w
    n_nodes = 2 * n

    best: dict = {"reward": -math.inf, "route": None, "dist": math.inf}

    def step_cost(src: int, dst: int, arrive_t: int) -> float:
        dist = float(dm[src, dst].item())
        c = dist / dist_scale
        open_t, close_t = int(tw[dst, 0]), int(tw[dst, 1])
        if open_t > 0 and arrive_t < open_t:
            c += early_w * (open_t - arrive_t) ** 2
        if close_t > 0 and arrive_t > close_t:
            c += late_w * (arrive_t - close_t) ** 2
        return c

    def dfs(
        current: int,
        visited: frozenset[int],
        load: int,
        time: int,
        route: list[int],
        cost: float,
        dist: float,
    ) -> None:
        if len(visited) == n_nodes:
            # Final leg: return to depot.
            d = float(dm[current, 0].item())
            total_cost = cost + d / dist_scale
            total_dist = dist + d
            reward = -total_cost
            if reward > best["reward"]:
                best["reward"] = reward
                best["route"] = route + [0]
                best["dist"] = total_dist
            return

        for nxt in range(1, n_nodes + 1):
            if nxt in visited:
                continue
            if nxt > n and (nxt - n) not in visited:
                continue                       # precedence
            new_load = load + int(demands[nxt])
            if new_load > capacity or new_load < 0:
                continue                       # capacity

            d = float(dm[current, nxt].item())
            dt = int(np.ceil(d / max(speed, 1e-6)))
            arrive_t = time + dt
            c = step_cost(current, nxt, arrive_t)

            dfs(
                nxt,
                visited | {nxt},
                new_load,
                arrive_t,
                route + [nxt],
                cost + c,
                dist + d,
            )

    dfs(0, frozenset(), 0, 0, [0], 0.0, 0.0)

    if best["route"] is None:
        return None, -math.inf, math.inf
    return best["route"], best["reward"], best["dist"]


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #

def _route_cost(env: CPDPTWEnv, route: list[int]) -> float:
    """Re-compute the RL reward for any route (same formula as env.step)."""
    n = env.node
    speed = env.speed
    demands = env.demands
    tw = env.time_window
    dm = env.dist_matrix
    dist_scale = env.dist_scale
    early_w = env.early_w
    late_w = env.late_w

    cost = 0.0
    time = 0
    for i in range(len(route) - 1):
        src, dst = route[i], route[i + 1]
        d = float(dm[src, dst].item())
        dt = int(np.ceil(d / max(speed, 1e-6)))
        time += dt
        cost += d / dist_scale
        if dst == 0:
            continue
        open_t, close_t = int(tw[dst, 0]), int(tw[dst, 1])
        if open_t > 0 and time < open_t:
            cost += early_w * (open_t - time) ** 2
        if close_t > 0 and time > close_t:
            cost += late_w * (time - close_t) ** 2
    return -cost


def compare(
    node: int = 5,
    capacity: int = 5,
    seed: int = 0,
    n_qubits: int = 6,
    n_layers: int = 3,
    checkpoints: Optional[dict[str, str]] = None,
) -> None:
    """
    Load trained checkpoints, run greedy rollout for each, validate constraints,
    and compare against the exact optimum.

    checkpoints: dict mapping model_kind -> path, e.g.
      {"quantum": "qrl_quantum.pt", "classical": "qrl_classical.pt"}
    If None, auto-discovers files matching the default naming convention.
    """
    env = CPDPTWEnv(node=node, vehicle_capacity=capacity, rng_seed=seed)
    env.reset(regenerate=True)

    # Auto-discover checkpoints if not provided.
    if checkpoints is None:
        candidates = {
            "quantum":        "qrl_quantum.pt",
            "qaoa":           "qrl_qaoa.pt",
            "classical":      "qrl_classical.pt",
            "classical-qaoa": "qrl_classical-qaoa.pt",
        }
        checkpoints = {k: v for k, v in candidates.items() if os.path.exists(v)}
        reinforce_candidates = {
            "reinforce_quantum":   "reinforce_quantum.pt",
            "reinforce_classical": "reinforce_classical.pt",
        }
        for k, v in reinforce_candidates.items():
            if os.path.exists(v):
                checkpoints[k] = v

    print(f"\nCPDPTW n={node}, Q={capacity}, seed={seed}")
    print(f"Instance: {node} request pairs, {2*node} delivery nodes\n")

    # Exact optimal.
    print("Running exact DFS solver...", end=" ", flush=True)
    opt_route, opt_reward, opt_dist = exact_solve(env)
    print("done.")
    if opt_route is None:
        print("No feasible route found by exact solver.")
        return

    opt_val = validate_route(env, opt_route)
    print(f"Exact optimal: reward={opt_reward:.4f}  dist={opt_dist:.4f}  "
          f"hard={'PASS' if opt_val['hard_ok'] else 'FAIL'}  "
          f"time_pen={opt_val['time_penalty']:.4f}")
    print(f"  route: {opt_route}\n")

    # Header.
    print(f"{'model':20s} {'reward':>9s} {'vs opt':>7s} {'dist':>8s} "
          f"{'hard':>6s} {'time_pen':>10s} {'late_viol':>9s} {'route'}")
    print("-" * 100)

    # Print exact row.
    print(f"{'exact (DFS)':20s} {opt_reward:9.4f} {'--':>7s} {opt_dist:8.4f} "
          f"{'PASS':>6s} {opt_val['time_penalty']:10.4f} "
          f"{len(opt_val['late_viol']):9d}  {opt_route}")

    for model_kind, ckpt_path in checkpoints.items():
        # Determine algo from name prefix.
        algo = "reinforce" if model_kind.startswith("reinforce") else "dqn"
        actual_kind = model_kind.replace("reinforce_", "")

        try:
            net = load_checkpoint(ckpt_path, actual_kind, env, n_qubits, n_layers)
        except Exception as e:
            print(f"{'  ' + model_kind:20s}  ERROR loading: {e}")
            continue

        env.reset(regenerate=False)     # same instance as exact solver
        route, reward, dist = greedy_rollout(net, env, algo=algo)
        val = validate_route(env, route)
        gap = (reward - opt_reward) / abs(opt_reward) * 100 if opt_reward != 0 else 0.0

        print(f"{model_kind:20s} {reward:9.4f} {gap:+6.1f}% {dist:8.4f} "
              f"{'PASS' if val['hard_ok'] else 'FAIL':>6s} "
              f"{val['time_penalty']:10.4f} "
              f"{len(val['late_viol']):9d}  {route}")

    print("-" * 100)
    print("vs opt: gap to exact optimal (0% = optimal, negative = worse)")
    print("hard:   capacity + precedence + completeness constraints")
    print("time_pen: sum of quadratic time-window penalties (lower is better)")
    print("late_viol: delivery nodes where vehicle arrived after deadline")


# --------------------------------------------------------------------------- #
# Detailed single-route report
# --------------------------------------------------------------------------- #

def report_route(env: CPDPTWEnv, route: list[int], label: str = "route") -> None:
    """Print a step-by-step constraint trace for one route."""
    n = env.node
    speed = env.speed
    demands = env.demands
    tw = env.time_window
    dm = env.dist_matrix

    print(f"\n--- {label} ---")
    print(f"route: {route}")
    load = 0
    time = 0
    for idx in range(len(route) - 1):
        src, dst = route[idx], route[idx + 1]
        d = float(dm[src, dst].item())
        dt = int(np.ceil(d / max(speed, 1e-6)))
        time += dt

        if dst == 0:
            print(f"  step {idx+1:2d}: {src:2d} -> {dst:2d}  "
                  f"dist={d:.3f}  t={time}  (return to depot)")
            continue

        load += int(demands[dst])
        open_t, close_t = int(tw[dst, 0]), int(tw[dst, 1])
        kind = "pickup  " if dst <= n else "delivery"

        tw_status = "ok"
        if open_t > 0 and time < open_t:
            tw_status = f"EARLY by {open_t - time}"
        elif close_t > 0 and time > close_t:
            tw_status = f"LATE  by {time - close_t}"

        print(f"  step {idx+1:2d}: {src:2d} -> {dst:2d}  {kind}  "
              f"dist={d:.3f}  t={time}  load={load}  "
              f"tw=[{open_t},{close_t}]  {tw_status}")
    print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CPDPTW route validator and exact comparison")
    ap.add_argument("--node",        type=int, default=5)
    ap.add_argument("--capacity",    type=int, default=5)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--n-qubits",    type=int, default=6)
    ap.add_argument("--n-layers",    type=int, default=3)
    ap.add_argument("--checkpoint",  type=str, default=None,
                    help="Path to a single .pt checkpoint to evaluate.")
    ap.add_argument("--model",       type=str, default="quantum",
                    choices=["quantum", "qaoa", "classical", "classical-qaoa",
                             "node-quantum", "node-qaoa"],
                    help="Model kind for --checkpoint.")
    ap.add_argument("--verbose",     action="store_true",
                    help="Print step-by-step constraint trace.")
    args = ap.parse_args()

    env = CPDPTWEnv(node=args.node, vehicle_capacity=args.capacity,
                    rng_seed=args.seed)
    env.reset(regenerate=True)

    if args.checkpoint:
        # Single checkpoint mode.
        net = load_checkpoint(args.checkpoint, args.model, env,
                              args.n_qubits, args.n_layers)
        env.reset(regenerate=False)
        route, reward, dist = greedy_rollout(net, env)
        val = validate_route(env, route)
        print(f"\nModel:  {args.model}  ({args.checkpoint})")
        print(f"Route:  {route}")
        print(f"Reward: {reward:.4f}   Dist: {dist:.4f}")
        print(f"Hard constraints: {'PASS' if val['hard_ok'] else 'FAIL'}")
        print(f"  capacity violations:  {val['capacity_viol'] or 'none'}")
        print(f"  precedence violations:{val['precedence_viol'] or 'none'}")
        print(f"  complete:             {val['complete']}")
        print(f"Time-window penalty: {val['time_penalty']:.4f}")
        print(f"  early arrivals: {val['early_viol'] or 'none'}")
        print(f"  late arrivals:  {val['late_viol'] or 'none'}")

        if args.verbose:
            report_route(env, route, label=f"{args.model} greedy")

        # Also show exact for reference.
        print("\nRunning exact DFS...", end=" ", flush=True)
        env.reset(regenerate=False)
        opt_route, opt_reward, opt_dist = exact_solve(env)
        print("done.")
        print(f"Exact: {opt_route}  reward={opt_reward:.4f}  dist={opt_dist:.4f}")
        gap = (reward - opt_reward) / abs(opt_reward) * 100 if opt_reward != 0 else 0.0
        print(f"Gap to optimal: {gap:+.1f}%")

        if args.verbose and opt_route:
            env.reset(regenerate=False)
            report_route(env, opt_route, label="exact optimal")
    else:
        # Full comparison mode.
        compare(node=args.node, capacity=args.capacity, seed=args.seed,
                n_qubits=args.n_qubits, n_layers=args.n_layers)
