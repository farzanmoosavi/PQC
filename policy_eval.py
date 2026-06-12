"""
policy_eval.py

Policy generalisation evaluation for the CPDPTW PQC chapter.

Trains a model with fixed_instance=False (policy learning — sees a different
random instance each episode) then evaluates it on a set of held-out seeds that
were never seen during training.  Compares against:
  - route-learning baseline (fixed_instance=True, trained on a single instance)
  - exact DFS solver (for gap computation)

Usage
-----
    # Quick sanity-check (20 train eps, 5 held-out seeds)
    python policy_eval.py --model quantum --quick

    # Full run
    python policy_eval.py --model quantum --train-episodes 300 --eval-seeds 20

    # Compare all models
    python policy_eval.py --compare --quick
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

import numpy as np
import torch

from cpdptw_env import CPDPTWEnv
from train_qrl import build_net, train
from route_validator import exact_solve


# --------------------------------------------------------------------------- #
# Greedy rollout on a given env/net pair
# --------------------------------------------------------------------------- #

def greedy_rollout(net, env, max_steps: Optional[int] = None) -> tuple[list[int], float, float, bool]:
    """
    Roll out the greedy policy on env with the given net.

    Returns
    -------
    route       : list of node indices visited (including depot at start/end)
    total_reward: accumulated RL reward (includes time-window penalties)
    total_dist  : total travel distance (as reported by env.total_distance)
    done        : True if all nodes were visited
    """
    state, _ = env.reset(regenerate=False)
    device = next(net.parameters()).device
    state = state.to(device)
    done = False
    n_steps = 0
    total_reward = 0.0
    limit = max_steps or 4 * env.n_total

    with torch.no_grad():
        while not done and n_steps < limit:
            mask = env.action_mask().to(device)
            if not mask.any():
                break
            q = net(state)
            q = q.masked_fill(~mask.unsqueeze(0), -float("inf"))
            action = int(q.max(1).indices.item())
            state, reward, done, _, _ = env.step(action)
            state = state.to(device)
            total_reward += reward.item()
            n_steps += 1

    return list(env.route), total_reward, env.total_distance, done


# --------------------------------------------------------------------------- #
# Evaluate a trained net on held-out seeds
# --------------------------------------------------------------------------- #

def evaluate_on_seeds(
    net,
    node: int,
    capacity: int,
    eval_seeds: list[int],
    exact: bool = True,
) -> dict:
    """
    Roll out greedy policy on each seed instance.  Optionally compute gap to exact.

    Returns
    -------
    dict with keys:
        rewards   : list[float]
        dists     : list[float]
        feas_rate : float  (fraction of seeds where all nodes were visited)
        dist_mean : float
        dist_std  : float
        gap_mean  : float  (% above exact; 0 if exact not computed)
        gap_std   : float
    """
    dists, rewards_list, gaps = [], [], []
    n_done = 0

    for seed in eval_seeds:
        env = CPDPTWEnv(node=node, vehicle_capacity=capacity, rng_seed=seed)
        env.reset(regenerate=True)

        if exact and node <= 5:
            _, opt_reward, opt_dist = exact_solve(env)
            env.reset(regenerate=False)
        else:
            opt_reward = None
            opt_dist = None

        route, agent_reward, dist, done = greedy_rollout(net, env)
        n_done += int(done)
        dists.append(dist)
        rewards_list.append(agent_reward)

        if opt_reward is not None and opt_reward > -math.inf:
            # Gap: how far below optimal reward (positive = agent is worse)
            gaps.append((opt_reward - agent_reward) / max(abs(opt_reward), 1e-9) * 100.0)

    feas = n_done / len(eval_seeds)

    return {
        "rewards":   rewards_list,
        "dists":     dists,
        "feas_rate": feas,
        "dist_mean": float(np.mean(dists)),
        "dist_std":  float(np.std(dists)),
        "gap_mean":  float(np.mean(gaps)) if gaps else 0.0,
        "gap_std":   float(np.std(gaps))  if gaps else 0.0,
    }


# --------------------------------------------------------------------------- #
# Train + evaluate one configuration
# --------------------------------------------------------------------------- #

def train_and_evaluate(
    model_kind: str,
    node: int = 5,
    capacity: int = 5,
    train_episodes: int = 300,
    eval_seeds: Optional[list[int]] = None,
    train_seed: int = 42,
    n_qubits: int = 6,
    n_layers: int = 3,
    fixed_instance: bool = False,  # False = policy learning
    out_prefix: str = "policy_eval",
) -> dict:
    """
    Train a model then evaluate on held-out seeds.

    fixed_instance=False  -> policy learning (generalisation)
    fixed_instance=True   -> route learning (memorisation baseline)
    """
    if eval_seeds is None:
        # Use seeds distinct from training seeds
        eval_seeds = list(range(100, 120))

    label = "policy" if not fixed_instance else "route"
    print(f"\n=== {model_kind} ({label} learning) ===")
    print(f"  train_episodes={train_episodes}, node={node}, "
          f"n_qubits={n_qubits}, n_layers={n_layers}")
    print(f"  eval on {len(eval_seeds)} held-out seeds: {eval_seeds[:3]}...")

    result = train(
        model_kind=model_kind,
        node=node,
        capacity=capacity,
        episodes=train_episodes,
        seed=train_seed,
        fixed_instance=fixed_instance,
        out_prefix=f"{out_prefix}_{label}_{model_kind}",
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    net = result["net"]
    net.eval()

    metrics = evaluate_on_seeds(net, node, capacity, eval_seeds, exact=(node <= 5))
    metrics["model"] = model_kind
    metrics["mode"] = label
    metrics["train_episodes"] = train_episodes

    return metrics


# --------------------------------------------------------------------------- #
# Compare route-learning vs policy-learning for a single model
# --------------------------------------------------------------------------- #

def compare_modes(
    model_kind: str,
    node: int = 5,
    capacity: int = 5,
    train_episodes: int = 300,
    eval_seeds: Optional[list[int]] = None,
    n_qubits: int = 6,
    n_layers: int = 3,
) -> tuple[dict, dict]:
    """
    Train and evaluate both fixed_instance=True (route) and False (policy),
    return (route_metrics, policy_metrics).
    """
    route_m = train_and_evaluate(
        model_kind, node, capacity, train_episodes, eval_seeds,
        n_qubits=n_qubits, n_layers=n_layers, fixed_instance=True,
        out_prefix="cmp",
    )
    policy_m = train_and_evaluate(
        model_kind, node, capacity, train_episodes, eval_seeds,
        n_qubits=n_qubits, n_layers=n_layers, fixed_instance=False,
        out_prefix="cmp",
    )
    return route_m, policy_m


def _print_metrics(label: str, m: dict) -> None:
    print(f"\n  [{label}]")
    print(f"    dist  = {m['dist_mean']:.3f} +/- {m['dist_std']:.3f}")
    print(f"    feas  = {m['feas_rate']:.2f}")
    if m['gap_mean'] != 0.0:
        print(f"    gap   = {m['gap_mean']:.1f}% +/- {m['gap_std']:.1f}%")


def print_comparison(model_kind: str, route_m: dict, policy_m: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Model: {model_kind}")
    print(f"  Route learning  (fixed_instance=True):  single-instance memorisation")
    print(f"  Policy learning (fixed_instance=False): generalises to new instances")
    print(f"{'='*60}")
    _print_metrics("route  learning", route_m)
    _print_metrics("policy learning", policy_m)
    print()
    delta = policy_m["dist_mean"] - route_m["dist_mean"]
    sign = "+" if delta > 0 else ""
    print(f"  -> Policy vs route: dist delta = {sign}{delta:.3f}  "
          f"(positive = policy is worse, expected for fewer training steps)")
    feas_delta = policy_m["feas_rate"] - route_m["feas_rate"]
    sign2 = "+" if feas_delta > 0 else ""
    print(f"  -> Feasibility delta = {sign2}{feas_delta:.2f}")


# --------------------------------------------------------------------------- #
# Multi-model comparison sweep
# --------------------------------------------------------------------------- #

def full_compare(
    models: list[str],
    node: int = 5,
    capacity: int = 5,
    train_episodes: int = 300,
    eval_seeds: Optional[list[int]] = None,
    n_qubits: int = 6,
    n_layers: int = 3,
) -> list[dict]:
    if eval_seeds is None:
        eval_seeds = list(range(100, 120))

    rows = []
    for model_kind in models:
        route_m, policy_m = compare_modes(
            model_kind, node, capacity, train_episodes, eval_seeds,
            n_qubits=n_qubits, n_layers=n_layers,
        )
        print_comparison(model_kind, route_m, policy_m)
        rows.extend([route_m, policy_m])
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Policy generalisation evaluation")
    ap.add_argument("--model", default="quantum",
                    choices=["quantum", "qaoa", "classical", "classical-qaoa",
                             "node-quantum", "node-qaoa", "classical-large"])
    ap.add_argument("--compare", action="store_true",
                    help="Compare route vs policy learning for the chosen model")
    ap.add_argument("--all-models", action="store_true",
                    help="Run compare for all model types")
    ap.add_argument("--node",            type=int, default=5)
    ap.add_argument("--capacity",        type=int, default=5)
    ap.add_argument("--train-episodes",  type=int, default=300)
    ap.add_argument("--eval-seeds",      type=int, default=20,
                    help="Number of held-out seeds for evaluation")
    ap.add_argument("--n-qubits",        type=int, default=6)
    ap.add_argument("--n-layers",        type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="20 train episodes, 5 eval seeds — fast sanity check")
    ap.add_argument("--out-prefix", default="policy_eval",
                    help="Output path prefix for generated .txt and .pt files.")
    args = ap.parse_args()

    if args.quick:
        args.train_episodes = 20
        args.eval_seeds = 5

    eval_seeds = list(range(100, 100 + args.eval_seeds))

    if args.all_models:
        # node-quantum/node-qaoa are REINFORCE-only; DQN comparison uses flat models.
        full_compare(
            models=["quantum", "qaoa", "classical"],
            node=args.node, capacity=args.capacity,
            train_episodes=args.train_episodes,
            eval_seeds=eval_seeds,
            n_qubits=args.n_qubits, n_layers=args.n_layers,
        )
    elif args.compare:
        route_m, policy_m = compare_modes(
            model_kind=args.model,
            node=args.node, capacity=args.capacity,
            train_episodes=args.train_episodes,
            eval_seeds=eval_seeds,
            n_qubits=args.n_qubits, n_layers=args.n_layers,
        )
        print_comparison(args.model, route_m, policy_m)
    else:
        m = train_and_evaluate(
            model_kind=args.model,
            node=args.node, capacity=args.capacity,
            train_episodes=args.train_episodes,
            eval_seeds=eval_seeds,
            n_qubits=args.n_qubits, n_layers=args.n_layers,
            fixed_instance=False,
            out_prefix=args.out_prefix,
        )
        print(f"\nPolicy evaluation ({args.model}):")
        _print_metrics("policy", m)
