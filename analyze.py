"""
analyze.py — Unified experiment analysis for Chapter 6.

Reads training metric files ({prefix}_{model}_s{seed}_{metric}.txt) and
optionally checkpoint files ({prefix}_{model}_s{seed}.pt) to produce one
aggregated CSV row per model with all metrics needed for the thesis:

  feasible_rate   — fraction of last-20% episodes where route completes
  final_reward    — mean reward in last 20% of training
  convergence_ep  — episode first reaching 80% of improvement from worst→best
  gap_pct         — (rl_cost - ref_cost)/ref_cost % (requires --gap flag)
  total_params    — total trainable parameter count
  pqc_params      — PQC-only parameter count (0 for classical models)

Usage
-----
  # Fast: training metrics only (no checkpoint loading)
  python analyze.py \\
      --prefix  results/exp_arch/arch_n3_tw0p0 \\
      --models  quantum qaoa node-quantum node-qaoa classical classical-large \\
      --seeds   0 1 2 3 4 5 6 \\
      --node 3 --tw-tightness 0.0 --algo dqn \\
      --out-csv results/exp_arch/arch_race.csv --append

  # With gap analysis (loads checkpoints, calls B&B solver — slower)
  python analyze.py ... --gap --n-qubits 9 --n-layers 4 --encoding ry
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Optional

import numpy as np

TAIL_FRAC = 0.20   # fraction of training episodes used for "final" statistics


# --------------------------------------------------------------------------- #
# Metric file helpers
# --------------------------------------------------------------------------- #

def _load_metric(prefix: str, model: str, seed: int, metric: str
                 ) -> Optional[np.ndarray]:
    path = f"{prefix}_{model}_s{seed}_{metric}.txt"
    if not os.path.exists(path):
        return None
    try:
        return np.atleast_1d(np.loadtxt(path))
    except Exception:
        return None


def _convergence_ep(curve: np.ndarray, threshold: float = 0.80) -> int:
    """First episode where reward crosses threshold × (max - min) span."""
    if len(curve) == 0:
        return 0
    worst, best = float(curve.min()), float(curve.max())
    span = best - worst
    if span < 1e-6:
        return len(curve)
    thr = worst + threshold * span
    for i, v in enumerate(curve):
        if v >= thr:
            return i
    return len(curve)


# --------------------------------------------------------------------------- #
# Gap analysis (optional — loads .pt checkpoints and calls reference solver)
# --------------------------------------------------------------------------- #

def _compute_gap(prefix: str, model: str, seed: int, node: int,
                 n_qubits: int, n_layers: int, encoding: str,
                 entanglement: str, capacity: int, tw_tightness: float,
                 mode: str) -> tuple[float, int, int]:
    """Return (gap_pct, total_params, pqc_params) for one checkpoint.
    Returns (nan, 0, 0) on any failure."""
    ckpt = f"{prefix}_{model}_s{seed}.pt"
    if not os.path.exists(ckpt):
        return float("nan"), 0, 0
    try:
        import torch
        from cpdptw_env import CPDPTWEnv
        from train_qrl import build_net
        from gap_analysis import reference_solve, eval_greedy_policy, _POLICY_EVAL_SEEDS

        env = CPDPTWEnv(node=node, vehicle_capacity=capacity,
                        rng_seed=seed, tw_tightness=tw_tightness)
        env.reset(regenerate=True)
        net = build_net(model, env, n_qubits=n_qubits, n_layers=n_layers,
                        encoding=encoding, entanglement=entanglement)
        net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        net.eval()

        total_p = sum(p.numel() for p in net.parameters())
        pqc_p   = sum(p.numel() for p in net.qlayer.parameters()) \
                  if hasattr(net, "qlayer") else 0

        if mode == "fixed":
            _, ref_dist, ref_cost, _ = reference_solve(env)
            _rl_dist, rl_cost, _ = eval_greedy_policy(net, env)
            gap = (rl_cost - ref_cost) / max(ref_cost, 1e-6) * 100.0
        else:
            gaps = []
            for es in _POLICY_EVAL_SEEDS:
                eval_env = CPDPTWEnv(node=node, vehicle_capacity=capacity,
                                     rng_seed=es, tw_tightness=tw_tightness)
                eval_env.reset(regenerate=True)
                _, _rd, ref_c, _ = reference_solve(eval_env)
                _, rl_c, _ = eval_greedy_policy(net, eval_env)
                gaps.append((rl_c - ref_c) / max(ref_c, 1e-6) * 100.0)
            gap = float(np.mean(gaps))

        return gap, total_p, pqc_p
    except Exception as e:
        print(f"  [gap] {model} s{seed}: {e}")
        return float("nan"), 0, 0


# --------------------------------------------------------------------------- #
# Core per-model aggregation
# --------------------------------------------------------------------------- #

def analyze_model(
    prefix: str,
    model: str,
    seeds: list[int],
    node: int,
    n_qubits: int,
    n_layers: int,
    encoding: str,
    entanglement: str,
    mode: str,
    capacity: int,
    tw_tightness: float,
    algo: str,
    tag: str,
    run_gap: bool,
) -> Optional[dict]:
    """Aggregate metrics across seeds for one model. Returns None if no data."""
    reward_curves, complete_curves, conv_eps = [], [], []

    for seed in seeds:
        rew = _load_metric(prefix, model, seed, "rewards")
        cmp = _load_metric(prefix, model, seed, "complete")
        if rew is None:
            continue
        reward_curves.append(rew)
        if cmp is not None:
            complete_curves.append(cmp)
        conv_eps.append(_convergence_ep(rew))

    if not reward_curves:
        print(f"  [skip] {model}: no files at {prefix}_{model}_s*_rewards.txt")
        return None

    min_len = min(len(c) for c in reward_curves)
    rewards_mat = np.stack([c[:min_len] for c in reward_curves])   # (S, T)
    mean_rewards = rewards_mat.mean(axis=0)

    tail = max(1, int(min_len * TAIL_FRAC))
    final_reward     = float(mean_rewards[-tail:].mean())
    final_reward_std = float(rewards_mat[:, -tail:].mean(axis=1).std())

    if complete_curves:
        min_len_c    = min(len(c) for c in complete_curves)
        complete_mat = np.stack([c[:min_len_c] for c in complete_curves])
        tail_c       = max(1, int(min_len_c * TAIL_FRAC))
        feasible_rate     = float(complete_mat[:, -tail_c:].mean())
        feasible_rate_std = float(complete_mat[:, -tail_c:].mean(axis=1).std())
    else:
        feasible_rate = feasible_rate_std = float("nan")

    conv_ep = int(np.mean(conv_eps)) if conv_eps else 0

    gap_vals, total_ps, pqc_ps = [], [], []
    if run_gap:
        for seed in seeds:
            g, tp, pp = _compute_gap(prefix, model, seed, node, n_qubits,
                                     n_layers, encoding, entanglement,
                                     capacity, tw_tightness, mode)
            if not np.isnan(g):
                gap_vals.append(g)
            if tp > 0:
                total_ps.append(tp)
                pqc_ps.append(pp)

    # When --gap not set, still read param count from first available checkpoint
    if not run_gap and not total_ps:
        ckpt_params = _try_param_count(prefix, model, seeds, node, n_qubits,
                                       n_layers, encoding, entanglement)
        if ckpt_params:
            total_ps = [ckpt_params[0]]
            pqc_ps   = [ckpt_params[1]]

    gap_pct     = float(np.mean(gap_vals))  if gap_vals else float("nan")
    gap_pct_std = float(np.std(gap_vals))   if gap_vals else float("nan")
    total_params = int(np.mean(total_ps))   if total_ps else 0
    pqc_params   = int(np.mean(pqc_ps))     if pqc_ps   else 0

    return {
        "tag":               tag,
        "algo":              algo,
        "model":             model,
        "node":              node,
        "n_layers":          n_layers,
        "encoding":          encoding,
        "entanglement":      entanglement,
        "tw_tightness":      tw_tightness,
        "mode":              mode,
        "n_seeds":           len(reward_curves),
        "feasible_rate":     _fmt(feasible_rate,     4),
        "feasible_rate_std": _fmt(feasible_rate_std, 4),
        "final_reward":      _fmt(final_reward,      3),
        "final_reward_std":  _fmt(final_reward_std,  3),
        "convergence_ep":    conv_ep,
        "gap_pct":           _fmt(gap_pct,     3),
        "gap_pct_std":       _fmt(gap_pct_std, 3),
        "total_params":      total_params,
        "pqc_params":        pqc_params,
    }


def _fmt(v: float, decimals: int) -> object:
    """Return rounded float or empty string for nan."""
    return "" if (isinstance(v, float) and np.isnan(v)) else round(v, decimals)


def _try_param_count(prefix: str, model: str, seeds: list[int],
                     node: int, n_qubits: int, n_layers: int,
                     encoding: str, entanglement: str) -> Optional[tuple[int, int]]:
    """Try to load first available checkpoint just for parameter counting."""
    for seed in seeds:
        ckpt = f"{prefix}_{model}_s{seed}.pt"
        if not os.path.exists(ckpt):
            continue
        try:
            import torch
            from cpdptw_env import CPDPTWEnv
            from train_qrl import build_net
            env = CPDPTWEnv(node=node, vehicle_capacity=5, rng_seed=seed)
            env.reset(regenerate=True)
            net = build_net(model, env, n_qubits=n_qubits, n_layers=n_layers,
                            encoding=encoding, entanglement=entanglement)
            net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            total_p = sum(p.numel() for p in net.parameters())
            pqc_p   = sum(p.numel() for p in net.qlayer.parameters()) \
                      if hasattr(net, "qlayer") else 0
            return total_p, pqc_p
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def analyze(
    prefix: str,
    models: list[str],
    seeds: list[int],
    node: int,
    n_qubits: int = 9,
    n_layers: int = 4,
    encoding: str = "ry",
    entanglement: str = "ring",
    mode: str = "fixed",
    capacity: int = 5,
    tw_tightness: float = 0.0,
    algo: str = "dqn",
    tag: str = "",
    run_gap: bool = False,
    out_csv: str = "",
    append: bool = False,
) -> list[dict]:
    if not tag:
        tag = os.path.basename(prefix)

    print(f"\n{'='*72}")
    print(f"analyze | tag={tag}  node={node}  tw={tw_tightness}  "
          f"algo={algo}  mode={mode}  enc={encoding}  ent={entanglement}")
    print(f"{'='*72}")

    rows = []
    for model in models:
        row = analyze_model(prefix, model, seeds, node, n_qubits, n_layers,
                            encoding, entanglement, mode, capacity, tw_tightness,
                            algo, tag, run_gap)
        if row is None:
            continue
        rows.append(row)
        print(f"  {model:<16} feasible={row['feasible_rate'] or 'n/a':>6}  "
              f"reward={row['final_reward']:>8}  "
              f"conv={row['convergence_ep']:>5d}  "
              f"params={row['total_params']:>6d}  "
              f"pqc={row['pqc_params']:>5d}  "
              f"gap%={row['gap_pct'] or 'n/a'}")

    if out_csv and rows:
        _write_csv(rows, out_csv, append=append)
    return rows


def _write_csv(rows: list[dict], path: str, append: bool = False) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not append or not os.path.exists(path) \
                   or os.path.getsize(path) == 0
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    action = "appended" if append else "written"
    print(f"  -> {path}  ({len(rows)} rows {action})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unified experiment analysis — Chapter 6")
    p.add_argument("--prefix",       required=True,
                   help="File prefix: loads {prefix}_{model}_s{seed}_*.txt")
    p.add_argument("--models",       nargs="+", required=True)
    p.add_argument("--seeds",        type=int, nargs="+", required=True)
    p.add_argument("--node",         type=int, default=4)
    p.add_argument("--n-qubits",     type=int, default=9)
    p.add_argument("--n-layers",     type=int, default=4)
    p.add_argument("--encoding",     choices=["ry", "rz", "ryrz"], default="ry")
    p.add_argument("--entanglement", choices=["none", "ring", "brick", "all", "star"],
                   default="ring")
    p.add_argument("--mode",         choices=["fixed", "policy"], default="fixed")
    p.add_argument("--capacity",     type=int, default=5)
    p.add_argument("--tw-tightness", type=float, default=0.0)
    p.add_argument("--algo",         choices=["dqn", "reinforce", "ppo"], default="dqn")
    p.add_argument("--tag",          default="",
                   help="Label written to the 'tag' column in the CSV.")
    p.add_argument("--out-csv",      default="",
                   help="Output CSV path.")
    p.add_argument("--append",       action="store_true",
                   help="Append rows to existing CSV instead of overwriting.")
    p.add_argument("--gap",          action="store_true",
                   help="Load checkpoints and run gap analysis (slower).")
    args = p.parse_args()

    analyze(
        prefix       = args.prefix,
        models       = args.models,
        seeds        = args.seeds,
        node         = args.node,
        n_qubits     = args.n_qubits,
        n_layers     = args.n_layers,
        encoding     = args.encoding,
        entanglement = args.entanglement,
        mode         = args.mode,
        capacity     = args.capacity,
        tw_tightness = args.tw_tightness,
        algo         = args.algo,
        tag          = args.tag,
        run_gap      = args.gap,
        out_csv      = args.out_csv,
        append       = args.append,
    )
