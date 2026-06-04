"""
noise_eval.py

Post-training noise evaluation for the CPDPTW PQC.  Loads a saved checkpoint,
rebuilds the circuit on a density-matrix simulator (default.mixed), injects
per-gate depolarising noise, and reports the degradation curve.

Depolarising channel model
--------------------------
After every gate G on qubit q, we apply:

    DepolarizingChannel(p, wires=q)

which maps  ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ).

This is the standard single-qubit after-gate error model used in IBM/Google
calibration papers and in the Preskill (2018) NISQ discussion.  Two-qubit gates
(CNOT) receive independent noise on each qubit.

Hardware context (approximate, 2024 superconducting qubits):
    single-qubit gate error  p ≈ 0.001
    two-qubit gate error     p ≈ 0.005–0.01
    NISQ "threshold" where quality collapses: p ≈ 0.02–0.05 for shallow circuits

Metrics reported per noise level
---------------------------------
  reward_mean   — mean episodic return (greedy policy)
  dist_mean     — mean route distance
  feas_mean     — fraction of feasible actions taken
  q_rmse        — RMS deviation of Q-values from the clean baseline over a fixed
                  set of states; measures circuit output drift independently of
                  behavioural change
  zz_mean_abs   — mean |<Z_i Z_{i+1}>| over ring pairs, averaged across states;
                  directly measures surviving entanglement — trends to 0 as noise
                  destroys correlations

Usage
-----
    # Train first:
    python train_qrl.py --model quantum --node 5 --episodes 300 --fixed-instance
    # Evaluate:
    python noise_eval.py --checkpoint qrl_quantum.pt --node 5
    python noise_eval.py --checkpoint qrl_quantum.pt --node 5 --n-qubits 11
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import pennylane as qml
    _HAS_PENNYLANE = True
except ImportError:
    _HAS_PENNYLANE = False


# --------------------------------------------------------------------------- #
# Default noise levels to sweep
# --------------------------------------------------------------------------- #

# Covers: near-zero → single-qubit gate error (~1e-3) → two-qubit gate error
# (~5e-3) → early NISQ threshold (~0.02) → deep NISQ (~0.1) → highly noisy.
NOISE_LEVELS = [0.0, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.10, 0.15, 0.20]


# --------------------------------------------------------------------------- #
# Noisy PQC Q-network
# --------------------------------------------------------------------------- #

class NoisyQuantumQNetwork(nn.Module):
    """
    Identical architecture to QuantumQNetwork but uses default.mixed so that
    DepolarizingChannel operations are valid.

    State dict is compatible with QuantumQNetwork (same parameter names and
    shapes), so a clean checkpoint can be loaded directly:

        noisy = NoisyQuantumQNetwork(env, noise_level=0.01, ...)
        noisy.load_state_dict(torch.load("qrl_quantum.pt"))

    At noise_level=0.0 the channel is a no-op and results match the clean model.
    """

    def __init__(
        self,
        env,
        *,
        n_qubits: int = 6,
        n_layers: int = 3,
        noise_level: float = 0.0,
        qaoa: bool = False,
        torch_device: Optional[torch.device] = None,
    ):
        super().__init__()
        if not _HAS_PENNYLANE:
            raise ImportError("pennylane is required.")

        self.n_actions   = env.n_actions
        self.n_obs       = env.n_observations
        self.n_qubits    = int(n_qubits)
        self.n_layers    = int(n_layers)
        self.noise_level = float(noise_level)
        self.qaoa        = bool(qaoa)
        self.n_outputs   = 2 * self.n_qubits   # <Z_i> + <Z_i Z_{i+1}>
        self.device      = torch_device or torch.device("cpu")

        self.compressor = nn.Sequential(
            nn.Linear(self.n_obs, self.n_qubits),
            nn.Tanh(),
        )

        dev = qml.device("default.mixed", wires=self.n_qubits)
        p   = self.noise_level
        nq  = self.n_qubits
        nl  = self.n_layers

        if self.qaoa:
            # QAOA circuit: state dict uses keys "gamma" and "beta"
            weight_shapes = {
                "gamma": (nl, nq),
                "beta":  (nl, nq),
            }

            @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
            def circuit(inputs, gamma, beta):
                for q in range(nq):
                    qml.Hadamard(wires=q)
                    if p > 0:
                        qml.DepolarizingChannel(p, wires=q)
                for layer in range(nl):
                    for q in range(nq):
                        qml.RY(inputs[..., q], wires=q)
                        if p > 0:
                            qml.DepolarizingChannel(p, wires=q)
                    for q in range(nq):
                        qml.IsingZZ(gamma[layer, q], wires=[q, (q + 1) % nq])
                        if p > 0:
                            qml.DepolarizingChannel(p, wires=q)
                            qml.DepolarizingChannel(p, wires=(q + 1) % nq)
                    for q in range(nq):
                        qml.RX(beta[layer, q], wires=q)
                        if p > 0:
                            qml.DepolarizingChannel(p, wires=q)
                z_obs  = [qml.expval(qml.PauliZ(q)) for q in range(nq)]
                zz_obs = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % nq))
                          for q in range(nq)]
                return z_obs + zz_obs

        else:
            # HEA circuit: state dict uses key "weights"
            weight_shapes = {"weights": (nl, nq, 3)}

            @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
            def circuit(inputs, weights):
                for q in range(nq):
                    qml.Hadamard(wires=q)
                    if p > 0:
                        qml.DepolarizingChannel(p, wires=q)
                for layer in range(nl):
                    for q in range(nq):
                        qml.RY(inputs[..., q], wires=q)
                        if p > 0:
                            qml.DepolarizingChannel(p, wires=q)
                    for q in range(nq):
                        qml.CNOT(wires=[q, (q + 1) % nq])
                        if p > 0:
                            qml.DepolarizingChannel(p, wires=q)
                            qml.DepolarizingChannel(p, wires=(q + 1) % nq)
                    for q in range(nq):
                        qml.RX(weights[layer, q, 0], wires=q)
                        if p > 0:
                            qml.DepolarizingChannel(p, wires=q)
                        qml.RY(weights[layer, q, 1], wires=q)
                        if p > 0:
                            qml.DepolarizingChannel(p, wires=q)
                        qml.RZ(weights[layer, q, 2], wires=q)
                        if p > 0:
                            qml.DepolarizingChannel(p, wires=q)
                z_obs  = [qml.expval(qml.PauliZ(q)) for q in range(nq)]
                zz_obs = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % nq))
                          for q in range(nq)]
                return z_obs + zz_obs

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.head   = nn.Linear(self.n_outputs, self.n_actions)
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state  = state.to(self.device)
        angles = self.compressor(state) * math.pi
        z      = self.qlayer(angles.cpu())
        z      = z.to(self.device).float()
        return self.head(z)

    def zz_values(self, state: torch.Tensor) -> torch.Tensor:
        """Return only the ZZ portion of the circuit output — shape (B, n_qubits)."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        angles = self.compressor(state.to(self.device)) * math.pi
        out    = self.qlayer(angles.cpu()).to(self.device).float()
        return out[:, self.n_qubits:]   # second half is ZZ


# --------------------------------------------------------------------------- #
# Greedy evaluation
# --------------------------------------------------------------------------- #

def greedy_rollout(net, env, n_episodes: int = 30) -> dict:
    """Run n_episodes greedy episodes; return mean reward, dist, feas."""
    net.eval()
    dev = next(net.parameters()).device
    rewards, dists, feas_rates = [], [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            state, _ = env.reset(regenerate=False)
            state = state.to(dev)
            total_r, n_steps, n_infeas = 0.0, 0, 0

            for _ in range(4 * env.n_total + 1):
                mask = env.action_mask().to(dev)
                if not mask.any():
                    break
                q = net(state)
                q = q.masked_fill(~mask.unsqueeze(0), -float("inf"))
                action = q.max(1).indices.item()

                nxt, reward, done, _, info = env.step(action)
                n_infeas += int(info.get("infeasible", False))
                total_r  += reward.item()
                n_steps  += 1
                state = nxt.to(dev)
                if done:
                    break

            rewards.append(total_r)
            dists.append(env.total_distance)
            feas_rates.append(1.0 - n_infeas / max(n_steps, 1))

    return {
        "reward_mean": float(np.mean(rewards)),
        "dist_mean":   float(np.mean(dists)),
        "feas_mean":   float(np.mean(feas_rates)),
    }


# --------------------------------------------------------------------------- #
# State pool for Q-value RMSE and ZZ diagnostics
# --------------------------------------------------------------------------- #

def collect_eval_states(env, n_states: int = 64) -> torch.Tensor:
    """
    Run random rollouts to collect a fixed pool of states.  Used to compute
    Q-value RMSE (vs clean baseline) and mean |<ZZ>| across noise levels.
    """
    states = []
    while len(states) < n_states:
        s, _ = env.reset(regenerate=False)
        states.append(s)
        for _ in range(4 * env.n_total):
            feas = env.valid_actions()
            if not feas:
                break
            a = random.choice(feas)
            s, _, done, _, _ = env.step(a)
            states.append(s)
            if done or len(states) >= n_states:
                break
    return torch.cat(states[:n_states], dim=0)   # (n_states, F)


# --------------------------------------------------------------------------- #
# Degradation curve
# --------------------------------------------------------------------------- #

def degradation_curve(
    checkpoint_path: str,
    env,
    *,
    n_qubits: int = 6,
    n_layers: int = 3,
    qaoa: bool = False,
    noise_levels: list[float] = NOISE_LEVELS,
    n_eval_episodes: int = 30,
    n_eval_states: int = 64,
    out_csv: str = "noise_degradation.csv",
) -> list[dict]:
    """
    Main entry point.  For each noise level:
      1. Build NoisyQuantumQNetwork with that noise_level.
      2. Load the saved checkpoint weights.
      3. Run greedy evaluation and Q-value diagnostics.
      4. Record all metrics.

    Also builds a clean (p=0) baseline once for RMSE and ZZ reference values.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # ---- Clean baseline -------------------------------------------------- #
    arch = "QAOA" if qaoa else "HEA"
    print(f"Building clean baseline (p=0, {arch}) from {checkpoint_path} ...")
    clean_net = NoisyQuantumQNetwork(env, n_qubits=n_qubits,
                                     n_layers=n_layers, noise_level=0.0,
                                     qaoa=qaoa)
    clean_net.load_state_dict(state_dict)
    clean_net.eval()

    eval_states = collect_eval_states(env, n_eval_states)

    with torch.no_grad():
        clean_q  = clean_net(eval_states)              # (N, n_actions)
        clean_zz = clean_net.zz_values(eval_states)    # (N, n_qubits)

    clean_metrics = greedy_rollout(clean_net, env, n_eval_episodes)
    print(f"  clean reward={clean_metrics['reward_mean']:.3f}  "
          f"dist={clean_metrics['dist_mean']:.3f}  "
          f"feas={clean_metrics['feas_mean']:.3f}")
    del clean_net

    # ---- Sweep ------------------------------------------------------------ #
    rows: list[dict] = []
    gates = _count_gates(n_qubits, n_layers, qaoa=qaoa)
    arch = "QAOA" if qaoa else "HEA"
    print(f"  Circuit: {arch}  n_qubits={n_qubits}  n_layers={n_layers}  "
          f"SQ={gates['sq']}  TQ={gates['tq']}  depth={gates['depth']}")

    for p in noise_levels:
        print(f"  p={p:.4f} ...", end=" ", flush=True)
        noisy = NoisyQuantumQNetwork(env, n_qubits=n_qubits,
                                     n_layers=n_layers, noise_level=p, qaoa=qaoa)
        noisy.load_state_dict(state_dict)
        noisy.eval()

        with torch.no_grad():
            noisy_q  = noisy(eval_states)
            noisy_zz = noisy.zz_values(eval_states)

        q_rmse   = float(torch.sqrt(((noisy_q - clean_q) ** 2).mean()).item())
        zz_delta = float(torch.abs(noisy_zz - clean_zz).mean().item())
        zz_abs   = float(torch.abs(noisy_zz).mean().item())

        metrics = greedy_rollout(noisy, env, n_eval_episodes)
        del noisy

        reward_deg = ((metrics["reward_mean"] - clean_metrics["reward_mean"])
                      / max(abs(clean_metrics["reward_mean"]), 1e-9))

        row = {
            "noise_p":        p,
            "n_qubits":       n_qubits,
            "n_layers":       n_layers,
            "sq_gates":       gates["sq"],
            "tq_gates":       gates["tq"],
            "circuit_depth":  gates["depth"],
            "total_weighted": gates["total_weighted"],
            "reward_mean":    metrics["reward_mean"],
            "dist_mean":      metrics["dist_mean"],
            "feas_mean":      metrics["feas_mean"],
            "reward_deg_pct": 100.0 * reward_deg,
            "q_rmse":         q_rmse,
            "zz_mean_abs":    zz_abs,
            "zz_delta":       zz_delta,
        }
        rows.append(row)
        print(f"reward={metrics['reward_mean']:.3f}  "
              f"deg={100*reward_deg:.1f}%  "
              f"q_rmse={q_rmse:.4f}  "
              f"|ZZ|={zz_abs:.4f}")

    # ---- Save ------------------------------------------------------------- #
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDegradation curve -> {out_csv}")

    _print_degradation_table(rows, clean_metrics, label="QAOA" if qaoa else "HEA")
    return rows


def _count_gates(n_qubits: int, n_layers: int, qaoa: bool = False) -> dict:
    """
    Gate count breakdown for hardware feasibility analysis.

    Ring entanglement parallelism:
        Even n_qubits: 2 parallel rounds (even-indexed pairs, then odd-indexed).
        Odd  n_qubits: 3 parallel rounds (wraparound pair creates a conflict).

    Returns
    -------
    dict with keys:
        sq            : single-qubit gate count
        tq            : two-qubit gate count
        depth         : circuit depth (critical path, assuming ring parallelism)
        total_weighted: sq + 2*tq  (single-qubit-equivalent, backward compat)
    """
    ent_rounds = 2 if n_qubits % 2 == 0 else 3

    # H initialisation: n_qubits SQ gates, all parallel -> depth 1
    sq = n_qubits
    tq = 0

    if qaoa:
        # Per layer: RY encoding (SQ) + IsingZZ ring (TQ) + RX mixer (SQ)
        sq += n_layers * (n_qubits + n_qubits)   # RY + RX
        tq += n_layers * n_qubits                 # IsingZZ pairs
        depth = 1 + n_layers * (1 + ent_rounds + 1)
    else:
        # Per layer: RY encoding (SQ) + CNOT ring (TQ) + RX+RY+RZ (3 SQ)
        sq += n_layers * (n_qubits + 3 * n_qubits)  # RY + RX+RY+RZ
        tq += n_layers * n_qubits                    # CNOT pairs
        depth = 1 + n_layers * (1 + ent_rounds + 3)

    return {
        "sq":             sq,
        "tq":             tq,
        "depth":          depth,
        "total_weighted": sq + 2 * tq,
    }


def _print_degradation_table(rows: list[dict], clean: dict,
                             label: str = "") -> None:
    tag = f"  [{label}]" if label else ""
    print(f"\n{'='*70}{tag}")
    if rows:
        g = rows[0]
        print(f"  SQ={g['sq_gates']}  TQ={g['tq_gates']}  "
              f"depth={g['circuit_depth']}  weighted={g['total_weighted']}")
    print(f"{'p':8s}  {'reward':9s}  {'deg%':7s}  {'q_rmse':8s}  "
          f"{'|ZZ|':8s}  {'feas':6s}")
    print(f"{'clean':8s}  {clean['reward_mean']:9.3f}  "
          f"{'0.0%':>7s}  {'—':>8s}  {'—':>8s}  {clean['feas_mean']:6.3f}")
    print("-" * 70)
    for r in rows:
        print(f"{r['noise_p']:8.4f}  {r['reward_mean']:9.3f}  "
              f"{r['reward_deg_pct']:7.1f}%  {r['q_rmse']:8.4f}  "
              f"{r['zz_mean_abs']:8.4f}  {r['feas_mean']:6.3f}")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# HEA vs QAOA noise comparison
# --------------------------------------------------------------------------- #

def compare_noise(
    checkpoint_hea:  str,
    checkpoint_qaoa: str,
    env,
    *,
    n_qubits: int = 6,
    n_layers: int = 3,
    noise_levels: list[float] = NOISE_LEVELS,
    n_eval_episodes: int = 30,
    n_eval_states: int = 64,
    out_csv: str = "noise_comparison.csv",
) -> list[dict]:
    """
    Run the degradation curve for both HEA (quantum) and QAOA on the same env,
    then print a side-by-side comparison and save a combined CSV.

    This is the direct evidence for the thesis claim that QAOA's structured
    Hamiltonian makes it more noise-resilient than the generic HEA ansatz.
    """
    print("=== HEA (quantum) noise sweep ===")
    rows_hea = degradation_curve(
        checkpoint_hea, env,
        n_qubits=n_qubits, n_layers=n_layers, qaoa=False,
        noise_levels=noise_levels,
        n_eval_episodes=n_eval_episodes,
        n_eval_states=n_eval_states,
        out_csv=out_csv.replace(".csv", "_hea.csv"),
    )

    print("\n=== QAOA noise sweep ===")
    rows_qaoa = degradation_curve(
        checkpoint_qaoa, env,
        n_qubits=n_qubits, n_layers=n_layers, qaoa=True,
        noise_levels=noise_levels,
        n_eval_episodes=n_eval_episodes,
        n_eval_states=n_eval_states,
        out_csv=out_csv.replace(".csv", "_qaoa.csv"),
    )

    # --- side-by-side comparison table ---
    gates_hea  = _count_gates(n_qubits, n_layers, qaoa=False)
    gates_qaoa = _count_gates(n_qubits, n_layers, qaoa=True)

    print(f"\n{'='*80}")
    print(f"HEA vs QAOA noise comparison  "
          f"(n_qubits={n_qubits}  n_layers={n_layers})")
    print(f"  HEA : SQ={gates_hea['sq']:3d}  TQ={gates_hea['tq']:3d}  "
          f"depth={gates_hea['depth']:3d}  weighted={gates_hea['total_weighted']}")
    print(f"  QAOA: SQ={gates_qaoa['sq']:3d}  TQ={gates_qaoa['tq']:3d}  "
          f"depth={gates_qaoa['depth']:3d}  weighted={gates_qaoa['total_weighted']}")
    print(f"{'p':8s}  {'HEA deg%':9s}  {'QAOA deg%':10s}  "
          f"{'HEA |ZZ|':9s}  {'QAOA |ZZ|':10s}  {'diff deg%':10s}")
    print("-" * 80)

    combined = []
    for rh, rq in zip(rows_hea, rows_qaoa):
        diff = rq["reward_deg_pct"] - rh["reward_deg_pct"]
        sign = "+" if diff > 0 else ""
        print(f"{rh['noise_p']:8.4f}  {rh['reward_deg_pct']:9.1f}%  "
              f"{rq['reward_deg_pct']:10.1f}%  "
              f"{rh['zz_mean_abs']:9.4f}  {rq['zz_mean_abs']:10.4f}  "
              f"{sign}{diff:9.1f}%")
        row = {"noise_p": rh["noise_p"]}
        for k, v in rh.items():
            if k != "noise_p":
                row[f"hea_{k}"] = v
        for k, v in rq.items():
            if k != "noise_p":
                row[f"qaoa_{k}"] = v
        row["deg_diff_pct"] = diff
        combined.append(row)

    print("=" * 80)
    print("  diff deg% = QAOA deg% - HEA deg%  "
          "(negative = QAOA degrades less = more noise-resilient)")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[0].keys()),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    print(f"\nCombined comparison -> {out_csv}")
    return combined


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from cpdptw_env import CPDPTWEnv

    ap = argparse.ArgumentParser(description="PQC noise degradation curve")
    ap.add_argument("--checkpoint",      default=None,
                    help="HEA checkpoint (.pt) for single-model or --compare mode")
    ap.add_argument("--checkpoint-qaoa", default=None,
                    help="QAOA checkpoint (.pt) for --compare mode")
    ap.add_argument("--compare",  action="store_true",
                    help="Run HEA vs QAOA side-by-side comparison "
                         "(requires --checkpoint and --checkpoint-qaoa)")
    ap.add_argument("--node",       type=int, default=5)
    ap.add_argument("--capacity",   type=int, default=5)
    ap.add_argument("--n-qubits",   type=int, default=6,
                    help="Must match the checkpoint architecture. "
                         "Natural encoding = 2*node+1 (default: 6 compact)")
    ap.add_argument("--n-layers",   type=int, default=3,
                    help="Must match the checkpoint architecture")
    ap.add_argument("--qaoa",       action="store_true",
                    help="Single-model mode: checkpoint is a QAOAQNetwork")
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--eval-eps",   type=int, default=30,
                    help="Greedy episodes per noise level (default: 30)")
    ap.add_argument("--out",        default="noise_degradation.csv")
    ap.add_argument("--noise",      type=float, nargs="*", default=None,
                    help="Custom noise levels (default: full NOISE_LEVELS sweep)")
    args = ap.parse_args()

    env = CPDPTWEnv(node=args.node, vehicle_capacity=args.capacity,
                    rng_seed=args.seed)

    natural = 2 * args.node + 1
    enc_label = (f"natural ({natural}q = 2x{args.node}+1)"
                 if args.n_qubits == natural
                 else f"compact ({args.n_qubits}q, natural={natural})")
    print(f"Encoding: {enc_label}")

    noise_levels = args.noise if args.noise is not None else NOISE_LEVELS

    if args.compare:
        if not args.checkpoint or not args.checkpoint_qaoa:
            ap.error("--compare requires both --checkpoint and --checkpoint-qaoa")
        compare_noise(
            checkpoint_hea  = args.checkpoint,
            checkpoint_qaoa = args.checkpoint_qaoa,
            env             = env,
            n_qubits        = args.n_qubits,
            n_layers        = args.n_layers,
            noise_levels    = noise_levels,
            n_eval_episodes = args.eval_eps,
            out_csv         = args.out,
        )
    else:
        if not args.checkpoint:
            ap.error("--checkpoint is required for single-model mode")
        degradation_curve(
            checkpoint_path = args.checkpoint,
            env             = env,
            n_qubits        = args.n_qubits,
            n_layers        = args.n_layers,
            qaoa            = args.qaoa,
            noise_levels    = noise_levels,
            n_eval_episodes = args.eval_eps,
            out_csv         = args.out,
        )
