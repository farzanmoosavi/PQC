"""
reinforce_qrl.py

REINFORCE (Monte Carlo policy gradient) baseline for CPDPTW — paired with the
Double-DQN in train_qrl.py so Chapter 6 can compare two RL families:

    algo / model  |  DQN (Q-value)  |  REINFORCE (PG)
    --------------|-----------------|------------------
    Quantum       |  train_qrl.py   |  reinforce_qrl.py
    Classical     |  train_qrl.py   |  reinforce_qrl.py

Why REINFORCE for PQC?
  Jerbi et al. (2021) show that PQCs used as value-function approximators face
  the barren-plateau gradient problem more acutely than policy-gradient methods,
  because Q-value regression back-propagates through the Bellman residual with
  potentially vanishing gradients deep in the circuit.  REINFORCE computes
  gradients via the log-probability of sampled trajectories, which flows through
  only one forward pass per step — a gentler signal for near-term circuits.

Algorithm (masked softmax policy):
  * The PQC / MLP outputs a logit vector of shape (n_actions,).
  * Infeasible logits are set to -inf (hard action masking).
  * Softmax gives a valid categorical distribution over feasible actions.
  * Log-probabilities are accumulated per episode; a single backward pass at the
    episode end updates the network.
  * Reward-to-go (not full return) is used as the baseline-free advantage
    estimate: G_t = sum_{k=t}^{T} gamma^{k-t} * r_k.
  * An entropy bonus encourages exploration (replaces epsilon-greedy).
  * Incomplete routes receive a penalty proportional to unvisited nodes before
    the episode return is computed.

Run:
    python reinforce_qrl.py --model quantum --node 5 --episodes 300
    python reinforce_qrl.py --model classical --node 5 --episodes 300
    python reinforce_qrl.py --model qaoa --node 5 --n-qubits 11 --n-layers 3
"""

from __future__ import annotations

import argparse
import random
from typing import List

import numpy as np
import torch
import torch.optim as optim

import torch.nn as nn

from cpdptw_env import CPDPTWEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Value network (critic) — shared baseline for actor-critic REINFORCE
# --------------------------------------------------------------------------- #

class ValueHead(nn.Module):
    """Lightweight MLP that estimates V(s) for variance reduction."""

    def __init__(self, n_obs: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)   # (B,) or scalar


# --------------------------------------------------------------------------- #
# Policy wrapper: turns any Q-network into a stochastic softmax policy
# --------------------------------------------------------------------------- #

class MaskedCategorical:
    """Categorical distribution with -inf masking over infeasible actions."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        masked = logits.masked_fill(~mask, -float("inf"))
        self.dist = torch.distributions.Categorical(logits=masked)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()


# --------------------------------------------------------------------------- #
# Build network (reuses train_qrl.build_net)
# --------------------------------------------------------------------------- #

def build_net(model_kind: str, env: CPDPTWEnv, n_qubits: int = 6, n_layers: int = 3):
    """Delegates to the same factory used by DQN training."""
    from train_qrl import build_net as _build_net
    return _build_net(model_kind, env, n_qubits, n_layers)


# --------------------------------------------------------------------------- #
# REINFORCE training loop
# --------------------------------------------------------------------------- #

def train_reinforce(
    model_kind: str = "quantum",
    node: int = 5,
    capacity: int = 5,
    episodes: int = 300,
    gamma: float = 0.99,
    lr: float = 2e-4,
    entropy_coef: float = 0.05,   # starting value; decays to 10% by end of training
    value_coef: float = 0.5,      # critic loss weight; 0 disables actor-critic
    fixed_instance: bool = True,
    seed: int = 0,
    out_prefix: str = "reinforce",
    n_qubits: int = 6,
    n_layers: int = 3,
    save_every: int = 100,
    encoding: str = "ry",
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = CPDPTWEnv(node=node, vehicle_capacity=capacity, rng_seed=seed)
    net = build_net(model_kind, env, n_qubits, n_layers, encoding).to(device)
    critic = ValueHead(env.n_observations).to(device)
    optimizer = optim.AdamW(
        list(net.parameters()) + list(critic.parameters()), lr=lr, amsgrad=True
    )
    # Cosine decay: lr → lr/100 over all episodes so the policy stabilises
    # instead of oscillating around good solutions it already found.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=episodes, eta_min=lr * 0.01
    )

    if hasattr(net, "param_report"):
        print(f"[REINFORCE/{model_kind}] params: {net.param_report()}")

    rewards_log, dists_log, complete_log, loss_log = [], [], [], []

    for ep in range(episodes):
        state, _ = env.reset(regenerate=not fixed_instance)
        state = state.to(device)

        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        ep_rewards: List[float] = []
        states_buf: List[torch.Tensor] = []   # for critic baseline
        n_steps = 0

        for _ in range(4 * env.n_total):
            mask = env.action_mask().to(device)            # (n_actions,)
            if not mask.any():
                break

            logits = net(state).squeeze(0)                 # (n_actions,)
            policy = MaskedCategorical(logits, mask)
            action = policy.sample()                       # scalar tensor

            log_probs.append(policy.log_prob(action))
            entropies.append(policy.entropy())
            states_buf.append(state)                       # save for critic

            nxt, reward, done, _, info = env.step(action.item())
            ep_rewards.append(reward.item())
            n_steps += 1

            if done:
                break
            state = nxt.to(device)

        # Incompletion penalty: 5× unvisited so stopping early never beats
        # completing the route even with some time-window violations.
        if not env.visited[1:].all():
            unvisited = int((~env.visited[1:]).sum().item())
            ep_rewards.append(-5.0 * unvisited)

        # Reward-to-go G_t = sum_{k=t}^{T} gamma^{k-t} * r_k.
        T = len(ep_rewards)
        returns = torch.zeros(T, device=device)
        G = 0.0
        for t in reversed(range(T)):
            G = ep_rewards[t] + gamma * G
            returns[t] = G

        if not log_probs:
            # Episode produced no log-probs (all infeasible from step 1).
            rewards_log.append(sum(ep_rewards))
            dists_log.append(env.total_distance)
            complete_log.append(0.0)
            continue

        L = len(log_probs)                         # steps with actions taken
        log_prob_t = torch.stack(log_probs)        # (L,)
        entropy_t  = torch.stack(entropies)        # (L,)

        # Actor-critic baseline: critic fits actual (un-normalized) returns so
        # it can distinguish good episodes from bad across different instances.
        # Advantages are then normalized for stable PG gradient magnitude.
        states_t   = torch.cat(states_buf, dim=0).squeeze(1)  # (L, F)
        values     = critic(states_t)                           # (L,)
        ret_L      = returns[:L]
        advantages = ret_L - values.detach()
        if L > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Entropy decays linearly from entropy_coef → 10% of it by the final
        # episode so the agent explores early and exploits late.
        cur_ent = entropy_coef * max(0.1, 1.0 - ep / max(episodes - 1, 1))

        # Policy-gradient loss with advantage baseline.
        pg_loss      = -(log_prob_t * advantages).mean()
        value_loss   = 0.5 * torch.nn.functional.mse_loss(values, ret_L.detach())
        entropy_loss = -cur_ent * entropy_t.mean()
        loss         = pg_loss + value_coef * value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(net.parameters()) + list(critic.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()

        total_r = sum(ep_rewards)
        completed = bool(env.visited[1:].all().item())
        rewards_log.append(total_r)
        dists_log.append(env.total_distance)
        complete_log.append(1.0 if completed else 0.0)
        loss_log.append(loss.item())

        if save_every > 0 and (ep + 1) % save_every == 0 and (ep + 1) < episodes:
            torch.save(net.state_dict(),
                       f"{out_prefix}_{model_kind}_s{seed}_ep{ep+1}.pt")

        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1:4d} | R={total_r:7.2f} | dist={env.total_distance:6.2f} "
                  f"| complete={complete_log[-1]:.2f} | loss={loss.item():.4f} "
                  f"| lr={scheduler.get_last_lr()[0]:.2e} | ent={cur_ent:.4f}")

    tag = f"{out_prefix}_{model_kind}_s{seed}"
    np.savetxt(f"{tag}_rewards.txt",  rewards_log)
    np.savetxt(f"{tag}_dists.txt",    dists_log)
    np.savetxt(f"{tag}_losses.txt",   loss_log if loss_log else [0.0])
    np.savetxt(f"{tag}_complete.txt", complete_log)
    ckpt = f"{tag}.pt"
    torch.save(net.state_dict(), ckpt)
    print(f"Done. Saved {tag}_*.txt  checkpoint -> {ckpt}")
    return dict(rewards=rewards_log, dists=dists_log, losses=loss_log,
                complete=complete_log, net=net)


# --------------------------------------------------------------------------- #
# Comparison helper: run DQN and REINFORCE back-to-back, print table
# --------------------------------------------------------------------------- #

def compare(
    model_kind: str = "quantum",
    node: int = 5,
    capacity: int = 5,
    episodes: int = 200,
    seed: int = 0,
    n_qubits: int = 6,
    n_layers: int = 3,
) -> None:
    """Quick 2×2 (algorithm × model) comparison printed to stdout."""
    from train_qrl import train as train_dqn
    from sweep_experiment import _convergence_episode

    tail = max(1, episodes // 5)
    results = {}

    for algo in ("dqn", "reinforce"):
        for mk in (model_kind, f"classical"):
            print(f"\n{'='*60}")
            print(f"  {algo.upper()} / {mk}")
            print(f"{'='*60}")
            if algo == "dqn":
                r = train_dqn(model_kind=mk, node=node, capacity=capacity,
                              episodes=episodes, seed=seed,
                              fixed_instance=True,
                              out_prefix=f"cmp_{algo}",
                              n_qubits=n_qubits, n_layers=n_layers)
            else:
                r = train_reinforce(model_kind=mk, node=node, capacity=capacity,
                                    episodes=episodes, seed=seed,
                                    fixed_instance=True,
                                    out_prefix=f"cmp_{algo}",
                                    n_qubits=n_qubits, n_layers=n_layers)
            results[(algo, mk)] = r

    print("\n" + "="*72)
    print(f"{'algo':10s} {'model':12s} {'final_reward':13s} {'best_reward':11s} "
          f"{'complete':8s} {'converge_ep':11s}")
    print("-"*72)
    for (algo, mk), r in results.items():
        fr = sum(r["rewards"][-tail:]) / tail
        br = max(r["rewards"])
        fc = sum(r["complete"][-tail:]) / tail
        ce = _convergence_episode(r["rewards"])
        print(f"{algo:10s} {mk:12s} {fr:13.3f} {br:11.3f} {fc:8.3f} {ce:11d}")
    print("="*72)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="REINFORCE policy-gradient for CPDPTW")
    p.add_argument("--model",
                   choices=["quantum", "qaoa", "classical", "classical-qaoa",
                            "node-quantum", "node-qaoa"],
                   default="quantum")
    p.add_argument("--node",      type=int,   default=5)
    p.add_argument("--capacity",  type=int,   default=5)
    p.add_argument("--episodes",  type=int,   default=300)
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--n-qubits",  type=int,   default=6)
    p.add_argument("--n-layers",  type=int,   default=3)
    p.add_argument("--lr",        type=float, default=5e-4)
    p.add_argument("--entropy",     type=float, default=0.05,
                   help="Entropy regularisation coefficient.")
    p.add_argument("--value-coef",  type=float, default=0.5,
                   help="Critic loss weight (0 = pure REINFORCE, no baseline).")
    p.add_argument("--compare",   action="store_true",
                   help="Run DQN vs REINFORCE side-by-side comparison table.")
    p.add_argument("--fixed-instance", action="store_true",
                   help="Train on one fixed problem instance.")
    p.add_argument("--out-prefix", default="reinforce",
                   help="Path prefix for output .txt and .pt files.")
    p.add_argument("--save-every", type=int, default=100,
                   help="Save a checkpoint every N episodes (0 = disable).")
    p.add_argument("--encoding", choices=["ry", "rz", "ryrz"], default="ry",
                   help="Qubit encoding strategy (default: ry).")
    args = p.parse_args()

    if args.compare:
        compare(model_kind=args.model, node=args.node, capacity=args.capacity,
                episodes=args.episodes, seed=args.seed,
                n_qubits=args.n_qubits, n_layers=args.n_layers)
    else:
        train_reinforce(
            model_kind=args.model,
            node=args.node,
            capacity=args.capacity,
            episodes=args.episodes,
            gamma=0.99,
            lr=args.lr,
            entropy_coef=args.entropy,
            value_coef=args.value_coef,
            fixed_instance=args.fixed_instance,
            seed=args.seed,
            out_prefix=args.out_prefix,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            save_every=args.save_every,
            encoding=args.encoding,
        )
