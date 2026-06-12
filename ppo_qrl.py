"""
ppo_qrl.py

PPO (Proximal Policy Optimisation) for CPDPTW — the third RL algorithm in
Chapter 6, alongside DQN (train_qrl.py) and REINFORCE (reinforce_qrl.py).

Why PPO over REINFORCE for PQC?
  REINFORCE updates once per episode from a single trajectory, giving high-
  variance gradients that are especially damaging for PQC circuits (barren
  plateau). PPO collects a buffer of N complete episodes, then runs K epochs
  of minibatch updates with a clipped surrogate objective:

      L^CLIP = E[ min(r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t) ]

  where r_t = pi_new(a|s) / pi_old(a|s).  The clip prevents any single
  step from moving the policy too far, which is critical when quantum
  gradients are noisy and small.  Running K epochs on the same buffer also
  makes better use of expensive circuit evaluations.

  Advantages over REINFORCE:
    - Clipped objective: no catastrophic single-step policy collapse
    - Multi-epoch updates: amortises the cost of circuit forward passes
    - GAE (lambda=0.95): lower-variance advantage estimates than raw returns
    - Minibatch SGD: decorrelates gradient updates within each buffer

Algorithm:
  1. Collect `episodes_per_update` complete episodes into a flat buffer.
  2. Compute GAE advantages + discounted returns.
  3. Run `n_epochs` passes over the buffer in random minibatches.
     Each minibatch computes L^CLIP + value loss + entropy bonus.
  4. Repeat until `episodes` total episodes are trained.

Run:
    python ppo_qrl.py --model quantum --node 3 --episodes 300
    python ppo_qrl.py --model node-quantum --node 3 --episodes 300
    python ppo_qrl.py --model classical-large --node 3 --episodes 300
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cpdptw_env import CPDPTWEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Value head (critic) — identical to reinforce_qrl.py
# --------------------------------------------------------------------------- #

class ValueHead(nn.Module):
    def __init__(self, n_obs: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


# --------------------------------------------------------------------------- #
# Rollout buffer
# --------------------------------------------------------------------------- #

@dataclass
class RolloutBuffer:
    states:       List[torch.Tensor] = field(default_factory=list)
    actions:      List[int]          = field(default_factory=list)
    log_probs:    List[torch.Tensor] = field(default_factory=list)
    rewards:      List[float]        = field(default_factory=list)
    values:       List[torch.Tensor] = field(default_factory=list)
    dones:        List[bool]         = field(default_factory=list)

    def clear(self):
        self.states.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.values.clear(); self.dones.clear()

    def __len__(self):
        return len(self.rewards)


def _masked_categorical(logits: torch.Tensor, mask: torch.Tensor):
    """Categorical distribution with -inf masking over infeasible actions."""
    return torch.distributions.Categorical(
        logits=logits.masked_fill(~mask, -float("inf"))
    )


def _compute_gae(
    rewards: List[float],
    values:  List[torch.Tensor],
    dones:   List[bool],
    gamma:   float,
    lam:     float,
    device:  torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation (Schulman et al. 2016).
    Returns advantages and discounted returns, both shape (T,).
    """
    T = len(rewards)
    adv   = torch.zeros(T, device=device)
    ret   = torch.zeros(T, device=device)
    vals  = torch.stack(values).detach().to(device)   # (T,)

    gae = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if dones[t] else (vals[t + 1].item() if t + 1 < T else 0.0)
        delta = rewards[t] + gamma * next_val - vals[t].item()
        gae   = delta + gamma * lam * (0.0 if dones[t] else gae)
        adv[t] = gae
        ret[t] = gae + vals[t].item()

    if T > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


# --------------------------------------------------------------------------- #
# Network factory (delegates to train_qrl.build_net)
# --------------------------------------------------------------------------- #

def build_net(model_kind: str, env: CPDPTWEnv, n_qubits: int = 6,
              n_layers: int = 3, encoding: str = "ry",
              entanglement: str = "ring"):
    from train_qrl import build_net as _build_net
    return _build_net(model_kind, env, n_qubits, n_layers, encoding, entanglement)


# --------------------------------------------------------------------------- #
# PPO training loop
# --------------------------------------------------------------------------- #

def train_ppo(
    model_kind:           str   = "quantum",
    node:                 int   = 3,
    capacity:             int   = 5,
    episodes:             int   = 300,
    gamma:                float = 0.99,
    gae_lambda:           float = 0.95,
    lr:                   float = 2e-4,
    clip_eps:             float = 0.2,
    n_epochs:             int   = 4,
    episodes_per_update:  int   = 10,
    minibatch_size:       int   = 32,
    entropy_coef:         float = 0.05,
    value_coef:           float = 0.5,
    fixed_instance:       bool  = True,
    seed:                 int   = 0,
    out_prefix:           str   = "ppo",
    n_qubits:             int   = 6,
    n_layers:             int   = 3,
    save_every:           int   = 100,
    encoding:             str   = "ry",
    entanglement:         str   = "ring",
) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    env    = CPDPTWEnv(node=node, vehicle_capacity=capacity, rng_seed=seed)
    net    = build_net(model_kind, env, n_qubits, n_layers, encoding,
                       entanglement).to(device)
    critic = ValueHead(env.n_observations).to(device)
    optimizer = optim.AdamW(
        list(net.parameters()) + list(critic.parameters()), lr=lr, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=episodes, eta_min=lr * 0.01
    )

    if hasattr(net, "param_report"):
        print(f"[PPO/{model_kind}] params: {net.param_report()}")

    rewards_log, dists_log, complete_log, loss_log = [], [], [], []
    buffer = RolloutBuffer()
    ep = 0

    while ep < episodes:
        # ------------------------------------------------------------------ #
        # Phase 1: collect episodes_per_update episodes into the buffer
        # ------------------------------------------------------------------ #
        buffer.clear()
        ep_rewards_batch: List[float] = []
        ep_dists_batch:   List[float] = []
        ep_complete_batch: List[float] = []

        for _ in range(episodes_per_update):
            if ep >= episodes:
                break
            ep += 1

            state, _ = env.reset(regenerate=not fixed_instance)
            state = state.to(device)
            ep_rewards: List[float] = []

            for _ in range(4 * env.n_total):
                mask = env.action_mask().to(device)
                if not mask.any():
                    break

                with torch.no_grad():
                    logits = net(state).squeeze(0)
                    val    = critic(state.squeeze(0) if state.dim() > 1 else state)

                dist   = _masked_categorical(logits, mask)
                action = dist.sample()

                buffer.states.append(state.squeeze(0) if state.dim() > 1 else state)
                buffer.actions.append(action.item())
                buffer.log_probs.append(dist.log_prob(action).detach())
                buffer.values.append(val.detach())

                nxt, reward, done, _, _ = env.step(action.item())
                ep_rewards.append(reward.item())
                buffer.rewards.append(reward.item())
                buffer.dones.append(done)

                if done:
                    break
                state = nxt.to(device)

            # Incompletion penalty — same as reinforce_qrl
            if not env.visited[1:].all():
                unvisited = int((~env.visited[1:]).sum().item())
                pen = -5.0 * unvisited
                ep_rewards.append(pen)
                if buffer.rewards:
                    buffer.rewards[-1] += pen   # tack onto last step

            ep_rewards_batch.append(sum(ep_rewards))
            ep_dists_batch.append(env.total_distance)
            ep_complete_batch.append(1.0 if env.visited[1:].all() else 0.0)

        if len(buffer) == 0:
            continue

        # ------------------------------------------------------------------ #
        # Phase 2: compute GAE advantages for the whole buffer
        # ------------------------------------------------------------------ #
        advantages, returns = _compute_gae(
            buffer.rewards, buffer.values, buffer.dones, gamma, gae_lambda, device
        )

        states_t   = torch.stack(buffer.states).to(device)          # (T, F)
        actions_t  = torch.tensor(buffer.actions, device=device)     # (T,)
        logprob_old= torch.stack(buffer.log_probs).to(device)        # (T,)

        # ------------------------------------------------------------------ #
        # Phase 3: K epochs of minibatch PPO updates
        # ------------------------------------------------------------------ #
        T = len(buffer)
        idx = torch.randperm(T)
        cur_ent = entropy_coef * max(0.1, 1.0 - ep / max(episodes - 1, 1))
        epoch_losses: List[float] = []

        for _ in range(n_epochs):
            for start in range(0, T, minibatch_size):
                mb = idx[start: start + minibatch_size]

                mb_states  = states_t[mb]
                mb_actions = actions_t[mb]
                mb_adv     = advantages[mb]
                mb_ret     = returns[mb]
                mb_lp_old  = logprob_old[mb]

                # Re-evaluate policy and value on minibatch
                logits_new = net(mb_states).squeeze(1)              # (B, A)
                # Reconstruct mask from valid actions — use all-True since
                # we're re-evaluating already-taken (feasible) actions.
                # Entropy computed over full action space for regularisation.
                dist_new   = torch.distributions.Categorical(logits=logits_new)
                lp_new     = dist_new.log_prob(mb_actions)           # (B,)
                entropy    = dist_new.entropy().mean()

                ratio      = torch.exp(lp_new - mb_lp_old)          # (B,)
                surr1      = ratio * mb_adv
                surr2      = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                pg_loss    = -torch.min(surr1, surr2).mean()

                val_new    = critic(mb_states)                       # (B,)
                val_loss   = 0.5 * nn.functional.mse_loss(val_new, mb_ret)

                loss = pg_loss + value_coef * val_loss - cur_ent * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(net.parameters()) + list(critic.parameters()), 1.0
                )
                optimizer.step()
                epoch_losses.append(loss.item())

        scheduler.step()

        rewards_log.extend(ep_rewards_batch)
        dists_log.extend(ep_dists_batch)
        complete_log.extend(ep_complete_batch)
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        loss_log.extend([mean_loss] * len(ep_rewards_batch))

        if save_every > 0 and ep % save_every == 0 and ep < episodes:
            torch.save(net.state_dict(),
                       f"{out_prefix}_{model_kind}_s{seed}_ep{ep}.pt")

        if ep % 10 == 0 or ep == episodes:
            mr = sum(ep_rewards_batch) / len(ep_rewards_batch)
            md = sum(ep_dists_batch)   / len(ep_dists_batch)
            mc = sum(ep_complete_batch)/ len(ep_complete_batch)
            print(f"Ep {ep:4d} | R={mr:7.2f} | dist={md:6.2f} "
                  f"| complete={mc:.2f} | loss={mean_loss:.4f} "
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
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PPO for CPDPTW")
    p.add_argument("--model",
                   choices=["quantum", "qaoa", "classical", "classical-qaoa",
                            "node-quantum", "node-qaoa", "classical-large"],
                   default="quantum")
    p.add_argument("--node",      type=int,   default=3)
    p.add_argument("--capacity",  type=int,   default=5)
    p.add_argument("--episodes",  type=int,   default=300)
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--n-qubits",  type=int,   default=6)
    p.add_argument("--n-layers",  type=int,   default=3)
    p.add_argument("--lr",        type=float, default=2e-4)
    p.add_argument("--clip-eps",  type=float, default=0.2,
                   help="PPO surrogate clip epsilon (default: 0.2).")
    p.add_argument("--n-epochs",  type=int,   default=4,
                   help="Update epochs per collected buffer (default: 4).")
    p.add_argument("--eps-per-update", type=int, default=10,
                   help="Episodes to collect before each PPO update (default: 10).")
    p.add_argument("--minibatch", type=int,   default=32,
                   help="Minibatch size for PPO epochs (default: 32).")
    p.add_argument("--entropy",   type=float, default=0.05,
                   help="Entropy regularisation coefficient.")
    p.add_argument("--value-coef", type=float, default=0.5,
                   help="Critic loss weight.")
    p.add_argument("--gae-lambda", type=float, default=0.95,
                   help="GAE lambda (default: 0.95).")
    p.add_argument("--fixed-instance", action="store_true",
                   help="Train on one fixed problem instance.")
    p.add_argument("--out-prefix", default="ppo",
                   help="Path prefix for output .txt and .pt files.")
    p.add_argument("--save-every", type=int, default=100,
                   help="Save a checkpoint every N episodes (0 = disable).")
    p.add_argument("--encoding", choices=["ry", "rz", "ryrz"], default="ry",
                   help="Qubit encoding strategy (default: ry).")
    p.add_argument("--entanglement",
                   choices=["none", "ring", "brick", "all", "star"],
                   default="ring", help="Entanglement topology (default: ring).")
    args = p.parse_args()

    train_ppo(
        model_kind          = args.model,
        node                = args.node,
        capacity            = args.capacity,
        episodes            = args.episodes,
        gamma               = 0.99,
        gae_lambda          = args.gae_lambda,
        lr                  = args.lr,
        clip_eps            = args.clip_eps,
        n_epochs            = args.n_epochs,
        episodes_per_update = args.eps_per_update,
        minibatch_size      = args.minibatch,
        entropy_coef        = args.entropy,
        value_coef          = args.value_coef,
        fixed_instance      = args.fixed_instance,
        seed                = args.seed,
        out_prefix          = args.out_prefix,
        n_qubits            = args.n_qubits,
        n_layers            = args.n_layers,
        save_every          = args.save_every,
        encoding            = args.encoding,
        entanglement        = args.entanglement,
    )
