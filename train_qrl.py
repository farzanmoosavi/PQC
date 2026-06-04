"""
train_qrl.py

Double-DQN training loop for CPDPTW with either the quantum (PQC) or the
parameter-matched classical Q-network. Fixes the bugs in the original loop:

  * Proper minibatch forward (no per-sample Python loop).
  * Epsilon-greedy exploration with decay (the original was pure-greedy → collapse).
  * Action masking applied at BOTH selection and bootstrap (the original masked
    only at selection, so the target could bootstrap off infeasible actions).
  * Robust non-final handling (no .squeeze() shape bug on single non-final state).
  * Soft target update only after an optimisation step actually runs.

Run:
    python train_qrl.py --model quantum --node 5 --episodes 200
    python train_qrl.py --model classical --node 5 --episodes 200
The two are designed to be parameter-matched so results are comparable.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import namedtuple
from itertools import count
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cpdptw_env import CPDPTWEnv

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "next_mask"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Prioritised replay
# --------------------------------------------------------------------------- #

class PrioritizedReplayMemory:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.memory: List[Transition] = []
        self.priorities: List[float] = []
        self.pos = 0

    def push(self, *args):
        max_p = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
            self.priorities.append(max_p)
        else:
            self.memory[self.pos] = Transition(*args)
            self.priorities[self.pos] = max_p
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        probs = np.array(self.priorities, dtype=np.float64) ** self.alpha
        probs /= probs.sum()
        idx = np.random.choice(len(self.memory), batch_size, p=probs, replace=True)
        weights = (len(self.memory) * probs[idx]) ** (-self.beta)
        weights /= weights.max()
        samples = [self.memory[i] for i in idx]
        return samples, idx, torch.tensor(weights, dtype=torch.float32, device=device)

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = float(p) + 1e-5

    def __len__(self):
        return len(self.memory)


# --------------------------------------------------------------------------- #
# Action selection (masked epsilon-greedy)
# --------------------------------------------------------------------------- #

def select_action(net, env, state, eps: float) -> torch.Tensor:
    mask = env.action_mask().to(device)              # (n_actions,)
    if not mask.any():
        return torch.tensor([[0]], device=device, dtype=torch.long)
    if random.random() < eps:
        choices = mask.nonzero(as_tuple=True)[0]
        a = choices[random.randrange(len(choices))]
        return a.view(1, 1)
    with torch.no_grad():
        q = net(state)                               # (1, n_actions)
        q = q.masked_fill(~mask.unsqueeze(0), -float("inf"))
        return q.max(1).indices.view(1, 1)


# --------------------------------------------------------------------------- #
# Optimisation step (Double DQN, batched, masked bootstrap)
# --------------------------------------------------------------------------- #

def optimize(net, target_net, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return None
    trans, idx, weights = memory.sample(batch_size)
    batch = Transition(*zip(*trans))

    state_batch = torch.cat(batch.state).to(device)          # (B, F)
    action_batch = torch.cat(batch.action).to(device)        # (B, 1)
    reward_batch = torch.stack(batch.reward).to(device)      # (B,)

    non_final = torch.tensor([s is not None for s in batch.next_state],
                             device=device, dtype=torch.bool)
    q_values = net(state_batch).gather(1, action_batch).squeeze(1)   # (B,)

    next_values = torch.zeros(batch_size, device=device)
    nf_states = [s for s in batch.next_state if s is not None]
    if nf_states:
        nf_batch = torch.cat(nf_states).to(device)                   # (Bnf, F)
        nf_masks = torch.stack([m for s, m in
                                zip(batch.next_state, batch.next_mask)
                                if s is not None and m is not None]).to(device)  # (Bnf, A)
        with torch.no_grad():
            # Double DQN: online net picks the action, target net evaluates it.
            online_q = net(nf_batch).masked_fill(~nf_masks, -float("inf"))
            next_actions = online_q.argmax(1, keepdim=True)
            target_q = target_net(nf_batch)
            chosen = target_q.gather(1, next_actions).squeeze(1)
        next_values[non_final] = chosen

    expected = reward_batch + gamma * next_values
    td_error = q_values - expected
    loss = (weights * nn.functional.smooth_l1_loss(
        q_values, expected, reduction="none")).mean()

    memory.update_priorities(idx, td_error.detach().abs().cpu().numpy())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #

def build_net(model_kind: str, env, n_qubits: int = 6, n_layers: int = 3):
    if model_kind == "quantum":
        from quantum_qnet import QuantumQNetwork
        return QuantumQNetwork(env, n_qubits=n_qubits, n_layers=n_layers)
    elif model_kind == "qaoa":
        from quantum_qnet import QAOAQNetwork
        return QAOAQNetwork(env, n_qubits=n_qubits, n_layers=n_layers)
    elif model_kind == "node-quantum":
        # Per-node encoding: n_qubits = 2*node+1 auto, n_qubits arg ignored.
        from quantum_qnet import QuantumNodeQNetwork
        return QuantumNodeQNetwork(env, n_layers=n_layers)
    elif model_kind == "node-qaoa":
        # Per-node encoding + QAOA: strongest form of Hamiltonian approx claim.
        from quantum_qnet import QAOANodeQNetwork
        return QAOANodeQNetwork(env, n_layers=n_layers)
    elif model_kind == "classical":
        from quantum_qnet import QuantumQNetwork, ClassicalQNetwork, match_classical_width
        # Match params to HEA quantum at the same qubit/layer config.
        try:
            ref = QuantumQNetwork(env, n_qubits=n_qubits, n_layers=n_layers)
            target = ref.param_report()["total"]
            w = match_classical_width(env, target)
        except Exception:
            w = 32
        return ClassicalQNetwork(env, hidden=w)
    elif model_kind == "classical-qaoa":
        from quantum_qnet import QAOAQNetwork, ClassicalQNetwork, match_classical_width
        # Match params to QAOA (fewer PQC params than HEA).
        try:
            ref = QAOAQNetwork(env, n_qubits=n_qubits, n_layers=n_layers)
            target = ref.param_report()["total"]
            w = match_classical_width(env, target)
        except Exception:
            w = 16
        return ClassicalQNetwork(env, hidden=w)
    raise ValueError(model_kind)


def train(model_kind="quantum", node=5, capacity=5, episodes=200,
          batch_size=16, gamma=0.99, tau=0.005, lr=1e-3,
          eps_start=0.9, eps_end=0.05, eps_decay=600,
          fixed_instance=True, seed=0, out_prefix="qrl",
          n_qubits=6, n_layers=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = CPDPTWEnv(node=node, vehicle_capacity=capacity, rng_seed=seed)
    net = build_net(model_kind, env, n_qubits, n_layers).to(device)
    target_net = build_net(model_kind, env, n_qubits, n_layers).to(device)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(net.parameters(), lr=lr, amsgrad=True)
    memory = PrioritizedReplayMemory(10000)

    if hasattr(net, "param_report"):
        print(f"[{model_kind}] params: {net.param_report()}")

    losses, dists, rewards, feas_rates = [], [], [], []
    steps_done = 0

    for ep in range(episodes):
        # fixed_instance=True trains on one problem (learning a route);
        # False draws a new instance each episode (learning a policy).
        state, _ = env.reset(regenerate=not fixed_instance)
        state = state.to(device)
        total_r, n_steps, n_infeas = 0.0, 0, 0
        last_loss = None
        done = False
        action = torch.tensor([[0]], device=device, dtype=torch.long)
        eps = eps_end

        for t in count():
            eps = eps_end + (eps_start - eps_end) * math.exp(-steps_done / eps_decay)
            steps_done += 1
            action = select_action(net, env, state, eps)
            nxt, reward, done, _, info = env.step(action.item())
            n_infeas += int(info.get("infeasible", False))
            reward = reward.to(device)

            if done:
                next_state, next_mask = None, None
            else:
                next_state = nxt.to(device)
                next_mask = env.action_mask().to(device)

            memory.push(state, action, next_state, reward, next_mask)
            state = next_state if next_state is not None else state
            total_r += reward.item()
            n_steps += 1

            loss_val = optimize(net, target_net, memory, optimizer, batch_size, gamma)
            if loss_val is not None:
                last_loss = loss_val
                # Soft target update only when we actually optimised.
                with torch.no_grad():
                    for tp, p in zip(target_net.parameters(), net.parameters()):
                        tp.mul_(1 - tau).add_(tau * p)

            if done or n_steps > 4 * env.n_total:
                break

        # Penalise incomplete routes: one unit of reward per unvisited node,
        # so the agent learns completion is worth more than avoiding bad steps.
        if not done:
            unvisited = int((~env.visited[1:]).sum().item())
            # 5× multiplier so incompletion dominates partial-route time penalties
            # and the agent is always incentivised to complete the route.
            penalty = torch.tensor(-5.0 * unvisited, device=device)
            memory.push(state, action, None, penalty, None)
            total_r += penalty.item()

        if last_loss is not None:
            losses.append(last_loss)
        dists.append(env.total_distance)
        rewards.append(total_r)
        feas_rates.append(1.0 - n_infeas / max(n_steps, 1))

        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1:4d} | R={total_r:7.2f} | dist={env.total_distance:6.2f} "
                  f"| feas={feas_rates[-1]:.2f} | eps={eps:.3f} "
                  f"| loss={last_loss if last_loss else float('nan'):.4f}")

    np.savetxt(f"{out_prefix}_{model_kind}_rewards.txt", rewards)
    np.savetxt(f"{out_prefix}_{model_kind}_dists.txt", dists)
    np.savetxt(f"{out_prefix}_{model_kind}_losses.txt", losses)
    np.savetxt(f"{out_prefix}_{model_kind}_feas.txt", feas_rates)
    ckpt = f"{out_prefix}_{model_kind}.pt"
    torch.save(net.state_dict(), ckpt)
    print(f"Done. Saved {out_prefix}_{model_kind}_*.txt  checkpoint -> {ckpt}")
    return dict(rewards=rewards, dists=dists, losses=losses, feas=feas_rates, net=net)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   choices=["quantum", "qaoa", "classical", "classical-qaoa",
                            "node-quantum", "node-qaoa"],
                   default="quantum")
    p.add_argument("--node",      type=int, default=5)
    p.add_argument("--capacity",  type=int, default=5)
    p.add_argument("--episodes",  type=int, default=200)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--n-qubits",  type=int, default=6,
                   help="Qubit count. Natural encoding = 2*node+1.")
    p.add_argument("--n-layers",  type=int, default=3)
    p.add_argument("--fixed-instance", action="store_true",
                   help="Train on one fixed problem instance (route learning).")
    p.add_argument("--out-prefix", default="qrl",
                   help="Path prefix for output .txt and .pt files.")
    args = p.parse_args()
    train(model_kind=args.model, node=args.node, capacity=args.capacity,
          episodes=args.episodes, seed=args.seed,
          fixed_instance=args.fixed_instance,
          out_prefix=args.out_prefix,
          n_qubits=args.n_qubits, n_layers=args.n_layers)
