"""
cpdptw_env.py

Single-vehicle CPDPTW with a ONE-TIME-PER-NODE formulation, matched to the
quantum-RL experiments. This is intentionally a *simplified* sequential
decision formulation — it is the right scope for a quantum-learning feasibility
study, NOT a substitute for the full bi-modal CE-CPDPTW solved classically.

Node layout (indices):
    0                      depot
    1 .. n                 pickups        (demand > 0, single EARLIEST time)
    n+1 .. 2n              deliveries     (demand < 0, single LATEST time)

Precedence: delivery (i+n) is feasible only after pickup i has been visited.
Capacity:   load must stay within [0, Q] at all times.

State vector (returned by _get_state), shape (1, 4 + 6n):
    [ load/Q,
      time/T,
      cur_x, cur_y,                       (current node coordinates in [0,1]^2)
      pickup_open[1..n]   / T,
      delivery_close[1..n]/ T,
      demand[1..2n]       / Q,
      visited[1..2n]      (binary)  ]

Components cur_x/cur_y and visited[1..2n] were absent in the original design.
Without visited flags the MDP state is non-Markov: two states with identical
load/time but different visit histories are indistinguishable to the network.
Without position the agent cannot learn distance-minimising behaviour.

Reward is negative cost: travel distance plus a one-sided time-window penalty
(pickups penalised for EARLY arrival, deliveries for LATE arrival), matching the
asymmetric food-delivery reward used in the classical track.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


class CPDPTWEnv:
    def __init__(
        self,
        node: int = 5,
        vehicle_capacity: int = 5,
        *,
        horizon_min: int = 60,
        speed_units_per_min: float = 1.2,
        early_penalty_w: float = 0.004,   # pickup earliness weight (small)
        late_penalty_w: float = 0.010,    # delivery lateness weight (dominant)
        dist_scale: float = 1.0,          # lowered from 10.0: keeps distance
                                          # and time penalties on comparable scales
        rng_seed: Optional[int] = None,
    ):
        self.node = int(node)
        self.n_total = self.node * 2
        self.capacity = int(vehicle_capacity)
        self.time_frame = int(horizon_min)
        self.speed = float(speed_units_per_min)
        self.early_w = float(early_penalty_w)
        self.late_w = float(late_penalty_w)
        self.dist_scale = float(dist_scale)
        self._seed = rng_seed
        self._rng = np.random.default_rng(rng_seed)
        self.reset()

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def reset(self, *, regenerate: bool = True) -> Tuple[torch.Tensor, dict]:
        """Reset the episode.

        regenerate=True draws a fresh instance (new coords/demands/times).
        regenerate=False replays the same instance from the start — useful
        for evaluating a fixed policy on a fixed problem.
        """
        self.total_distance = 0.0
        self.vehicle_time = 0
        self.load = 0
        self.current_node = 0
        self.route: List[int] = [0]

        if regenerate or not hasattr(self, "city_coords"):
            self.city_coords = torch.tensor(
                self._rng.random((self.n_total + 1, 2)), dtype=torch.float32
            )
            pickup_demands = self._rng.integers(1, 4, size=self.node)
            self.demands = np.concatenate(
                ([0], pickup_demands, -pickup_demands)
            ).astype(np.int64)

            # One time per node: stochastic pickup readiness, delivery deadlines.
            lam = self.n_total
            inter = self._rng.exponential(scale=self.time_frame / lam, size=self.node)
            t_pickup = np.ceil(np.cumsum(inter)).astype(int)
            t_pickup = np.clip(t_pickup, 0, self.time_frame)
            t_delivery = np.clip(
                t_pickup + self._rng.integers(15, 30, size=self.node),
                0, self.time_frame,
            )

            self.time_window = np.zeros((self.n_total + 1, 2), dtype=int)
            self.time_window[1:self.node + 1, 0] = t_pickup          # pickup open
            self.time_window[self.node + 1:, 1] = t_delivery         # delivery close

            # Precompute the full distance matrix once per instance.
            diff = self.city_coords[:, None, :] - self.city_coords[None, :, :]
            self.dist_matrix = torch.sqrt((diff ** 2).sum(-1))       # (2n+1, 2n+1)

        self.visited = torch.zeros(self.n_total + 1, dtype=torch.bool)
        self.visited[0] = True
        return self._get_state(), {"route": self.route.copy()}

    def step(self, action: int):
        if action < 0 or action > self.n_total:
            raise IndexError(f"Action {action} out of range [0,{self.n_total}]")

        # Infeasible-action guard: return state unchanged with a fixed penalty.
        if not self._is_feasible(action):
            return (self._get_state(), torch.tensor(-1.0, dtype=torch.float32),
                    False, False, {"route": self.route.copy(), "infeasible": True})

        demand = int(self.demands[action])
        new_load = self.load + demand

        # Travel (use precomputed matrix).
        dist = float(self.dist_matrix[self.current_node, action].item())
        dt = int(np.ceil(dist / max(self.speed, 1e-6)))
        self.vehicle_time += dt
        self.total_distance += dist

        # One-sided time-window penalty.
        open_t, close_t = self.time_window[action]
        time_pen = 0.0
        if open_t > 0 and self.vehicle_time < open_t:          # pickup early
            time_pen += self.early_w * (open_t - self.vehicle_time) ** 2
        if close_t > 0 and self.vehicle_time > close_t:        # delivery late
            time_pen += self.late_w * (self.vehicle_time - close_t) ** 2

        reward = -(dist / self.dist_scale + time_pen)

        # Commit.
        self.load = new_load
        self.current_node = int(action)
        self.visited[action] = True
        self.route.append(int(action))

        done = bool(self.visited[1:].all().item())
        if done:
            # Return to depot: include final leg in cost.
            ret_dist = float(self.dist_matrix[self.current_node, 0].item())
            ret_dt = int(np.ceil(ret_dist / max(self.speed, 1e-6)))
            self.total_distance += ret_dist
            self.vehicle_time += ret_dt
            self.route.append(0)
            reward -= ret_dist / self.dist_scale
        return (self._get_state(), torch.tensor(reward, dtype=torch.float32),
                done, False, {"route": self.route.copy(), "infeasible": False})

    # ------------------------------------------------------------------ #
    # Feasibility / helpers
    # ------------------------------------------------------------------ #
    def _is_feasible(self, action: int) -> bool:
        if action == 0:
            return False                       # depot already visited
        if self.visited[action]:
            return False
        if action > self.node and not self.visited[action - self.node]:
            return False                       # delivery before its pickup
        new_load = self.load + int(self.demands[action])
        if new_load > self.capacity or new_load < 0:
            return False
        return True

    def valid_actions(self) -> List[int]:
        return [a for a in range(1, self.n_total + 1) if self._is_feasible(a)]

    def action_mask(self) -> torch.Tensor:
        """Boolean mask of shape (n_actions,), True where feasible."""
        mask = torch.zeros(self.n_actions, dtype=torch.bool)
        for a in self.valid_actions():
            mask[a] = True
        return mask

    def get_route(self) -> List[int]:
        return self.route.copy()

    def _get_state(self) -> torch.Tensor:
        Q = max(1.0, float(self.capacity))
        T = max(1.0, float(self.time_frame))
        vt = float(self.vehicle_time)
        load_enc = torch.tensor([self.load / Q], dtype=torch.float32)
        time_enc = torch.tensor([vt / T], dtype=torch.float32)
        pos_enc = self.city_coords[self.current_node]               # (2,) in [0,1]^2
        # Remaining time until pickup window opens (0 if window already open).
        pickup_remain = torch.tensor(
            np.maximum(0, self.time_window[1:self.node + 1, 0] - vt),
            dtype=torch.float32) / T
        # Remaining time before delivery deadline (0 if already late).
        delivery_remain = torch.tensor(
            np.maximum(0, self.time_window[self.node + 1:, 1] - vt),
            dtype=torch.float32) / T
        demand_enc = torch.tensor(self.demands[1:], dtype=torch.float32) / Q
        visited_enc = self.visited[1:].float()                      # (2n,) binary
        state = torch.cat([load_enc, time_enc, pos_enc, pickup_remain,
                           delivery_remain, demand_enc, visited_enc])
        return state.unsqueeze(0)

    @property
    def n_actions(self) -> int:
        return self.n_total + 1

    @property
    def n_observations(self) -> int:
        return 4 + 6 * self.node  # 2 global + 2 position + 4n time/demand + 2n visited

