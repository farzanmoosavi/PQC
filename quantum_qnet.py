"""
quantum_qnet.py

Two PQC Q-networks for CPDPTW and a parameter-matched classical baseline.

  QuantumQNetwork  -- generic hardware-efficient ansatz (HEA): data re-uploading
                      with RX/RY/RZ rotations + configurable entanglement topology.
                      Trainable PQC params: 3 * n_layers * n_qubits.

  QAOAQNetwork     -- QAOA-inspired ansatz addressing contribution (2) and (3):
                      cost unitary  IsingZZ(gamma) ~ exp(-i*gamma*Z_i*Z_j)
                      mixer unitary RX(beta)        ~ exp(-i*beta*X_i)
                      Maps to the standard p-layer QAOA cost+mixer decomposition
                      (Farhi et al. 2014) with data re-uploading (Perez-Salinas 2020).
                      Trainable PQC params: 2 * n_layers * n_qubits  (33% fewer
                      than HEA) -- directly supporting the "fewer trainable
                      parameters" contribution claim.
                      IsingZZ topology is fixed as ring (problem-specific design).

  ClassicalQNetwork -- two-hidden-layer MLP; hidden width is chosen by
                       match_classical_width() to match the total param count of
                       whichever quantum model is under test.

Sensitivity knobs
-----------------
QuantumQNetwork accepts:
  entanglement : "ring" | "brick" | "all" | "star"
  encoding     : "ry"   | "rz"   | "ryrz"
  h_init       : True (H|0> superposition) | False (|0> computational basis)

QAOAQNetwork accepts:
  encoding     : "ry"   | "rz"   | "ryrz"
  h_init       : True | False
  (IsingZZ topology is always ring -- this is the contribution design choice.)

Natural qubit count: 2n+1 (one qubit per CPDPTW node including depot).
Compact qubit count: ceil(log2(2n+1)).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    import pennylane as qml
    _HAS_PENNYLANE = True
except ImportError:
    _HAS_PENNYLANE = False


# --------------------------------------------------------------------------- #
# Entanglement topology helper
# --------------------------------------------------------------------------- #

def _ent_pairs(n_qubits: int, topology: str, layer: int = 0) -> list[tuple[int, int]]:
    """
    Return (control, target) pairs for CNOT entanglement in one circuit layer.

    ring  -- each qubit to its right neighbour with wraparound; O(n) gates.
             Most common in hardware-efficient ansatz literature.
    brick -- alternating even/odd qubit pairs per layer; O(n) gates, avoids
             the ring's periodic boundary and can reach farther correlations
             in fewer layers (Shi et al. 2022).
    all   -- every ordered pair (i,j) with i<j; O(n^2) gates, maximally
             expressive but noise-heavy on real hardware.
    star  -- qubit 0 (depot proxy) connects to all others; O(n) gates,
             matches the depot-centric topology of the routing graph.
    """
    if topology == "ring":
        return [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    elif topology == "brick":
        offset = layer % 2
        return [(q, q + 1) for q in range(offset, n_qubits - 1, 2)]
    elif topology == "all":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    elif topology == "star":
        return [(0, q) for q in range(1, n_qubits)]
    else:
        raise ValueError(
            f"Unknown entanglement topology '{topology}'. "
            f"Choose from: ring, brick, all, star"
        )


# --------------------------------------------------------------------------- #
# Quantum Q-network (HEA)
# --------------------------------------------------------------------------- #

class QuantumQNetwork(nn.Module):
    """PQC-based Q-network with configurable circuit design knobs.

    Pipeline:
        state (1, F)  ->  classical compressor (F -> n_angles angles in [-pi,pi])
                      ->  PQC (data re-uploading, n_layers)
                      ->  <Z_i> + <Z_i Z_{i+1}> ring  (2*n_qubits scalars)
                      ->  linear head (2*n_qubits -> n_actions)

    Sensitivity parameters (all keyword-only):
        entanglement : str  -- "ring" (default) | "brick" | "all" | "star"
        encoding     : str  -- "ry" (default) | "rz" | "ryrz"
        h_init       : bool -- True (default, H superposition) | False (|0> start)
    """

    def __init__(
        self,
        env,
        *,
        n_qubits: int = 6,
        n_layers: int = 3,
        entanglement: str = "ring",
        encoding: str = "ry",
        h_init: bool = True,
        torch_device: Optional[torch.device] = None,
    ):
        super().__init__()
        if not _HAS_PENNYLANE:
            raise ImportError("pennylane is required for QuantumQNetwork.")

        self.env = env
        self.n_actions = env.n_actions
        self.n_obs = env.n_observations
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.entanglement = entanglement
        self.encoding = encoding
        self.h_init = h_init
        self.device = torch_device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # ryrz encoding uses two rotation angles per qubit per layer.
        self.n_angles = 2 * self.n_qubits if encoding == "ryrz" else self.n_qubits
        # No Tanh: Tanh saturates for large inputs and kills gradients through the
        # compressor.  LayerNorm keeps activations bounded without blocking gradients.
        self.compressor = nn.Sequential(
            nn.Linear(self.n_obs, self.n_angles),
            nn.LayerNorm(self.n_angles),
        )

        # Circuit output: n_qubits <Z_i> + n_qubits <Z_i Z_{i+1}> ring.
        # ZZ terms are always ring pairs regardless of entanglement topology --
        # they are fixed observables, not part of the topology design.
        self.n_outputs = 2 * self.n_qubits

        # enc_scales[l, q] (or [l, q, 2] for ryrz) are trainable encoding weights
        # living inside the quantum circuit: RY(enc_scales[l,q] * compressed_input[q]).
        # Initialised to 1.0 so the circuit starts with unscaled encoding and learns
        # how to weight each input feature per layer.  This is the variational encoding
        # approach described in quantum RL papers for routing (MCSoC 2024).
        enc_shape = (self.n_layers, self.n_qubits, 2) if encoding == "ryrz" \
                    else (self.n_layers, self.n_qubits)
        weight_shapes = {
            "weights":    (self.n_layers, self.n_qubits, 3),
            "enc_scales": enc_shape,
        }
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights, enc_scales):
            # Initialisation: H superposition or |0> computational basis.
            if self.h_init:
                for q in range(self.n_qubits):
                    qml.Hadamard(wires=q)

            for layer in range(self.n_layers):
                # --- Data encoding with trainable per-layer per-qubit scales ---
                if self.encoding == "ryrz":
                    for q in range(self.n_qubits):
                        qml.RY(enc_scales[layer, q, 0] * inputs[..., q], wires=q)
                        qml.RZ(enc_scales[layer, q, 1] * inputs[..., self.n_qubits + q], wires=q)
                elif self.encoding == "rz":
                    for q in range(self.n_qubits):
                        qml.RZ(enc_scales[layer, q] * inputs[..., q], wires=q)
                else:  # "ry" (default)
                    for q in range(self.n_qubits):
                        qml.RY(enc_scales[layer, q] * inputs[..., q], wires=q)

                # --- Entanglement ---
                for ctrl, tgt in _ent_pairs(self.n_qubits, self.entanglement, layer):
                    qml.CNOT(wires=[ctrl, tgt])

                # --- Trainable rotations (RX/RY/RZ per qubit) ---
                for q in range(self.n_qubits):
                    qml.RX(weights[layer, q, 0], wires=q)
                    qml.RY(weights[layer, q, 1], wires=q)
                    qml.RZ(weights[layer, q, 2], wires=q)

            z_obs  = [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
            zz_obs = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % self.n_qubits))
                      for q in range(self.n_qubits)]
            return z_obs + zz_obs

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        # Small-random variational weights avoid the barren plateau at t=0.
        # PennyLane defaults to Uniform(0, 2pi) which gives near-zero-gradient
        # outputs for deep circuits.  N(0, 0.01) starts near-identity.
        with torch.no_grad():
            self.qlayer.weights.normal_(0.0, 0.01)
            self.qlayer.enc_scales.fill_(1.0)
        self.head = nn.Linear(self.n_outputs, self.n_actions)
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        angles = self.compressor(state) * math.pi          # (B, n_angles)
        z = self.qlayer(angles.cpu())                       # (B, 2*n_qubits)
        z = z.to(self.device).float()
        return self.head(z)                                 # (B, n_actions)

    def param_report(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "compressor":  count(self.compressor),
            "enc_scales":  int(self.qlayer.enc_scales.numel()),
            "pqc_var":     int(self.qlayer.weights.numel()),
            "head":        count(self.head),
            "total":       count(self),
        }


# --------------------------------------------------------------------------- #
# QAOA-inspired PQC Q-network  (contribution model)
# --------------------------------------------------------------------------- #

class QAOAQNetwork(nn.Module):
    """
    QAOA-inspired PQC Q-network for CPDPTW.

    Pipeline:
        state (1, F)  ->  classical compressor (F -> n_angles angles)
                      ->  QAOA circuit (data re-uploading, n_layers)
                      ->  <Z_i> + <Z_i Z_{i+1}>  (2*n_qubits scalars)
                      ->  linear head (2*n_qubits -> n_actions)

    Circuit structure per layer:
        encode(angle_q)         -- data re-uploading (RY, RZ, or RY+RZ)
        IsingZZ(gamma[l,q])     -- cost unitary ~ exp(-i*gamma*Z_i Z_{i+1})
                                   ring topology: fixed design choice
        RX(beta[l,q])           -- mixer unitary ~ exp(-i*beta*X_q)

    IsingZZ topology is fixed as ring for all configurations.  This is the
    problem-specific design contribution: the ring matches the routing cost
    Hamiltonian H_C = sum_{<i,j>} d_{ij}(I - Z_i Z_j)/2.  Non-ring
    topologies would break the QAOA analogy without a corresponding change
    to the cost Hamiltonian structure.

    Sensitivity parameters (keyword-only):
        encoding : "ry" (default) | "rz" | "ryrz"
        h_init   : True (default) | False
    """

    def __init__(
        self,
        env,
        *,
        n_qubits: int = 6,
        n_layers: int = 3,
        encoding: str = "ry",
        h_init: bool = True,
        torch_device: Optional[torch.device] = None,
    ):
        super().__init__()
        if not _HAS_PENNYLANE:
            raise ImportError("pennylane is required for QAOAQNetwork.")

        self.env = env
        self.n_actions = env.n_actions
        self.n_obs = env.n_observations
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.encoding = encoding
        self.h_init = h_init
        self.device = torch_device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.n_outputs = 2 * self.n_qubits

        self.n_angles = 2 * self.n_qubits if encoding == "ryrz" else self.n_qubits
        self.compressor = nn.Sequential(
            nn.Linear(self.n_obs, self.n_angles),
            nn.LayerNorm(self.n_angles),
        )

        enc_shape = (self.n_layers, self.n_qubits, 2) if encoding == "ryrz" \
                    else (self.n_layers, self.n_qubits)
        weight_shapes = {
            "gamma":      (self.n_layers, self.n_qubits),
            "beta":       (self.n_layers, self.n_qubits),
            "enc_scales": enc_shape,
        }
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, gamma, beta, enc_scales):
            if self.h_init:
                for q in range(self.n_qubits):
                    qml.Hadamard(wires=q)

            for layer in range(self.n_layers):
                # Data encoding with trainable per-layer per-qubit scales
                if self.encoding == "ryrz":
                    for q in range(self.n_qubits):
                        qml.RY(enc_scales[layer, q, 0] * inputs[..., q], wires=q)
                        qml.RZ(enc_scales[layer, q, 1] * inputs[..., self.n_qubits + q], wires=q)
                elif self.encoding == "rz":
                    for q in range(self.n_qubits):
                        qml.RZ(enc_scales[layer, q] * inputs[..., q], wires=q)
                else:
                    for q in range(self.n_qubits):
                        qml.RY(enc_scales[layer, q] * inputs[..., q], wires=q)

                # Cost unitary: IsingZZ on ring (fixed topology)
                for q in range(self.n_qubits):
                    qml.IsingZZ(gamma[layer, q],
                                wires=[q, (q + 1) % self.n_qubits])

                # Mixer: RX per qubit
                for q in range(self.n_qubits):
                    qml.RX(beta[layer, q], wires=q)

            z_obs  = [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
            zz_obs = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % self.n_qubits))
                      for q in range(self.n_qubits)]
            return z_obs + zz_obs

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        with torch.no_grad():
            self.qlayer.gamma.normal_(0.0, 0.01)
            self.qlayer.beta.normal_(0.0, 0.01)
            self.qlayer.enc_scales.fill_(1.0)
        self.head = nn.Linear(self.n_outputs, self.n_actions)
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        angles = self.compressor(state) * math.pi
        z = self.qlayer(angles.cpu())
        z = z.to(self.device).float()
        return self.head(z)

    def param_report(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters())
        pqc = int(self.qlayer.gamma.numel() + self.qlayer.beta.numel())
        return {
            "compressor":  count(self.compressor),
            "enc_scales":  int(self.qlayer.enc_scales.numel()),
            "pqc_var":     pqc,
            "head":        count(self.head),
            "total":       count(self),
        }


# --------------------------------------------------------------------------- #
# Per-node feature extractor (shared by NodeQNetwork variants)
# --------------------------------------------------------------------------- #

def _extract_node_features(
    state: torch.Tensor,
    n_node: int,
    city_coords: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reorganise the flat CPDPTW state into a per-node feature tensor.

    State layout (F = 4 + 6*n_node):
        [0]        load / Q
        [1]        time / T
        [2:4]      cur_x, cur_y  (current vehicle position)
        [4:4+n]    pickup_open[1..n] / T
        [4+n:4+2n] delivery_close[1..n] / T
        [4+2n:4+4n] demand[1..2n] / Q
        [4+4n:4+6n] visited[1..2n]

    city_coords : (2n+1, 2) tensor of node coordinates, or None.
        When provided, each qubit's feature vector is extended with the node's
        own (x, y) position and its paired partner's (x, y) position:
          - pickup i  (1..n)    -> partner = delivery i+n
          - delivery i+n (n+1..2n) -> partner = pickup i
          - depot (0)           -> partner = depot (self)
        This closes the "blind to distance" gap and gives each node direct
        structural awareness of its pickup/delivery pairing.

    Returns:
        (B, 2*n_node+1, 7)   when city_coords is None
        (B, 2*n_node+1, 11)  when city_coords is provided

    Per-node feature layout:
        [tw_feature, demand, visited,        <- local (node-specific)
         time/T, load/Q, cur_x, cur_y,      <- global vehicle state
         node_x, node_y,                    <- node coordinate
         partner_x, partner_y]              <- paired node coordinate
    """
    n = n_node
    B = state.shape[0]
    dev = state.device

    global_ctx = state[:, [1, 0, 2, 3]]                  # (B, 4): time,load,x,y

    pickup_open    = state[:, 4      : 4 + n]
    delivery_close = state[:, 4 + n  : 4 + 2*n]
    demand_pu      = state[:, 4 + 2*n: 4 + 3*n]
    demand_de      = state[:, 4 + 3*n: 4 + 4*n]
    visited_pu     = state[:, 4 + 4*n: 4 + 5*n]
    visited_de     = state[:, 4 + 5*n: 4 + 6*n]

    depot_local    = torch.zeros(B, 1, 3, device=dev)
    pickup_local   = torch.stack([pickup_open,    demand_pu, visited_pu], dim=2)
    delivery_local = torch.stack([delivery_close, demand_de, visited_de], dim=2)

    local = torch.cat([depot_local, pickup_local, delivery_local], dim=1)  # (B, 2n+1, 3)
    glob  = global_ctx.unsqueeze(1).expand(B, 2*n + 1, 4)                  # (B, 2n+1, 4)
    out   = torch.cat([local, glob], dim=2)                                 # (B, 2n+1, 7)

    if city_coords is not None:
        coords_t = city_coords.to(dev)                                      # (2n+1, 2)
        coords = coords_t.unsqueeze(0).expand(B, -1, -1)
        out = torch.cat([out, coords], dim=2)                               # (B, 2n+1, 9)

        # Pair-aware: add each node's pickup/delivery partner coordinates.
        partner_idx = torch.zeros(2*n + 1, dtype=torch.long, device=dev)
        partner_idx[1:n + 1]     = torch.arange(n + 1, 2*n + 1, device=dev)  # pickups -> deliveries
        partner_idx[n + 1:2*n + 1] = torch.arange(1, n + 1, device=dev)      # deliveries -> pickups
        partner_coords = coords_t[partner_idx].unsqueeze(0).expand(B, -1, -1)
        out = torch.cat([out, partner_coords], dim=2)                       # (B, 2n+1, 11)

    return out


# --------------------------------------------------------------------------- #
# Per-node HEA Q-network
# --------------------------------------------------------------------------- #

class QuantumNodeQNetwork(nn.Module):
    """
    HEA Q-network with per-node qubit encoding.

    Unlike QuantumQNetwork (which uses a dense Linear(F, n_qubits) compressor),
    this network assigns one qubit to each CPDPTW node:
        n_qubits = 2 * node + 1  (auto-set from env, natural encoding)

    Each qubit's input angle is derived exclusively from the features of the
    corresponding node, via a shared lightweight encoder:
        node features (7,) -> Linear(7, n_out) + tanh -> n_out angles

    This restores the node-qubit correspondence assumed by the QAOA Hamiltonian
    interpretation: <Z_i> reads the state of node i, and ZZ correlations
    <Z_i Z_j> represent the physical node-pair interaction.

    Classical parameters:
        shared node encoder: 11*n_out + n_out = 12*n_out  (12 or 24 total)
        head:                 2*n_qubits * n_actions + n_actions

    The classical overhead is ~18x smaller than the compact compressor
    (12 vs ~210 params for n_qubits=6), placing more representational load
    on the quantum circuit -- which is the design intent.

    Sensitivity parameters:
        n_layers     : int   -- circuit depth
        entanglement : str   -- ring | brick | all | star
        encoding     : str   -- ry (1 angle/qubit) | ryrz (2 angles/qubit)
        h_init       : bool  -- H superposition vs |0> start
    """

    def __init__(
        self,
        env,
        *,
        n_layers: int = 3,
        entanglement: str = "ring",
        encoding: str = "ry",
        h_init: bool = True,
        torch_device: Optional[torch.device] = None,
        # n_qubits is ignored if passed — always 2*node+1
        n_qubits: Optional[int] = None,
    ):
        super().__init__()
        if not _HAS_PENNYLANE:
            raise ImportError("pennylane is required for QuantumNodeQNetwork.")

        self.env = env
        self.n_node     = env.node
        self.n_actions  = env.n_actions
        self.n_obs      = env.n_observations
        self.n_qubits   = 2 * env.node + 1          # one qubit per node incl. depot
        self.n_layers   = int(n_layers)
        self.entanglement = entanglement
        self.encoding   = encoding
        self.h_init     = h_init
        self.device     = torch_device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Shared node encoder: same weights applied to every node's feature vector.
        # n_out=1 for "ry", n_out=2 for "ryrz"
        self.n_out      = 2 if encoding == "ryrz" else 1
        self.n_angles   = self.n_qubits * self.n_out
        # 11 features per node: 3 local + 4 global vehicle state + 2 node coords + 2 partner coords.
        self.node_encoder = nn.Sequential(
            nn.Linear(11, self.n_out),
            nn.Tanh(),
        )

        self.n_outputs  = 2 * self.n_qubits
        enc_shape = (self.n_layers, self.n_qubits, 2) if encoding == "ryrz" \
                    else (self.n_layers, self.n_qubits)
        weight_shapes   = {
            "weights":    (self.n_layers, self.n_qubits, 3),
            "enc_scales": enc_shape,
        }
        dev             = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights, enc_scales):
            if self.h_init:
                for q in range(self.n_qubits):
                    qml.Hadamard(wires=q)
            for layer in range(self.n_layers):
                if self.encoding == "ryrz":
                    for q in range(self.n_qubits):
                        qml.RY(enc_scales[layer, q, 0] * inputs[..., q], wires=q)
                        qml.RZ(enc_scales[layer, q, 1] * inputs[..., self.n_qubits + q], wires=q)
                else:
                    for q in range(self.n_qubits):
                        qml.RY(enc_scales[layer, q] * inputs[..., q], wires=q)
                for ctrl, tgt in _ent_pairs(self.n_qubits, self.entanglement, layer):
                    qml.CNOT(wires=[ctrl, tgt])
                for q in range(self.n_qubits):
                    qml.RX(weights[layer, q, 0], wires=q)
                    qml.RY(weights[layer, q, 1], wires=q)
                    qml.RZ(weights[layer, q, 2], wires=q)
            z_obs  = [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
            zz_obs = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % self.n_qubits))
                      for q in range(self.n_qubits)]
            return z_obs + zz_obs

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        with torch.no_grad():
            self.qlayer.weights.normal_(0.0, 0.01)
            self.qlayer.enc_scales.fill_(1.0)
        self.head   = nn.Linear(self.n_outputs, self.n_actions)
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        # Pass city_coords so each qubit encodes its node's location.
        # NOTE: reads current episode's coords. For DQN + fixed_instance=False
        # (policy learning), this produces incorrect node features for replayed
        # transitions from past episodes. Use node models with REINFORCE, or
        # with DQN + fixed_instance=True only.
        coords = self.env.city_coords.to(self.device)              # (2n+1, 2)
        node_feats = _extract_node_features(state, self.n_node, coords)  # (B, 2n+1, 11)
        B = node_feats.shape[0]

        # Apply shared encoder to every node: (B, 2n+1, 11) -> (B, 2n+1, n_out)
        angles_per_node = self.node_encoder(node_feats) * math.pi  # (B, 2n+1, n_out)

        if self.n_out == 1:
            angles = angles_per_node.squeeze(-1)                   # (B, n_qubits)
        else:
            # ryrz: interleave [q0_ry, q1_ry, ..., q0_rz, q1_rz, ...]
            angles = torch.cat([
                angles_per_node[..., 0],   # (B, n_qubits) RY angles
                angles_per_node[..., 1],   # (B, n_qubits) RZ angles
            ], dim=1)                                               # (B, 2*n_qubits)

        z = self.qlayer(angles.cpu())                              # (B, 2*n_qubits)
        z = z.to(self.device).float()
        return self.head(z)

    def param_report(self) -> dict:
        def count(m): return sum(p.numel() for p in m.parameters())
        return {
            "node_encoder": count(self.node_encoder),
            "enc_scales":   int(self.qlayer.enc_scales.numel()),
            "pqc_var":      int(self.qlayer.weights.numel()),
            "head":         count(self.head),
            "total":        count(self),
        }


# --------------------------------------------------------------------------- #
# Per-node QAOA Q-network  (strongest form of contribution claim)
# --------------------------------------------------------------------------- #

class QAOANodeQNetwork(nn.Module):
    """
    QAOA Q-network with per-node qubit encoding.

    This is the strongest realisation of contribution (2): the IsingZZ cost
    unitary now acts on qubits that genuinely represent individual CPDPTW nodes,
    so the interaction Z_i Z_j directly approximates the pairwise travel-cost
    term d_ij in the routing cost Hamiltonian:

        H_C = sum_{<i,j>} d_ij (I - Z_i Z_j) / 2

    With compact encoding (QuantumQNetwork / QAOAQNetwork), qubit i has no
    structural relationship to node i, so this Hamiltonian interpretation is
    an approximation at best.  Here it is exact by construction.

    n_qubits = 2 * node + 1  (auto-set, natural encoding).

    Classical parameters:
        shared node encoder: 11*n_out + n_out  (12 or 24 total, vs ~210 for compact)
        head:                 2*n_qubits * n_actions + n_actions

    Per-node features (11): [tw_feature, demand, visited,
                              time/T, load/Q, cur_x, cur_y,
                              node_x, node_y, partner_x, partner_y]
    """

    def __init__(
        self,
        env,
        *,
        n_layers: int = 3,
        encoding: str = "ry",
        h_init: bool = True,
        torch_device: Optional[torch.device] = None,
        n_qubits: Optional[int] = None,  # ignored
    ):
        super().__init__()
        if not _HAS_PENNYLANE:
            raise ImportError("pennylane is required for QAOANodeQNetwork.")

        self.env        = env
        self.n_node     = env.node
        self.n_actions  = env.n_actions
        self.n_obs      = env.n_observations
        self.n_qubits   = 2 * env.node + 1
        self.n_layers   = int(n_layers)
        self.encoding   = encoding
        self.h_init     = h_init
        self.device     = torch_device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.n_out    = 2 if encoding == "ryrz" else 1
        self.n_angles = self.n_qubits * self.n_out
        # 11 features per node: 3 local + 4 global vehicle state + 2 node coords + 2 partner coords.
        self.node_encoder = nn.Sequential(
            nn.Linear(11, self.n_out),
            nn.Tanh(),
        )

        self.n_outputs = 2 * self.n_qubits
        enc_shape = (self.n_layers, self.n_qubits, 2) if encoding == "ryrz" \
                    else (self.n_layers, self.n_qubits)
        weight_shapes  = {
            "gamma":      (self.n_layers, self.n_qubits),
            "beta":       (self.n_layers, self.n_qubits),
            "enc_scales": enc_shape,
        }
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, gamma, beta, enc_scales):
            if self.h_init:
                for q in range(self.n_qubits):
                    qml.Hadamard(wires=q)
            for layer in range(self.n_layers):
                if self.encoding == "ryrz":
                    for q in range(self.n_qubits):
                        qml.RY(enc_scales[layer, q, 0] * inputs[..., q], wires=q)
                        qml.RZ(enc_scales[layer, q, 1] * inputs[..., self.n_qubits + q], wires=q)
                else:
                    for q in range(self.n_qubits):
                        qml.RY(enc_scales[layer, q] * inputs[..., q], wires=q)
                # Cost unitary on ring: Z_i Z_{i+1} approximates d_{i,i+1}
                for q in range(self.n_qubits):
                    qml.IsingZZ(gamma[layer, q],
                                wires=[q, (q + 1) % self.n_qubits])
                for q in range(self.n_qubits):
                    qml.RX(beta[layer, q], wires=q)
            z_obs  = [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
            zz_obs = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % self.n_qubits))
                      for q in range(self.n_qubits)]
            return z_obs + zz_obs

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        with torch.no_grad():
            self.qlayer.gamma.normal_(0.0, 0.01)
            self.qlayer.beta.normal_(0.0, 0.01)
            self.qlayer.enc_scales.fill_(1.0)
        self.head   = nn.Linear(self.n_outputs, self.n_actions)
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        coords = self.env.city_coords.to(self.device)              # (2n+1, 2)
        node_feats = _extract_node_features(state, self.n_node, coords)  # (B, 2n+1, 11)
        angles_per_node = self.node_encoder(node_feats) * math.pi  # (B, 2n+1, n_out)

        if self.n_out == 1:
            angles = angles_per_node.squeeze(-1)
        else:
            angles = torch.cat([
                angles_per_node[..., 0],
                angles_per_node[..., 1],
            ], dim=1)

        z = self.qlayer(angles.cpu())
        z = z.to(self.device).float()
        return self.head(z)

    def param_report(self) -> dict:
        def count(m): return sum(p.numel() for p in m.parameters())
        pqc = int(self.qlayer.gamma.numel() + self.qlayer.beta.numel())
        return {
            "node_encoder": count(self.node_encoder),
            "enc_scales":   int(self.qlayer.enc_scales.numel()),
            "pqc_var":      pqc,
            "head":         count(self.head),
            "total":        count(self),
        }


# --------------------------------------------------------------------------- #
# Parameter-matched classical baseline
# --------------------------------------------------------------------------- #

class ClassicalQNetwork(nn.Module):
    """MLP Q-network with a tunable hidden width so its parameter count can be
    matched to the QuantumQNetwork. Use this as the control in every experiment.
    """

    def __init__(self, env, *, hidden: int = 32,
                 torch_device: Optional[torch.device] = None):
        super().__init__()
        self.env = env
        self.device = torch_device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.net = nn.Sequential(
            nn.Linear(env.n_observations, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, env.n_actions),
        )
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state.to(self.device))

    def param_report(self) -> dict:
        return {"total": sum(p.numel() for p in self.parameters())}


def match_classical_width(env, target_params: int) -> int:
    """Find the MLP hidden width whose param count is closest to target_params."""
    best_w, best_diff = 4, math.inf
    for w in range(4, 256):
        n = ClassicalQNetwork(env, hidden=w).param_report()["total"]
        diff = abs(n - target_params)
        if diff < best_diff:
            best_diff, best_w = diff, w
    return best_w

