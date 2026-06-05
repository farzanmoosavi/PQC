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

# Probe lightning.qubit once at import time so every network class reuses the
# same result without repeating the detection.  Three possible backends:
#   lightning.qubit + adjoint  — C++ statevector, ~10-50x faster than default.qubit
#   default.qubit   + backprop — pure-Python/torch statevector, always available
#   (no pennylane)             — ImportError raised when a network is constructed
_LIGHTNING_OK: bool = False
if _HAS_PENNYLANE:
    try:
        import pennylane_lightning          # noqa: F401  (check the package exists)
        _test_dev = qml.device("lightning.qubit", wires=2)

        # Verify adjoint diff actually works end-to-end (version-mismatch guard).
        import pennylane as _qml
        @_qml.qnode(_test_dev, interface="torch", diff_method="adjoint")
        def _probe(x):
            _qml.RY(x[0], wires=0)
            return _qml.expval(_qml.PauliZ(0))
        import torch as _torch
        _x = _torch.tensor([0.1], requires_grad=True)
        _probe(_x)
        del _test_dev, _probe, _x, _qml, _torch
        _LIGHTNING_OK = True
    except Exception:
        _LIGHTNING_OK = False


def _make_qdevice(n_qubits: int):
    """Return (device, diff_method), preferring lightning.qubit when available.

    Falls back silently to default.qubit + backprop on CPU-only machines or
    when pennylane-lightning is not installed.
    """
    if not _HAS_PENNYLANE:
        raise ImportError("pennylane is required for quantum networks.")
    if _LIGHTNING_OK:
        return qml.device("lightning.qubit", wires=n_qubits), "adjoint"
    return qml.device("default.qubit", wires=n_qubits), "backprop"


def qdevice_info() -> str:
    """One-line string describing the active quantum simulation backend."""
    if not _HAS_PENNYLANE:
        return "pennylane: NOT installed"
    backend = "lightning.qubit (adjoint)" if _LIGHTNING_OK else "default.qubit (backprop)"
    gpu = "CUDA available" if (
        __import__("torch").cuda.is_available()) else "CPU only"
    return f"quantum backend: {backend}  |  torch: {gpu}"


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
        dev, _diff = _make_qdevice(self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=_diff)
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
        with torch.no_grad():
            self.qlayer.weights.normal_(0.0, 0.01)
            self.qlayer.enc_scales.fill_(1.0)
        self.head = nn.Linear(self.n_outputs, self.n_actions)
        self.to(self.device)
        self.qlayer.to('cpu')   # circuit always runs on CPU (Bug #6)

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
        classical = count(self.compressor) + count(self.head)
        total = count(self)
        return {
            "compressor":    count(self.compressor),
            "enc_scales":    int(self.qlayer.enc_scales.numel()),
            "pqc_var":       int(self.qlayer.weights.numel()),
            "head":          count(self.head),
            "total":         total,
            "classical_frac": round(classical / total, 3),
        }


# --------------------------------------------------------------------------- #
# QAOA-inspired PQC Q-network  (flat / compact encoding)
# --------------------------------------------------------------------------- #

class QAOAQNetwork(nn.Module):
    """
    HEA-ZZ PQC Q-network with QAOA-style circuit structure.

    NOTE: this is NOT a true QAOA implementation.  The IsingZZ angles
    gamma[l, q] are free trainable parameters per qubit per layer — the
    actual instance distances d_ij appear nowhere.  Because the flat
    compressor maps all 2n+1 nodes onto n_qubits < 2n+1 qubits, there is
    no one-to-one qubit-node correspondence, so real distances cannot be
    embedded.  The circuit structure (encode → ZZ-cost → RX-mixer) is
    QAOA-inspired but the cost layer is variational, not physics-derived.

    For the genuine distance-aware QAOA model see QAOANodeQNetwork.

    Pipeline:
        state (1, F)  ->  classical compressor (F -> n_angles angles)
                      ->  HEA-ZZ circuit (data re-uploading, n_layers)
                      ->  <Z_i> + <Z_i Z_{i+1}>  (2*n_qubits scalars)
                      ->  linear head (2*n_qubits -> n_actions)

    Circuit structure per layer:
        encode(angle_q)         -- data re-uploading (RY, RZ, or RY+RZ)
        IsingZZ(gamma[l,q])     -- free variational ZZ rotation (ring)
        RX(beta[l,q])           -- mixer unitary

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
        dev, _diff = _make_qdevice(self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=_diff)
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
        self.qlayer.to('cpu')   # circuit always runs on CPU (Bug #6)

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
        classical = count(self.compressor) + count(self.head)
        total = count(self)
        return {
            "compressor":    count(self.compressor),
            "enc_scales":    int(self.qlayer.enc_scales.numel()),
            "pqc_var":       pqc,
            "head":          count(self.head),
            "total":         total,
            "classical_frac": round(classical / total, 3),
        }


# --------------------------------------------------------------------------- #
# Per-node feature extractor (shared by NodeQNetwork variants)
# --------------------------------------------------------------------------- #

def _extract_node_features(state: torch.Tensor, n_node: int) -> torch.Tensor:
    """
    Reorganise the flat CPDPTW state into a per-node feature tensor (B, 2n+1, 11).

    State layout (F = 6 + 10*n_node):
        [0]              load / Q
        [1]              time / T
        [2:4]            cur_x, cur_y
        [4:4+n]          pickup_open / T
        [4+n:4+2n]       delivery_close / T
        [4+2n:4+4n]      demand / Q
        [4+4n:4+6n]      visited
        [4+6n:4+6n+2n+1] node_x[0..2n]
        [4+6n+2n+1:end]  node_y[0..2n]

    Coordinates are always present in the state (added by cpdptw_env), so the
    node models never need to reach into a stale env reference.

    Per-node output (11 features):
        [tw_feature, demand, visited,          local node-specific
         time/T, load/Q, cur_x, cur_y,        global vehicle state
         node_x, node_y,                       this node's coords
         partner_x, partner_y]                 paired node's coords
    """
    n = n_node
    B = state.shape[0]
    dev = state.device

    global_ctx = state[:, [1, 0, 2, 3]]           # (B, 4): time, load, cur_x, cur_y

    pickup_open    = state[:, 4      : 4 + n]
    delivery_close = state[:, 4 + n  : 4 + 2*n]
    demand_pu      = state[:, 4 + 2*n: 4 + 3*n]
    demand_de      = state[:, 4 + 3*n: 4 + 4*n]
    visited_pu     = state[:, 4 + 4*n: 4 + 5*n]
    visited_de     = state[:, 4 + 5*n: 4 + 6*n]

    # Coords are embedded in the state — no env reference needed.
    cs = 4 + 6 * n                                 # coord block start
    node_x = state[:, cs          : cs + (2*n+1)]  # (B, 2n+1)
    node_y = state[:, cs + (2*n+1): cs + 2*(2*n+1)]
    coords = torch.stack([node_x, node_y], dim=-1) # (B, 2n+1, 2)

    depot_local    = torch.zeros(B, 1, 3, device=dev)
    pickup_local   = torch.stack([pickup_open,    demand_pu, visited_pu], dim=2)
    delivery_local = torch.stack([delivery_close, demand_de, visited_de], dim=2)

    local = torch.cat([depot_local, pickup_local, delivery_local], dim=1)  # (B, 2n+1, 3)
    glob  = global_ctx.unsqueeze(1).expand(B, 2*n + 1, 4)                  # (B, 2n+1, 4)
    out   = torch.cat([local, glob, coords], dim=2)                        # (B, 2n+1, 9)

    partner_idx = torch.zeros(2*n + 1, dtype=torch.long, device=dev)
    partner_idx[1:n + 1]       = torch.arange(n + 1, 2*n + 1, device=dev)
    partner_idx[n + 1:2*n + 1] = torch.arange(1, n + 1, device=dev)
    partner_coords = coords[:, partner_idx, :]     # (B, 2n+1, 2)
    return torch.cat([out, partner_coords], dim=2) # (B, 2n+1, 11)


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
        dev, _diff      = _make_qdevice(self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=_diff)
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
        self.qlayer.to('cpu')   # circuit always runs on CPU; keep params there (Bug #6)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        # Coords are embedded in state — no stale env reference needed (Bug #1 fix).
        node_feats = _extract_node_features(state, self.n_node)    # (B, 2n+1, 11)
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
    True QAOA Q-network with per-node qubit encoding.

    Each qubit corresponds to one CPDPTW node (depot + n pickups + n deliveries),
    so the IsingZZ cost unitary implements a genuine instance-aware cost layer:

        H_C = sum_{q} d_{q, q+1} Z_q Z_{q+1}   (ring topology)

    where d_{q, q+1} is the actual travel distance between adjacent nodes,
    read from the state vector each forward pass.  gamma[layer] is a single
    trainable scalar per QAOA layer — the true QAOA parameterisation
    exp(-i gamma_l H_C) — not a free per-qubit weight.

    This is the key architectural distinction from QAOAQNetwork (flat), where
    qubits have no node correspondence and distances cannot be embedded.

    n_qubits = 2*node+1  (auto-set, natural encoding).
    """

    def __init__(
        self,
        env,
        *,
        n_layers: int = 3,
        encoding: str = "ry",
        h_init: bool = True,
        torch_device: Optional[torch.device] = None,
        n_qubits: Optional[int] = None,  # ignored — always 2*node+1
    ):
        super().__init__()
        if not _HAS_PENNYLANE:
            raise ImportError("pennylane is required for QAOANodeQNetwork.")

        self.n_node    = env.node
        self.n_actions = env.n_actions
        self.n_obs     = env.n_observations
        self.n_qubits  = 2 * env.node + 1
        self.n_layers  = int(n_layers)
        self.encoding  = encoding
        self.h_init    = h_init
        self.device    = torch_device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.n_out     = 2 if encoding == "ryrz" else 1
        self.n_angles  = self.n_qubits * self.n_out
        self.node_encoder = nn.Sequential(
            nn.Linear(11, self.n_out),
            nn.Tanh(),
        )

        self.n_outputs = 2 * self.n_qubits
        enc_shape = (self.n_layers, self.n_qubits, 2) if encoding == "ryrz" \
                    else (self.n_layers, self.n_qubits)
        weight_shapes = {
            "gamma":      (self.n_layers,),          # one scalar per QAOA layer
            "beta":       (self.n_layers, self.n_qubits),
            "enc_scales": enc_shape,
        }
        dev, _diff = _make_qdevice(self.n_qubits)
        n_q = self.n_qubits
        n_ang = self.n_angles

        @qml.qnode(dev, interface="torch", diff_method=_diff)
        def circuit(inputs, gamma, beta, enc_scales):
            # inputs layout: [angles (n_ang), ring_dist_norm (n_qubits)]
            if self.h_init:
                for q in range(n_q):
                    qml.Hadamard(wires=q)
            for layer in range(self.n_layers):
                if self.encoding == "ryrz":
                    for q in range(n_q):
                        qml.RY(enc_scales[layer, q, 0] * inputs[..., q], wires=q)
                        qml.RZ(enc_scales[layer, q, 1] * inputs[..., n_q + q], wires=q)
                else:
                    for q in range(n_q):
                        qml.RY(enc_scales[layer, q] * inputs[..., q], wires=q)
                # True QAOA cost layer: exp(-i gamma_l d_{q,q+1} Z_q Z_{q+1})
                for q in range(n_q):
                    qml.IsingZZ(gamma[layer] * inputs[..., n_ang + q],
                                wires=[q, (q + 1) % n_q])
                for q in range(n_q):
                    qml.RX(beta[layer, q], wires=q)
            z_obs  = [qml.expval(qml.PauliZ(q)) for q in range(n_q)]
            zz_obs = [qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % n_q))
                      for q in range(n_q)]
            return z_obs + zz_obs

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        with torch.no_grad():
            self.qlayer.gamma.normal_(0.0, 0.01)
            self.qlayer.beta.normal_(0.0, 0.01)
            self.qlayer.enc_scales.fill_(1.0)
        self.head = nn.Linear(self.n_outputs, self.n_actions)
        self.to(self.device)
        self.qlayer.to('cpu')   # circuit always runs on CPU (Bug #6)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        B = state.shape[0]

        # Coords extracted from state — no stale env reference (Bug #1 fix).
        node_feats = _extract_node_features(state, self.n_node)    # (B, 2n+1, 11)
        angles_per_node = self.node_encoder(node_feats) * math.pi  # (B, 2n+1, n_out)

        if self.n_out == 1:
            angles = angles_per_node.squeeze(-1)                   # (B, n_qubits)
        else:
            angles = torch.cat([
                angles_per_node[..., 0],
                angles_per_node[..., 1],
            ], dim=1)                                               # (B, 2*n_qubits)

        # Extract ring distances from state coords and normalise to [0, pi].
        n = self.n_node
        cs = 4 + 6 * n
        node_x = state[:, cs          : cs + (2*n+1)]
        node_y = state[:, cs + (2*n+1): cs + 2*(2*n+1)]
        coords = torch.stack([node_x, node_y], dim=-1)             # (B, 2n+1, 2)
        ring_dist = torch.zeros(B, self.n_qubits, device=self.device)
        for q in range(self.n_qubits):
            nq = (q + 1) % self.n_qubits
            ring_dist[:, q] = (coords[:, q] - coords[:, nq]).norm(dim=-1)
        max_d = ring_dist.max(dim=1, keepdim=True).values.clamp(min=1e-6)
        ring_dist_norm = ring_dist / max_d * math.pi               # (B, n_qubits)

        circuit_inputs = torch.cat([angles, ring_dist_norm], dim=1)
        z = self.qlayer(circuit_inputs.cpu())
        z = z.to(self.device).float()
        return self.head(z)

    def param_report(self) -> dict:
        def count(m): return sum(p.numel() for p in m.parameters())
        pqc = int(self.qlayer.gamma.numel() + self.qlayer.beta.numel())
        classical = count(self.node_encoder) + count(self.head)
        total = count(self)
        return {
            "node_encoder":  count(self.node_encoder),
            "enc_scales":    int(self.qlayer.enc_scales.numel()),
            "pqc_var":       pqc,
            "head":          count(self.head),
            "total":         total,
            "classical_frac": round(classical / total, 3),
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

