"""
test_cluster.py

Run this on the cluster before the main experiment to catch environment issues,
missing dependencies, and broken code paths early.

Usage:
    python test_cluster.py          # full check (~60s on CPU)
    python test_cluster.py --quick  # minimal check (~10s)

Exit code 0 = all tests passed.  Non-zero = at least one test failed.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import torch

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        msg = fn()
        results.append((name, True, msg or ""))
        print(f"  {PASS}  {name}" + (f"  [{msg}]" if msg else ""))
    except Exception:
        tb = traceback.format_exc().strip().splitlines()[-1]
        results.append((name, False, tb))
        print(f"  {FAIL}  {name}  <<  {tb}")


# --------------------------------------------------------------------------- #
# 1. Imports
# --------------------------------------------------------------------------- #

def test_imports():
    import pennylane as qml          # noqa: F401
    import torch.optim               # noqa: F401
    from cpdptw_env import CPDPTWEnv # noqa: F401
    from quantum_qnet import (       # noqa: F401
        QuantumQNetwork, QAOAQNetwork,
        QuantumNodeQNetwork, QAOANodeQNetwork, ClassicalQNetwork,
    )
    from reinforce_qrl import train_reinforce, ValueHead  # noqa: F401
    from train_qrl import build_net, train               # noqa: F401
    return f"pennylane {qml.__version__}, torch {torch.__version__}"


# --------------------------------------------------------------------------- #
# 2. Environment
# --------------------------------------------------------------------------- #

def test_env_reset():
    from cpdptw_env import CPDPTWEnv
    env = CPDPTWEnv(node=3, vehicle_capacity=5, rng_seed=42)
    state, info = env.reset()
    assert state.shape == (1, env.n_observations), f"bad shape {state.shape}"
    assert env.n_observations == 4 + 6 * 3, f"expected {4+18}, got {env.n_observations}"
    return f"state ({state.shape[1]},)  n_actions={env.n_actions}"


def test_env_step():
    from cpdptw_env import CPDPTWEnv
    env = CPDPTWEnv(node=3, vehicle_capacity=5, rng_seed=42)
    env.reset()
    actions = env.valid_actions()
    assert actions, "no valid actions after reset"
    nxt, r, done, _, info = env.step(actions[0])
    assert nxt.shape == (1, env.n_observations)
    assert isinstance(r.item(), float)
    return f"valid_actions={len(actions)}, first_step_reward={r.item():.3f}"


def test_env_dynamic_time_features():
    """Verify state time features decrease as vehicle_time increases."""
    from cpdptw_env import CPDPTWEnv
    env = CPDPTWEnv(node=3, vehicle_capacity=5, rng_seed=7)
    s0, _ = env.reset()
    # Force vehicle_time forward; features should shrink toward 0.
    env.vehicle_time = int(env.time_frame * 0.5)
    s1 = env._get_state()
    # Pickup-remain and delivery-remain windows at position [4..4+n] should
    # on average be smaller after time has advanced.
    n = env.node
    pickup_0 = s0[0, 4:4 + n].sum().item()
    pickup_1 = s1[0, 4:4 + n].sum().item()
    assert pickup_1 <= pickup_0, "remaining-time features did not shrink"
    return "remaining-time features decrease as vehicle_time advances"


# --------------------------------------------------------------------------- #
# 3. Network forward passes
# --------------------------------------------------------------------------- #

def _make_env():
    from cpdptw_env import CPDPTWEnv
    return CPDPTWEnv(node=3, vehicle_capacity=5, rng_seed=0)


def test_forward_quantum():
    from quantum_qnet import QuantumQNetwork
    env = _make_env()
    net = QuantumQNetwork(env, n_qubits=4, n_layers=2)
    state, _ = env.reset()
    out = net(state)
    assert out.shape == (1, env.n_actions)
    pr = net.param_report()
    assert "enc_scales" in pr and pr["enc_scales"] == 2 * 4  # n_layers * n_qubits
    return f"out {out.shape}  enc_scales={pr['enc_scales']}  total={pr['total']}"


def test_forward_qaoa():
    from quantum_qnet import QAOAQNetwork
    env = _make_env()
    net = QAOAQNetwork(env, n_qubits=4, n_layers=2)
    state, _ = env.reset()
    out = net(state)
    assert out.shape == (1, env.n_actions)
    pr = net.param_report()
    assert "enc_scales" in pr
    return f"out {out.shape}  enc_scales={pr['enc_scales']}  total={pr['total']}"


def test_forward_node_quantum():
    from quantum_qnet import QuantumNodeQNetwork
    env = _make_env()
    net = QuantumNodeQNetwork(env, n_layers=2)
    assert net.n_qubits == 2 * env.node + 1
    state, _ = env.reset()
    out = net(state)
    assert out.shape == (1, env.n_actions)
    pr = net.param_report()
    assert "enc_scales" in pr
    return f"n_qubits={net.n_qubits}  out {out.shape}  enc_scales={pr['enc_scales']}"


def test_forward_node_qaoa():
    from quantum_qnet import QAOANodeQNetwork
    env = _make_env()
    net = QAOANodeQNetwork(env, n_layers=2)
    state, _ = env.reset()
    out = net(state)
    assert out.shape == (1, env.n_actions)
    return f"out {out.shape}"


def test_forward_classical():
    from quantum_qnet import ClassicalQNetwork
    env = _make_env()
    net = ClassicalQNetwork(env, hidden=32)
    state, _ = env.reset()
    out = net(state)
    assert out.shape == (1, env.n_actions)
    return f"out {out.shape}"


def test_enc_scales_init():
    """enc_scales must start at 1.0, not random."""
    from quantum_qnet import QuantumQNetwork, QAOAQNetwork
    env = _make_env()
    for cls in (QuantumQNetwork, QAOAQNetwork):
        net = cls(env, n_qubits=4, n_layers=2)
        val = net.qlayer.enc_scales
        assert torch.allclose(val, torch.ones_like(val)), \
            f"{cls.__name__} enc_scales not 1.0: {val}"
    return "all enc_scales initialised to 1.0"


def test_pair_aware_features():
    """Node encoder receives 11 features (includes partner coords)."""
    from quantum_qnet import QuantumNodeQNetwork
    env = _make_env()
    net = QuantumNodeQNetwork(env, n_layers=1)
    # The first Linear in node_encoder must accept 11 inputs.
    in_features = net.node_encoder[0].in_features
    assert in_features == 11, f"expected 11, got {in_features}"
    return f"node_encoder in_features={in_features}"


# --------------------------------------------------------------------------- #
# 4. REINFORCE actor-critic training
# --------------------------------------------------------------------------- #

def test_reinforce_runs(quick: bool = False):
    from reinforce_qrl import train_reinforce
    r = train_reinforce(
        model_kind="quantum", node=3, n_qubits=4, n_layers=2,
        episodes=3 if quick else 10,
        fixed_instance=True, seed=0, out_prefix="_test_ci",
        value_coef=0.5,
    )
    assert len(r["rewards"]) > 0
    assert len(r["losses"]) > 0
    assert all(f >= 0.0 for f in r["feas"])
    return f"episodes={len(r['rewards'])}  last_loss={r['losses'][-1]:.4f}"


def test_reinforce_gradients():
    """Check that enc_scales actually change after a few gradient steps."""
    from cpdptw_env import CPDPTWEnv
    from quantum_qnet import QuantumQNetwork
    from reinforce_qrl import MaskedCategorical, ValueHead
    import torch.optim as optim

    env = CPDPTWEnv(node=3, vehicle_capacity=5, rng_seed=1)
    net = QuantumQNetwork(env, n_qubits=4, n_layers=2)
    critic = ValueHead(env.n_observations)
    opt = optim.AdamW(list(net.parameters()) + list(critic.parameters()), lr=1e-2)

    scales_before = net.qlayer.enc_scales.detach().clone()

    for _ in range(3):
        state, _ = env.reset()
        log_probs, entropies, rewards, states_buf = [], [], [], []
        for _ in range(4 * env.n_total):
            mask = env.action_mask()
            if not mask.any():
                break
            logits = net(state).squeeze(0)
            policy = MaskedCategorical(logits, mask)
            action = policy.sample()
            log_probs.append(policy.log_prob(action))
            entropies.append(policy.entropy())
            states_buf.append(state)
            nxt, r, done, _, _ = env.step(action.item())
            rewards.append(r.item())
            if done:
                break
            state = nxt
        if not log_probs:
            continue
        T = len(rewards)
        ret = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + 0.99 * G
            ret[t] = G
        L = len(log_probs)
        lp = torch.stack(log_probs)
        vals = critic(torch.cat(states_buf).squeeze(1))
        adv = ret[:L] - vals.detach()
        loss = -(lp * adv).mean() + 0.5 * torch.nn.functional.mse_loss(vals, ret[:L])
        opt.zero_grad()
        loss.backward()
        opt.step()

    scales_after = net.qlayer.enc_scales.detach()
    changed = not torch.allclose(scales_before, scales_after, atol=1e-6)
    assert changed, "enc_scales did not change after 3 gradient steps"
    delta = (scales_after - scales_before).abs().max().item()
    return f"enc_scales max_delta={delta:.6f} (gradients flowing)"


def test_value_coef_zero():
    """value_coef=0 disables critic loss, should still run."""
    from reinforce_qrl import train_reinforce
    r = train_reinforce(
        model_kind="classical", node=3, episodes=3,
        fixed_instance=True, seed=0, out_prefix="_test_ci", value_coef=0.0,
    )
    assert len(r["losses"]) > 0
    return "pure REINFORCE (value_coef=0) ran without error"


# --------------------------------------------------------------------------- #
# 5. Cleanup test artefacts
# --------------------------------------------------------------------------- #

def cleanup():
    import glob, os
    for f in glob.glob("_test_ci_*"):
        try:
            os.remove(f)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Minimal run (fewer episodes, skips gradient check)")
    args = ap.parse_args()

    print("\n=== cluster smoke-test ===\n")

    print("[ imports ]")
    check("imports", test_imports)

    print("\n[ environment ]")
    check("env reset",              test_env_reset)
    check("env step",               test_env_step)
    check("dynamic time features",  test_env_dynamic_time_features)

    print("\n[ network forward passes ]")
    check("QuantumQNetwork forward",     test_forward_quantum)
    check("QAOAQNetwork forward",        test_forward_qaoa)
    check("QuantumNodeQNetwork forward", test_forward_node_quantum)
    check("QAOANodeQNetwork forward",    test_forward_node_qaoa)
    check("ClassicalQNetwork forward",   test_forward_classical)
    check("enc_scales init = 1.0",       test_enc_scales_init)
    check("pair-aware 11 features",      test_pair_aware_features)

    print("\n[ REINFORCE actor-critic ]")
    check("reinforce runs",
          lambda: test_reinforce_runs(quick=args.quick))
    check("value_coef=0 (pure PG)",  test_value_coef_zero)
    if not args.quick:
        check("enc_scales receive gradients", test_reinforce_gradients)

    cleanup()

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*40}")
    print(f"  {n_pass} passed  /  {n_fail} failed")
    print(f"{'='*40}\n")

    if n_fail:
        print("FAILED tests:")
        for name, ok, msg in results:
            if not ok:
                print(f"  {FAIL} {name}: {msg}")
        sys.exit(1)
    else:
        print("All tests passed. Safe to run the main experiment.")
        sys.exit(0)


if __name__ == "__main__":
    main()
