"""Shared RL utilities across Tasks 1-4."""

import numpy as np
import matplotlib.pyplot as plt
from mdp import (
    gamma, DEPOT, AT_1, AT_2, REP_1, REP_2, xi1, xi2,
    states, state_index, n_states,
    ACTIONS, action_index, n_actions, feasible_actions,
    cost, transitions, simulate_step,
)


def compute_reference_policy(tol=1e-10):
    """Run A1 policy iteration. Returns (policy, V) dicts."""
    V = {s: 0.0 for s in states}
    policy = {s: feasible_actions(s)[0] for s in states}

    while True:
        # Policy evaluation
        while True:
            delta = 0
            for s in states:
                a = policy[s]
                v_new = cost(s, a)
                for sp, p in transitions(s, a).items():
                    v_new += gamma * p * V[sp]
                delta = max(delta, abs(V[s] - v_new))
                V[s] = v_new
            if delta < tol:
                break

        # Policy improvement
        stable = True
        for s in states:
            best_a = policy[s]
            best_q = float("inf")
            for a in feasible_actions(s):
                q = cost(s, a)
                for sp, p in transitions(s, a).items():
                    q += gamma * p * V[sp]
                if q < best_q:
                    best_q = q
                    best_a = a
            if best_a != policy[s]:
                policy[s] = best_a
                stable = False

        if stable:
            break
    
    return policy, V


def epsilon_greedy(Q, state, epsilon):
    """Epsilon-greedy over feasible actions. Returns action string.

    Q: array (n_states, n_actions). Minimization.
    """
    acts = feasible_actions(state)
    act_idxs = [action_index[a] for a in acts]
    if np.random.random() < epsilon:
        return np.random.choice(acts).item()
    return acts[np.argmin(Q[state_index[state], act_idxs])]


def extract_policy_tabular(Q):
    """Greedy policy from Q-table. Returns dict state -> action string."""
    policy = {}
    for s in states:
        s_idx = state_index[s]
        acts = feasible_actions(s)
        act_idxs = [action_index[a] for a in acts]
        best_a = np.argmin(Q[s_idx, act_idxs])
        policy[s] = acts[best_a]
    return policy


def print_policy_table(policy, location, name):
    """Print policy table for a given engineer location."""
    print(f"\n{name}")
    header = "".ljust(8) + "".join(f"x2={j}".ljust(16) for j in range(xi2 + 1))
    print(header)
    for x1 in range(xi1 + 1):
        row = f"x1={x1}".ljust(8)
        for x2 in range(xi2 + 1):
            s = (x1, x2, location)
            row += policy.get(s, "N/A").ljust(16)
        print(row)


def print_all_policy_tables(policy):
    """Print policy tables for DEPOT, AT_1, AT_2."""
    for loc, name in [(DEPOT, "Depot"), (AT_1, "At Machine 1"), (AT_2, "At Machine 2")]:
        print_policy_table(policy, loc, name)


def compare_policies(policy, ref_policy, label=""):
    """Compare policy to reference. Returns number of differing states."""
    diffs = []
    for s in states:
        a = policy.get(s)
        a_ref = ref_policy.get(s)
        if a != a_ref:
            diffs.append((s, a, a_ref))

    tag = f" ({label})" if label else ""
    print(f"Policy comparison{tag}: {len(diffs)}/{len(states)} states differ.")
    for i, (s, a, a_ref) in enumerate(diffs):
        if i < 20:
            print(f"  {s}: {a} vs ref {a_ref}")
        else:
            print(f"{len(diffs) - i} more")
            break
    return len(diffs)


def plot_convergence(values, ylabel, title, xlabel="Episode", savefig=None):
    """Plot a convergence curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=150)
    plt.show()
