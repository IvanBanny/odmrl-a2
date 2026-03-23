"""Shared RL utilities across Tasks 1-4."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mdp import (
    gamma, DEPOT, AT_1, AT_2, REP_1, REP_2, xi1, xi2,
    states, state_index, n_states,
    ACTIONS, action_index, n_actions, feasible_actions,
    feasible_action_indices,
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
    """Epsilon-greedy over feasible actions. Returns action index (int).

    Q: array (n_states, n_actions). Minimization.
    """
    act_idxs = feasible_action_indices[state]
    if np.random.random() < epsilon:
        return act_idxs[np.random.randint(len(act_idxs))]
    return act_idxs[np.argmin(Q[state_index[state], act_idxs])]


def extract_policy_tabular(Q):
    """Greedy policy from Q-table. Returns dict state -> action string."""
    policy = {}
    for s in states:
        act_idxs = feasible_action_indices[s]
        best = act_idxs[np.argmin(Q[state_index[s], act_idxs])]
        policy[s] = ACTIONS[best]
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
    print(f"\nPolicy comparison{tag}: {len(diffs)}/{len(states)} states differ.")
    for i, (s, a, a_ref) in enumerate(diffs):
        if i < 20:
            print(f"  {s}: {a} vs ref {a_ref}")
        else:
            print(f"{len(diffs) - i} more")
            break
    return len(diffs)


def _apply_log_axes(ax):
    """Configure log-scale y-axis: labeled major ticks, unlabeled minor grid."""
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=12))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.15, which="minor")


def plot_convergence(x, y, ylabel, title, xlabel="Episode", savefig=None):
    """Plot a convergence curve (single run with smoothing)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x_arr = np.asarray(x)
    y_arr = np.asarray(y, dtype=float)
    if len(y_arr) >= 20:
        ax.plot(x_arr, y_arr, alpha=0.3, color="C0", label="Raw")
        window = max(len(y_arr) // 20, 2)
        kernel = np.ones(window) / window
        smoothed = np.convolve(y_arr, kernel, mode="valid")
        pad = window // 2
        ax.plot(x_arr[pad:pad + len(smoothed)], smoothed,
                color="C0", linewidth=2, label=f"Smoothed (w={window})")
        ax.legend()
    else:
        ax.plot(x_arr, y_arr, color="C0", marker="o", markersize=4)
    _apply_log_axes(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    if savefig:
        os.makedirs("images", exist_ok=True)
        fig.savefig(os.path.join("images", savefig), dpi=150)
    plt.show()


def plot_convergence_multi(runs, ylabel, title, xlabel="Episode", savefig=None):
    """Plot convergence from multiple parallel runs.

    Args:
        runs: list of (x_array, y_array) per seed.
    Individual runs drawn at very low alpha; mean overlaid in full color.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    # Common x grid spanning the intersection of all runs.
    x_min = max(r[0][0] for r in runs)
    x_max = min(r[0][-1] for r in runs)
    common_x = np.linspace(x_min, x_max, 400)
    ys = []
    for x, y in runs:
        yi = np.interp(common_x, x, y)
        ys.append(yi)
        ax.plot(common_x, yi, alpha=0.08, color="C0", linewidth=0.8)
    mean_y = np.mean(ys, axis=0)
    ax.plot(common_x, mean_y, color="C0", linewidth=2,
            label=f"Mean ({len(runs)} runs)")
    ax.legend()
    _apply_log_axes(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    if savefig:
        os.makedirs("images", exist_ok=True)
        fig.savefig(os.path.join("images", savefig), dpi=150)
    plt.show()
