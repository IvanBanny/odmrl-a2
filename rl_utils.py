"""Shared RL utilities across Tasks 1-4."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
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


def _plot_runs_on_ax(ax, runs, color="C0", label=None,
                     run_alpha=0.08, mean_alpha=1.0, mean_lw=2):
    """Overlay individual runs (low alpha) with mean on a single axis."""
    x_min = max(r[0][0] for r in runs)
    x_max = min(r[0][-1] for r in runs)
    common_x = np.linspace(x_min, x_max, 400)
    ys = []
    for x, y in runs:
        yi = np.interp(common_x, x, y)
        ys.append(yi)
        ax.plot(common_x, yi, alpha=run_alpha, color=color, linewidth=0.8)
    mean_y = np.mean(ys, axis=0)
    if label is None:
        label = f"Mean ({len(runs)} runs)"
    ax.plot(common_x, mean_y, color=color, linewidth=mean_lw,
            alpha=mean_alpha, label=label)


def _apply_log_axes(ax):
    """Configure log-scale y-axis: labeled major ticks, unlabeled minor grid."""
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=12))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.15, which="minor")


def policy_match_fraction(Q, ref_policy):
    """Fraction of states where greedy Q-policy matches ref_policy."""
    match = 0
    for s in states:
        act_idxs = feasible_action_indices[s]
        best = act_idxs[np.argmin(Q[state_index[s], act_idxs])]
        if ACTIONS[best] == ref_policy[s]:
            match += 1
    return match / len(states) * 100.0


def plot_convergence_two_panel(runs_rmse, runs_match,
                               ylabel_left=r"$\mathrm{RMSE}$ to $V^*$",
                               ylabel_right="Policy match %",
                               title="", xlabel="Episode", savefig=None,
                               color="C0", label=None,
                               bg_runs_rmse=None, bg_runs_match=None,
                               bg_color="C0", bg_label="Classic Q"):
    """Two-panel convergence plot: RMSE to V* (log) and policy match % (linear).

    Args:
        runs_rmse: list of (x, y) for RMSE to V* per run.
        runs_match: list of (x, y) for policy match % per run.
        color: color for the main runs.
        label: label for the main mean line (default: "Mean (N runs)").
        bg_runs_rmse: optional background runs for RMSE (faint overlay).
        bg_runs_match: optional background runs for match % (faint overlay).
        bg_color: color for background runs.
        bg_label: legend label for background mean line.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Left: RMSE to V* on log scale
    if bg_runs_rmse is not None:
        _plot_runs_on_ax(ax1, bg_runs_rmse, color=bg_color, label=bg_label,
                         run_alpha=0.03, mean_alpha=0.35, mean_lw=1.5)
    _plot_runs_on_ax(ax1, runs_rmse, color=color, label=label)
    _apply_log_axes(ax1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_left)
    ax1.set_title(title + " - RMSE convergence")
    ax1.legend()

    # Right: policy match % on linear scale
    if bg_runs_match is not None:
        _plot_runs_on_ax(ax2, bg_runs_match, color=bg_color, label=bg_label,
                         run_alpha=0.03, mean_alpha=0.35, mean_lw=1.5)
    _plot_runs_on_ax(ax2, runs_match, color=color, label=label)
    ax2.axhline(100.0, color="C3", linestyle="--", linewidth=1.5,
                label="PI optimal (100%)")
    ax2.set_ylim(0, 105)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_right)
    ax2.set_title(title + " - Policy match")
    ax2.grid(True, alpha=0.3)

    # Annotate final mean value
    x_min = max(r[0][0] for r in runs_match)
    x_max = min(r[0][-1] for r in runs_match)
    final_vals = [np.interp(x_max, x, y) for x, y in runs_match]
    final_mean = np.mean(final_vals)
    ax2.annotate(f"{final_mean:.1f}%",
                 xy=(x_max, final_mean), fontsize=9, fontweight="bold",
                 color=color, ha="right", va="top",
                 xytext=(-8, -6), textcoords="offset points")
    ax2.legend()

    fig.tight_layout()
    if savefig:
        os.makedirs("images", exist_ok=True)
        fig.savefig(os.path.join("images", savefig), dpi=150)
    plt.show()


def min_q_values(Q):
    """Return array of min_a Q(s,a) over feasible actions for each state."""
    vals = np.empty(n_states)
    for s in states:
        si = state_index[s]
        vals[si] = Q[si, feasible_action_indices[s]].min()
    return vals


# Map location int -> color for per-location scatter coding
_LOC_COLORS = {DEPOT: "C0", AT_1: "C2", AT_2: "C4", REP_1: "C1", REP_2: "C3"}
_LOC_NAMES = {DEPOT: "DEPOT", AT_1: "AT_1", AT_2: "AT_2",
              REP_1: "REP_1", REP_2: "REP_2"}


def plot_bias_scatter(Q_mean, ref_V, title="", savefig=None, color="C0",
                      label=None, Q_mean_bg=None, bg_color="C0",
                      bg_label=None):
    """Scatter plot of min_a Q(s,a) vs V_PI(s), color-coded by location."""
    fig, ax = plt.subplots(figsize=(6.5, 6))

    v_pi = np.array([ref_V[s] for s in states])
    locs = np.array([s[2] for s in states])

    if Q_mean_bg is not None:
        min_q_bg = min_q_values(Q_mean_bg)
        ax.scatter(v_pi, min_q_bg, alpha=0.15, s=16, color=bg_color,
                   label=bg_label, zorder=2, edgecolors="none")

    min_q = min_q_values(Q_mean)
    # Plot each location with its own color
    for loc in [DEPOT, AT_1, AT_2, REP_1, REP_2]:
        mask = locs == loc
        if not mask.any():
            continue
        lbl = f"{label} - {_LOC_NAMES[loc]}" if label else _LOC_NAMES[loc]
        ax.scatter(v_pi[mask], min_q[mask], alpha=0.7, s=22,
                   color=_LOC_COLORS[loc], label=lbl, zorder=3,
                   edgecolors="none")

    lo = min(v_pi.min(), min_q.min()) - 0.5
    hi = max(v_pi.max(), min_q.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x")
    ax.set_xlabel(r"$V^{\pi^*}(s)$ (PI)")
    ax.set_ylabel(r"$\min_a Q(s,a)$ (Q-learning)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if savefig:
        os.makedirs("images", exist_ok=True)
        fig.savefig(os.path.join("images", savefig), dpi=150)
    plt.show()


_ACTION_SHORT = {
    "nothing": "no",
    "travel_1": "t1",
    "travel_2": "t2",
    "travel_depot": "td",
    "maintain_1": "m1",
    "maintain_2": "m2",
    "continue maintenance": "cm",
}


def plot_policy_diff(policy, ref_policy, title="", savefig=None):
    """3-panel heatmap: green = matches PI, red = differs.

    Each cell shows a two-character action abbreviation:
    no=nothing, t1/t2=travel_1/2, td=travel_depot, m1/m2=maintain, cm=continue.
    """
    locs = [(DEPOT, "DEPOT"), (AT_1, "AT_1"), (AT_2, "AT_2")]
    cmap = ListedColormap(["#d94040", "#40b040"])
    nrows, ncols = xi1 + 1, xi2 + 1
    cell_h = 0.45
    fig, axes = plt.subplots(
        1, 3, figsize=(3.2 * 3 + 1.0, nrows * cell_h + 1.0))

    for ax, (loc, loc_name) in zip(axes, locs):
        grid = np.zeros((nrows, ncols))
        for x1 in range(nrows):
            for x2 in range(ncols):
                s = (x1, x2, loc)
                if s in policy and s in ref_policy:
                    grid[x1, x2] = 1.0 if policy[s] == ref_policy[s] else 0.0
                else:
                    grid[x1, x2] = 0.5

        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="equal",
                  origin="upper")
        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.set_xticklabels([str(j) for j in range(ncols)], fontsize=8)
        ax.set_yticklabels([str(i) for i in range(nrows)], fontsize=8)
        ax.set_xlabel("x2")
        ax.set_ylabel("x1")
        ax.set_title(loc_name)

        for x1 in range(nrows):
            for x2 in range(ncols):
                s = (x1, x2, loc)
                a = policy.get(s, "")
                short = _ACTION_SHORT.get(a, "?")
                ax.text(x2, x1, short, ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if savefig:
        os.makedirs("images", exist_ok=True)
        fig.savefig(os.path.join("images", savefig), dpi=150)
    plt.show()


def _fmt_count(v):
    """Short readable count: 0, 42, 1.2k, 45k, 1.2M, etc."""
    if v < 1:
        return "0"
    if v < 1_000:
        return f"{v:.0f}"
    if v < 10_000:
        return f"{v / 1_000:.1f}k"
    if v < 1_000_000:
        return f"{v / 1_000:.0f}k"
    if v < 10_000_000:
        return f"{v / 1_000_000:.1f}M"
    return f"{v / 1_000_000:.0f}M"


def plot_visits(visit_counts, title="", savefig=None):
    """Heatmap of mean visit counts for DEPOT/AT_1/AT_2 with shared colorbar.

    REP_1/REP_2 totals are shown as text annotations below the figure since
    those locations have only a single valid row/column each.
    """
    main_locs = [(DEPOT, "DEPOT"), (AT_1, "AT_1"), (AT_2, "AT_2")]
    nrows, ncols = xi1 + 1, xi2 + 1

    # Collect grids to find global vmin/vmax for shared color scale
    grids = []
    for loc, _ in main_locs:
        grid = np.zeros((nrows, ncols))
        for x1 in range(nrows):
            for x2 in range(ncols):
                s = (x1, x2, loc)
                if s in state_index:
                    grid[x1, x2] = visit_counts[state_index[s]]
        grids.append(grid)
    vmin = min(g.min() for g in grids)
    vmax = max(g.max() for g in grids)

    # REP totals
    rep1_total = sum(
        visit_counts[state_index[s]]
        for s in states if s[2] == REP_1)
    rep2_total = sum(
        visit_counts[state_index[s]]
        for s in states if s[2] == REP_2)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
    for i, (ax, (loc, loc_name), grid) in enumerate(
            zip(axes[:3], main_locs, grids)):
        im = ax.imshow(grid, aspect="equal", origin="upper", cmap="YlOrRd",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.set_xticklabels([str(j) for j in range(ncols)], fontsize=8)
        ax.set_yticklabels([str(i) for i in range(nrows)], fontsize=8)
        ax.set_ylabel("x1")
        ax.set_title(loc_name)

        # Annotate cells with shortened visit counts
        thresh = vmin + 0.6 * (vmax - vmin)
        for r in range(nrows):
            for c in range(ncols):
                v = grid[r, c]
                color = "white" if v > thresh else "black"
                ax.text(c, r, _fmt_count(v), ha="center", va="center",
                        fontsize=6, color=color, fontweight="bold")

        if i == 1:
            ax.set_xlabel(
                f"x2\n\n"
                f"REP_1 total: {_fmt_count(rep1_total)}   "
                f"REP_2 total: {_fmt_count(rep2_total)}",
                fontsize=8)
        else:
            ax.set_xlabel("x2")

    fig.colorbar(im, cax=axes[3], label="visits")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if savefig:
        os.makedirs("images", exist_ok=True)
        fig.savefig(os.path.join("images", savefig), dpi=150)
    plt.show()
