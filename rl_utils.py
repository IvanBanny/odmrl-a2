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


def _fmt_ep(ep):
    """Format episode number compactly: 1k, 5k, 99.5k, etc."""
    if ep < 1000:
        return str(ep)
    k = ep / 1000
    return f"{k:g}k"


def plot_bias_evolution(t1_runs_se, t2_runs_se,
                        t1_snapshots, t2_snapshots,
                        ref_v_arr,
                        title="", savefig=None):
    """Two-panel bias figure: signed error time-series + scatter grid.

    Left: mean signed error vs episode for both methods.
    Right: 2x3 grid of scatter plots at checkpoint episodes showing
    min_a Q(s,a) vs V*(s), with shared axes across all subplots.

    Args:
        t1_runs_se: list of (x, y) signed error per run for classic Q.
        t2_runs_se: list of (x, y) signed error per run for double Q.
        t1_snapshots: list (per run) of list of (episode, Q) tuples.
        t2_snapshots: same for double Q.
        ref_v_arr: 1-D array of V*(s) for all states.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter

    C_CLASSIC = "C0"
    C_DOUBLE = "#C84430"
    snapshot_episodes = [1000, 5000, 10000, 25000, 50000, 99500]

    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.22,
                           left=0.065, right=0.99, top=0.88, bottom=0.11)

    # --- Left panel: signed error time-series ---
    ax_left = fig.add_subplot(gs[0])
    _plot_runs_on_ax(ax_left, t1_runs_se, color=C_CLASSIC, label="Classic Q")
    _plot_runs_on_ax(ax_left, t2_runs_se, color=C_DOUBLE, label="Double Q")
    ax_left.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax_left.set_xlabel("Episode")
    ax_left.set_ylabel(r"Mean signed error  "
                        r"$\mathbb{E}_s[\min_a Q - V^*]$")
    ax_left.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: _fmt_ep(int(x))))
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=9, loc="lower right")

    # Mark snapshot episodes on the time-series
    for ep in snapshot_episodes:
        ax_left.axvline(ep, color="grey", linestyle=":", linewidth=0.5,
                        alpha=0.4)

    # --- Right panel: 2x3 grid of scatter mini-plots ---
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs[1], hspace=0.18, wspace=0.08)

    # Average Q-tables at each snapshot episode across runs
    def _mean_snapshot_q(all_snaps, ep):
        qs = []
        for run_snaps in all_snaps:
            for e, q in run_snaps:
                if e == ep:
                    qs.append(q)
                    break
        if not qs:
            return None
        return np.mean(qs, axis=0)

    # Precompute all min-Q arrays to find shared axis limits
    all_min_qs = []
    scatter_data = []
    for ep in snapshot_episodes:
        q1_mean = _mean_snapshot_q(t1_snapshots, ep)
        q2_mean = _mean_snapshot_q(t2_snapshots, ep)
        mq1 = min_q_values(q1_mean) if q1_mean is not None else None
        mq2 = min_q_values(q2_mean) if q2_mean is not None else None
        scatter_data.append((mq1, mq2))
        if mq1 is not None:
            all_min_qs.append(mq1)
        if mq2 is not None:
            all_min_qs.append(mq2)

    # Shared limits: union of V* range and all Q ranges, rounded outward
    lo = min(ref_v_arr.min(), *(a.min() for a in all_min_qs)) - 0.5
    hi = max(ref_v_arr.max(), *(a.max() for a in all_min_qs)) + 0.5
    # ~3 clean ticks across the shared range
    tick_vals = np.linspace(np.ceil(lo), np.floor(hi), 4)

    for idx, (ep, (mq1, mq2)) in enumerate(
            zip(snapshot_episodes, scatter_data)):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs_right[row, col])

        if mq1 is not None:
            ax.scatter(ref_v_arr, mq1, s=8, alpha=0.25, color=C_CLASSIC,
                       edgecolors="none", zorder=2, rasterized=True)
        if mq2 is not None:
            ax.scatter(ref_v_arr, mq2, s=8, alpha=0.45, color=C_DOUBLE,
                       edgecolors="none", zorder=3, rasterized=True)

        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.6, alpha=0.35)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)
        ax.tick_params(labelsize=6, length=2, pad=1)
        ax.grid(True, alpha=0.15, linewidth=0.4)

        # Compact episode label in top-left corner
        ax.text(0.05, 0.95, _fmt_ep(ep), transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7,
                          pad=1.5))

        # Only label outer edges
        if row == 1:
            ax.set_xlabel(r"$V^*$", fontsize=8, labelpad=2)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(r"$\min_a Q$", fontsize=8, labelpad=2)
        else:
            ax.set_yticklabels([])

    # Scatter legend: compact, tucked into bottom-right of last subplot
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", color=C_CLASSIC, linestyle="none",
               markersize=4, alpha=0.6, label="Classic Q"),
        Line2D([], [], marker="o", color=C_DOUBLE, linestyle="none",
               markersize=4, alpha=0.6, label="Double Q"),
    ]
    ax.legend(handles=handles, fontsize=6.5, loc="lower right",
              handletextpad=0.3, borderpad=0.3, framealpha=0.8)

    fig.suptitle(title, fontsize=12)
    if savefig:
        os.makedirs("images", exist_ok=True)
        fig.savefig(os.path.join("images", savefig), dpi=150,
                    bbox_inches="tight")
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


def plot_value_comparison(v_hat_fn, ref_V, title="", savefig=None):
    """3x3 grid: rows = V*, V_hat, error; columns = DEPOT, AT_1, AT_2.

    Args:
        v_hat_fn: callable(state) -> float, the approximate value.
        ref_V: dict state -> float, the PI reference values.
    """
    locs = [(DEPOT, "DEPOT"), (AT_1, "AT_1"), (AT_2, "AT_2")]
    nrows, ncols = xi1 + 1, xi2 + 1
    row_labels = [r"$V^*(s)$ (PI)", r"$\hat{V}(s)$ (Linear FA)",
                  r"$\hat{V} - V^*$ (error)"]

    # Build grids
    grids_star = []
    grids_hat = []
    grids_err = []
    for loc, _ in locs:
        g_star = np.zeros((nrows, ncols))
        g_hat = np.zeros((nrows, ncols))
        for x1 in range(nrows):
            for x2 in range(ncols):
                s = (x1, x2, loc)
                g_star[x1, x2] = ref_V[s]
                g_hat[x1, x2] = v_hat_fn(s)
        grids_star.append(g_star)
        grids_hat.append(g_hat)
        grids_err.append(g_hat - g_star)

    # Shared color limits for V* and V_hat rows
    vmin_val = min(g.min() for g in grids_star + grids_hat)
    vmax_val = max(g.max() for g in grids_star + grids_hat)
    # Symmetric limits for error row
    err_abs = max(abs(g).max() for g in grids_err)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
    all_grids = [grids_star, grids_hat, grids_err]
    cmaps = ["viridis", "viridis", "RdBu_r"]
    vmins = [vmin_val, vmin_val, -err_abs]
    vmaxs = [vmax_val, vmax_val, err_abs]

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            im = ax.imshow(all_grids[row][col], aspect="equal", origin="upper",
                           cmap=cmaps[row], vmin=vmins[row], vmax=vmaxs[row])
            ax.set_xticks(range(ncols))
            ax.set_yticks(range(nrows))
            ax.set_xticklabels([str(j) for j in range(ncols)], fontsize=7)
            ax.set_yticklabels([str(i) for i in range(nrows)], fontsize=7)
            if row == 2:
                ax.set_xlabel("x2")
            if col == 0:
                ax.set_ylabel("x1\n\n" + row_labels[row], fontsize=9)
            if row == 0:
                ax.set_title(locs[col][1], fontsize=11)

            # Annotate cells with values
            thresh = vmins[row] + 0.6 * (vmaxs[row] - vmins[row])
            for r in range(nrows):
                for c in range(ncols):
                    v = all_grids[row][col][r, c]
                    color = "white" if v > thresh else "black"
                    if row == 2:
                        color = "white" if abs(v) > 0.6 * err_abs else "black"
                    fmt = f"{v:.2f}" if row == 2 else f"{v:.1f}"
                    ax.text(c, r, fmt, ha="center", va="center",
                            fontsize=6, color=color, fontweight="bold")

        # Colorbar in the 4th column
        fig.colorbar(im, cax=axes[row, 3])

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if savefig:
        os.makedirs("images", exist_ok=True)
        fig.savefig(os.path.join("images", savefig), dpi=150)
    plt.show()


def plot_feature_decomposition(W, PHI_all, n_fourier, n_loc, n_fail,
                               title="", savefig=None):
    """4x3 grid: rows = Fourier, location, failure, total; cols = DEPOT/AT_1/AT_2.

    Shows how the three feature groups contribute to V_hat = min_a Q(s,a).

    Args:
        W: weight matrix (n_actions, n_features).
        PHI_all: precomputed features (n_states, n_features).
        n_fourier, n_loc, n_fail: feature group sizes.
    """
    locs = [(DEPOT, "DEPOT"), (AT_1, "AT_1"), (AT_2, "AT_2")]
    nrows, ncols = xi1 + 1, xi2 + 1
    row_labels = ["Fourier basis", "Location indicators",
                  "Failure indicators", "Total (sum)"]

    # Feature index ranges
    slices = [
        slice(0, n_fourier),
        slice(n_fourier, n_fourier + n_loc),
        slice(n_fourier + n_loc, n_fourier + n_loc + n_fail),
    ]

    # Build grids: for each component, compute min_a [W[a, slice] @ phi[slice]]
    # But decomposition of min is tricky - instead, find the greedy action from
    # full Q, then decompose that action's Q-value into components
    all_grids = [[], [], [], []]  # fourier, loc, fail, total
    for loc, _ in locs:
        g = [np.zeros((nrows, ncols)) for _ in range(4)]
        for x1 in range(nrows):
            for x2 in range(ncols):
                s = (x1, x2, loc)
                si = state_index[s]
                act_idxs = feasible_action_indices[s]
                # Greedy action under full weights
                q_full = W[act_idxs] @ PHI_all[si]
                best = act_idxs[np.argmin(q_full)]
                # Decompose the greedy action's Q-value
                for k, sl in enumerate(slices):
                    g[k][x1, x2] = W[best, sl] @ PHI_all[si, sl]
                g[3][x1, x2] = q_full.min()
        for k in range(4):
            all_grids[k].append(g[k])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
    cmaps = ["viridis", "coolwarm", "coolwarm", "viridis"]

    for row in range(4):
        grids = all_grids[row]
        if row in (1, 2):
            abs_max = max(abs(g).max() for g in grids)
            if abs_max == 0:
                abs_max = 1.0
            vmin, vmax = -abs_max, abs_max
        else:
            vmin = min(g.min() for g in grids)
            vmax = max(g.max() for g in grids)

        for col in range(3):
            ax = axes[row, col]
            im = ax.imshow(grids[col], aspect="equal", origin="upper",
                           cmap=cmaps[row], vmin=vmin, vmax=vmax)
            ax.set_xticks(range(ncols))
            ax.set_yticks(range(nrows))
            ax.set_xticklabels([str(j) for j in range(ncols)], fontsize=7)
            ax.set_yticklabels([str(i) for i in range(nrows)], fontsize=7)
            if row == 3:
                ax.set_xlabel("x2")
            if col == 0:
                ax.set_ylabel("x1\n\n" + row_labels[row], fontsize=9)
            if row == 0:
                ax.set_title(locs[col][1], fontsize=11)

            for r in range(nrows):
                for c in range(ncols):
                    v = grids[col][r, c]
                    if row in (1, 2):
                        color = "white" if abs(v) > 0.6 * abs_max else "black"
                    else:
                        color = "white" if v > vmin + 0.6 * (vmax - vmin) else "black"
                    ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                            fontsize=6, color=color, fontweight="bold")

        fig.colorbar(im, cax=axes[row, 3])

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if savefig:
        os.makedirs("images", exist_ok=True)
        fig.savefig(os.path.join("images", savefig), dpi=150)
    plt.show()
