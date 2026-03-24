"""Hyperparameter sweep: learning rate vs exploration floor for Q-learning."""

import multiprocessing as mp
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mdp import (
    gamma, states, state_index, n_states,
    ACTIONS, action_index, n_actions,
    feasible_action_indices, simulate_step,
)
from rl_utils import (
    compute_reference_policy,
    epsilon_greedy,
    policy_match_fraction,
)

REF_POLICY, REF_V = compute_reference_policy()

LRS = [0.001, 0.005, 0.01, 0.05]
EPS_ENDS = [0.0, 0.0001, 0.001, 0.01]
N_SEEDS = 5
EPISODES = 50000
STEPS = 1000
LR_DECAY = 2e-4
EPS_START = 1.0
PING_EP = 500


def q_learning_sweep(lr, eps_end, seed):
    """Single Q-learning run. Returns (lr, eps_end, seed, match_history)."""
    np.random.seed(seed)
    Q = np.zeros((n_states, n_actions))
    match_hist = []

    for episode in range(EPISODES):
        eps = EPS_START + (eps_end - EPS_START) * episode / max(EPISODES - 1, 1)
        alpha = lr / (1.0 + LR_DECAY * episode)
        s = states[np.random.randint(n_states)]
        for _ in range(STEPS):
            s_idx = state_index[s]
            a_idx = epsilon_greedy(Q, s, eps)
            s_next, c = simulate_step(s, ACTIONS[a_idx])
            next_idxs = feasible_action_indices[s_next]
            min_q_next = Q[state_index[s_next], next_idxs].min()
            td_target = c + gamma * min_q_next
            Q[s_idx, a_idx] += alpha * (td_target - Q[s_idx, a_idx])
            s = s_next

        if episode % PING_EP == 0:
            match_pct = policy_match_fraction(Q, REF_POLICY)
            match_hist.append((episode, match_pct))

    return lr, eps_end, seed, match_hist


def _worker(args):
    return q_learning_sweep(*args)


if __name__ == "__main__":
    tasks = [
        (lr, eps_end, seed)
        for lr, eps_end in itertools.product(LRS, EPS_ENDS)
        for seed in range(N_SEEDS)
    ]
    print(f"Launching {len(tasks)} runs ({len(LRS)}x{len(EPS_ENDS)}x{N_SEEDS}) "
          f"on {mp.cpu_count()} cores...")

    with mp.get_context("fork").Pool(min(len(tasks), mp.cpu_count())) as pool:
        results = pool.map(_worker, tasks)

    # Organize results: (lr, eps_end) -> list of match_histories
    from collections import defaultdict
    grouped = defaultdict(list)
    for lr, eps_end, seed, hist in results:
        grouped[(lr, eps_end)].append(hist)

    # ---- Figure 1: Heatmap of final policy match % ----
    final_mean = np.zeros((len(LRS), len(EPS_ENDS)))
    final_std = np.zeros((len(LRS), len(EPS_ENDS)))
    for i, lr in enumerate(LRS):
        for j, eps_end in enumerate(EPS_ENDS):
            finals = [h[-1][1] for h in grouped[(lr, eps_end)]]
            final_mean[i, j] = np.mean(finals)
            final_std[i, j] = np.std(finals)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(final_mean, cmap="RdYlGn", vmin=90, vmax=100,
                   aspect="auto", origin="upper")
    ax.set_xticks(range(len(EPS_ENDS)))
    ax.set_xticklabels([str(e) for e in EPS_ENDS])
    ax.set_yticks(range(len(LRS)))
    ax.set_yticklabels([str(lr) for lr in LRS])
    ax.set_xlabel(r"$\epsilon_{\mathrm{end}}$")
    ax.set_ylabel(r"$\alpha_0$ (initial learning rate)")
    ax.set_title("Final policy match % (mean over 5 seeds, 50k episodes)")

    for i in range(len(LRS)):
        for j in range(len(EPS_ENDS)):
            ax.text(j, i, f"{final_mean[i,j]:.1f}\n$\\pm${final_std[i,j]:.1f}",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="black" if final_mean[i, j] > 90 else "white")

    fig.colorbar(im, ax=ax, label="Policy match %")
    fig.tight_layout()
    fig.savefig("images/hp_heatmap2.png", dpi=150)
    print("Saved images/hp_heatmap2.png")

    # ---- Figure 2: Convergence curves per config ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.cm.viridis
    configs = list(itertools.product(LRS, EPS_ENDS))
    norm = Normalize(vmin=0, vmax=len(configs) - 1)

    for idx, (lr, eps_end) in enumerate(configs):
        hists = grouped[(lr, eps_end)]
        # Interpolate all seeds onto common x
        xs = [np.array([h[0] for h in hist]) for hist in hists]
        ys = [np.array([h[1] for h in hist]) for hist in hists]
        common_x = xs[0]
        interp_ys = [np.interp(common_x, x, y) for x, y in zip(xs, ys)]
        mean_y = np.mean(interp_ys, axis=0)
        color = cmap(norm(idx))
        ax.plot(common_x, mean_y, color=color, linewidth=1.2, alpha=0.8,
                label=f"lr={lr}, eps={eps_end}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Policy match %")
    ax.set_title("Convergence by hyperparameter config (mean of 5 seeds)")
    ax.set_ylim(50, 105)
    ax.axhline(100, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig("images/hp_convergence.png", dpi=150)
    print("Saved images/hp_convergence.png")
    plt.show()
