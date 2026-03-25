"""Task 2: Double Q-Learning (maximization bias investigation)."""

import multiprocessing as mp
import numpy as np
from mdp import (
    gamma, DEPOT, AT_1, AT_2, REP_1, REP_2, xi1, xi2,
    states, state_index, n_states,
    ACTIONS, action_index, n_actions, feasible_actions,
    feasible_action_indices,
    cost, transitions, simulate_step,
)
from rl_utils import (
    compute_reference_policy,
    epsilon_greedy,
    extract_policy_tabular,
    print_all_policy_tables,
    compare_policies,
    policy_match_fraction,
    min_q_values,
    plot_convergence_two_panel,
    plot_bias_scatter,
    plot_policy_diff,
    plot_visits,
)

# Precompute reference policy once (inherited by forked workers)
REF_POLICY, REF_V = compute_reference_policy()
REF_V_ARR = np.array([REF_V[s] for s in states])


def double_q_learning(lr=0.1, lr_decay=1e-4, episodes=100000, steps=1000,
                      eps_start=1.0, eps_end=0.01, ping_ep=None):
    """Double Q-learning to address minimization bias."""
    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))
    rmse_hist = []
    match_hist = []
    visit_counts = np.zeros(n_states, dtype=np.int64)

    for episode in range(episodes):
        eps = eps_start + (eps_end - eps_start) * episode / max(episodes - 1, 1)
        alpha = lr / (1.0 + lr_decay * episode)
        s = states[np.random.randint(n_states)]
        for _ in range(steps):
            s_idx = state_index[s]
            visit_counts[s_idx] += 1

            # Choose action by behavior policy using current Q1+Q2
            a_idx = epsilon_greedy(Q1 + Q2, s, eps)

            # Simulate step
            s_next, c = simulate_step(s, ACTIONS[a_idx])
            s_next_idx = state_index[s_next]
            next_idxs = feasible_action_indices[s_next]

            if np.random.random() < 0.5:
                best_a = next_idxs[np.argmin(Q1[s_next_idx, next_idxs])]
                td_target = c + gamma * Q2[s_next_idx, best_a]
                Q1[s_idx, a_idx] += alpha * (td_target - Q1[s_idx, a_idx])
            else:
                best_a = next_idxs[np.argmin(Q2[s_next_idx, next_idxs])]
                td_target = c + gamma * Q1[s_next_idx, best_a]
                Q2[s_idx, a_idx] += alpha * (td_target - Q2[s_idx, a_idx])

            s = s_next
        if ping_ep and episode % ping_ep == 1:
            Q_combined = (Q1 + Q2) / 2
            rmse = np.sqrt(np.mean((min_q_values(Q_combined) - REF_V_ARR) ** 2))
            rmse_hist.append((episode, rmse))
            match_pct = policy_match_fraction(Q_combined, REF_POLICY)
            match_hist.append((episode, match_pct))
            if episode % (episodes // 10) < ping_ep:
                print(f"  Episode {episode:>6d}/{episodes}  eps={eps:.3f}"
                      f"  RMSE={rmse:.4f}  match={match_pct:.1f}%")

    return Q1, Q2, rmse_hist, match_hist, visit_counts


N_RUNS = 20
RUN_KWARGS = dict(lr=0.01, lr_decay=2e-4, episodes=100000, steps=1000,
                  eps_start=1.0, eps_end=0.0001, ping_ep=500)


def _worker(seed):
    np.random.seed(seed)
    Q1, Q2, rmse_hist, match_hist, visits = double_q_learning(**RUN_KWARGS)
    Q = (Q1 + Q2) / 2
    x_rmse = np.array([h[0] for h in rmse_hist])
    y_rmse = np.array([h[1] for h in rmse_hist])
    x_m = np.array([h[0] for h in match_hist])
    y_m = np.array([h[1] for h in match_hist])
    return Q, x_rmse, y_rmse, x_m, y_m, visits


if __name__ == "__main__":
    print(f"Launching {N_RUNS} parallel runs on {mp.cpu_count()} cores...")
    with mp.get_context("fork").Pool(min(N_RUNS, mp.cpu_count())) as pool:
        results = pool.map(_worker, range(N_RUNS))

    all_Q = np.array([r[0] for r in results])
    Q_mean = all_Q.mean(axis=0)
    runs_rmse = [(r[1], r[2]) for r in results]
    runs_match = [(r[3], r[4]) for r in results]
    all_visits = np.array([r[5] for r in results])
    mean_visits = all_visits.mean(axis=0)

    # Save intermediate data
    np.save("images/t2_runs_rmse.npy", np.array(runs_rmse, dtype=object))
    np.save("images/t2_runs_match.npy", np.array(runs_match, dtype=object))
    np.save("images/t2_Q_mean.npy", Q_mean)
    np.save("images/t2_Q_all.npy", all_Q)
    np.save("images/t2_visits.npy", mean_visits)

    policy = extract_policy_tabular(Q_mean)
    print(f"\nReference V(0,0,DEPOT) = {REF_V[(0,0,DEPOT)]:.3f}")

    # Load Task 1 data for background comparison
    t1_runs_rmse = [(np.array(r[0], dtype=float), np.array(r[1], dtype=float))
                  for r in np.load("images/t1_runs_rmse.npy", allow_pickle=True)]
    t1_runs_match = [(np.array(r[0], dtype=float), np.array(r[1], dtype=float))
                     for r in np.load("images/t1_runs_match.npy", allow_pickle=True)]
    t1_Q_mean = np.load("images/t1_Q_mean.npy")

    print_all_policy_tables(policy)
    compare_policies(policy, REF_POLICY)

    # Two-panel convergence plot
    plot_convergence_two_panel(
        runs_rmse, runs_match,
        title="Double Q-Learning", savefig="t2.png",
        color="#C84430", label="Double Q",
        bg_runs_rmse=t1_runs_rmse, bg_runs_match=t1_runs_match,
        bg_color="C0", bg_label="Classic Q",
    )

    # Bias scatter - overlay both methods
    plot_bias_scatter(
        Q_mean, REF_V,
        title="Value bias - Double Q vs Classic Q", savefig="t2_bias.png",
        color="#C84430", label="Double Q",
        Q_mean_bg=t1_Q_mean, bg_color="C0", bg_label="Classic Q",
    )

    # Policy diff heatmap
    plot_policy_diff(
        policy, REF_POLICY,
        title="Double Q - Policy diff vs PI", savefig="t2_policy_diff.png",
    )

    # State visitation
    plot_visits(
        mean_visits,
        title="Double Q - State visitation (mean over runs)",
        savefig="t2_visits.png",
    )
