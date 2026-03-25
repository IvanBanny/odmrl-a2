"""Task 1: Tabular Q-Learning."""

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
    plot_convergence_two_panel,
    plot_bias_scatter,
    plot_policy_diff,
    plot_visits,
)

# Precompute reference policy once (will be inherited by forked workers)
REF_POLICY, REF_V = compute_reference_policy()


def q_learning(lr=0.1, lr_decay=1e-4, episodes=100000, steps=1000,
               eps_start=1.0, eps_end=0.01, ping_ep=None):
    """Classic Q-learning with decaying learning rate."""
    Q = np.zeros((n_states, n_actions))
    hist = []
    match_hist = []
    visit_counts = np.zeros(n_states, dtype=np.int64)
    Q_old = Q.copy() if ping_ep else np.array(None)

    for episode in range(episodes):
        eps = eps_start + (eps_end - eps_start) * episode / max(episodes - 1, 1)
        alpha = lr / (1.0 + lr_decay * episode)
        s = states[np.random.randint(n_states)]
        for _ in range(steps):
            s_idx = state_index[s]
            visit_counts[s_idx] += 1

            # Choose action by behavior policy (returns int index)
            a_idx = epsilon_greedy(Q, s, eps)

            # Simulate step
            s_next, c = simulate_step(s, ACTIONS[a_idx])

            # Min Q over feasible actions at next state
            next_idxs = feasible_action_indices[s_next]
            min_q_next = Q[state_index[s_next], next_idxs].min()

            # Update Q table
            td_target = c + gamma * min_q_next
            Q[s_idx, a_idx] += alpha * (td_target - Q[s_idx, a_idx])
            s = s_next
        if ping_ep and episode % ping_ep == 1:
            delta = np.abs(Q - Q_old).max()
            hist.append((episode, delta))
            match_pct = policy_match_fraction(Q, REF_POLICY)
            match_hist.append((episode, match_pct))
            Q_old = Q.copy()
            if episode % (episodes // 10) < ping_ep:
                print(f"  Episode {episode:>6d}/{episodes}  eps={eps:.3f}"
                      f"  max|dQ|={delta:.4f}  match={match_pct:.1f}%")

    return Q, hist, match_hist, visit_counts


N_RUNS = 20
RUN_KWARGS = dict(lr=0.01, lr_decay=2e-4, episodes=100000, steps=1000,
                  eps_start=1.0, eps_end=0.0001, ping_ep=500)


def _worker(seed):
    np.random.seed(seed)
    Q, hist, match_hist, visits = q_learning(**RUN_KWARGS)
    x_dq = np.array([h[0] for h in hist])
    y_dq = np.array([h[1] for h in hist])
    x_m = np.array([h[0] for h in match_hist])
    y_m = np.array([h[1] for h in match_hist])
    return Q, x_dq, y_dq, x_m, y_m, visits


if __name__ == "__main__":
    print(f"Launching {N_RUNS} parallel runs on {mp.cpu_count()} cores...")
    with mp.get_context("fork").Pool(min(N_RUNS, mp.cpu_count())) as pool:
        results = pool.map(_worker, range(N_RUNS))

    all_Q = np.array([r[0] for r in results])
    Q_mean = all_Q.mean(axis=0)
    runs_dq = [(r[1], r[2]) for r in results]
    runs_match = [(r[3], r[4]) for r in results]
    all_visits = np.array([r[5] for r in results])
    mean_visits = all_visits.mean(axis=0)

    # Save intermediate data
    np.save("images/t1_runs_dq.npy", np.array(runs_dq, dtype=object))
    np.save("images/t1_runs_match.npy", np.array(runs_match, dtype=object))
    np.save("images/t1_Q_mean.npy", Q_mean)
    np.save("images/t1_Q_all.npy", all_Q)
    np.save("images/t1_visits.npy", mean_visits)

    policy = extract_policy_tabular(Q_mean)
    print(f"\nReference V(0,0,DEPOT) = {REF_V[(0,0,DEPOT)]:.3f}")

    print_all_policy_tables(policy)
    compare_policies(policy, REF_POLICY)

    # Two-panel convergence plot
    plot_convergence_two_panel(
        runs_dq, runs_match,
        title="Classic Q-Learning", savefig="t1.png",
    )

    # Bias scatter
    plot_bias_scatter(
        Q_mean, REF_V,
        title="Classic Q - Value bias", savefig="t1_bias.png",
        color="C0", label="Classic Q",
    )

    # Policy diff heatmap
    plot_policy_diff(
        policy, REF_POLICY,
        title="Classic Q - Policy diff vs PI", savefig="t1_policy_diff.png",
    )

    # State visitation
    plot_visits(
        mean_visits,
        title="Classic Q - State visitation (mean over runs)",
        savefig="t1_visits.png",
    )
