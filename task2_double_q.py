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
    plot_convergence_multi,
)

def double_q_learning(lr=0.1, lr_decay=1e-4, episodes=100000, steps=1000,
                      eps_start=1.0, eps_end=0.01, ping_ep=None):
    """Double Q-learning to address minimization bias."""
    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))
    hist = []
    Q_old = (Q1 + Q2).copy() if ping_ep else np.array(None)

    for episode in range(episodes):
        eps = eps_start + (eps_end - eps_start) * episode / max(episodes - 1, 1)
        alpha = lr / (1.0 + lr_decay * episode)
        s = states[np.random.randint(n_states)]
        for _ in range(steps):
            s_idx = state_index[s]

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
            delta = np.abs(Q_combined - Q_old).max()
            hist.append((episode, delta))
            Q_old = Q_combined
            if episode % (episodes // 10) < ping_ep:
                print(f"  Episode {episode:>6d}/{episodes}  eps={eps:.3f}  max|dQ|={delta:.4f}")

    return (Q1, Q2, hist) if ping_ep else (Q1, Q2)

N_RUNS = 20
RUN_KWARGS = dict(lr=0.1, lr_decay=2e-4, episodes=500000, steps=1000,
                  eps_start=1.0, eps_end=0.01, ping_ep=2000)


def _worker(seed):
    np.random.seed(seed)
    Q1, Q2, hist = double_q_learning(**RUN_KWARGS)
    x = np.array([h[0] for h in hist])
    y = np.array([h[1] for h in hist])
    return (Q1 + Q2) / 2, x, y


if __name__ == "__main__":
    print(f"Launching {N_RUNS} parallel runs on {mp.cpu_count()} cores...")
    with mp.get_context("fork").Pool(min(N_RUNS, mp.cpu_count())) as pool:
        results = pool.map(_worker, range(N_RUNS))

    Q_mean = np.mean([r[0] for r in results], axis=0)
    runs = [(r[1], r[2]) for r in results]
    
    np.save("images/t2_runs.npy", np.array([(r[1], r[2]) for r in results], dtype=object))
    np.save("images/t2_Q_mean.npy", Q_mean)

    policy = extract_policy_tabular(Q_mean)
    ref_policy, ref_V = compute_reference_policy()
    print_all_policy_tables(policy)
    compare_policies(policy, ref_policy)
    plot_convergence_multi(
        runs, xlabel="Episode", ylabel=r"$\max |\Delta Q|$",
        title="Double Q-Learning convergence", savefig="t2.png"
    )
