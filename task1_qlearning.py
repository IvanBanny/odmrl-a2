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
    plot_convergence_multi,
)

def q_learning(lr=0.1, lr_decay=1e-4, episodes=100000, steps=1000,
               eps_start=1.0, eps_end=0.01, ping_ep=None):
    """Classic Q-learning with decaying learning rate."""
    Q = np.zeros((n_states, n_actions))
    hist = []
    Q_old = Q.copy() if ping_ep else np.array(None)

    for episode in range(episodes):
        eps = eps_start + (eps_end - eps_start) * episode / max(episodes - 1, 1)
        alpha = lr / (1.0 + lr_decay * episode)
        s = states[np.random.randint(n_states)]
        for _ in range(steps):
            s_idx = state_index[s]

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
            Q_old = Q.copy()
            if episode % (episodes // 10) < ping_ep:
                print(f"  Episode {episode:>6d}/{episodes}  eps={eps:.3f}  max|dQ|={delta:.4f}")

    return (Q, hist) if ping_ep else Q

N_RUNS = 20
RUN_KWARGS = dict(lr=0.1, lr_decay=2e-4, episodes=500000, steps=1000,
                  eps_start=1.0, eps_end=0.01, ping_ep=2000)


def _worker(seed):
    np.random.seed(seed)
    Q, hist = q_learning(**RUN_KWARGS)
    x = np.array([h[0] for h in hist])
    y = np.array([h[1] for h in hist])
    return Q, x, y


if __name__ == "__main__":
    print(f"Launching {N_RUNS} parallel runs on {mp.cpu_count()} cores...")
    with mp.get_context("fork").Pool(min(N_RUNS, mp.cpu_count())) as pool:
        results = pool.map(_worker, range(N_RUNS))

    Q_mean = np.mean([r[0] for r in results], axis=0)
    runs = [(r[1], r[2]) for r in results]

    np.save("images/t1_runs.npy", np.array([(r[1], r[2]) for r in results], dtype=object))
    np.save("images/t1_Q_mean.npy", Q_mean)

    policy = extract_policy_tabular(Q_mean)
    ref_policy, ref_V = compute_reference_policy()
    print_all_policy_tables(policy)
    compare_policies(policy, ref_policy)
    plot_convergence_multi(
        runs, xlabel="Episode", ylabel=r"$\max |\Delta Q|$",
        title="Classic Q-Learning convergence", savefig="t1.png"
    )
