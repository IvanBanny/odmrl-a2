"""Task 3: Q-Learning with Linear Value Function Approximation.

Uses Fourier cosine basis (order 3) for the two numerical state dimensions
(x1, x2), augmented with location one-hots and failure indicators to handle
the categorical location variable and the discontinuity at the failure thresholds.
Per-action weight vectors: Q(s, a_j) = w_j^T phi(s).

Episodic semi-gradient Q-learning with random start states (matching Task 1
protocol for clean comparison).

Per-feature step-size scaling for Fourier features (Sutton & Barto 9.5.2):
alpha_i = alpha / ||c^i||
"""

import multiprocessing as mp
import numpy as np
from itertools import product
from mdp import (
    gamma, DEPOT, AT_1, AT_2, REP_1, REP_2, xi1, xi2,
    states, state_index, n_states,
    ACTIONS, action_index, n_actions, feasible_actions,
    feasible_action_indices,
    cost, simulate_step,
)
from rl_utils import (
    compute_reference_policy,
    min_q_values,
    print_all_policy_tables,
    compare_policies,
    plot_convergence_two_panel,
    plot_bias_scatter,
    plot_policy_diff,
    plot_visits,
    plot_value_comparison,
    plot_feature_decomposition,
)


# Precompute reference policy once (inherited by forked workers)
REF_POLICY, REF_V = compute_reference_policy()
REF_V_ARR = np.array([REF_V[s] for s in states])


# ---------------------------------------------------------------------------
# Feature construction: Fourier cosine basis + location/failure indicators
# ---------------------------------------------------------------------------

FOURIER_ORDER = 3
# All (c1, c2) pairs with c1, c2 in {0, ..., order}
FOURIER_COEFFS = np.array(
    list(product(range(FOURIER_ORDER + 1), repeat=2)), dtype=np.float64
)  # shape ((order + 1) ^ 2, 2) = (16, 2)
N_FOURIER = len(FOURIER_COEFFS)
N_LOC = 5  # one-hot: DEPOT, AT_1, AT_2, REP_1, REP_2
N_FAIL = 3  # I(x1=xi1), I(x2=xi2), I(both failed)
N_FEATURES = N_FOURIER + N_LOC + N_FAIL

# Per-feature lr scaling (Sutton & Barto 9.5.2)
ALPHA_SCALE = np.ones(N_FEATURES)
for _i, _c in enumerate(FOURIER_COEFFS):
    _norm = np.linalg.norm(_c)
    if _norm > 0:
        ALPHA_SCALE[_i] = 1.0 / _norm


def phi(state):
    """Feature vector for a single state. Returns shape (N_FEATURES,)."""
    x1, x2, l = state
    f = np.empty(N_FEATURES)

    # Fourier cosine basis: cos(pi * [c1, c2] * [x1/xi1, x2/xi2])
    s_norm = np.array([x1 / xi1, x2 / xi2])
    f[:N_FOURIER] = np.cos(np.pi * (FOURIER_COEFFS @ s_norm))

    # Location one-hot
    k = N_FOURIER
    f[k: k + N_LOC] = 0.0
    f[k + l] = 1.0

    # Failure indicators
    k += N_LOC
    f1 = float(x1 == xi1)
    f2 = float(x2 == xi2)
    f[k] = f1
    f[k + 1] = f2
    f[k + 2] = f1 * f2

    return f


# Precompute feature vectors for every state
PHI = np.array([phi(s) for s in states])  # (n_states, N_FEATURES)


# ---------------------------------------------------------------------------
# Linear Q helpers
# ---------------------------------------------------------------------------

def epsilon_greedy_linear(W, s, eps):
    """Epsilon-greedy over feasible actions using linear Q. Returns int idx."""
    act_idxs = feasible_action_indices[s]
    if np.random.random() < eps:
        return act_idxs[np.random.randint(len(act_idxs))]
    return act_idxs[np.argmin(W[act_idxs] @ PHI[state_index[s]])]


def extract_policy(W):
    """Greedy policy from weight matrix. Returns dict state -> action str."""
    policy = {}
    for s in states:
        act_idxs = feasible_action_indices[s]
        q_vals = W[act_idxs] @ PHI[state_index[s]]
        policy[s] = ACTIONS[act_idxs[np.argmin(q_vals)]]
    return policy


def policy_match(W, ref_policy):
    """% of states where greedy linear policy matches ref_policy."""
    match = 0
    for s in states:
        act_idxs = feasible_action_indices[s]
        q_vals = W[act_idxs] @ PHI[state_index[s]]
        if ACTIONS[act_idxs[np.argmin(q_vals)]] == ref_policy[s]:
            match += 1
    return match / len(states) * 100.0


def compute_ve(W, visit_counts):
    """VE(w) = sum_s mu(s) [V*(s) - V_hat(s, w)] ^ 2, with mu from visits."""
    total = visit_counts.sum()
    if total == 0:
        return float("inf")
    mu = visit_counts / total
    v_hat = np.empty(n_states)
    for s in states:
        si = state_index[s]
        act_idxs = feasible_action_indices[s]
        v_hat[si] = np.min(W[act_idxs] @ PHI[si])
    return np.sum(mu * (v_hat - REF_V_ARR) ** 2)


def w_to_qtable(W):
    """Expand weight matrix to (n_states, n_actions) Q-table for plotting."""
    Q = np.full((n_states, n_actions), np.inf)
    for s in states:
        si = state_index[s]
        for a_idx in feasible_action_indices[s]:
            Q[si, a_idx] = W[a_idx] @ PHI[si]
    return Q


# ---------------------------------------------------------------------------
# Semi-gradient Q-learning with linear FA
# ---------------------------------------------------------------------------

def q_learning_linear(lr=0.01, lr_decay=2e-4, episodes=100_000, steps=1000,
                      eps_start=1.0, eps_end=0.0001, ping_ep=None):
    """Episodic semi-gradient Q-learning with linear FA.

    Update rule (per-action weights, minimization):
        delta = c + gamma * min_{a'} W[a']^T phi(s') - W[a]^T phi(s)
        W[a] += alpha * delta * (ALPHA_SCALE * phi(s))

    The ALPHA_SCALE vector implements per-feature step-size scaling.
    """
    W = np.zeros((n_actions, N_FEATURES))
    visit_counts = np.zeros(n_states, dtype=np.int64)
    ve_hist = []
    match_hist = []

    for episode in range(episodes):
        eps = eps_start + (eps_end - eps_start) * episode / max(episodes - 1, 1)
        alpha = lr / (1.0 + lr_decay * episode)
        s = states[np.random.randint(n_states)]

        for _ in range(steps):
            si = state_index[s]
            visit_counts[si] += 1
            phi_s = PHI[si]

            # Action selection
            a_idx = epsilon_greedy_linear(W, s, eps)

            # Environment step
            s_next, c = simulate_step(s, ACTIONS[a_idx])

            # Semi-gradient TD update (only W[a_idx] is modified)
            next_acts = feasible_action_indices[s_next]
            min_q_next = (W[next_acts] @ PHI[state_index[s_next]]).min()

            td_error = c + gamma * min_q_next - (W[a_idx] @ phi_s)
            W[a_idx] += alpha * td_error * (ALPHA_SCALE * phi_s)

            s = s_next

        if ping_ep and episode % ping_ep == 1:
            ve = compute_ve(W, visit_counts)
            ve_hist.append((episode, ve))
            match_pct = policy_match(W, REF_POLICY)
            match_hist.append((episode, match_pct))
            if episode % (episodes // 10) < ping_ep:
                print(f"  Episode {episode:>6d}/{episodes}  eps={eps:.4f}"
                      f"  VE={ve:.4f}  match={match_pct:.1f}%")

    return W, ve_hist, match_hist, visit_counts


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

N_RUNS = 20
RUN_KWARGS = dict(lr=0.01, lr_decay=2e-4, episodes=100_000, steps=1000,
                  eps_start=1.0, eps_end=0.0001, ping_ep=500)


def _worker(seed):
    np.random.seed(seed)
    W, ve_hist, match_hist, visits = q_learning_linear(**RUN_KWARGS)
    x_ve = np.array([h[0] for h in ve_hist])
    y_ve = np.array([h[1] for h in ve_hist])
    x_m = np.array([h[0] for h in match_hist])
    y_m = np.array([h[1] for h in match_hist])
    return W, x_ve, y_ve, x_m, y_m, visits


if __name__ == "__main__":
    print("Task 3: Linear FA Q-Learning "
          f"(Fourier order {FOURIER_ORDER})")
    print(f"  Features per state: {N_FEATURES}  "
          f"({N_FOURIER} Fourier + {N_LOC} loc + {N_FAIL} fail)")
    print(f"  Weight matrix: {n_actions} x {N_FEATURES} "
          f"= {n_actions * N_FEATURES} parameters")
    print(f"  (vs {n_states * n_actions} Q-table entries in tabular)")
    print(f"Launching {N_RUNS} runs on {mp.cpu_count()} cores...")

    with mp.get_context("fork").Pool(min(N_RUNS, mp.cpu_count())) as pool:
        results = pool.map(_worker, range(N_RUNS))

    all_W = np.array([r[0] for r in results])
    W_mean = all_W.mean(axis=0)
    runs_ve = [(r[1], r[2]) for r in results]
    runs_match = [(r[3], r[4]) for r in results]
    all_visits = np.array([r[5] for r in results])
    mean_visits = all_visits.mean(axis=0)

    # Save intermediate data
    np.save("images/t3_runs_ve.npy", np.array(runs_ve, dtype=object))
    np.save("images/t3_runs_match.npy", np.array(runs_match, dtype=object))
    np.save("images/t3_W_mean.npy", W_mean)
    np.save("images/t3_W_all.npy", all_W)
    np.save("images/t3_visits.npy", mean_visits)

    policy = extract_policy(W_mean)
    print(f"\nReference V(0,0,DEPOT) = {REF_V[(0, 0, DEPOT)]:.3f}")

    print_all_policy_tables(policy)
    compare_policies(policy, REF_POLICY)

    # Convergence: VE (left, log) + policy match % (right, linear)
    plot_convergence_two_panel(
        runs_ve, runs_match,
        ylabel_left=r"$\overline{\mathrm{VE}}$",
        title=f"Linear FA Q-Learning (Fourier order {FOURIER_ORDER})",
        xlabel="Episode",
        savefig="t3.png",
    )

    # Bias scatter (reuse tabular plotting via expanded Q-table)
    Q_equiv = w_to_qtable(W_mean)
    plot_bias_scatter(
        Q_equiv, REF_V,
        title="Linear FA - Value bias",
        savefig="t3_bias.png",
        color="C2", label="Linear FA",
    )

    # Policy diff heatmap
    plot_policy_diff(
        policy, REF_POLICY,
        title="Linear FA - Policy diff vs PI",
        savefig="t3_policy_diff.png",
    )

    # State visitation
    plot_visits(
        mean_visits,
        title="Linear FA - State visitation (mean over runs)",
        savefig="t3_visits.png",
    )

    # V* vs V_hat vs error
    def v_hat_fn(s):
        si = state_index[s]
        act_idxs = feasible_action_indices[s]
        return (W_mean[act_idxs] @ PHI[si]).min()

    plot_value_comparison(
        v_hat_fn, REF_V,
        title="Linear FA - Value approximation vs PI",
        savefig="t3_value_comparison.png",
    )

    # Feature decomposition
    plot_feature_decomposition(
        W_mean, PHI, N_FOURIER, N_LOC, N_FAIL,
        title="Linear FA - Feature decomposition",
        savefig="t3_decomposition.png",
    )
