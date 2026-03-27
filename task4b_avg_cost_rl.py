"""Task 4: Part B: Average Cost RL (model-free, unknown P)."""

import numpy as np
import matplotlib.pyplot as plt
from mdp import (
    DEPOT, AT_1, AT_2, REP_1, REP_2, xi1, xi2,
    states, state_index, n_states,
    ACTIONS, action_index, n_actions, feasible_actions,
    cost, transitions, simulate_step,
)
from rl_utils import (
    epsilon_greedy,
    extract_policy_tabular,
    plot_convergence_multi_compare,
    print_all_policy_tables,
    compare_policies,
    plot_convergence,
    plot_convergence_multi
)

np.random.seed(42)

def RVI_Q_learning(num_steps, alpha, alpha_decay, epsilon_start, epsilon_min, epsilon_decay):
    Q = np.zeros((n_states, n_actions))
    s_ref = (0, 0, DEPOT)  # Reference state for average cost estimation
    a_ref = "nothing"  # Reference action for average cost estimation

    s = (0,0,DEPOT)  # Starting state
    g = []  # To track average cost convergence
    td = []  # To track TD error convergence
    visits = np.zeros((n_states, n_actions))

    for t in range(num_steps):
        epsilon = epsilon_start / t ** epsilon_decay if t > 0 else epsilon_start
        if epsilon < epsilon_min:
            epsilon = epsilon_min
        epsilon_greedy_action = epsilon_greedy(Q, s, epsilon)
        a = ACTIONS[epsilon_greedy_action]
        s_next, c = simulate_step(s, a)
        next_act = [a2 for a2 in feasible_actions(s_next)]
        q_next_min = np.min(Q[state_index[s_next], [action_index[a2] for a2 in next_act]])
        q_ref = Q[state_index[s_ref], action_index[a_ref]]

        td_target = c - q_ref + q_next_min
        td_error = td_target - Q[state_index[s], action_index[a]]

        visits[state_index[s], action_index[a]] += 1
        
        alpha_t = alpha / (visits[state_index[s], action_index[a]] ** alpha_decay)  
        Q[state_index[s], action_index[a]] += alpha_t * td_error

        g.append(q_ref)  # Average cost estimate
        td.append(abs(td_error))  # TD error for convergence tracking

        s = s_next

        if (t % 5000) == 0:
            s = (0,0,DEPOT)  # Reset to the starting state every 5000 steps to ensure exploration  

    return Q, g, td  

def parameter_testing(param_name, param_values, num_steps=300000):
    g_runs = []
    td_runs = []
    labels = []

    # Define the base configuration for RVI Q-learning
    config = {
        'alpha': 0.5,
        'alpha_decay': 0.75,
        'epsilon_decay': 0.5,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01
    }

    for val in param_values:
        
        current_config = config.copy()
        current_config[param_name] = val
        
        print(f"Testing {param_name} = {val}...")
        
        
        Q, g, td = RVI_Q_learning(num_steps, **current_config)
        
        
        x_axis = np.arange(len(g))
        g_runs.append([x_axis, g])

        td_window = 10000
        td_smoothed = np.array([np.mean(td[i:i+td_window]) for i in range(0, len(td), td_window)])
        x_td = np.arange(len(td_smoothed)) * td_window
        td_runs.append([x_td, td_smoothed])  # Average TD error per 10000 steps
        labels.append(f"{param_name}={val}")

        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    
    for i, (x, y) in enumerate(g_runs):
        ax1.plot(x, y, label=f"{labels[i]}")
    ax1.set_title(f"Convergence of Average Cost ({param_name})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average cost")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for i, (x, y) in enumerate(td_runs):
        ax2.plot(x, y, label=f"{labels[i]}", alpha=0.7)
    ax2.set_title(f"TD Error Magnitude ({param_name})")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Mean Absolute TD Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

#parameter_testing('alpha', [0.1, 0.3, 0.5, 0.7, 0.9], num_steps=1000000)
#parameter_testing('alpha_decay', [0.5, 0.6, 0.7, 0.8, 0.9], num_steps=1000000)
#parameter_testing('epsilon_decay', [0.1, 0.3, 0.5, 0.7, 0.9], num_steps=1000000)

Q, g, td = RVI_Q_learning(10000000, alpha=0.3, alpha_decay=0.8, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.3)

def auto_calibrate_thresholds(series, window=1000, tail_fraction=0.2, level_multiplier=1.1):
    """Estimate drift and level thresholds from the tail noise floor of a series."""
    smoothed = np.array([np.mean(series[i:i+window]) for i in range(0, len(series), window)], dtype=float)
    if len(smoothed) < 6:
        return 1e-5, float(np.mean(smoothed)) if len(smoothed) > 0 else 0.0

    tail_len = max(5, int(len(smoothed) * tail_fraction))
    tail = smoothed[-tail_len:]
    tail_diffs = np.abs(np.diff(tail))

    # Drift threshold: around 2-sigma of tail drift noise
    drift_threshold = max(2.0 * np.std(tail_diffs), 1e-12)

    # Optional level threshold (useful for |TD|)
    level_threshold = float(np.median(tail) * level_multiplier)
    return drift_threshold, level_threshold


def convergence_tester(series, window=1000, threshold=1e-5, patience=8,
                       level_threshold=None, min_start_fraction=0.6,
                       tail_relax=1.25):
    smoothed = np.array([np.mean(series[i:i+window]) for i in range(0, len(series), window)])
    diffs = np.abs(np.diff(smoothed))
    start_i = int(min_start_fraction * len(diffs))
    stable = 0
    converge_index, converge_value, diff_value = None, None, None
    for i in range(start_i, len(diffs)):
        diff = diffs[i]
        drift_ok = diff < threshold
        level_ok = True if level_threshold is None else (smoothed[i + 1] < level_threshold)

        if drift_ok and level_ok:
            stable += 1
            if stable >= patience:
                # Extra guard: tail after detection should stay mostly stable
                tail_diffs = diffs[i + 1:]
                if len(tail_diffs) == 0 or np.mean(tail_diffs) < threshold * tail_relax:
                    converge_index = (i + 1) * window
                    converge_value = smoothed[i + 1]
                    diff_value = diff
                    break
        else:
            stable = 0

    if converge_index is None:
        print("No convergence detected with current thresholds.")
    else:
        print(f"Convergence detected at index {converge_index} with value {converge_value:.6f} and diff {diff_value:.2e}")
    return converge_value
    
print("ewa")





#g_threshold, _ = auto_calibrate_thresholds(g, window=1000)
#td_threshold, td_level_threshold = auto_calibrate_thresholds(td, window=1000)

print(f"Auto thresholds -> g drift: {g_threshold:.2e}, td drift: {td_threshold:.2e}, td level: {td_level_threshold:.4f}")

#convergence_tester(g, window=1000, threshold=g_threshold, patience=8, min_start_fraction=0.2)
#convergence_tester(td, window=1000, threshold=td_threshold, patience=8, level_threshold=td_level_threshold, min_start_fraction=0.2)

plot_convergence([x for x in range(len(g))], g, "Average Cost Estimate", "Average Cost Estimate Convergence", "Steps")
plot_convergence([x for x in range(len(td))], td, "TD Error", "TD Error Convergence", "Steps")