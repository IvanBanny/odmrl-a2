"""Task 3: Q-Learning with Linear Value Function Approximation."""

import numpy as np
import matplotlib.pyplot as plt
from mdp import (
    gamma, DEPOT, AT_1, AT_2, REP_1, REP_2, xi1, xi2,
    states, state_index, n_states,
    ACTIONS, action_index, n_actions, feasible_actions,
    cost, transitions, simulate_step,
)
from rl_utils import (
    compute_reference_policy,
    print_all_policy_tables,
    compare_policies,
    plot_convergence,
)
