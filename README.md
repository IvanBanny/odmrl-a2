# Assignment 2 - RL for Two-Machine Maintenance MDP

Course 2AMS40.

## Tasks
1. **Tabular Q-Learning** - learn optimal policy model-free.
2. **Double Q-Learning** - investigate/fix maximization bias.
3. **Linear Function Approximation** - Q-learning with features instead of table.
4. **Average Cost** - (A) model-based PI/VI, (B) model-free RL. No discount factor.

## Files
| File | Purpose |
|------|---------|
| `Ass1_PI.py` | A1 model solution - only for reference |
| `mdp.py` | MDP model from the aforementioned A1 model solution (do not modify) |
| `rl_utils.py` | Shared utilities: reference policy, epsilon-greedy, policy display/comparison, plotting |
| `task1_qlearning.py` | Task 1 |
| `task2_double_q.py` | Task 2 |
| `task3_linear_approx.py` | Task 3 (needs custom action selection - weights, not Q-table) |
| `task4a_avg_cost_pi.py` | Task 4A (model-based, no gamma) |
| `task4b_avg_cost_rl.py` | Task 4B (model-free, no gamma) |
