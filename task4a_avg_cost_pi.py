"""Task 4: Part A: Average Cost Optimal Policy (model-based, full knowledge of P)."""

import numpy as np
import matplotlib.pyplot as plt
import random
from mdp import (
    DEPOT, AT_1, AT_2, REP_1, REP_2, xi1, xi2,
    states, state_index, n_states,
    ACTIONS, action_index, n_actions, feasible_actions,
    cost, transitions, simulate_step,
)
from rl_utils import (
    print_all_policy_tables,
    compare_policies,
    plot_convergence,
)

import numpy as np

# P[state_idx, action_idx, next_state_idx] = probability
P = np.zeros((n_states, n_actions, n_states))

for state_idx, state in enumerate(states):
    for action_idx, action in enumerate(ACTIONS):
        # Get transition probabilities for this (state, action) pair
        trans_probs = transitions(state, action)
        
        # Fill the matrix
        for next_state, prob in trans_probs.items():
            next_state_idx = state_index[next_state]
            P[state_idx, action_idx, next_state_idx] = prob

# Now you can access transition probabilities as:
# P[state_idx, action_idx, :] gives all probabilities from state_idx under action_idx
# P[state_idx, action_idx, next_state_idx] gives the specific probability

v = [0 for _ in range(n_states)]  # Initial value function

pi = ["nothing" for _ in range(n_states)] # Initial policy: Action 0 for all states, except when at the end of maintenance cycle.
for s in states:
    if s[0] == 5 and s[1] == 7:
        pi[state_index[s]] = random.choice(["travel_1", "travel_2"]) # if both at end of maintenance cycle, choose randomly between travel_1 and travel_2
    elif s[0] == 5:
        pi[state_index[s]] = "travel_1"  
    elif s[1] == 7:
        pi[state_index[s]] = "travel_2"  
    else:
        feasible_actions_list = feasible_actions(s)
        if "nothing" in feasible_actions_list:
            pi[state_index[s]] = "nothing" 
        else:
            pi[state_index[s]] = random.choice(feasible_actions_list)  # If "nothing" is not feasible, choose randomly from the feasible actions
           
max_policy_iterations = 50
max_value_iterations = 1000
delta = 1e-6
ref_state = (0, 0, DEPOT) # Reference state

g_convergence = [0 for _ in range(max_policy_iterations)] # To track average cost convergence

for i in range(max_policy_iterations):
    print("Iteration {}".format(i+1))
    optimal_policy_found = True
    g = 0
    v_old = v.copy()
    for _ in range(max_value_iterations):
        max_diff = 0
        g_new = 0 # Average cost estimate
        for s in states:
            val = cost(s, pi[state_index[s]])
            for next_s in states:
                val += P[state_index[s], action_index[pi[state_index[s]]], state_index[next_s]] * v_old[state_index[next_s]] # Transition effects   
            if s == ref_state: 
                g_new = val 
                
            v[state_index[s]] = val - g_new # Update value function
            max_diff = max(max_diff, abs(v[state_index[s]] - v_old[state_index[s]]))
            

        v_old = v.copy()    
        if max_diff < delta:
            g = g_new
            print("Value function converged after {} iterations with max_diff {:.2e}.".format(_, max_diff))
            break 
    for s in states:
        best_action = pi[state_index[s]]
        best_value = v[state_index[s]]
        feasible_actions_list = feasible_actions(s)
        for a in feasible_actions_list:
            val = cost(s, a)
            for next_s in states:
                val += P[state_index[s], action_index[a], state_index[next_s]] * v[state_index[next_s]] # Transition effects
            if val < best_value:
                best_value = val
                best_action = a
                optimal_policy_found = False
        pi[state_index[s]] = best_action

    g_convergence[i] = g

    if optimal_policy_found:
        print("Optimal policy found after {} iterations.".format(i+1))
        break
    

print("Average Cost per Stage:", g)
print("Optimal Policy:", pi)
print("Relative Value Function:", v)


policy_by_state = {s: pi[state_index[s]] for s in states}
print_all_policy_tables(policy_by_state)
            
plot_convergence(range(1, len(g_convergence)+1), g_convergence, title="Average Cost Convergence", ylabel="Average Cost per Stage")



    

