import numpy as np
from itertools import product
import math
from math import exp, factorial
import pandas as pd

# -----------------------------
# Parameters
# -----------------------------
gamma = 0.9
lam=0.5
xi1, xi2 = 5, 7

# Location states
DEPOT = 0
AT_1 = 1   # engineer is @ machine 1
AT_2 = 2   # engineer is @ machine 2
REP_1 = 3   # 1 unit of remaining time for corrective maintenance @1. Note that this state only exists when x1=xi1. 
REP_2 = 4   # 1 unit of remaining time for corrective maintenance @2. Note that this state only exists when x2=xi2. 


# -----------------------------
# State space
# -----------------------------
"""
Note that if we define the state-space as follows
states = [(x1, x2, l)
          for x1 in range(xi1 + 1)
          for x2 in range(xi2 + 1)
          for l in range(5)]
we allow for several states of the form (x1,x2,REP_1) or (x1,x2,REP_2) that do not have meaning. 
Its best to not include these states. So, we do as follows:
"""

states = []

for x1 in range(xi1 + 1):
    for x2 in range(xi2 + 1):

        # always allowed locations
        for l in [DEPOT, AT_1, AT_2]:
            states.append((x1, x2, l))

        # corrective maintenance state for machine 1
        if x1 == xi1:
            states.append((x1, x2, REP_1))

        # corrective maintenance state for machine 2
        if x2 == xi2:
            states.append((x1, x2, REP_2))


# -----------------------------
# Poisson probabilities
# -----------------------------
def poisson_pmf(y,lam, max_y):
    """
    Returns the y-th element of the list:
    [P(X=0), P(X=1), ..., P(X=max_y-1), P(X>=max_y)]
    with X ~ Poisson(lam)
    """

    probs = []

    if max_y>0:
        # P(X=0)
        p = math.exp(-lam)
        probs.append(p)

        # Compute P(X=k) recursively
        for k in range(1, max_y):
            p = p * lam / k
            probs.append(p)

        # Tail probability
        tail = 1.0 - sum(probs)
        probs.append(tail)
    elif max_y==0:
        probs=[1]
    return probs[y]

# -----------------------------
# Action space
# -----------------------------
#Note that this is further refined below when considering the possible acitons
ACTIONS = ["nothing", "travel_1", "travel_2", "travel_depot",
           "maintain_1", "maintain_2","continue maintenance"]
"""
The action "continue maintenance" is a pseudo-action to only indicate that we are forced to continue the corrective maintenance for a second period of time and should not impact the results        
"""    

# -----------------------------
# Feasible actions
# -----------------------------
def feasible_actions(state):
    x1, x2, l = state
    
    acts = []
    
    # forced continuation during repair
    if l in (REP_1, REP_2):
        acts = ["continue maintenance"]

    elif l == DEPOT:
        if x2 == xi2 and x1 == xi1:
            acts = ["travel_1", "travel_2"]
        elif x2 == xi2 and x1 < xi1:
            acts = ["travel_2"]
        elif x1 == xi1 and x2 < xi2:
            acts = ["travel_1"]
        else:
            acts = ["nothing", "travel_1", "travel_2"]

    elif l == AT_1:
        if x1 == xi1:
            acts = ["maintain_1"]
        elif x1 < xi1 and x2 == xi2: 
            acts = ["travel_depot"]
        else:
            acts = ["travel_depot", "maintain_1","nothing"]  
        

    elif l == AT_2:
        if x2 == xi2:
            acts = ["maintain_2"]
        elif x2 < xi2 and x1 == xi1: 
            acts = ["travel_depot"]
        else:
            acts = ["travel_depot", "maintain_2","nothing"] 

    return acts

# -----------------------------
# Cost
# -----------------------------
def cost(state, action):
    x1, x2, l = state

    c = 0

    if action == "maintain_1":
        if x1 < xi1:
            c += 1 
        else:
            c += 5 
                
    if action == "maintain_2":
        if x2 < xi2:
            c += 1 
        else:
            c += 5 

    # unavailability cost
    if x1 == xi1 :
        c += 1
    if x2 == xi2 :
        c += 1

    return c

# -----------------------------
# Transitions
# -----------------------------
def transitions(state, action):
    x1, x2, l = state
    trans = {}

    # ---------------- forced repair completion
    if l == REP_1:
        # in the next unit, maintenance completed → machine 1 becomes healthy
        for y in range(xi2-x2+1):
            p = poisson_pmf(y,lam, xi2-x2)
            x2n = min(x2 + y, xi2)
            trans[(0, x2n, AT_1)] = \
                trans.get((0, x2n, AT_1), 0) + p
        return trans

    elif l == REP_2:
        # in the next unit, maintenance completed → machine 2 becomes healthy
        for y in range(xi1-x1+1):
            p = poisson_pmf(y,lam,xi1-x1)
            x1n = min(x1 + y, xi1)
            trans[(x1n, 0, AT_2)] = \
                trans.get((x1n, 0, AT_2), 0) + p
        return trans

    # ---------------- travel
    elif action == "travel_1":
        for y1 in range(xi1-x1+1):
            for y2 in range(xi2-x2+1):
                p = poisson_pmf(y1,lam,xi1-x1) * poisson_pmf(y2,lam,xi2-x2)
                x1n = min(x1 + y1, xi1)
                x2n = min(x2 + y2, xi2)
                trans[(x1n, x2n, AT_1)] = \
                    trans.get((x1n, x2n, AT_1), 0) + p
        return trans

    elif action == "travel_2":
        for y1 in range(xi1-x1+1):
            for y2 in range(xi2-x2+1):
                p = poisson_pmf(y1,lam,xi1-x1) * poisson_pmf(y2,lam,xi2-x2)
                x1n = min(x1 + y1, xi1)
                x2n = min(x2 + y2, xi2)
                trans[(x1n, x2n, AT_2)] = \
                    trans.get((x1n, x2n, AT_2), 0) + p
        return trans
    
    elif action == "travel_depot":
        for y1 in range(xi1-x1+1):
            for y2 in range(xi2-x2+1):
                p = poisson_pmf(y1,lam,xi1-x1) * poisson_pmf(y2,lam,xi2-x2)
                x1n = min(x1 + y1, xi1)
                x2n = min(x2 + y2, xi2)
                trans[(x1n, x2n, DEPOT)] = \
                    trans.get((x1n, x2n, DEPOT), 0) + p

    # ---------------- maintenance start
    elif action == "maintain_1":
        # preventive = 1 period
        if x1 < xi1:
            for y in range(xi2-x2+1):
                p = poisson_pmf(y,lam, xi2-x2)
                x2n = min(x2 + y, xi2)
                trans[(0, x2n, AT_1)] = \
                    trans.get((0, x2n, AT_1), 0) + p
        elif x1 == xi1:
            # corrective = 2 periods
            for y in range(xi2-x2+1):
                p = poisson_pmf(y,lam,xi2-x2)
                x2n = min(x2 + y, xi2)
                trans[(xi1, x2n, REP_1)] = \
                    trans.get((xi1, x2n, REP_1), 0) + p
        return trans

    elif action == "maintain_2":
        if x2 < xi2:
            for y in range(xi1-x1+1):
                p = poisson_pmf(y,lam,xi1-x1)
                x1n = min(x1 + y, xi1)
                trans[(x1n, 0, AT_2)] = \
                    trans.get((x1n, 0, AT_2), 0) + p
        elif x2==xi2:
            for y in range(xi1-x1+1):
                p = poisson_pmf(y,lam,xi1-x1)
                x1n = min(x1 + y, xi1)
                trans[(x1n, xi2, REP_2)] = \
                    trans.get((x1n, xi2, REP_2), 0) + p
        return trans

    elif action == "nothing":
        for y1 in range(xi1-x1+1):
            for y2 in range(xi2-x2+1):
                p = poisson_pmf(y1,lam,xi1-x1) * poisson_pmf(y2,lam,xi2-x2)
                x1n = min(x1 + y1, xi1)
                x2n = min(x2 + y2, xi2)
                trans[(x1n, x2n, l)] = \
                    trans.get((x1n, x2n, l), 0) + p
    return trans

# -----------------------------
# Policy Iteration
# -----------------------------
V = {s: 0.0 for s in states}
policy = {s: feasible_actions(s)[0] for s in states}

def policy_evaluation(policy, V, tol=1e-10):
    while True:
        delta = 0
        for s in states:
            a = policy[s]
            v_new = cost(s, a)
            for sp, p in transitions(s, a).items():
                v_new += gamma * p * V[sp]
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        if delta < tol:
            break

def policy_improvement(policy, V):
    stable = True
    for s in states:
        best_a = policy[s]
        best_v = float("inf")
        for a in feasible_actions(s):
            q = cost(s, a)
            for sp, p in transitions(s, a).items():
                q += gamma * p * V[sp]
                #print(s,a,sp,cost(s, a),p)
            if q < best_v:
                best_v = q
                best_a = a
        if best_a != policy[s]:
            policy[s] = best_a
            stable = False
    return stable

# -----------------------------
# Run algorithm
# -----------------------------
iterations = 0
while True:
    iterations += 1
    policy_evaluation(policy, V)
    
    if policy_improvement(policy, V):
        break

print("Converged in", iterations, "iterations")

# -----------------------------
# Output results
# -----------------------------
for s in states:
    #if s[2] == DEPOT:
    print(f"State {s}: Action={policy[s]}, Value={V[s]:.3f}")


# -----------------------------
# Function to print policy table for a location
# -----------------------------
def print_policy_table(location, name):

    table = []

    for x1 in range(xi1 + 1):
        row = []
        for x2 in range(xi2 + 1):
            row.append(policy[(x1, x2, location)])
        table.append(row)

    df = pd.DataFrame(
        table,
        index=[f"x1={i}" for i in range(xi1 + 1)],
        columns=[f"x2={j}" for j in range(xi2 + 1)]
    )

    print(f"\nOptimal Policy ({name})")
    print(df)



# -----------------------------
# Export policy tables to Excel
# -----------------------------
def export_policy_tables():

    with pd.ExcelWriter("optimal_policy_tables.xlsx") as writer:

        for location, name in [(DEPOT, "Depot"),
                               (AT_1, "At Machine 1"),
                               (AT_2, "At Machine 2")]:

            table = []

            for x1 in range(xi1 + 1):
                row = []
                for x2 in range(xi2 + 1):
                    row.append(policy[(x1, x2, location)])
                table.append(row)

            df = pd.DataFrame(
                table,
                index=[f"x1={i}" for i in range(xi1 + 1)],
                columns=[f"x2={j}" for j in range(xi2 + 1)]
            )

            df.to_excel(writer, sheet_name=name)

    print("Policy tables exported to optimal_policy_tables.xlsx")


export_policy_tables()

# -----------------------------
# Print tables
# -----------------------------
print_policy_table(DEPOT, "Depot")
print_policy_table(AT_1, "At Machine 1")
print_policy_table(AT_2, "At Machine 2")




"""  
# -----------------------------
# Checking: Create Excel file with all transitions per action
# -----------------------------
def transition_tables():

    with pd.ExcelWriter("transition_tables.xlsx") as writer:

        for action in ACTIONS:

            rows = []
            index = []

            for s in states:

                if action not in feasible_actions(s):
                    continue

                trans = transitions(s, action)

                row = []
                for sp in states:
                    row.append(trans.get(sp, 0))

                rows.append(row)
                index.append(s)

            df = pd.DataFrame(rows, index=index, columns=states)

            # export to Excel sheet
            sheet = action[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet)

    print("Transition tables exported to transition_tables.xlsx")


transition_tables()

""" 

""" 
# -----------------------------
# Checking: Check row sums of transitions if they add to 1
# -----------------------------
def check_transition_row_sums():

    for action in ACTIONS:

        print(f"\nRow sums for action: {action}")

        for s in states:

            if action not in feasible_actions(s):
                continue

            trans = transitions(s, action)

            row_sum = sum(trans.values())

            print(f"State {s} -> sum = {row_sum:.6f}")
check_transition_row_sums()            
""" 