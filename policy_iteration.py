"""
    This file comtains the code implrmentation of the Policy Iteration Algorithm
"""

import numpy as np
from gridworld import standard_grid, ACTION_SPACE
# from IterativePolicyEvaluation_probabilistic import print_in_gridworld, print_policy

grid = standard_grid() # initializing the grid
gamma = 0.9
tol = 1e-3

def print_in_gridworld(v):
    '''
        Function to print the values in the gridworld
    '''
    # making the grid list
    out = []
    for i in range(7):
        if i%2 != 0:
            inv = []
            for j in range(9):
                if j%2 == 0:
                    inv.append("|")
                else:
                    if (i//2, j//2) == (1,1):
                        inv.append(0.00)
                    else:
                        _ = v[(i//2, j//2)]
                        # _ = round(_, 2)
                        inv.append(_)
            out.append(inv)
        else:
            out.append(["_" for x in range(9)])

    for i in out:
        print(*i,"\t", end = "\n") 

def init_rewards_and_transition_probs(grid):
    '''
        get the transition probs and the rewards
            rewards: r(s,a,s')
            transition probs: p(s' | a,s) = 1 in each case since the gridworld is deterministic
    '''
    rewards = {}
    transition_probs = {}
    
    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s, a)
                    transition_probs[(s, a, s2)] = 1
                    if s2 in grid.rewards:
                        rewards[(s, a, s2)] = grid.rewards.get(s2, 0)
    return transition_probs, rewards

transition_probs, rewards = init_rewards_and_transition_probs(grid)
### INIT RANDOM POLICY AND VALUE ###
states = grid.all_states() 
policy = {}

for s in states: # making a random policy
    policy[s] = np.random.choice(a = list(ACTION_SPACE))


### EVALUATE VALUE ###

def value_eval(grid, policy):
    '''
        Evaluate the value given a policy using IterativePolicyEvaluation
    '''
    
    values = {st:0 for st in states}

    while True:
        r = 0
        delta = 0
        for s in states:             
            if grid.is_terminal(s):
                continue
            old_v = values[s]
            new_v = 0
            for a in ACTION_SPACE:
                action_prob = 1 if policy.get(s) == a else 0
                for s2 in states:
                    if s2 == s:
                        continue
                    r = rewards.get((s,a,s2), 0)
                    env_prob = transition_probs.get((s,a,s2),0)

                    new_v += action_prob * env_prob * (r + gamma*values[s2])
            values[s] = new_v
            delta = max(delta, np.absolute(old_v - values[s]))
        if delta < tol:
            break
    return values


### POLICY ITERATION ###
def policy_iter(grid, policy):
    '''
        Perform policy iteration to find the best policy and its value
    '''
    i = 1   
    while True:
        policy_converged = True # flag to check if the policy imporved and break out of the loop
        # evaluate the value function for the older policy
        old_v = value_eval(grid, policy)
        
        # evaluate the new policy
        for s in states:
            v_ls = []
            new_a = ""
            best_v = float("-inf")
            if grid.is_terminal(s):
                continue
            old_a = policy[s]
            for a in ACTION_SPACE:
                v = 0
                for s2 in states:
                    env_prob = transition_probs.get((s,a,s2), 0)
                    reward = rewards.get((s,a,s2), 0)

                    v += env_prob * (reward + gamma*old_v[s2])
                v_ls.append(v) # appending the values for each action 
            v_ls = np.array(v_ls)
            i = np.argmax(v_ls) 
            new_a = list(ACTION_SPACE)[i] # best action
            best_v = v_ls[i] # best value
            policy[s] = new_a
            if new_a != old_a:
                policy_converged = False
        print(i, "th iteration")
        i += 1
        if policy_converged == True:
            break

    return policy

policy_n = policy_iter(grid, policy)
print("values:")
print_in_gridworld(policy_n)
