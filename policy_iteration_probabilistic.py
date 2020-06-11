'''
    This file contains the code implementation of policy iteration algortihm in the windy gridworld
'''

from windy_gridworld import windy_grid, windy_grid_penalized, ACTION_SPACE
import numpy as np

grid = windy_grid_penalized(-0.5)
tol = 1e-3
gamma = 0.9

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

def init_rewards_and_iteration_probs(grid):
    # init the following
    # transition_probabilities ===> p(s' | s,a)
    # rewards ===> r(s,a,s')
    rewards = {}
    transition_probs = {}
    for (s, a), v in grid.probs.items():
        for s2, p in v.items():
            transition_probs[(s, a, s2)] = p
            rewards[(s, a, s2)] = grid.rewards.get(s2, 0)

    return transition_probs, rewards

# init the rewards and probs
transition_probs, rewards = init_rewards_and_iteration_probs(grid)

def value_eval(grid, policy):
    # evaluate the value given a policy in the gridworld
    # init value
    states = grid.all_states()
    values = {s:0 for s in states}

    # calculate the values
    while True:
        delta = 0
        r = 0
        for s in states:
            if grid.is_terminal(s):
                continue
            v_new = 0 # for storing the updated value
            old_v = values[s]
            for a in ACTION_SPACE:
                action_probs = 1 if policy[s] == a else 0
                for s2 in states:
                    if s2 == s:
                        continue
                    env_probs = transition_probs.get((s,a,s2),0)
                    r = rewards.get((s,a,s2),0)
                    v_new += action_probs * env_probs * (r + gamma*values[s2])
            values[s] = v_new
            delta = max(delta, np.absolute(old_v - values[s]))
        if delta < tol:
            break
    return values

policy = {}
states = grid.all_states()
for s in states: # making a random policy
    policy[s] = np.random.choice(a = list(ACTION_SPACE))
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
                if v > best_v:
                    new_a = a
                    best_v = v
            policy[s] = new_a
            if new_a != old_a:
                policy_converged = False
        print("Iteration", i)
        i += 1
        if policy_converged == True:
            break

    return policy

policy_n = policy_iter(grid, policy)
print("values:")
print_in_gridworld(policy_n)

