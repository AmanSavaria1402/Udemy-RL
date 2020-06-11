"""
    This file contains code implementation of the iterative policy evaluation of the Iterative Policy Evaluation algorithm for a windy gridworld
"""

import numpy as np
from windy_gridworld import windy_grid, ACTION_SPACE

# intializing the gridworld
grid = windy_grid()

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
                        _ = round(_, 2)
                        inv.append(_)
            out.append(inv)
        else:
            out.append(["__" for x in range(9)])

    for i in out:
        print(*i,"\t", end = "\n")   


def print_policy(P, g):
    for i in range(g.rows):
        print("---------------------------")
    for j in range(g.cols):
        a = P.get((i,j), ' ')
        print("  %s  |" % a, end="")
    print("")

# some variables
"""
    NEEDED: 1. p(s' | s, a) = transition probs
            2. p(a | s) = policy
            3. r(s, a, s') = can be found out inside the loop
"""
gamma = 0.9 # the discount factor
states = grid.all_states() # all the states
envt_probs = grid.probs
tol = 1e-3 # tolerance

### probabilistic policy ###
policy = {
    (2, 0): {'U': 0.5, 'R': 0.5},
    (1, 0): {'U': 1.0},
    (0, 0): {'R': 1.0},
    (0, 1): {'R': 1.0},
    (0, 2): {'R': 1.0},
    (1, 2): {'U': 1.0},
    (2, 1): {'R': 1.0},
    (2, 2): {'U': 1.0},
    (2, 3): {'L': 1.0},
  }

# initializing the values
values = {st:0 for st in states}

# doing Iterativepolicty evaluation
def iterative_policy_evaluation():
    '''
        Funtion to do the iterative policy evaluaotion in code
    '''
   
    while True:
        delta = 0 
        old_v = 0
        
        r = 0
        act_dict = {}
        for s in states:
            old_v = values[s]
            new_v = 0
            action_dict = {} # for storing p(s' | s, a) after an action
            if grid.is_terminal(s):
                continue # continue if the state is terminal without updation
                
            act_dict = policy[s] # pi(a | s)
            acts = act_dict.keys() # actions in the policy
            for a in acts:
                action_dict = envt_probs.get((s, a), 0) # dict mapping of p(s' | s, a)
                a_prob = act_dict[a] # prob of action
                for s_pr in action_dict.keys():
                    r = grid.rewards.get(s_pr, 0) # getting the reward
                    s_pr_prob = action_dict[s_pr] # p(s' | s,a)                   
                    new_v += (a_prob * s_pr_prob * (r + (gamma*values[s_pr]))) # bellmans equation
            values[s] = new_v # update the value after the iteration
            delta = max(delta, np.absolute(old_v - values[s]))
        
        if delta < tol:
            break
        print("Maximum difference: ", delta)
        print_in_gridworld(values)
    
    return values

iterative_policy_evaluation()
# print(policy)