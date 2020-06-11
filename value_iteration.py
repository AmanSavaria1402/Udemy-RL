'''
    The file contains code implementation of the value iteration algorithm
'''

import numpy as np
from gridworld import standard_grid, ACTION_SPACE

grid = standard_grid() # defining the standard 3*4 grid
gamma=  0.9
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

def init_rewards_and_transition_probs():
    '''
        Initialize the rewardsand the transition probabilities dictionary
        rewards ===> r(s,a,s') 
        transition probabilities ===> p(s' | s,a) - key = (s,a,s') --- in this case, 1 since the gridworld is deterministic
    '''
    rewards = {}
    transition_probs = {}

    for s in grid.all_states():
        if grid.is_terminal(s):
            continue
        for a in ACTION_SPACE:
            s2 = grid.get_next_state(s,a)
            transition_probs[(s,a,s2)] = 1
            if s2 in grid.rewards:
                rewards[(s,a,s2)] = grid.rewards.get(s2,0)

    return transition_probs, rewards

# getting the transition probs and rewards
transition_probs, rewards = init_rewards_and_transition_probs()

def value_iteration():
    # function to perform value iteration in the gridworld environment
    # init a value
    states = grid.all_states()
    V = {st:0 for st in states}
    policy = {st:"" for st in states} # dict to store policy in
    i = 1

    # main loop
    while True:
        delta = 0
        print("Iteration: ", i)
        for s in states:                 
            if grid.is_terminal(s):
                continue
            v_old = V[s]
            val_ = [] # store the values for each action
            for a in ACTION_SPACE:
                v_new = 0                
                for s2 in states: 
                    env_probs = transition_probs.get((s,a,s2), 0)
                    r = rewards.get((s,a,s2), 0)
                    # updating the value for the action
                    v_new += env_probs * (r + gamma*V[s2])
                val_.append(v_new)          
            i = np.argmax(val_) # the index of maximum value
            best_v = val_[i] # getting the best value
            V[s] = best_v # updating the best value for the state
            best_a = list(ACTION_SPACE)[i] # the best action for the state
            delta = max(delta, np.absolute(v_old - V[s]))
            # store the best action in the policy
            policy[s] = best_a
        print("delta: ", delta)
        print_in_gridworld(policy)
        i += 1
        # breaking condition
        if delta < tol:
            break


value_iteration()

