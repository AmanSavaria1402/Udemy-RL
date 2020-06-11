'''
This is the code file for the IterativePpolicyIteration algoritm for finding out the optimal value function

NOTE :  1. The environment is deterministic, all the actions will have only one result, ie , what it is intended to do
        2. We are finding the optimal value function by following a give policy.
        3. Policy will be of two types  i. Fixed
                                        ii. Randomly distributed

PSEUDOCODE : 
init values for each state
del = 0
while True: # infinte loop
    for s in states:
        evaluate the value for state
        del = max(0, |new_val-old_val|)
    if del < threshold:
        break
return values

TODO: Run all the code and analyse it.
'''
import numpy as np
from gridworld import standard_grid
import prettytable as pt
import math

# importing the satndard grid and defining some variables
grid = standard_grid() # the standard grid has rewards only at the terminal states and 0 rewards for all other states
rewards = grid.rewards
tolerance = 1e-3 # the tolerance

gamma = 1.0     # the gamma value

# initialize value for each state
states = grid.all_states()
values = {st:0.00 for st in states} # set to be 0 for each state

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

### fixed policy ###
policy_d = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }
### uniform policy ###
'''
    This policy will have a uniform probability for each action and the probability can be given by 1/no of actions for each state
'''
policy_p = grid.actions 

###DOING THE ITERATIVE POLICY EVALUATION-for determinsitic policy###
def IPI_deterministic():
    """
            Function to do the Iterative Policy evaluation for a determinsitic policy
    """
    while True:
        delta = 0      # for storing the difference
        old_v = 0
        r = 0
        a = ""
        for s in states: # looping over all the states
            
            if grid.is_terminal(s):
                continue # terminal values dont need updation as they are already equal to 0
            a = policy_d[s]   
            old_v = values[s] # old value of the state 
            grid.set_state(s)
            r = grid.move(a) # reward for the state
            values[s] = r + (gamma * values[grid.current_state()]) # updating the value based on the bellman equation
            delta = max(delta, np.abs(old_v - values[s]))
        if delta <= tolerance:
            break
    
    return values

###DOING THE ITERATIVE POLICY EVALUATION-for probabilistic policy###
def IPI_probabilistic():
    """
        Function to do the Iterative Policy Evaluation for probabilistic policy
    """
    while True:
            delta = 0      # for storing the difference
            old_v = 0
            r = 0
            for s in states: # looping over all the states
                new_v = 0
                if grid.is_terminal(s):
                    continue # terminal values dont need updation as they are already equal to 0
                policy_prob = 1/len(policy_p[s]) #the probability for each action
                act = policy_p[s]   
                old_v = values[s] # old value of the state 

                for a in act:   
                    grid.set_state(s)
                    r = grid.move(a) # reward for the state
                    new_v += policy_prob * (r + (gamma * values[grid.current_state()])) # updating the value based on the bellman equation
                values[s] = new_v
                delta = max(delta, np.abs(old_v - values[s]))

            if delta <= tolerance:
                break
        
    return values


########################################################################################################################################################
if __name__ == "__main__":
    # doing the iteration and printing for deterministic policy
    IPI_probabilistic()
    print_in_gridworld(values)
