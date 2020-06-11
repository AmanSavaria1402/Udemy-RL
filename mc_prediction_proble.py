"""
    This file contains the code implementation of the prediction problem/policy evaluation using the Monte Carlo Methods

    Algorithm:
            pi = policy to be evaluated
            V = arbitrary value function
            returns = empty list to store returns for each episode for all s ∈ all states
            repeat:
                    generate episode using pi
                    for each state appearing in the episode:
                        G = return for the first encounter of s
                        append G to returns(s)
                        V[s] = average(returns(s))

    How to find the returns for each state - Before that we need each state in an episode and its rewards
    ########################################################################################
    Making a list of each state and its reward
        s = set_starting_state
        states_and_rewards = [(s, 0)] # no reward for getting in the first state
        while game is not over:
            a = policy[s]
            r = grid.move(s) # move to the state and get the reward
            s2 = get the current state
            states_and_rewards.append((s2,r))
    ########################################################################################
    Getting the returns for each state
        G = 0 # return for the terminal state 
        returns = []
        for s in reverse order of states:
            returns.append(s,G) # terminal state has a return of 0
            G = r + γ*G
        reverse the returns
    ########################################################################################
"""


from gridworld import standard_grid, ACTION_SPACE
import numpy as np
import matplotlib.pyplot as plt

# making the gridworld environment
grid = standard_grid() 
gamma = 0.9 # discount factor

### fixed policy ###
policy = {
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
# funtion to print the value function in gridworld
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

def init_states_and_rewards(policy):
    # function to make the states and rewards list for each episode
    states_and_rewards = []
    # resetting the game to start at any random state
    states = [st for st in grid.all_states() if not grid.is_terminal(st)] # all possible states the agent can be in
    rand_idx = np.random.choice(len(states)) # a random index 
    grid.set_state(states[rand_idx]) # setting the random state as the start state

    # get the starting state
    s = grid.current_state()
    states_and_rewards.append(((s,0))) # no reward for getting into the starting state
    while not grid.is_terminal(s):
        grid.set_state(s)
        a = policy[s] # get the action
        r = grid.move(a) # move to the next state and get the reward
        s2 = grid.current_state() # next state
        states_and_rewards.append((s2,r))
        s = s2 # update part

    return states_and_rewards

def get_states_and_returns(policy):
    # calculate the returns for each state given the rewards
    returns = [] 
    states_and_rewards = init_states_and_rewards(policy)
    states_and_rewards.reverse() # reverse the list for calculation of the returns and start calculating from the terminal state
    g = 0 # variable for storing the return
    for s,r in states_and_rewards:
        returns.append((s,g)) # return will be 0 for terminal state
        g = r + gamma*g # updating the value of returns
    returns.reverse() # reversing the list
    return returns


def monte_carlo_prediction(policy, N, grid):
    # find the optimal value function using MC methods
    # init a value function
    states = grid.all_states()
    V = {st:0 for st in states}
    
    returns = {}# dict for storing all the returns for each episode
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            returns[s] = 0
    
    # main loop
    for i in range(N):
        # play the episode and get the returns for the first visit to each state
        ret = get_states_and_returns(policy)
        print(ret)
        seen_states = set()
        curr = 0 # for storing the current return value
        for s,g in ret: # looping over each state
            if grid.is_terminal(s):
                continue # we do not need to update the value function of the terminal states as they are already 0
            if s not in seen_states:
                curr = g # first visit return for the state
                # append curr
                returns[s].append(curr)
                # getting the average
                V[s] = np.mean(returns[s])# we only need the first visit returns for calculations
                seen_states.add(s) # adding to the seen states set if it is the first visit
    print_in_gridworld(V)            
    print_policy(policy,grid)

monte_carlo_prediction(policy, 100, grid)