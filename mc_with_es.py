"""
    This file contains code implementation of the Monte Carlo Policy Control problem with exploring starts.

    Pseudocode:
                Init:
                    returns(s,a) ===> empty list for each (s,a) pair for storing the returns for each episode
                    q(s,a) ===> arbitrary state-action value 
                    pi(s) ===> arbitrary policy

                repeat:
                    get random (s,a) from all sεS and aεA s.t prob(s,a)>0
                    generate an episode using pi having (s,a) as starting state-action pair --- Note: For the starting state we are taking a random action and following policy pi thereafter
                    for all s,a in the episode:
                        G = return for the first visit of s,a
                        append G to returns(s,a)
                        q(s,a) = average(returns(s,a))                                  /////// this is the policy evaluation part
                    
                    for all s in an episode:
                        pi(s) = argmax[a](q(s,a))                                       /////// this is the policy improvement part
"""


import numpy as np
from grid_world import standard_grid,negative_grid
gamma = 0.9
ACTION_SPACE = ("U", "D", "L", "R")
# funtions to print the value function and policy in gridworld
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
    # for printing the policy in the gridworld
    for i in range(g.rows):
        print("\n---------------------------")
        for j in range(g.cols):
            a = P.get((i,j), ' ')
            print("  %s  |" % a, end="")
    print("")
    
def print_values(V, g):
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            v = V.get((i,j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="") # -ve sign takes up an extra space
        print("")

def play_game(grid, policy):
    '''
        function to play the game with randomized starting states, based on a policy and give the returns for each (state,action) pair
    '''
    # selecting a random starting state-action pair
    states = list(grid.actions.keys())
    rand_idx = np.random.choice(range(0,len(states)))
    s = states[rand_idx]
    # setting the start state
    grid.set_state(s) 
    a = np.random.choice(list(ACTION_SPACE)) # choosing random action for starting state
    
    sa_pair_rewards = [(s,a,0)] #  no reward for getting into the starting state 
    seen_states = set()
    seen_states.add(grid.current_state()) #adding the starting state
    num_steps = 0
    # getting the rewards for each state action pair given the policy
    '''
        Mind the timning:
                we are storing s(t), a(t) and r(t) triple
                but r(t) is a result of s(t-1) and r(t-1)
    '''
    while not grid.game_over(): # loop untill the game is not over
        r = grid.move(a) # take action and get the reward
        num_steps += 1 # count for step taken
        s = grid.current_state()

        '''
            This is a little hack to prevent the agent from getting stuck and ending up in an infinitely long episode

            if num_steps == 1 and new_state == old state:
                give a reward of -10 and break
            else:
                reward drops by -10/num of steps taken before reaching that state
        '''
        if s in seen_states:
            reward = -10. / num_steps
            sa_pair_rewards.append((s, None, reward))
            break
        elif grid.game_over():
            sa_pair_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            sa_pair_rewards.append((s, a, r))
            seen_states.add(s)
    
    sa_pair_returns = [] # for storing the returns
    first = True
    G = 0 # for the rewards
    for s,a,r in reversed(sa_pair_rewards):
        if first:
            first = False
        else:
            sa_pair_returns.append((s,a,G))
        G = r + gamma*G
    sa_pair_returns.reverse()
    return sa_pair_returns

def get_best_a_v(d):
  # returns the argmax (key) and max (value) from a dictionary
  # put this into a function since we are using it so often
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

if __name__ == "__main__":
    # defining the gridworld and some variables
    grid = negative_grid(step_cost=-0.9)
    # grid = standard_grid()
    

    
    ##################################################
    ###########          MC CONTROL       ############
    ##################################################
    # getting all state action pairs
    V = {} # for storing the values
    returns = {}
    states = grid.all_states()
    # arbitrary action value function
    Q = {} # q(s,a) 
    '''
        q(s,a) is of the form:
            q[s] = {a1:q, a2:q....}
    '''
    for s in states:
        if s in grid.actions: # not a terminal state
            Q[s] = {}
            for a in ACTION_SPACE:
                Q[s][a] = 0 # needs to be initialized to something so we can argmax it
                returns[(s,a)] = []
        else: # terminal state or state we can't otherwise get to
            pass
   
    ###RANDOM DETERMINISTIC POLICY###
    policy = {}
    # choosing random action from the action space for each policy
    for s in grid.actions.keys():
        policy[s] = np.random.choice(list(ACTION_SPACE))

    deltas = []
    for i in range(2000): # main loop
        # generate an episode using pi
        sa_pair_returns = play_game(grid, policy) 
        seen_states = set()
        for s,a,G in sa_pair_returns: # evaluation
            sa = (s, a)
            if sa in seen_states:
                continue
            
            returns[sa].append(G)
            Q[s][a] = np.mean(returns[sa])
            seen_states.add(sa)
        
        for s in policy.keys(): # improvement
            policy[s] = get_best_a_v(Q[s])[0]
    # find V
    V = {}
    for s in Q.keys():
        V[s] = get_best_a_v(Q[s])[1]
    print("POLICY")
    print_policy(policy, grid)
    print("VALUE FUNCTION")
    print_values(V, grid)   
    # print("Q: ", Q)
    # print("-"*100)
    # print(policy)