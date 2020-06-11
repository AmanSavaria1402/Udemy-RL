'''
        Tasks:
            1. Create a class CasinoMachine that returns the score for winning or losing based on a probaility given by the user.
            2. Make 2 objects with different winning scores
            3. Use the epsilon-greedy method to estimate the better casino machine
'''

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# defining the casino machine class
class CasinoMachine:
    def __init__(self, p):   
        self.p = p # the actual win rate
        self.p_estimate = 0 # the current estimate 
        self.N = 0 # the number of iterations completed

    def pull(self): # pulling the machine arm
        # return score 1 having a probability of p
        return np.random.random() < self.p

    def update(self, x): # update the mean score and the value of N
        self.N += 1 # updating the no of samples
        self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N # updating the mean score

# defining some variables
iters = [10000, 50000, 100000]
p_vals = [0.35, 0.50, 0.75]
eps = [0.01, 0.05, 0.1] # the different values of each epsilon

# Function for greedy-epsilon algorithm
def greedy_epsilon(eps, steps):
    """
        This Function finds the correct winning rate of the three casino machines and gives out the reward of pulling
        the arm at each epoch and the winning rate estimation of each machines.
        It does this by using the greedy-epsilon algorithm.
    """
    rewards = np.zeros(steps) # array that stores the rewards for each iteration
    n_corr_bandit = 0 # no of times the correct bandit was choosen
    explor = 0 # no of times algorithm eplored
    exploit = 0 # no of times algorithm exploited
    x = 0
    ch = 0
    iter_scores = np.zeros(3) # list to store the current probability scores for each
    
    # defining the three machines
    machine_list = [CasinoMachine(p_vals[0]), CasinoMachine(p_vals[1]), CasinoMachine(p_vals[2])]
    machine = '' # storing the machine object
   
    
    for i in tqdm(range(steps)):
        # declaring x
        x = np.random.uniform() # the random value
        ch = 2 # best bandit

        # randomly choosing one of the machines when x < eps
        if x < eps:
            ch = np.random.randint(3) 
            machine = machine_list[ch] # selecting the machine randomly
            explor+=1
        
        else: # if x > eps and its not the first iteration
            ch = np.argmax(iter_scores) # choosing the machine with the highest estimated p (greedy choice)
            machine = machine_list[ch]
            exploit+=1

        # pulling the machine arm
        pull = int(machine.pull()) # 1 if won 0 if lost

        # updating the score of the machine
        machine.update(pull)
        # updating the rewards
        rewards[i] = pull        
        # changing the score value
        iter_scores[ch] = machine.p_estimate
        
        # No of times the correct bandit ran
        if ch == np.argmax(iter_scores):
            n_corr_bandit+=1
        
    # printing some details
    print("*"*100)
    print("EPSILON VALUE:", eps)
    print("Correct", n_corr_bandit)
    print("Explored:",explor)
    print("Exploited:",exploit)
    print("*"*100)
    # Comparing the estimated to actual scores
    print(f"Machine1 actual = {p_vals[0]} estimated = {iter_scores[0]}")
    print(f"Machine2 actual = {p_vals[1]} estimated = {iter_scores[1]}")
    print(f"Machine3 actual = {p_vals[2]} estimated = {iter_scores[2]}")
    print("*"*100)

    return rewards # return the rewards

# making an array for storing the rewards in each list
reward_array = []

for i in iters:
    for e in eps:
        reward_array.append(greedy_epsilon(e, i))

reward_array = np.array(reward_array)
# plotting the curves

"""
        Its like this ((10k, 50k, 100k), .01), ((10k, 50k, 100k), .05), ((10k, 50k, 100k), 0.1)

"""
iter_nums = [10000]*3 + [50000]*3 + [100000]*3
cumm_reward_array = np.array([np.cumsum(reward_array[rar]) for rar in range(9)])
win_rate = np.array([cumm_reward_array[i]/(np.arange(iter_nums[i]) + 1) for i in range(9)])
####################################################################3
# plotting the epsilons
plt.subplot(121)
plt.title("For EPSILON and 50k EPOCHS")
plt.plot(win_rate[1], label = f"Epsilon={eps[0]}")
plt.plot(win_rate[4], label = f"Epsilon={eps[1]}")
plt.plot(win_rate[7], label = f"Epsilon={eps[2]}")
plt.legend()
plt.xscale("log")
####################################################################
# plotting the epochs
plt.subplot(122)
plt.title("FOR EPOCHS and EPSILON = 0.05") 
plt.plot(win_rate[3], label = f"Epsilon={iters[0]}")
plt.plot(win_rate[4], label = f"Epsilon={iters[1]}")
plt.plot(win_rate[5], label = f"Epsilon={iters[2]}")
plt.legend()
plt.xscale("log")
####################################################################
plt.show()

