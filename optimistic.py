"""
    This script has the code implementation of the Optimistic Initial Values in python:
        Pseudocode:
                    1. Take the best bandit based on the current mean estimations
                    2. Pull its arm
                    3. Update the value of the mean estimates of that bandit
                    4, Run processess 1-3 man_iter number of times 
"""
# importing some libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Maling the class bandit
class Bandit():
    '''
        This class represents a bandit and it has two class funtions for pulling its arm and updating the 
        values after the aarm is pulled.
    '''
    def __init__(self, p_true, p_est):
        self.p_true = p_true  # the true success rate
        self.p_est = p_est # the estimated mean value
        self.N = 1   

    def pull(self):
        '''
            Function to pull the arm of the bandit
        '''
        return np.random.random() < self.p_true # return True with a probability of p_true

    def update(self, x):
        '''
            Function to update parameters
        '''
        self.N += 1
        self.p_est = ((self.N - 1)*self.p_est + x)/ self.N

# defining the variables
iters = 100000 # no of iterations
p_true = [0.45, 0.60, 0.75] # true probabilities of the bandits
p_est = [6, 4, 5] # mean estimatons for bandits
bandits = [Bandit(p_true[0], p_est[0]), Bandit(p_true[1], p_est[1]), Bandit(p_true[2], p_est[2])]

# performing the algorithm
def optimistic():
    n_opt = 0 # number of times optimal bandit was choosen
    n_subopt = 0
    rewards0 = []
    rewards1 = [] # storing rewards for each iteration
    rewards2 = []
    
    best_list = []

    for i in tqdm(range(iters)):
        best_ = np.argmax([b.p_est for b in bandits]) # index of current best bandit
        # pulling the best bandits arm
        pull = bandits[best_].pull()
        # updating the best bandit
        bandits[best_].update(pull)
        # appending in rewards
        # rewards.append(int(pull))
        if best_ == 2:
            rewards2.append(int(pull))
            n_opt += 1
        elif best_ ==1:
            rewards1.append(int(pull))
            n_subopt += 1
        else:
            rewards0.append(int(pull))
            n_subopt += 1

    print("Number of times optimal bandit was pulled:", n_opt)
    print("Number of times sub-optimal bandit was pulled:", n_subopt)
    print(f"For Bandit 1, true_val = {p_true[0]} estimates = {bandits[0].p_est}")
    print(f"For Bandit 2, true_val = {p_true[1]} estimates = {bandits[1].p_est}")
    print(f"For Bandit 3, true_val = {p_true[2]} estimates = {bandits[2].p_est}")

    return rewards0, rewards1, rewards2

# calling the function
rewards0, rewards1, rewards2 = optimistic()

# plotting some plots 
cumm_rewards0 = np.cumsum(rewards0)
reward_cdf0 = cumm_rewards0 / (np.arange(len(rewards0))+1)
cumm_rewards1 = np.cumsum(rewards1)
reward_cdf1 = cumm_rewards1 / (np.arange(len(rewards1))+1)
cumm_rewards2 = np.cumsum(rewards2)
reward_cdf2 = cumm_rewards2 / (np.arange(len(rewards2))+1)

plt.plot(reward_cdf0, label = "Bandit 0")
plt.plot(np.ones(iters)*0.45, label = "Bandit 0 true score")
plt.plot(reward_cdf1, label = "Bandit 1")
plt.plot(np.ones(iters)*0.60, label = "Bandit 1 true score")
plt.plot(reward_cdf2, label = "Bandit 2")
plt.plot(np.ones(iters)*0.75, label = "Bandit 2 true score")
plt.xscale("log")
plt.legend()
plt.show()