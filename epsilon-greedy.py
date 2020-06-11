'''
        Tasks:
            1. Create a class CasinoMachine that returns the score for winning or losing based on a probaility given by the user.
            2. Make 2 objects with different winning scores
            3. Use the epsilon-greedy method to estimate the better casino machine
'''

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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
max_iter = 10000
p_vals = [0.35, 0.50, 0.75]
EPS = 0.1

# Function for greedy-epsilon algorithm
def greedy_epsilon():
    rewards = np.zeros(max_iter) # array that stores the rewards for each iteration
    n_corr_bandit = 0 # no of times the correct bandit was choosen
    explor = 0 # no of times algorithm eplored
    exploit = 0 # no of times algorithm exploited
    x = 0
    ch = 0
    iter_scores = np.zeros(3) # list to store the current probability scores for each
    
    # defining the three machines
    machine_list = [CasinoMachine(p_vals[0]), CasinoMachine(p_vals[1]), CasinoMachine(p_vals[2])]
    machine = '' # storing the machine object
   
    
    for i in tqdm(range(max_iter)):
        # declaring x
        x = np.random.uniform() # the random value
        ch = 2 # best bandit

        # randomly choosing one of the machines when x < EPS
        if x < EPS:
            ch = np.random.randint(3) 
            machine = machine_list[ch] # selecting the machine randomly
            explor+=1
        
        else: # if x > EPS and its not the first iteration
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
    print("Correct", n_corr_bandit)
    print("Explored:",explor)
    print("Exploited:",exploit)
    return iter_scores, rewards

iter_scores, rewards = greedy_epsilon()
# Comparing the estimated to actual scores
print(f"Machine1 actual = {p_vals[0]} estimated = {iter_scores[0]}")
print(f"Machine2 actual = {p_vals[1]} estimated = {iter_scores[1]}")
print(f"Machine3 actual = {p_vals[2]} estimated = {iter_scores[2]}")

# plotting the curves
import seaborn as sns
cumm_rewards = np.cumsum(rewards)
win_rate = cumm_rewards/(np.arange(max_iter) + 1)
plt.plot(win_rate)
plt.plot(np.ones(max_iter)*0.75)
plt.show()
