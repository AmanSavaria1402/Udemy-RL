'''
    This script contains code implementation for the Upper Confiddence Bound algorithm
'''

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit

# declaring some variables
p_vals =  [0.45, 0.60, 0.75] # list for the probabilities of each bandit
bandits  = [Bandit(p) for p in p_vals] # list containing the bandits
iters = 50000 # number of iterations

# defining the function
def ucb1():
    total_plays = 0 # for storing the total number of plays
    rewards = [] # for storing the rewards after each epoch
    explored = 0
    exploited = 0
    optimal_bandit = 0 # for counting pulls of optimal bandit

    # running every bandit once
    total_plays = 3
    pulls = [b.pull() for b in bandits] # pulling each bandits arm
    # updating each bandit
    for i, b in enumerate(bandits):
        b.update(pulls[i])


    # for i in range(iters):
    #     # finding the optimal bandit for cureent epoch
    #     '''
    #         this is found out b using the following formula     
    #                             j = argmax(X_bar + sqrt(2*logN/nj))

    #     '''
        
ucb1()
print([b.N for b in bandits])    

