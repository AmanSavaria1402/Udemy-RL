'''
    This script contains the class bandit that is used for different algorithms in this course
'''
import numpy as np
# making the class
class Bandit():
    '''
        This is the bandit class.
    '''
    
    def __init__(self, p):
        self.p = p # the true probability of the bandit
        self.p_est = 0 # estimated probability of the bandit, this will be estimated using algos
        self.N = 0  # number of times Bandit's arm is pulled

    def pull(self):
        '''
            Function to pull the bandits arm and return the reward
        '''
        return np.random.random() < self.p # returning true with a probability of p

    def update(self, x):
        '''
            This function updates the paramters p_est and N
        '''
        self.N += 1 # updating the value for N
        self.p_est = ((self.N - 1)*self.p_est + x) / (self.N) # updating the value for estimated p


if __name__ == "__main__":
    print("Module script")