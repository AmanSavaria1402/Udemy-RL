"""
This file contains the code for the windy gridworld environment and its definition
"""

import numpy as np

ACTION_SPACE = ('U', 'D', 'L', 'R')

class WindyGrid: # the environment
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0] # start x and y coordinates stored in different variables
        self.j = start[1]

    def set(self, rewards, actions, probs): 
        '''
            This function sets all the actions and rewards of the gridworld
            Rewards=> dict mapping state(row,col) : rewards
            Actions=> dict mapping state(row, col) : list of all possible actions in that state
            Probs => a dict mapping the state and action to the next state and its probabilities
                    i.e. ..................               {
                                                            (state, action) : {next_state1 : probability, next_state2 : probability ...}
                                                          }
        '''
        self.rewards = rewards
        self.actions = actions
        self.probs = probs

    def set_state(self, s):
        '''
            This function sets the current state to the given state
        '''
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        '''
            Returns the current state of the agent
        '''
        return (self.i, self.j)

    def is_terminal(self, s):
        '''
            Checks whether the given state is terminal or not and returns a
            boolean accordingly
        '''
        return s not in self.actions 
        # if there are no actions, then this might be the terminal state

    def get_next_state(self, s, a):
        # this answers: where would I end up if I perform action 'a' in state 's'?
        self.set_state(s)
        r = self.move(a)
        return (self.i, self.j)

    def move(self, action):
        '''
            Function to move the agent, updates the x and y coordinates acordingly
        '''
        # get current state
        s = (self.i, self.j)
        a = action

        # get the probabilities and the next states
        s_next_dict = self.probs[(s,a)]
        s_next_probs = s_next_dict.values()
        s_next_states = s_next_dict.keys()

        # choose a state based on the probability
        s_next = np.random.choice(s_next_states, p = s_next_probs)

        # update the current state
        self.i, self.j = s_next
            
        # return a reward (if any)
        return self.rewards.get(s_next, 0)


    def undo(self,action): # This is probably not needed anymore
        '''
            Go back one move in the given direction
        '''
        if action == "U":
            self.i += 1
        elif action == "D":
            self.i -= 1
        elif action == "L":
            self.j += 1
        elif action == "R":
            self.j -= 1
        # sanity check - if we arrive at a state that does not exist
        assert(self.current_state() in self.all_states())
        
    def game_over(self):
    # returns true if game is over, else false
    # true if we are in a state where no actions are possible
        return (self.i, self.j) not in self.actions

    def all_states(self):
    # possibly buggy but simple way to get all states
    # either a position that has possible next actions
    # or a position that yields a reward
        return set(self.actions.keys()) | set(self.rewards.keys())


def windy_grid():
  g = WindyGrid(3, 4, (2, 0))
  rewards = {(0, 3): 1, (1, 3): -1}
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }

  # p(s' | s, a) represented as:
  # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
  probs = {
    ((2, 0), 'U'): {(1, 0): 1.0},
    ((2, 0), 'D'): {(2, 0): 1.0},
    ((2, 0), 'L'): {(2, 0): 1.0},
    ((2, 0), 'R'): {(2, 1): 1.0},
    ((1, 0), 'U'): {(0, 0): 1.0},
    ((1, 0), 'D'): {(2, 0): 1.0},
    ((1, 0), 'L'): {(1, 0): 1.0},
    ((1, 0), 'R'): {(1, 0): 1.0},
    ((0, 0), 'U'): {(0, 0): 1.0},
    ((0, 0), 'D'): {(1, 0): 1.0},
    ((0, 0), 'L'): {(0, 0): 1.0},
    ((0, 0), 'R'): {(0, 1): 1.0},
    ((0, 1), 'U'): {(0, 1): 1.0},
    ((0, 1), 'D'): {(0, 1): 1.0},
    ((0, 1), 'L'): {(0, 0): 1.0},
    ((0, 1), 'R'): {(0, 2): 1.0},
    ((0, 2), 'U'): {(0, 2): 1.0},
    ((0, 2), 'D'): {(1, 2): 1.0},
    ((0, 2), 'L'): {(0, 1): 1.0},
    ((0, 2), 'R'): {(0, 3): 1.0},
    ((2, 1), 'U'): {(2, 1): 1.0},
    ((2, 1), 'D'): {(2, 1): 1.0},
    ((2, 1), 'L'): {(2, 0): 1.0},
    ((2, 1), 'R'): {(2, 2): 1.0},
    ((2, 2), 'U'): {(1, 2): 1.0},
    ((2, 2), 'D'): {(2, 2): 1.0},
    ((2, 2), 'L'): {(2, 1): 1.0},
    ((2, 2), 'R'): {(2, 3): 1.0},
    ((2, 3), 'U'): {(1, 3): 1.0},
    ((2, 3), 'D'): {(2, 3): 1.0},
    ((2, 3), 'L'): {(2, 2): 1.0},
    ((2, 3), 'R'): {(2, 3): 1.0},
    ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
    ((1, 2), 'D'): {(2, 2): 1.0},
    ((1, 2), 'L'): {(1, 2): 1.0},
    ((1, 2), 'R'): {(1, 3): 1.0},
  }
  g.set(rewards, actions, probs)
  return g



def windy_grid_penalized(step_cost=-0.1):
  g = WindyGrid(3, 4, (2, 0))
  rewards = {
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
    (0, 3): 1,
    (1, 3): -1
  }
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'D', 'R'),
    (1, 0): ('U', 'D'),
    (1, 2): ('U', 'D', 'R'),
    (2, 0): ('U', 'R'),
    (2, 1): ('L', 'R'),
    (2, 2): ('L', 'R', 'U'),
    (2, 3): ('L', 'U'),
  }

  # p(s' | s, a) represented as:
  # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}
  probs = {
    ((2, 0), 'U'): {(1, 0): 1.0},
    ((2, 0), 'D'): {(2, 0): 1.0},
    ((2, 0), 'L'): {(2, 0): 1.0},
    ((2, 0), 'R'): {(2, 1): 1.0},
    ((1, 0), 'U'): {(0, 0): 1.0},
    ((1, 0), 'D'): {(2, 0): 1.0},
    ((1, 0), 'L'): {(1, 0): 1.0},
    ((1, 0), 'R'): {(1, 0): 1.0},
    ((0, 0), 'U'): {(0, 0): 1.0},
    ((0, 0), 'D'): {(1, 0): 1.0},
    ((0, 0), 'L'): {(0, 0): 1.0},
    ((0, 0), 'R'): {(0, 1): 1.0},
    ((0, 1), 'U'): {(0, 1): 1.0},
    ((0, 1), 'D'): {(0, 1): 1.0},
    ((0, 1), 'L'): {(0, 0): 1.0},
    ((0, 1), 'R'): {(0, 2): 1.0},
    ((0, 2), 'U'): {(0, 2): 1.0},
    ((0, 2), 'D'): {(1, 2): 1.0},
    ((0, 2), 'L'): {(0, 1): 1.0},
    ((0, 2), 'R'): {(0, 3): 1.0},
    ((2, 1), 'U'): {(2, 1): 1.0},
    ((2, 1), 'D'): {(2, 1): 1.0},
    ((2, 1), 'L'): {(2, 0): 1.0},
    ((2, 1), 'R'): {(2, 2): 1.0},
    ((2, 2), 'U'): {(1, 2): 1.0},
    ((2, 2), 'D'): {(2, 2): 1.0},
    ((2, 2), 'L'): {(2, 1): 1.0},
    ((2, 2), 'R'): {(2, 3): 1.0},
    ((2, 3), 'U'): {(1, 3): 1.0},
    ((2, 3), 'D'): {(2, 3): 1.0},
    ((2, 3), 'L'): {(2, 2): 1.0},
    ((2, 3), 'R'): {(2, 3): 1.0},
    ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
    ((1, 2), 'D'): {(2, 2): 1.0},
    ((1, 2), 'L'): {(1, 2): 1.0},
    ((1, 2), 'R'): {(1, 3): 1.0},
  }
  g.set(rewards, actions, probs)
  return g



def grid_5x5(step_cost=-0.1):
  g = WindyGrid(5, 5, (4, 0))
  rewards = {(0, 4): 1, (1, 4): -1}
  actions = {
    (0, 0): ('D', 'R'),
    (0, 1): ('L', 'R'),
    (0, 2): ('L', 'R'),
    (0, 3): ('L', 'D', 'R'),
    (1, 0): ('U', 'D', 'R'),
    (1, 1): ('U', 'D', 'L'),
    (1, 3): ('U', 'D', 'R'),
    (2, 0): ('U', 'D', 'R'),
    (2, 1): ('U', 'L', 'R'),
    (2, 2): ('L', 'R', 'D'),
    (2, 3): ('L', 'R', 'U'),
    (2, 4): ('L', 'U', 'D'),
    (3, 0): ('U', 'D'),
    (3, 2): ('U', 'D'),
    (3, 4): ('U', 'D'),
    (4, 0): ('U', 'R'),
    (4, 1): ('L', 'R'),
    (4, 2): ('L', 'R', 'U'),
    (4, 3): ('L', 'R'),
    (4, 4): ('L', 'U'),
  }
  g.set(rewards, actions)

  # non-terminal states
  visitable_states = actions.keys()
  for s in visitable_states:
    g.rewards[s] = step_cost

  return g
