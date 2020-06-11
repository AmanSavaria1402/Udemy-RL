"""
This file contains the code for the gridworld environment and its definition
"""

ACTION_SPACE = {"U" ,"D", "L", "R"}

class Grid: # the environment
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0] # start x and y coordinates stored in different variables
        self.j = start[1]

    def set(self, rewards, actions): 
        '''
            This function sets all the actions and rewards of the gridworld
            Rewards=> dict mapping state(row,col) : rewards
            Actions=> dict mapping state(row, col) : list of all possible actions in that state
        '''
        self.rewards = rewards
        self.actions = actions

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

    def move(self, action):
        '''
            Function to move the agent, updates the x and y coordinates acordingly
        '''
        # check if legal move first
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        # return a reward (if any)
        return self.rewards.get((self.i, self.j), 0)


    def undo(self,action):
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
    
    def get_next_state(self, s, a):
        # this answers: where would I end up if I perform action 'a' in state 's'?
        self.set_state(s)
        self.move(a)
        return (self.i, self.j)
        

    def all_states(self):
    # possibly buggy but simple way to get all states
    # either a position that has possible next actions
    # or a position that yields a reward
        return set(self.actions.keys()) | set(self.rewards.keys())

    def check_act_avail(self, a, s):
        '''
        checks whether an action can be taken
        '''
        return a in self.actions[s[0], s[1]]


def standard_grid():
  # define a grid that describes the reward for arriving at each state
  # and possible actions at each state
  # the grid looks like this
  # x means you can't go there
  # s means start position
  # number means reward at that state
  # .  .  .  1
  # .  x  . -1
  # s  .  .  .
  g = Grid(3, 4, (2, 0))
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
  g.set(rewards, actions)
  return g


def negative_grid(step_cost=-0.1):
  # in this game we want to try to minimize the number of moves
  # so we will penalize every move
  g = standard_grid()
  g.rewards.update({
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
  })
  return g

