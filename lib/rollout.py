import random


##########################################################################
########                        TASK 0                            ########
##########################################################################
# Implement ReplayBuffer class. See docstrings for details               #

class ReplayBuffer(object):
    def __init__(self, capacity):
        """
        Arguments:
            capacity: Max number of elements in buffer
        """
        self.capacity = capacity
        self.buffer = []
        
    def push(self, s0, a, s1, r, d):
        """Push an element to the buffer.

        Arguments:
            s0: State before action
            a: Action picked by the agent
            s1: State after performing the action
            r: Reward recieved is state s1.
            d: Whether the episode terminated after in the state s1.

        If the buffer is full, start to rewrite elements
        starting from the oldest ones.
        """
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0, a, s1, r, d))

    def sample(self, batch_size):
        """Return `batch_size` randomly chosen elements."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return size of the buffer."""
        return len(self.buffer)

##########################################################################
########                        TASK 0                            ########
##########################################################################
