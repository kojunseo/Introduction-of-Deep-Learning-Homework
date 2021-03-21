import numpy as np

"""
DO NOT EDIT ANY PARTS OTHER THAN "EDIT HERE" !!!

[Description]
__init__ - Initialize necessary variables for optimizer class
input   : gamma, epsilon
return  : X

update   - Update weight for one minibatch
input   : w - current weight, grad - gradient for w, lr - learning rate
return  : updated weight
"""

class SGD:
    def __init__(self, gamma, epsilon):
        self.gamma = gamma
        self.epsilon = epsilon

    def update(self, w, grad, lr):
        updated_weight = None
        updated_weight = w - (grad * lr)
        return updated_weight
