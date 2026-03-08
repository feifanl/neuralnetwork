import numpy as np
import random
import Value

# Neuron class to hold the weights, bias, and value of each neuron
class Neuron:
    def __init__(self, n_weights):
        # Random weights and bias at initialization
        self.weights = [Value(random.uniform(-1, 1), label = f'w{i}') for i in range(n_weights)]
        self.bias = Value(random.uniform(-1, 1), label = 'b')

    def __call__(self, x): 
        activation = sum(weight * value for weight, value in zip(self.weights, x)) + self.bias
        
        return activation.tanh()