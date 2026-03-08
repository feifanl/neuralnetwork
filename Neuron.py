import numpy as np
import Value

# Neuron class to hold the weights, bias, and value of each neuron
class Neuron:
    def __init__(self, n_weights):
        # Random weights and bias at initialization
        self.weights = [Value(np.random.rand(), label = f'w{i}') for i in range(n_weights)]
        self.bias = Value(np.random.rand(), label = 'b')