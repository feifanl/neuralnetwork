import numpy as np

# Value class instead of scalars; backprop requires knowing operations and nodes that produced values
class Value: 
    def __init__(self, v, _children = (), _op = '', label = ''):
        self.v = v
        # Nodes that point to this one
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(v = {self.v})"

    # Operator overloading
    def __add__(self, other):
        res = Value(self.v + other.v, (self, other), '+')
        return res

    def __mul__(self, other):
        res = Value(self.v * other.v, (self, other), '*')
        return res

# Neuron class to hold the weights, bias, and value of each neuron
class Neuron:
    def __init__(self, n_weights):
        # Random weights and bias at initialization
        self.weights = [Value(np.random.rand(), label = f'w{i}') for i in range(n_weights)]
        self.bias = Value(np.random.rand(), label = 'b')

class Layer: 
    def __init__(self):
        pass

class NN:
    def __init__(self):
        pass