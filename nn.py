import numpy as np

# Using normal scalars doesn't work; backprop requires knowing operations and nodes that produced certain values
class Value: 
    def __init__(self, v, _children, _op, label = ""):
        self.v = v
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(v = {self.v})"

    def __add__(self, other):
        res = Value(self.v + other.v, (self, other), '+')
        return res

    def __mul__(self, other):
        res = Value(self.v * other.v, (self, other), '*')
        return res

class Neuron:
    def __init__(self, n_weights):
        self.weights = [Value(np.random.rand(), (), '', f'w{i}') for i in range(n_weights)]
        self.bias = Value(np.random.rand(), (), '', 'b')

class Layer: 
    def __init__(self):
        pass

class NN:
    def __init__(self):
        pass