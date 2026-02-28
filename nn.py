# Using normal scalars doesn't work; backprop requires knowing operations and nodes that produced certain values
class Value: 
    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return f"Value(v = {self.v})"

    def __add__(self, other):
        res = Value(self.v + other.v)
        return res

    def __mul__(self, other):
        res = Value(self.v * other.v)
        return res

class Node: 
    def __init__(self):
        pass

class Layer: 
    def __init__(self):
        pass

class NN:
    def __init__(self):
        pass