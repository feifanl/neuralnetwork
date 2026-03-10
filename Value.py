import numpy as np
import math

# Value class instead of scalars; backprop requires knowing operations and nodes that produced values
class Value: 
    def __init__(self, v, _children = (), _op = '', label = ''):
        self.v = v
        # Gradient initialized to 0
        self.grad = 0.0
        self._backward = lambda: None

        # Nodes that point to this one
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(v = {self.v})"

    # Operator overloading
    def __add__(self, other):
        # Handle case where other is not a Value object
        other = other if isinstance(other, Value) else Value(other)

        res = Value(self.v + other.v, (self, other), '+')

        def _backward():
            self.grad += res.grad
            other.grad += res.grad
        
        res._backward = _backward

        return res

    # If int + self, Python looks for radd otherwise it'll break because it doesn't know how to do int + Value
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + other * -1
    
    # Same case as radd
    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        # Handle case where other is not a Value object
        other = other if isinstance(other, Value) else Value(other)

        res = Value(self.v * other.v, (self, other), '*')

        def _backward():
            self.grad += other.v * res.grad
            other.grad += self.v * res.grad

        res._backward = _backward

        return res
    
    # Same logic as radd
    def __rmul__(self, other):
        return self * other
    
    # Can be represented as multiplication with 1/other
    def __truediv__(self, other):
        return self * other ** -1
    
    # Only if other is a scalar
    def __pow__(self, other):
        res = Value(self.v ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * self.v ** (other - 1) * res.grad

        res._backward = _backward
        
        return res

    # e^self
    def exp(self):
        res = Value(math.exp(self.v), (self, ), 'exp')

        def _backward():
            self.grad += math.exp(self.v) * res.grad

        res._backward = _backward

        return res
    
    def tanh(self):
        x = self.v
        t = (math.exp(2 * x) - 1)/(math.exp(2 * x) + 1)
        res = Value(t, (self, ), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * res.grad

        res._backward = _backward

        return res
    
    # Using ReLU as activation function to avoid vanishing gradient problem (really large/really small inputs cause gradient of tanh to approach 0)
    def relu(self): 
        x = self.v if self.v > 0 else 0
        res = Value(x, (self, ), "ReLU")

        def _backward():
            self.grad += (1 if x > 0 else 0) * res.grad
        
        res._backward = _backward

        return res
    
    # Log function for loss 
    def log(self):
        res = Value(math.log(self.v), (self, ), "log")

        def _backward():
            self.grad += 1 / self.v * res.grad

        res._backward = _backward

        return res
    
    # Backprop
    def backward(self): 
        topo = []
        visited = set()

        def build_topo(value):
            if value not in visited:
                visited.add(value)
                for child in value._prev:
                    build_topo(child)
                # Only add once all of children are in
                topo.append(value)
        build_topo(self)

        # Gradient with respect to oneself is 1
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()