import math
from Value import Value, Neuron

def backprop(root):
    # Gradient of root with respect to root is just 1 
    root.grad = 1

# cross-entropy loss
def loss(probs):
    return -math.log(probs[label])