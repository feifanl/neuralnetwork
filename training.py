import math
import Value
import Neuron

def forward_pass():
    pass

def backward_pass():
    pass

def backprop(root):
    # Gradient of root with respect to root is just 1 
    root.grad = 1

# cross-entropy loss
def loss(probs):
    return -math.log(probs[label])

def accuracy():
    pass