from nn import Value, Neuron

def backprop(root):
    # Gradient of root with respect to root is just 1 
    root.grad = 1

    