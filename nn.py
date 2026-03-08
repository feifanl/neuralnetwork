import Layer

class NN:
    # nin is the size of the input "layer", nouts is the sizes of every layer
    def __init__(self, nin, nouts):
        # append the input layer to the start
        nn = [nin] + nouts
        self.layers = [Layer(nn[i], nn[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
