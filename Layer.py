from Neuron import Neuron

class Layer: 
    # nout is number of neurons in this layer, nin is num in previous layer
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs
    
    def get_params(self):
        params = []
        for n in self.neurons:
            params.extend(n.get_params())

        return params