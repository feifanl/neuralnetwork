# Neural Network

Constructing a neural network without PyTorch to learn how to implement backpropagation, gradient descent, and learn more about model architecture.

### Installation
```bash
pip install -r requirements.txt
```

### Results
Model was successful in classifying tiny datasets and simple inputs.

I then tried to classify MNIST images using a 3-hidden-layer MLP. The training time was incredibly slow and the model seemed to cap out ~1.5 loss and ~25-30% accuracy. Changing model parameters yielded some better results, but it's clear that using just Python without matrix multiplication and vectorization is inefficient.