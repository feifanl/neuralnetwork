# cross-entropy loss
def compute_loss(probs, label):
    return probs[label].log() * -1

def accuracy():
    pass