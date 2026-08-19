import numpy as np

def relu(x):
    return np.maximum(0, x)

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)

    identity = x
    
    h1 = relu(x@W1.T)
    h2 = h1@W2.T

    out = relu(h2+identity)

    return out
