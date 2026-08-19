import numpy as np

def relu(x): return np.maximum(0, x)

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    # YOUR CODE HERE
    x = np.asarray(x)
    w1 = np.asarray(W1)
    w2 = np.asarray(W2)
    ws = np.asarray(Ws)

    short = x@ws
    h = relu(x@w1)
    z = h@w2

    return relu(z+short)