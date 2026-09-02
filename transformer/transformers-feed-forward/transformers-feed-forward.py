import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    hidden = np.dot(x,W1) + b1

    relu_ = np.maximum(0, hidden)

    op = np.dot(relu_, W2) + b2

    return op