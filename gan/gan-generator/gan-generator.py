import numpy as np

def generator(z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns generated samples as a float64 array with shape (B, D).
    """
    affine_out = np.matmul(z,W) + b

    out = np.tanh(affine_out)

    return out