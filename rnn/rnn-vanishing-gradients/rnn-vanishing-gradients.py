import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> np.ndarray:
    """
    Returns T float64 spectral-norm powers.
    """
    W_hh = np.asarray(W_hh, dtype=np.float64)

    rho = np.linalg.norm(W_hh, ord=2)

    exponents = np.arange(T, dtype=np.float64)
    bounds = np.power(rho, exponents)

    return bounds.astype(np.float64)