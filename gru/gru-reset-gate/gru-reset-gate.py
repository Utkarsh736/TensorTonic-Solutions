import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
          W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """
    Returns the float64 reset-gate values.
    """
    joined = np.concatenate([h_prev, x_t], axis=-1)          # (H+D,) or (N, H+D)
    affine = joined @ W_r.T + b_r                            # (H,) or (N, H)
    r_t = sigmoid(affine)

    return np.asarray(r_t, dtype=np.float64)